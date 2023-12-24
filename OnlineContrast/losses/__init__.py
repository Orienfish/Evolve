import torch
from .supcon import SupConLoss, IRDLoss
from .simclr import SimCLRLoss
from .cka import CKALoss
from .barlowtwins import BarlowTwinsLoss
from .byol import BYOL
from .vicreg import VICRegLoss
from .simsiam import SimSiamLoss

def get_loss(opt):
    if opt.criterion == 'simclr':
        criterion = SimCLRLoss(model=opt.model,
                               lifelong_method=opt.lifelong_method,
                               temperature=opt.temp_cont)
    elif opt.criterion == 'supcon':
        criterion = SupConLoss(stream_bsz=opt.batch_size,
                               model=opt.model,
                               temperature=opt.temp_cont)
    elif opt.criterion == 'cka':
        criterion = CKALoss()
    elif opt.criterion == 'barlowtwins':
        criterion = BarlowTwinsLoss(model=opt.model,
                                    lifelong_method=opt.lifelong_method)
    elif opt.criterion == 'byol':
        criterion = BYOL(model=opt.model,
                         lifelong_method=opt.lifelong_method)
    elif opt.criterion == 'vicreg':
        criterion = VICRegLoss(model=opt.model,
                               lifelong_method=opt.lifelong_method)
    elif opt.criterion == 'simsiam':
        criterion = SimSiamLoss(model=opt.model,
                                lifelong_method=opt.lifelong_method)
    else:
        raise ValueError('loss method not supported: {}'.format(opt.criterion))

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    criterion_reg = IRDLoss(projector=criterion.projector,
                            current_temperature=opt.current_temp,
                            past_temperature=opt.past_temp)
    if torch.cuda.is_available():
        criterion_reg = criterion_reg.cuda()

    criterion_expert = None
    if opt.distill_method == 'cka':
        criterion_expert = CKALoss()
    elif opt.distill_method == 'barlowtwins':
        criterion_expert = BarlowTwinsLoss(model=opt.model)
    elif opt.distill_method == 'vicreg':
        criterion_expert = VICRegLoss(model=opt.model)
    elif opt.distill_method == 'simsiam':
        criterion_expert = SimSiamLoss(model=opt.model)

    if criterion_expert is not None and torch.cuda.is_available():
        criterion_expert = criterion_expert.cuda()

    return criterion, criterion_reg, criterion_expert
