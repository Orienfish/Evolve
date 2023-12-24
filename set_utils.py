import math
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from networks.resnet_big import ResNet, Bottleneck, BasicBlock
from networks.resnet_pnn import resnet18_pnn
from networks.resnet_big import ConvEncoder
from networks.vision_transformer import VisionTransformer
from networks.swin_transformer import SwinTransformer, SwinTransformerBlockV2, PatchMergingV2

dataset_num_classes = {
    'nabird': 555,
    'oxford_pets': 37,
    'cub200': 200,
    'caltech101': 101,
    'stanford_dogs': 120,
    'voc2007': 21,
    'cifar10': 10,
    'cifar100': 20,
    'imagenet': 1000,
    'tinyimagenet': 100,  # temp setting
    'stream51': 51,
    'core50': 50,
    'mnist': 10
}

def create_model(model_type: str,
                 method: str,
                 dataset: str,
                 **kwargs):
    if model_type == 'resnet18':
        if method == 'pnn':
            model = resnet18_pnn(dataset_num_classes[dataset])
        else:
            model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif model_type == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif model_type == 'resnet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif model_type == 'resnet101':
        ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif model_type == 'cnn':
        model = ConvEncoder()
    else:
        raise ValueError(model_type)

    return model


def load_student_backbone(model_type: str,
                          method: str,
                          dataset: str,
                          ckpt: str,
                          **kwargs):
    """
    Load student model of model_type and pretrained weights from ckpt.
    """
    # Set models
    model = create_model(model_type, method, dataset, **kwargs)

    model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    if ckpt is not None:
        state = torch.load(ckpt)
        state_dict = {}
        for k in list(state.keys()):
            if k.startswith("fc."):
                continue
            state_dict['module.' + k] = state[k]
            del state[k]
        model.load_state_dict(state_dict)

    return model


def load_teacher_backbone(model_path: str,
                          image_size: int,
                          num_teachers: int,
                          **kwargs):
    """
    Load all pretrained models in the model_path.
    Note, that all models are just backbone without the final Linear layer.
    """
    pretrained_model_path = {
        'imageNet_v2': 'resnet50-11ad3fa6.pth',
        'swin_b': 'swin_b-68c6b09e.pth',
        'moco_v1': 'moco_v1_200ep_pretrain.pth.tar',
        'moco_v2_200ep': 'moco_v2_200ep_pretrain.pth.tar',
    }

    assert len(pretrained_model_path) >= num_teachers, \
        "Insufficient number of experts!"

    teacher_models = {}
    for name in list(pretrained_model_path.keys())[:num_teachers]:

        if name == 'vit_b_32' and image_size != 224:
            # The vit_b_32 model can only work if image size is 224
            continue

        print('Loading teacher model {}...'.format(name))
        pkl = torch.load(os.path.join(model_path, pretrained_model_path[name]))
        state_dict = {}

        if 'imageNet' in name:
            model = create_model('resnet50', **kwargs)
            for k, v in pkl.items():
                if k.startswith("fc."):
                    continue
                state_dict['module.' + k] = v

        elif 'moco' in name:
            model = create_model('resnet50', **kwargs)
            pkl = pkl['state_dict']
            state_dict = {}
            for k, v in pkl.items():
                if not k.startswith("module.encoder_q."):
                    continue
                k = k.replace("module.encoder_q.", "")
                if k.startswith("fc."):
                    continue
                state_dict['module.'+ k] = v

        elif 'maskrcnn' in name:
            model = create_model('resnet50', **kwargs)
            for k, v in pkl.items():
                if not k.startswith("backbone.body."):
                    continue
                k = k.replace("backbone.body.", "")
                state_dict['module.' + k] = v

        elif 'deeplab' in name:
            model = create_model('resnet50', **kwargs)
            for k, v in pkl.items():
                if not k.startswith("backbone."):
                    continue
                k = k.replace("backbone.", "")
                state_dict['module.' + k] = v

        elif 'keyPoint' in name:
            model = create_model('resnet50', **kwargs)
            for k, v in pkl.items():
                if not k.startswith("backbone.body."):
                    continue
                k = k.replace("backbone.body.", "")
                state_dict['module.' + k] = v

        elif name == 'vit_b_32':
            model = VisionTransformer(image_size=224, # Need to use 224x224 image input
                                      patch_size=32,
                                      num_layers=12,
                                      num_heads=12,
                                      hidden_dim=768,
                                      mlp_dim=3072,
                                      **kwargs)
            for k, v in pkl.items():
                if k.startswith("heads."):
                    continue
                state_dict['module.' + k] = v

        elif name == 'swin_b':
            model = SwinTransformer(patch_size=[4, 4],
                                    embed_dim=128,
                                    depths=[2, 2, 18, 2],
                                    num_heads=[4, 8, 16, 32],
                                    window_size=[7, 7],
                                    stochastic_depth_prob=0.5,
                                    **kwargs)
            for k, v in pkl.items():
                if k.startswith("head."):
                    continue
                state_dict['module.' + k] = v

        elif name == 'swin_v2_b':
            model = SwinTransformer(patch_size=[4, 4],
                                    embed_dim=128,
                                    depths=[2, 2, 18, 2],
                                    num_heads=[4, 8, 16, 32],
                                    window_size=[8, 8],
                                    stochastic_depth_prob=0.5,
                                    block=SwinTransformerBlockV2,
                                    downsample_layer=PatchMergingV2,
                                    **kwargs)
            for k, v in pkl.items():
                if k.startswith("head."):
                    continue
                state_dict['module.' + k] = v


        model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model = model.cuda()
            cudnn.benchmark = True

        model.load_state_dict(state_dict)
        teacher_models[name] = model

    return teacher_models


def set_model(opt):
    # Load student model, with pretrained weights if any
    model = load_student_backbone(opt.model,
                                  opt.lifelong_method,
                                  opt.dataset,
                                  opt.ckpt)

    # Load teacher models
    teacher_models = {}
    if opt.expert_power > .0:
        teacher_models = load_teacher_backbone(opt.ckpt_folder,
                                               opt.size,
                                               opt.num_experts)

    return model, teacher_models

def set_constant_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(lr, momentum, weight_decay, model, **kwargs):
    parameters = [{
        'name': 'backbone',
        'params': [param for name, param in model.named_parameters()],
    }]

    # Include the model as heads as part of the parameters to optimize
    if 'criterion' in kwargs:
        parameters = [{
            'name': 'backbone',
            'params': [param for name, param in model.named_parameters()],
        }, {
            'name': 'heads',
            'params': [param for name, param in kwargs['criterion'].named_parameters()],
        }]

    optimizer = optim.SGD(parameters,
                          lr=lr,
                          momentum=momentum,
                          weight_decay=weight_decay)
    return optimizer
