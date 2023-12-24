import os
import types
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer, seed_everything
# from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchvision.models import resnet18, resnet50

from cassle.args.setup import parse_args_linear

try:
    from cassle.methods.dali import ClassificationABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True
from cassle.methods.linear import LinearModel
#from cassle.utils.classification_dataloader import prepare_data
#from cassle.utils.checkpointer import Checkpointer
from data_utils import set_linear_loader
from set_utils import load_teacher_backbone

# The following lines are required to avoid errors during torch.load(),
# because torch.load() require the model module to be in the same folder
# More info is here: https://github.com/pytorch/pytorch/issues/3678
import sys
sys.path.insert(0, './OnlineContrast')
sys.path.insert(0, './UCL')

def main():
    seed_everything(5)

    parser = argparse.ArgumentParser('additional argument')
    parser.add_argument('--data_folder', type=str, default='./datasets/', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--train_samples_ratio', type=float, default=1.0,
                        help="the ratio of total training samples used in training")
    parser.add_argument('--test_samples_ratio', type=float, default=1.0,
                        help="the ratio of total testing samples used in testing")
    parser.add_argument('--train_test_split', type=float, default=0.8,
                        help="the ratio of train/test split")
    parser.add_argument('--load_at_beginning', default=False,
                        action="store_true",
                        help="whether to load all data at the beginning, only used in core50 and stream51")
    
    parser.add_argument('--ckpt_folder', type=str, default='../pretrained_models',
                        help='path to the folder locating pre-trained models')
    
    args = parse_args_linear(parser)

    # split classes into tasks
    tasks = None
    #if args.split_strategy == "class":
    #    assert args.num_classes % args.num_tasks == 0
    #    tasks = torch.randperm(args.num_classes).chunk(args.num_tasks)

    if args.encoder == "resnet18":
        backbone = resnet18()
    elif args.encoder == "resnet50":
        backbone = resnet50()
    else:
        raise ValueError("Only [resnet18, resnet50] are currently supported.")

    #if args.cifar:
    #    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    #    backbone.maxpool = nn.Identity()

    backbone.fc = nn.Identity()

    #assert (
    #    args.pretrained_feature_extractor.endswith(".ckpt")
    #    or args.pretrained_feature_extractor.endswith(".pth")
    #    or args.pretrained_feature_extractor.endswith(".pt")
    #)
    #ckpt_path = args.pretrained_feature_extractor

    # train_loader, val_loader = prepare_data(
    #    args.dataset,
    #    data_dir=args.data_dir,
    #    train_dir=args.train_dir,
    #    val_dir=args.val_dir,
    #    batch_size=args.batch_size,
    #    num_workers=args.num_workers,
    #    semi_supervised=args.semi_supervised,
    # )
    train_loader, val_loader = set_linear_loader(args)

    # Find the ckpt path from args.ckpt_folder
    for res_dir in os.listdir(args.ckpt_folder):
        print('Evaluating ', res_dir)
        args.ckpt_dir_path = os.path.join(args.ckpt_folder, res_dir)

        # If this directory has already been evaluated, skip
        if os.path.isfile(
                os.path.join(args.ckpt_dir_path, 'linear_result.txt')):
            with open(os.path.join(args.ckpt_dir_path, 'linear_result.txt'),
                      'r') as f:
                line_cnt = len(f.readlines())

            if line_cnt >= args.max_epochs:
                print('Skip current directory as linear_result.txt exists '
                      'with {} lines (more than max epochs {})!'.format(
                    line_cnt, args.max_epochs))
                continue

        try:

            all_ckpts = os.listdir(args.ckpt_dir_path)
            all_ckpts = [ckpt for ckpt in all_ckpts if ckpt.endswith('.pth')]
            ckpt_step = [int(ckpt.split('.')[0].split('_')[1]) for ckpt in
                         all_ckpts]

            ckpt_path = os.path.join(args.ckpt_dir_path,
                                     '1_{}.pth'.format(max(ckpt_step)))

            print('ckpt_path: ', ckpt_path)

            # state = torch.load(ckpt_path) #["model"]
            # state_dict = {}
            # for k, v in state.items():
            #    if k.startswith("fc."):
            #        continue
            #    state_dict[k] = v
            state = torch.load(ckpt_path)["model"]
            # print(state.keys())
            state_dict = {}
            for k, v in state.items():
                if k.startswith("fc."):
                    continue

                # Depending on the source ckpt, use different loading
                if 'UCL' in ckpt_path and 'backbone' in k:
                    state_dict[k.replace("net.module.backbone.module.", "")] = v
                elif 'OnlineContrast' in ckpt_path:  # OnlineContrast
                    state_dict[k.replace("module.", "")] = v

            backbone.load_state_dict(state_dict, strict=True)

            if args.dali:
                assert _dali_avaliable, "Dali is not currently avaiable, please install it first."
                MethodClass = types.new_class(
                    f"Dali{LinearModel.__name__}",
                    (ClassificationABC, LinearModel)
                )
            else:
                MethodClass = LinearModel

            model = MethodClass(backbone, **args.__dict__, tasks=tasks)

            callbacks = []

            # wandb logging
            #if args.wandb:
            #    wandb_logger = WandbLogger(
            #        name=args.name, project=args.project, offline=args.offline
            #    )
            #    wandb_logger.watch(model, log="gradients", log_freq=100)
            #    wandb_logger.log_hyperparams(args)

            # lr logging
            #    lr_monitor = LearningRateMonitor(logging_interval="epoch")
            #    callbacks.append(lr_monitor)

            # save checkpoint on last epoch only
            #    ckpt = Checkpointer(
            #        args,
            #        logdir=os.path.join(args.checkpoint_dir, "linear"),
            #        frequency=args.checkpoint_frequency,
            #    )
            #    callbacks.append(ckpt)

            trainer = Trainer.from_argparse_args(
                args,
                logger=None, # wandb_logger if args.wandb else None,
                callbacks=callbacks,
                plugins=DDPPlugin(find_unused_parameters=True),
                checkpoint_callback=False,
                terminate_on_nan=True,
                accelerator="ddp",
            )
            if args.dali:
                trainer.fit(model, val_dataloaders=val_loader)
            else:
                trainer.fit(model, train_loader, val_loader)

        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")

    # The logging function is integrated in LinearModel


if __name__ == "__main__":
    main()
