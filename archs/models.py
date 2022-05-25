import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.functional import interpolate
import torch.nn.functional as F
#from datasets import get_num_classes
from archs.mobilenet import MobileNetV1CC as MobileNetV1
from archs.preactresnet import PreActResNet18
from archs.resnet_cifar import resnet20, resnet20_dverge
from archs.vggnet import VGG
from archs.wrn import wrn_28_4 as wideresnet_28_4
import functools



def get_num_classes(dataset: str):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'svhn':
        return 10
    else:
        raise ValueError('Invalid dataset name.')

def get_input_size(dataset: str):
    if dataset == 'cifar10':
        return 32
    elif dataset == 'cifar100':
        return 32
    elif dataset == 'imagenet':
        return 224
    else:
        raise ValueError('Invalid dataset name.')

def get_architecture(arch: str, dataset: str, width_mult: float, expansion_factor: int, channel_mult: float, use_stat_layers: bool) -> torch.nn.Module:
    num_classes = get_num_classes(dataset)
    print(num_classes)
    if arch == 'preactresnet18':
        model = PreActResNet18(num_classes=num_classes, width_mult= width_mult, use_stat_layers=use_stat_layers)
    elif arch == 'mobilenetv1':
        model = MobileNetV1(width_mult= width_mult, num_classes=num_classes, use_stat_layers=use_stat_layers)
    elif arch == 'wideresnet_28_4':
        model = wideresnet_28_4()
        model.sub_block1 = None
    elif arch == 'resnet20':
        #model = torchvision.models.resnet18(pretrained=False)
        model = resnet20(num_classes=num_classes, width_mult= width_mult, use_stat_layers=use_stat_layers)
    elif arch == 'resnet20_dverge':
        #model = torchvision.models.resnet18(pretrained=False)
        model = resnet20_dverge(num_classes=num_classes, width_mult= width_mult, use_stat_layers=use_stat_layers)
    elif arch.startswith('vgg'):
        #model = torchvision.models.resnet18(pretrained=False)
        model = VGG(vgg_name=arch, num_classes=num_classes, width_mult= width_mult, use_stat_layers=use_stat_layers)
    else:
        raise ValueError('Invalid model name.')
    return model
