import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.functional import interpolate
import torch.nn.functional as F
from archs.models import get_architecture as get_cifar_model
import numpy as np
from collections import namedtuple
import torchvision
import os
import shutil
import functools

def get_architecture(arch: str, dataset: str, width_mult: float, expansion_factor: int, channel_mult: float, use_stat_layers: bool) -> torch.nn.Module:
    if dataset == 'imagenet':
        return get_imagenet_model(arch, width_mult)
    else:
        return get_cifar_model(arch,dataset, width_mult, expansion_factor, channel_mult, use_stat_layers)
def get_input_size(args):
    if args.dataset == 'imagenet':
        return (args.batch_size,3,224,224)
    elif args.dataset == 'cifar10':
        return (args.batch_size,3,32,32)
    else:
        raise ValueError('Invalid dataset: ' + args.dataset)

def remove_module(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_state_dict['.'.join(key.split('.')[1:])] = state_dict[key]
    return new_state_dict

def load_params(model, args, fname):
    is_dp = True if isinstance(model, nn.DataParallel) else False
    state_dict_has_dp = True
    print('Model DataParallel status: {}'.format(is_dp))
    if args.dataset in ['cifar10', 'cifar100', 'svhn']:
        ckpt = torch.load(os.path.join(fname, f'model_best.pth'))
        state_dict = ckpt['state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict_has_dp = True
        else:
            state_dict_has_dp = False
        print('model loaded from: ', fname)
        if is_dp == False and state_dict_has_dp == True:
            state_dict = remove_module(state_dict)
        model.load_state_dict(state_dict)

    elif args.dataset == 'imagenet':
        print("=> loading default best model '{}'".format(os.path.join(fname, 'model_best.pth.tar')))
        checkpoint = torch.load(os.path.join(fname, 'model_best.pth.tar'))
        state_dict = checkpoint['state_dict']
        best_prec1 = checkpoint['best_prec1']
        if is_dp == False:
            state_dict = remove_module(state_dict)
        model.load_state_dict(state_dict)
        print('model loaded from: ', fname)
    else:
        raise ValueError('Invalid dataset: ' + args.dataset)
