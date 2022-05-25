import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from typing import Tuple, Any
import os


CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

svhn_mean = (0.5, 0.5, 0.5)
svhn_std = (0.5, 0.5, 0.5)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)



def get_normalize(args):
    if args.no_norm:
        normalize = lambda X: X
        return normalize
    if args.dataset == 'cifar100':
        mu = torch.tensor(CIFAR100_MEAN).view(3,1,1).cuda()
        std = torch.tensor(CIFAR100_STD).view(3,1,1).cuda()
    elif args.dataset == 'cifar10':
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
    elif args.dataset == 'svhn':
        mu = torch.tensor(svhn_mean).view(3,1,1).cuda()
        std = torch.tensor(svhn_std).view(3,1,1).cuda()
    elif args.dataset == 'imagenet':
        mu = torch.tensor(IMAGENET_MEAN).view(3,1,1).cuda()
        std = torch.tensor(IMAGENET_STD).view(3,1,1).cuda()
    else:
        raise ValueError("Invalid dataset name")
    normalize = lambda X: (X - mu)/std
    return normalize

def get_dataloaders(args):
    if args.dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


        train_dataset = datasets.CIFAR100(
            '/scratch/CIFAR100', train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            '/scratch/CIFAR100', train=False, transform=test_transform, download=True)

    elif args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        train_dataset = datasets.ImageFolder(os.path.join('/scratch/IMAGENET/data-dir/raw-data', 'train'), train_transform)
        test_dataset = datasets.ImageFolder(os.path.join('/scratch/IMAGENET/data-dir/raw-data', 'validation'), test_transform)


    elif args.dataset == 'svhn':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


        train_dataset = datasets.SVHN(
            '/scratch/SVHN', split='train', transform=train_transform, download=True)
        test_dataset = datasets.SVHN(
            '/scratch/SVHN', split='test', transform=test_transform, download=True)

    elif args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.CIFAR10(
            '/scratch/CIFAR10', train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            '/scratch/CIFAR10', train=False, transform=test_transform, download=True)
    else:
        raise ValueError("Invalid dataset name")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    return test_loader, train_loader
