import math
import functools
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
import torch.nn.init as init
from archs.stat_modules import Conv2dStat, LinearStat, BatchNorm2dStat
####################### ResNets for CIFAR10 ####################################################


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', Conv2d = nn.Conv2d, BatchNorm2d = nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'Z': #custom to handle 4x4 model
                self.shortcut = LambdaLayer(lambda x: x[:, :, ::2, ::2])
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        #out = F.relu(out)
        out = self.relu2(out)
        return out


class BasicBlockDVERGE(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B', Conv2d = nn.Conv2d, BatchNorm2d = nn.BatchNorm2d):
        super(BasicBlockDVERGE, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'Z': #custom to handle 4x4 model
                self.shortcut = LambdaLayer(lambda x: x[:, :, ::2, ::2])
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        #out = F.relu(out)
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_mult= 1, use_stat_layers = False, last_layer_4x4=False):
        super(ResNet, self).__init__()
        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d
        self.in_planes = int(round(16*width_mult))

        self.conv1 = Conv2d(3, int(round(16*width_mult)), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(int(round(16*width_mult)))
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, int(round(16*width_mult)), num_blocks[0], stride=1, Conv2d = Conv2d, BatchNorm2d = BatchNorm2d)
        self.layer2 = self._make_layer(block, int(round(32*width_mult)), num_blocks[1], stride=2, Conv2d = Conv2d, BatchNorm2d = BatchNorm2d)
        if last_layer_4x4:
            self.layer3 = self._make_layer_2(block, int(round(64*width_mult)), num_blocks[2], stride=2, Conv2d = Conv2d, BatchNorm2d = BatchNorm2d)
        else:
            self.layer3 = self._make_layer(block, int(round(64*width_mult)), num_blocks[2], stride=2, Conv2d = Conv2d, BatchNorm2d = BatchNorm2d)
        self.linear = Linear(int(round(64*width_mult)), num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, Conv2d = nn.Conv2d, BatchNorm2d = nn.BatchNorm2d):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, Conv2d = Conv2d, BatchNorm2d = BatchNorm2d))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    def _make_layer_2(self, block, planes, num_blocks, stride, Conv2d = nn.Conv2d, BatchNorm2d = nn.BatchNorm2d):
        strides = [stride] + [1]*(num_blocks-2) + [stride]
        layers = []
        for stride in strides:
            if stride != 1 and self.in_planes == planes:
                layers.append(block(self.in_planes, planes, stride, Conv2d = Conv2d, BatchNorm2d = BatchNorm2d, option='Z'))
            else:
                layers.append(block(self.in_planes, planes, stride, Conv2d = Conv2d, BatchNorm2d = BatchNorm2d))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10, width_mult= 1, use_stat_layers=False):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, width_mult= width_mult, use_stat_layers=use_stat_layers)

def resnet20_dverge(num_classes=10, width_mult= 1, use_stat_layers=False):
    return ResNet(BasicBlockDVERGE, [3, 3, 3], num_classes=num_classes, width_mult= width_mult, use_stat_layers=use_stat_layers)

def resnet20_4x4(num_classes=10, width_mult= 1, use_stat_layers=False): #resnet20 with an added stride in the last few layers
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, width_mult= width_mult, use_stat_layers=use_stat_layers, last_layer_4x4=True)

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])
