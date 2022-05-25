import math
import functools
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
import torch.nn.init as init
from archs.stat_modules import Conv2dStat, LinearStat, BatchNorm2dStat

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name='vgg16', num_classes=10, width_mult=1, use_stat_layers=False):
        super(VGG, self).__init__()
        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d
        self.features = self._make_layers(cfg[vgg_name], Conv2d=Conv2d, BatchNorm2d=BatchNorm2d, width_mult=width_mult)
        self.classifier = Linear(int(round(512*width_mult)), num_classes)
        #print(self.features)
        #assert(1==2)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        #print(out)
        out = self.classifier(out)
        #print(out)
        #print('weight',self.classifier.weight.data)
        #print('bias',self.classifier.bias.data)
        #assert(1==2)
        return out

    def _make_layers(self, cfg, Conv2d=nn.Conv2d, BatchNorm2d=nn.BatchNorm2d, width_mult = 1):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = int(round(x*width_mult))
                layers += [Conv2d(in_channels, x, kernel_size=3, padding=1),
                           BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)


def test():
    net = VGG('vgg11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
'''
########################### VGGNets ##########################################
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'tinyVGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG7': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes = 10, use_stat_layers=False):
        super(VGG, self).__init__()
        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d
        self.features = self._make_layers(cfg[vgg_name], Conv2d=Conv2d, BatchNorm2d=BatchNorm2d)
        if vgg_name == 'tinyVGG16':
            self.classifier = nn.Sequential(Linear(512*7*7, 4096),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            Linear(4096, 2048),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            Linear(2048, num_classes)
                                            )
        elif vgg_name =='VGG7':
            self.classifier = nn.Sequential(Linear(512*4*4, 1024),
                                            BatchNorm1d(1024),
                                            nn.ReLU(inplace=True),
                                            Linear(1024, num_classes),
                                            BatchNorm1d(10),
                                            )
        else:
            self.classifier = Linear(512, num_classes)

    def forward(self, x):
        #print(x.shape)
        out = self.features(x)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, Conv2d=nn.Conv2d, BatchNorm2d=nn.BatchNorm2d):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Conv2d(in_channels, x, kernel_size=3, padding=1),
                           BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
'''
