import math
import functools
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
import torch.nn.init as init
from archs.stat_modules import Conv2dStat, LinearStat, BatchNorm2dStat



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, Conv2d=nn.Conv2d, BatchNorm2d=nn.BatchNorm2d):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, Conv2d=nn.Conv2d, BatchNorm2d=nn.BatchNorm2d):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, Conv2d=Conv2d, BatchNorm2d=BatchNorm2d))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, Conv2d=Conv2d, BatchNorm2d=BatchNorm2d),
            # pw-linear
            Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



def MobileNetV2CIFAR10(num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, expansion_factor = 6, channel_mult = 1.0, block=None, use_stat_layers = False):
    if inverted_residual_setting is None:
        inverted_residual_setting = [
            # t, c, n, s
            [1, int(channel_mult*16), 1, 1],
            [expansion_factor, int(channel_mult*24), 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
            [expansion_factor, int(channel_mult*32), 3, 2],
            [expansion_factor, int(channel_mult*64), 4, 2],
            [expansion_factor, int(channel_mult*96), 3, 1],
            [expansion_factor, int(channel_mult*160), 3, 2],
            [expansion_factor, int(channel_mult*320), 1, 1],
        ]
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult, inverted_residual_setting=inverted_residual_setting, round_nearest=round_nearest, expansion_factor=expansion_factor, channel_mult=channel_mult, block=block, use_stat_layers=use_stat_layers)

def MobileNetV2CIFAR10_v2(num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, expansion_factor = 6, channel_mult = 1.0, block=None, use_stat_layers = False):
    if inverted_residual_setting is None:
        inverted_residual_setting = [
            # t, c, n, s
            [1, int(channel_mult*16), 1, 1],
            [expansion_factor, int(channel_mult*24), 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
            [expansion_factor, int(channel_mult*32), 3, 1], # NOTE: change stride 2 -> 1 for CIFAR10
            [expansion_factor, int(channel_mult*64), 4, 2],
            [expansion_factor, int(channel_mult*96), 3, 1],
            [expansion_factor, int(channel_mult*160), 3, 2],
            [expansion_factor, int(channel_mult*320), 1, 1],
        ]
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult, inverted_residual_setting=inverted_residual_setting, round_nearest=round_nearest, expansion_factor=expansion_factor, channel_mult=channel_mult, block=block, use_stat_layers=use_stat_layers)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 expansion_factor = 6,
                 channel_mult = 1.0,
                 block=None,
                 use_stat_layers = False):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, int(channel_mult*16), 1, 1],
                [expansion_factor, int(channel_mult*24), 2, 2],
                [expansion_factor, int(channel_mult*32), 3, 2],
                [expansion_factor, int(channel_mult*64), 4, 2],
                [expansion_factor, int(channel_mult*96), 3, 1],
                [expansion_factor, int(channel_mult*160), 3, 2],
                [expansion_factor, int(channel_mult*320), 1, 1],
            ]
        #print(inverted_residual_setting)
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, Conv2d=Conv2d, BatchNorm2d=BatchNorm2d)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, Conv2d=Conv2d, BatchNorm2d=BatchNorm2d))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, Conv2d=Conv2d, BatchNorm2d=BatchNorm2d))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0),
            Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        x = x.mean([2, 3])
        #x = self.pool(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

class MobileNetV1(nn.Module):
    def __init__(self, width_mult = 1, res_mul =1, num_classes=200, use_stat_layers = False):
        super(MobileNetV1, self).__init__()
        #assert(width_mult <= 1 and width_mult >0)

        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d

        def conv_bn(inp, oup, stride):
            inp = round(inp*1.0)
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                Conv2d(inp, oup, 3, stride, 1, bias=False),
                BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            inp = int(round(inp*width_mult))
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                Conv2d(inp, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn(  3,  32, 2),
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.d_in = int(round(1024*width_mult))
        self.classifier = Linear(int(round(1024*width_mult)), num_classes)
    def forward(self, x):
            x = self.features(x)

            x = x.view(-1, self.d_in)
            x = self.classifier(x)
            return x


class MobileNetV1C(nn.Module):
    def __init__(self, width_mult = 1, res_mul =1, num_classes=200, use_stat_layers = False):
        super(MobileNetV1C, self).__init__()
        #assert(width_mult <= 1 and width_mult >0)

        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d

        def conv_bn(inp, oup, stride):
            inp = round(inp*1.0)
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                Conv2d(inp, oup, 3, stride, 1, bias=False),
                BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            inp = int(round(inp*width_mult))
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                Conv2d(inp, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn(  3,  32, 1),
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.d_in = int(round(1024*width_mult))
        self.classifier = Linear(int(round(1024*width_mult)), num_classes)
    def forward(self, x):
            x = self.features(x)
            x = x.mean([2, 3])
            x = self.classifier(x)
            return x


class MobileNetV1CC(nn.Module):
    def __init__(self, width_mult = 1, res_mul =1, num_classes=200, use_stat_layers = False):
        super(MobileNetV1CC, self).__init__()
        #assert(width_mult <= 1 and width_mult >0)

        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d

        def conv_bn(inp, oup, stride):
            inp = round(inp*1.0)
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                Conv2d(inp, oup, 3, stride, 1, bias=False),
                BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            inp = int(round(inp*width_mult))
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                Conv2d(inp, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn(  3,  32, 1),
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.d_in = int(round(1024*width_mult))
        self.classifier = Linear(int(round(1024*width_mult)), num_classes)
    def forward(self, x):
            x = self.features(x)
            x = x.mean([2, 3])
            x = self.classifier(x)
            return x



class MobileNetV1R(nn.Module):
    def __init__(self, width_mult = 1, res_mul =1, num_classes=200, use_stat_layers = False):
        super(MobileNetV1R, self).__init__()
        #assert(width_mult <= 1 and width_mult >0)

        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d

        def conv_bn(inp, oup, stride):
            inp = round(inp*1.0)
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                Conv2d(inp, oup, 3, stride, 1, bias=False),
                BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dwr(inp, oup, stride):
            inp = int(round(inp*width_mult))
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                Conv2d(inp, oup, 3, stride, 1, bias=False),
                BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            conv_bn(  3,  32, 2),
            conv_dwr( 32,  64, 1),
            conv_dwr( 64, 128, 2),
            conv_dwr(128, 128, 1),
            conv_dwr(128, 256, 2),
            conv_dwr(256, 256, 1),
            conv_dwr(256, 512, 2),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 1024, 2),
            conv_dwr(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.d_in = int(round(1024*width_mult))
        self.classifier = Linear(int(round(1024*width_mult)), num_classes)
    def forward(self, x):
            x = self.features(x)

            x = x.view(-1, self.d_in)
            x = self.classifier(x)
            return x


class MobileNetV1RC(nn.Module):
    def __init__(self, width_mult = 1, res_mul =1, num_classes=200, use_stat_layers = False):
        super(MobileNetV1RC, self).__init__()
        #assert(width_mult <= 1 and width_mult >0)

        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d

        def conv_bn(inp, oup, stride):
            inp = round(inp*1.0)
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                Conv2d(inp, oup, 3, stride, 1, bias=False),
                BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dwr(inp, oup, stride):
            inp = int(round(inp*width_mult))
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                Conv2d(inp, oup, 3, stride, 1, bias=False),
                BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            conv_bn(  3,  32, 1),
            conv_dwr( 32,  64, 1),
            conv_dwr( 64, 128, 2),
            conv_dwr(128, 128, 1),
            conv_dwr(128, 256, 2),
            conv_dwr(256, 256, 1),
            conv_dwr(256, 512, 2),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 1024, 2),
            conv_dwr(1024, 1024, 1),
        )
        self.d_in = int(round(1024*width_mult))
        self.classifier = Linear(int(round(1024*width_mult)), num_classes)
    def forward(self, x):
            x = self.features(x)
            x = x.mean([2, 3])
            #x = x.view(-1, self.d_in)
            x = self.classifier(x)
            return x


class MobileNetV1RCC(nn.Module):
    def __init__(self, width_mult = 1, res_mul =1, num_classes=10, use_stat_layers = False):
        super(MobileNetV1RCC, self).__init__()
        #assert(width_mult <= 1 and width_mult >0)

        Conv2d = Conv2dStat if use_stat_layers else nn.Conv2d
        Linear = LinearStat if use_stat_layers else nn.Linear
        BatchNorm2d = BatchNorm2dStat if use_stat_layers else nn.BatchNorm2d

        def conv_bn(inp, oup, stride):
            inp = round(inp*1.0)
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                Conv2d(inp, oup, 3, stride, 1, bias=False),
                BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dwr(inp, oup, stride):
            inp = int(round(inp*width_mult))
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                Conv2d(inp, oup, 3, stride, 1, bias=False),
                BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            conv_bn(  3,  32, 1),
            conv_dwr( 32,  64, 1),
            conv_dwr( 64, 128, 2),
            conv_dwr(128, 128, 1),
            conv_dwr(128, 256, 1),
            conv_dwr(256, 256, 1),
            conv_dwr(256, 512, 2),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 512, 1),
            conv_dwr(512, 1024, 2),
            conv_dwr(1024, 1024, 1),
        )
        self.d_in = int(round(1024*width_mult))
        self.classifier = Linear(int(round(1024*width_mult)), num_classes)
    def forward(self, x):
            x = self.features(x)
            x = x.mean([2, 3])
            x = self.classifier(x)
            return x
