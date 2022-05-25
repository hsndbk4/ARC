import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
import numpy as np


class ModuleStat(metaclass=ABCMeta):
    @abstractmethod
    def get_num_ops(self):
        raise NotImplementedError


class Conv2dStat(nn.Conv2d, ModuleStat):
    def __init__(self, *kargs, **kwargs):
        super(Conv2dStat, self).__init__(*kargs, **kwargs)
        self.num_mult = 0
        self.num_add = 0
        self.num_ops = 0
        self.in_shape = 0
        self.out_shape = 0

    def get_num_ops(self):
        return self.num_ops
    def get_in_shape(self):
        return self.in_shape
    def get_out_shape(self):
        return self.out_shape
    def forward(self, input):
        self.in_shape = tuple(input.shape[1:])
        out = super(Conv2dStat,self).forward(input)
        self.out_shape = tuple(out.shape[1:])
        num_out_elements = out.data[0,:].numel()
        cin = self.in_channels // self.groups
        kh, kw = self.kernel_size
        kernel_mul = kh * kw * cin #D_l
        bias_ops = 1 if self.bias is not None else 0
        kernel_add = kh * kw * cin - 1 + bias_ops
        self.num_mult = kernel_mul*num_out_elements
        self.num_add = kernel_add*num_out_elements
        self.num_ops = self.num_mult+self.num_add
        return out

class LinearStat(nn.Linear, ModuleStat):
    def __init__(self, *kargs, **kwargs):
        super(LinearStat, self).__init__(*kargs, **kwargs)
        self.num_mult = 0
        self.num_add = 0
        self.num_ops = 0
        self.in_shape = 0
        self.out_shape = 0
    def forward(self, input):
        self.in_shape = tuple(input.shape[1:])
        out = super(LinearStat,self).forward(input)
        self.out_shape = tuple(out.shape[1:])
        num_out_elements = out.data[0,:].numel()
        cin = self.in_features
        kernel_mul = cin #D_l
        bias_ops = 1 if self.bias is not None else 0
        kernel_add = cin - 1 + bias_ops
        self.num_mult = kernel_mul*num_out_elements
        self.num_add = kernel_add*num_out_elements
        self.num_ops = self.num_mult+self.num_add
        return out
    def get_num_ops(self):
        return self.num_ops
    def get_in_shape(self):
        return self.in_shape
    def get_out_shape(self):
        return self.out_shape

class BatchNorm2dStat(nn.BatchNorm2d, ModuleStat):
    def __init__(self, *kargs, **kwargs):
        super(BatchNorm2dStat, self).__init__(*kargs, **kwargs)
        self.num_mult = 0
        self.num_add = 0
        self.num_ops = 0
        self.in_shape = 0
        self.out_shape = 0
    def forward(self, input):
        self.in_shape = tuple(input.shape[1:])
        out = super(BatchNorm2dStat,self).forward(input)
        self.out_shape = tuple(out.shape[1:])
        num_out_elements = out.data[0,:].numel()
        self.num_mult = num_out_elements
        self.num_add = num_out_elements
        self.num_ops = self.num_mult+self.num_add
        return out
    def get_num_ops(self):
        return self.num_ops
    def get_in_shape(self):
        return self.in_shape
    def get_out_shape(self):
        return self.out_shape



class BatchNorm1dStat(nn.BatchNorm1d, ModuleStat):
    def __init__(self, *kargs, **kwargs):
        super(BatchNorm1dStat, self).__init__(*kargs, **kwargs)
        self.num_mult = 0
        self.num_add = 0
        self.num_ops = 0
        self.in_shape = 0
        self.out_shape = 0
    def forward(self, input):
        self.in_shape = tuple(input.shape[1:])
        out = super(BatchNorm1dStat,self).forward(input)
        self.out_shape = tuple(out.shape[1:])
        num_out_elements = out.data[0,:].numel()
        self.num_mult = num_out_elements
        self.num_add = num_out_elements
        self.num_ops = self.num_mult+self.num_add
        return out
    def get_num_ops(self):
        return self.num_ops
    def get_in_shape(self):
        return self.in_shape
    def get_out_shape(self):
        return self.out_shape
