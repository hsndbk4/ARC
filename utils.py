import numpy as np
from collections import namedtuple
import torch
from torch import nn
import torchvision
import os
import shutil
import functools




def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()

def read_logfile(filename: str):
    if not os.path.isfile(filename):
        raise ValueError("=> no logfile found at '{}'".format(filename))
    f = open(filename,'r')
    line = f.readline()
    cnt = 1
    return_lists = {}
    keys = line.strip('\n').split('\t')
    for i in range(len(keys)):
        return_lists[keys[i]] = []
    line = f.readline()
    while line:
        #print(line)
        s = line.split('\t')
        for i in range(len(s)):
            return_lists[keys[i]].append(float(s[i]))
        line = f.readline()
        cnt += 1
    f.close()
    return return_lists

## ignores first line
## gets all the params in a dict
## after that gets the sweep
def read_logfile_sweep(filename: str):
    if not os.path.isfile(filename):
        raise ValueError("=> no logfile found at '{}'".format(filename))
    f = open(filename,'r')
    line = f.readline()
    delim = '--------------------'
    line = f.readline()
    cnt = 2
    configs = {}
    while line:
        line = line.strip('\n')
        if line == delim:
            break
        key, val = line.split('\t')
        configs[key] = float(val)
        line = f.readline()
        cnt += 1
    line = f.readline()
    return_lists = {}
    keys = line.strip('\n').split('\t')
    for i in range(len(keys)):
        return_lists[keys[i]] = []
    line = f.readline()
    while line:
        #print(line)
        s = line.split('\t')
        for i in range(len(s)):
            return_lists[keys[i]].append(float(s[i]))
        line = f.readline()
        cnt += 1
    f.close()
    return configs,return_lists

########## setters and getters


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
