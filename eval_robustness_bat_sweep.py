
import argparse
import logging
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

from utils import *
from architectures import get_architecture, load_params
from datasets import get_normalize, get_dataloaders
from attack import attack_arc

upper_limit, lower_limit = 1,0

def get_args_local():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'svhn', 'imagenet'])
    parser.add_argument('--model', default='resnet20')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--num-workers', default=16, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none', 'arc'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--eta', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--g', default=-1, type=int, help='how many hyperplanes to consider in the ARC_lite algorithm, -1 means default arc')
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--rand-init', action='store_true', help='if true, the adv algorithm will add random perturbation to delta')
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--use-stat-layers', action='store_true', help='if true, replaces nn modules with stat modules')
    parser.add_argument('--no-norm', action='store_true', help='if true, no data normalization would be used for evaluating the model')


    parser.add_argument('--width-mult', default=1.0, type=float, help='MobileNetV1 width multiplier')
    parser.add_argument('--expansion-factor', default=6, type=int, help='MobileNetV2 expansion factor')
    parser.add_argument('--channel-mult', default=1.0, type=float, help='MobileNetV2 channel multiplier')


    parser.add_argument('--logfilename', default='', type=str, help='choose the output filename')


    parser.add_argument('--outdir', default='cifar_model', type=str)
    parser.add_argument('--fname', default='cifar_model', type=str, nargs='+')
    parser.add_argument('--attacker-strategy', default='same', type=str, choices=['adv','same'])
    parser.add_argument('--alpha', default=1,type=float) # the probability of the defender choosing model 1
    parser.add_argument('--sweep-param', default='alpha', type=str, choices=['alpha', 'eta', 'K', 'epsilon', 'g'])
    return parser.parse_args()

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def expected_loss(loss_func, prob):
    return lambda output, y, reduction='mean': sum([loss_func(output[indx],y,reduction=reduction)*prob[indx] for indx in range(len(prob))])


class Parallel(nn.Module):
    def __init__(self, modules = []):
        super(Parallel, self).__init__()
        if len(modules) == 0:
            raise ValueError('Cannot initialize Parallel module with 0 modules')
        self.modulelist = nn.ModuleList(modules)
    def forward(self, x):
        out = []
        for m in self.modulelist:
            y = m(x)
            out.append(y)
        return out



def expected_acc(models, alpha, X, y, normalize=None, reduction='mean'):
    acc = 0
    for i in range(len(alpha)):
        y_i = models[i](normalize(X))
        t_i = (y_i.max(1)[1] == y).float()
        acc += alpha[i]*t_i
    return acc.mean() if reduction == 'mean' else acc


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False, normalize=None, loss_func=F.cross_entropy, rand_init=True):

    max_loss = torch.zeros(X.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        #model.module.sample()
        #print(model.module.model_indx)
        delta = torch.zeros_like(X).cuda()
        if rand_init:
            if norm == "l_inf":
                delta.uniform_(-epsilon, epsilon)
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*epsilon
            else:
                raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = loss_func(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = loss_func(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta



def get_num_classes(dataset: str):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'svhn':
        return 10
    elif dataset == 'imagenet':
        return 1000
    else:
        raise ValueError('Invalid dataset name.')


def main():
    args = get_args_local()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    M = len(args.fname)
    assert(M==2) # only supports two models for BAT

    epsilon = (args.epsilon / 255.)
    eta = (args.eta / 255.)
    test_loader, train_loader = get_dataloaders(args)

    normalize = get_normalize(args)
    print('Loading the models ...')

    ## construct the ensemble model
    ## fname is a list of file paths, where each path contains a model_best.pth checkpoint with the parameters
    models = []
    for fname in args.fname:
        model =  get_architecture(args.model, args.dataset, args.width_mult, args.expansion_factor, args.channel_mult, args.use_stat_layers)
        load_params(model, args, fname)
        for param in model.parameters():
            param.requires_grad = False
        if args.dataset == 'imagenet':
            model = nn.DataParallel(model).cuda()
        models.append(model)



    alpha = np.array([args.alpha, 1-args.alpha])

    model_attacker = Parallel(models)
    if args.dataset != 'imagenet':
        model_attacker = nn.DataParallel(model_attacker).cuda()
    model_attacker.eval()

    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True


    if args.logfilename == '':
        added_str = ''
        if args.rand_init:
            added_str = '-rand_seed_'+str(args.seed)
        logfilename = os.path.join(args.outdir, 'SWEEP_'+args.sweep_param+'-attack_'+args.attack+added_str+'.txt')
    else:
        logfilename = os.path.join(args.outdir, args.logfilename)
    args_to_print = {'alpha': args.alpha, 'eta': eta, 'K': args.attack_iters, 'epsilon': epsilon, 'g': args.g}
    if not os.path.isfile(logfilename):
        print("=> need to create the logfile titled '{}'".format(logfilename))
        init_logfile(logfilename, "---------- Static Args ----------")
        for arg_key in args_to_print:
            if arg_key == args.sweep_param:
                continue
            log(logfilename, arg_key+'\t'+"{:.8f}".format(args_to_print[arg_key]))
        log(logfilename, "--------------------")
        log(logfilename, args.sweep_param+"\ttest_robust_acc")

    criterion = lambda output, y, reduction='mean': F.cross_entropy(output, y, reduction=reduction)
    attackr_loss = expected_loss(criterion, prob = alpha)
    test_acc = 0
    test_robust_acc = 0
    test_n = 0
    #loss_eval = lambda output, y, reduction='mean': F.nll_loss(torch.log(output),y, reduction=reduction)
    num_classes = get_num_classes(args.dataset)
    for i, (X, y) in enumerate(test_loader):

        X, y = X.cuda(), y.cuda()
        # Random initialization
        if args.attack == 'none':
            delta = torch.zeros_like(X)
        elif args.attack == 'pgd':
            delta = attack_pgd(model_attacker, X, y, epsilon, eta, args.attack_iters, args.restarts, args.norm, normalize=normalize, loss_func=attackr_loss, rand_init=args.rand_init)
        elif args.attack == 'arc':
            delta = attack_arc(models, X, y, epsilon, alpha, args.attack_iters, eta, args.norm, num_classes=num_classes, normalize=normalize, g=args.g, rand_init=args.rand_init)
        delta = delta.detach()

        for i in range(M):
            robust_output_i = models[i](normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            test_robust_acc+= (robust_output_i.max(1)[1] == y).sum().item()*alpha[i]

            output_i = models[i](normalize(X))
            test_acc += (output_i.max(1)[1] == y).sum().item()*alpha[i]
        test_n += y.size(0)
    log(logfilename, "{:.8f}\t{:.3f}".format(args_to_print[args.sweep_param], test_robust_acc*100/test_n))

if __name__ == "__main__":
    main()
