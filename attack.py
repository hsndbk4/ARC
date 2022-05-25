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

upper_limit, lower_limit = 1,0

def expected_acc(models, alpha, X, y, normalize=None, reduction='mean'):
    acc = 0
    for i in range(len(alpha)):
        y_i = models[i](normalize(X))
        t_i = (y_i.max(1)[1] == y).float()
        acc += alpha[i]*t_i
    return acc.mean() if reduction == 'mean' else acc

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


## given models = [f_1, ..., f_M] and alpha=[alpha_1, ..., alpha_M] as the random ensemble
## X is the [B\times D] batch of D-dimensional inputs, y is the vector of lables, y_i \in [C]
## epsilon: maximum allowed radius for adversarial perturbation
## attack_iters: also knows as K, total number of iterations in the attack
## step_size: or eta, the 'localized' epsilon, maximum norm for the local perturbation, usually made smaller than epsilon to avoid approximation inaccuracies due to linearization
## num_classes: number of classes, also referred to as C

def attack_ARC_l2(models, X, y, epsilon, alpha, attack_iters, step_size, norm, num_classes, normalize=None, rand_init=True):
    assert(norm == 'l_2') #no support yet for l_inf norm
    # first, get an indexing I such that alpha[i]>=alpha[j] \forall i,j\inI s.t. i<=j
    I = np.flip(np.argsort(alpha))
    delta_min = torch.zeros_like(X).cuda()
    C = num_classes
    beta_m = step_size
    rho = 0.05*epsilon #positive constant to avoind boundary points
    loss_min = expected_acc(models,alpha,X,y,normalize=normalize,reduction='none')
    if rand_init:
        if norm == "l_inf":
            delta_min.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta_min.normal_()
            d_flat = delta_min.view(delta_min.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta_min.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta_min *= r/n*epsilon
        else:
            raise ValueError
    delta_min = clamp(delta_min, lower_limit-X, upper_limit-X)
    for _ in range(attack_iters):
        delta_local = torch.zeros_like(X).cuda() ##local perutbration living in the smaller ball of radius beta_m and center X+delta_min
        loss_min_local = loss_min.clone()
        for i in I:
            f_i = models[i]
            ## compute the shortest distance to the linearized decision boundary of X+delta_min
            ## also compute the unit norm direction that leads to that boundary
            delta = (delta_min+delta_local).clone()
            delta.requires_grad = True
            out = f_i(normalize(X+delta))
            y_hat = out.argmax(dim=1) ## the lables assigned to X+delta by f_i
            out_m,_ = out.max(dim=1) ## the maximum logits [f_i(X+delta)]_m
            out_m.backward(torch.ones_like(out_m),retain_graph=True)
            del_y = delta.grad.detach().clone() #compute [grad f_i(X+delta)]_m
            delta.grad.zero_()
            zeta = torch.ones_like(out_m)*np.infty #shortest distance to decision boundary
            g = torch.zeros_like(X).cuda() #direction to the closest decision boundary, unit norm
            #print(zeta)
            for j in range(C):
                is_self = (y_hat == j)
                out_j = out[:,j]
                out_j.backward(torch.ones_like(out_j),retain_graph=True)
                del_j = delta.grad.detach() #compute [grad f_i(X+delta)]_j for logit index j
                with torch.no_grad():
                    zeta_j = torch.abs(out_m-out_j)
                    zeta_j[is_self] = np.infty
                    w_j = del_y-del_j
                    del_norm = torch.norm(w_j.view(w_j.shape[0],-1),dim=1)
                    #del_norm = (w_j).norm(dim=1)
                    zeta_j = zeta_j/(del_norm+1e-10)
                    g_j = -w_j/(del_norm.view(-1,1,1,1)+1e-10)
                    update_indx = zeta>zeta_j
                    zeta[update_indx]=zeta_j[update_indx]
                    g[update_indx] = g_j[update_indx]
                delta.grad.zero_()
            ## now we have g and zeta needed for our algorithm
            ## we compute beta for every image in the batch
            beta = torch.ones_like(y).float().cuda()
            beta = beta*beta_m
            #'''
            if i!=I[0]: #not the first classifier in the ensemble
                vec_dot = (g.view(g.size(0),-1)*delta_local.view(delta_local.size(0),-1)).sum(dim=1)

                beta_c = (beta_m)/(beta_m-zeta+1e-10)*(torch.abs(-vec_dot+zeta)) + rho
                beta[zeta<beta_m]=beta_c[zeta<beta_m]

            delta_local_hat = delta_local + beta.view(-1,1,1,1)*g
            delta_local_hat = beta_m*delta_local_hat/(torch.norm(delta_local_hat.view(w_j.shape[0],-1),dim=1).view(-1,1,1,1)+1e-10)
            ## delta_local_hat is now a beta_m norm peturbation away from X+delta_min
            d = (delta_min + delta_local_hat).view(delta_min.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(delta_min)
            ## now we have an epsilon norm perturbation away from X
            d = clamp(d, lower_limit - X, upper_limit - X)
            ## now it is also clipped to satisfy phyiscal constraints (X_adv \in [lower_limit, upper_limit])
            new_loss =  expected_acc(models,alpha,X+d,y,normalize=normalize,reduction='none')
            delta_local[new_loss <= loss_min_local] = delta_local_hat.detach()[new_loss <= loss_min_local]
            loss_min_local = torch.min(loss_min_local, new_loss)

        ## after going over the M classifiers, update delta_min and loss_min
        d = (delta_min + delta_local).view(delta_min.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(delta_min)
        ## now we have an epsilon norm perturbation away from X
        d = clamp(d, lower_limit - X, upper_limit - X)
        ## now it is also clipped to satisfy phyiscal constraints (X_adv \in [lower_limit, upper_limit])
        new_loss =  expected_acc(models,alpha,X+d,y,normalize=normalize,reduction='none')
        delta_min[new_loss <= loss_min] = d.detach()[new_loss <= loss_min]
        loss_min = torch.min(loss_min, new_loss)

    return delta_min



def attack_ARC_linf(models, X, y, epsilon, alpha, attack_iters, step_size, norm, num_classes, normalize=None, rand_init=True):
    assert(norm == 'l_inf') #no support yet for l_inf norm
    # first, get an indexing I such that alpha[i]>=alpha[j] \forall i,j\inI s.t. i<=j
    I = np.flip(np.argsort(alpha))
    delta_min = torch.zeros_like(X).cuda()
    C = num_classes
    beta_m = step_size
    rho = 0.05*epsilon #positive constant to avoind boundary points
    #loss_min = torch.ones_like(y).float()
    loss_min = expected_acc(models,alpha,X,y,normalize=normalize,reduction='none')
    if rand_init:
        if norm == "l_inf":
            delta_min.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta_min.normal_()
            d_flat = delta_min.view(delta_min.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta_min.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta_min *= r/n*epsilon
        else:
            raise ValueError
    delta_min = clamp(delta_min, lower_limit-X, upper_limit-X)
    # in each iteration, try all different directions, and pick the one that minimizes the expected accuracy the most
    for _ in range(attack_iters):
        delta_local = torch.zeros_like(X).cuda() ##local perutbration living in the smaller ball of radius beta_m and center X+delta_min
        loss_min_local = loss_min.clone()
        for i in I:
            f_i = models[i]
            ## compute the shortest distance to the linearized decision boundary of X+delta_min
            ## also compute the unit norm direction that leads to that boundary
            delta = (delta_min+delta_local).clone()
            delta.requires_grad = True
            out = f_i(normalize(X+delta))
            y_hat = out.argmax(dim=1) ## the lables assigned to X+delta by f_i
            out_m,_ = out.max(dim=1) ## the maximum logits [f_i(X+delta)]_m
            out_m.backward(torch.ones_like(out_m),retain_graph=True)
            del_y = delta.grad.detach().clone() #compute [grad f_i(X+delta)]_m
            delta.grad.zero_()
            zeta = torch.ones_like(out_m)*np.infty #shortest distance to decision boundary
            g = torch.zeros_like(X).cuda() #direction to the closest decision boundary, unit norm
            g_2 = torch.zeros_like(X).cuda() #required for beta computation, g_2 = g if p=2
            #print(zeta)
            for j in range(C):
                is_self = (y_hat == j)
                out_j = out[:,j]
                out_j.backward(torch.ones_like(out_j),retain_graph=True)
                del_j = delta.grad.detach() #compute [grad f_i(X+delta)]_j for logit index j
                with torch.no_grad():
                    zeta_j = torch.abs(out_m-out_j)
                    zeta_j[is_self] = np.infty
                    w_j = del_y-del_j
                    del_norm = torch.norm(w_j.view(w_j.shape[0],-1),p=1,dim=1)
                    #del_norm = (w_j).norm(dim=1)
                    zeta_j = zeta_j/(del_norm+1e-10)
                    g_j = -torch.sign(w_j)
                    g_2_j = -w_j/(del_norm.view(-1,1,1,1)+1e-10)
                    update_indx = zeta>zeta_j
                    zeta[update_indx]=zeta_j[update_indx]
                    g[update_indx] = g_j[update_indx]
                    g_2[update_indx] = g_2_j[update_indx]
                delta.grad.zero_()
            ## now we have g and zeta needed for our algorithm
            ## we compute beta for every image in the batch
            beta = torch.ones_like(y).float().cuda()
            beta = beta*beta_m
            #'''
            if i!=I[0]: #not the first classifier in the ensemble
                vec_dot = (g_2.view(g_2.size(0),-1)*delta_local.view(delta_local.size(0),-1)).sum(dim=1)
                #print(vec_dot)

                beta_c = (beta_m)/(beta_m-zeta+1e-10)*(torch.abs(-vec_dot+zeta)) + rho
                #print(beta_c)

                beta[zeta<beta_m]=beta_c[zeta<beta_m]
                #print(beta)
                #assert(1==2)
            #print(beta)
            #'''
            delta_local_hat = delta_local + beta.view(-1,1,1,1)*g
            delta_local_hat = beta_m*delta_local_hat/(torch.norm(delta_local_hat.view(w_j.shape[0],-1),p=float('inf'),dim=1).view(-1,1,1,1)+1e-10)
            ## delta_local_hat is now a beta_m norm peturbation away from X+delta_min

            #d = (delta_min + delta_local_hat).view(delta_min.size(0),-1).renorm(p=float('inf'),dim=0,maxnorm=epsilon).view_as(delta_min)
            d = torch.clamp(delta_min + delta_local_hat, min=-epsilon, max=epsilon)
            ## now we have an epsilon norm perturbation away from X
            d = clamp(d, lower_limit - X, upper_limit - X)
            ## now it is also clipped to satisfy phyiscal constraints (X_adv \in [lower_limit, upper_limit])
            new_loss =  expected_acc(models,alpha,X+d,y,normalize=normalize,reduction='none')
            delta_local[new_loss <= loss_min_local] = delta_local_hat.detach()[new_loss <= loss_min_local]
            loss_min_local = torch.min(loss_min_local, new_loss)

        ## after going over the M classifiers, update delta_min and loss_min
        #d = (delta_min + delta_local).view(delta_min.size(0),-1).renorm(p=float('inf'),dim=0,maxnorm=epsilon).view_as(delta_min)
        d = torch.clamp(delta_min + delta_local, min=-epsilon, max=epsilon)
        ## now we have an epsilon norm perturbation away from X
        d = clamp(d, lower_limit - X, upper_limit - X)
        ## now it is also clipped to satisfy phyiscal constraints (X_adv \in [lower_limit, upper_limit])
        new_loss =  expected_acc(models,alpha,X+d,y,normalize=normalize,reduction='none')
        delta_min[new_loss <= loss_min] = d.detach()[new_loss <= loss_min]
        loss_min = torch.min(loss_min, new_loss)

    return delta_min






## lighter version of the above function, instead of searching over all C-1 hyper-planes, we search over k largest logits
def attack_ARC_l2_lite(models, X, y, epsilon, alpha, attack_iters, step_size, norm, num_classes, normalize=None, rand_init=True, topk=2):
    assert(norm == 'l_2') #no support yet for l_inf norm
    assert(topk>=2)
    assert(topk<=num_classes)
    # first, get an indexing I such that alpha[i]>=alpha[j] \forall i,j\inI s.t. i<=j
    I = np.flip(np.argsort(alpha))
    delta_min = torch.zeros_like(X).cuda()
    C = num_classes
    beta_m = step_size
    rho = 0.05*epsilon #positive constant to avoind boundary points
    #loss_min = torch.ones_like(y).float()
    loss_min = expected_acc(models,alpha,X,y,normalize=normalize,reduction='none')
    if rand_init:
        if norm == "l_inf":
            delta_min.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta_min.normal_()
            d_flat = delta_min.view(delta_min.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta_min.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta_min *= r/n*epsilon
        else:
            raise ValueError
    delta_min = clamp(delta_min, lower_limit-X, upper_limit-X)
    # in each iteration, try all different directions, and pick the one that minimizes the expected accuracy the most
    for _ in range(attack_iters):
        delta_local = torch.zeros_like(X).cuda() ##local perutbration living in the smaller ball of radius beta_m and center X+delta_min
        loss_min_local = loss_min.clone()
        for i in I:
            f_i = models[i]
            ## compute the shortest distance to the linearized decision boundary of X+delta_min
            ## also compute the unit norm direction that leads to that boundary
            delta = (delta_min+delta_local).clone()
            delta.requires_grad = True
            out = f_i(normalize(X+delta))
            out_top_k,_ = torch.topk(out,topk,dim=1)
            out_m = out_top_k[:,0]
            out_m.backward(torch.ones_like(out_m),retain_graph=True)
            del_y = delta.grad.detach().clone() #compute [grad f_i(X+delta)]_m
            delta.grad.zero_()
            zeta = torch.ones_like(out_m)*np.infty #shortest distance to decision boundary
            g = torch.zeros_like(X).cuda() #direction to the closest decision boundary, unit norm
            for j in range(1,topk):
                out_j = out_top_k[:,j]
                out_j.backward(torch.ones_like(out_j),retain_graph=True)
                del_j = delta.grad.detach() #compute [grad f_i(X+delta)]_j for logit index j
                with torch.no_grad():
                    zeta_j = torch.abs(out_m-out_j)
                    w_j = del_y-del_j
                    del_norm = torch.norm(w_j.view(w_j.shape[0],-1),dim=1)
                    #del_norm = (w_j).norm(dim=1)
                    zeta_j = zeta_j/(del_norm+1e-10)
                    g_j = -w_j/(del_norm.view(-1,1,1,1)+1e-10)
                    update_indx = zeta>zeta_j
                    zeta[update_indx]=zeta_j[update_indx]
                    g[update_indx] = g_j[update_indx]
                delta.grad.zero_()
            ## now we have g and zeta needed for our algorithm
            ## we compute beta for every image in the batch
            beta = torch.ones_like(y).float().cuda()
            beta = beta*beta_m
            #'''
            if i!=I[0]: #not the first classifier in the ensemble
                vec_dot = (g.view(g.size(0),-1)*delta_local.view(delta_local.size(0),-1)).sum(dim=1)
                #print(vec_dot)

                beta_c = (beta_m)/(beta_m-zeta+1e-10)*(torch.abs(-vec_dot+zeta)) + rho
                #print(beta_c)

                beta[zeta<beta_m]=beta_c[zeta<beta_m]
                #print(beta)
                #assert(1==2)
            #print(beta)
            #'''
            delta_local_hat = delta_local + beta.view(-1,1,1,1)*g
            delta_local_hat = beta_m*delta_local_hat/(torch.norm(delta_local_hat.view(w_j.shape[0],-1),dim=1).view(-1,1,1,1)+1e-10)
            ## delta_local_hat is now a beta_m norm peturbation away from X+delta_min

            d = (delta_min + delta_local_hat).view(delta_min.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(delta_min)
            ## now we have an epsilon norm perturbation away from X
            d = clamp(d, lower_limit - X, upper_limit - X)
            ## now it is also clipped to satisfy phyiscal constraints (X_adv \in [lower_limit, upper_limit])
            new_loss =  expected_acc(models,alpha,X+d,y,normalize=normalize,reduction='none')
            delta_local[new_loss <= loss_min_local] = delta_local_hat.detach()[new_loss <= loss_min_local]
            loss_min_local = torch.min(loss_min_local, new_loss)

        ## after going over the M classifiers, update delta_min and loss_min
        d = (delta_min + delta_local).view(delta_min.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(delta_min)
        ## now we have an epsilon norm perturbation away from X
        d = clamp(d, lower_limit - X, upper_limit - X)
        ## now it is also clipped to satisfy phyiscal constraints (X_adv \in [lower_limit, upper_limit])
        new_loss =  expected_acc(models,alpha,X+d,y,normalize=normalize,reduction='none')
        delta_min[new_loss <= loss_min] = d.detach()[new_loss <= loss_min]
        loss_min = torch.min(loss_min, new_loss)

    return delta_min





def attack_ARC_linf_lite(models, X, y, epsilon, alpha, attack_iters, step_size, norm, num_classes, normalize=None, rand_init=True, topk=2):
    assert(norm == 'l_inf') #no support yet for l_inf norm
    assert(topk>=2)
    assert(topk<=num_classes)
    #assert(1==3) #remove after the l_inf fix
    # first, get an indexing I such that alpha[i]>=alpha[j] \forall i,j\inI s.t. i<=j
    I = np.flip(np.argsort(alpha))
    delta_min = torch.zeros_like(X).cuda()
    C = num_classes
    beta_m = step_size
    rho = 0.05*epsilon #positive constant to avoind boundary points
    #loss_min = torch.ones_like(y).float()
    loss_min = expected_acc(models,alpha,X,y,normalize=normalize,reduction='none')
    if rand_init:
        if norm == "l_inf":
            delta_min.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta_min.normal_()
            d_flat = delta_min.view(delta_min.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta_min.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta_min *= r/n*epsilon
        else:
            raise ValueError
    delta_min = clamp(delta_min, lower_limit-X, upper_limit-X)
    # in each iteration, try all different directions, and pick the one that minimizes the expected accuracy the most
    for _ in range(attack_iters):
        delta_local = torch.zeros_like(X).cuda() ##local perutbration living in the smaller ball of radius beta_m and center X+delta_min
        loss_min_local = loss_min.clone()
        for i in I:
            f_i = models[i]
            ## compute the shortest distance to the linearized decision boundary of X+delta_min
            ## also compute the unit norm direction that leads to that boundary
            delta = (delta_min+delta_local).clone()
            delta.requires_grad = True
            out = f_i(normalize(X+delta))
            out_top_k,_ = torch.topk(out,topk,dim=1)
            out_m = out_top_k[:,0]
            out_m.backward(torch.ones_like(out_m),retain_graph=True)
            del_y = delta.grad.detach().clone() #compute [grad f_i(X+delta)]_m
            delta.grad.zero_()
            zeta = torch.ones_like(out_m)*np.infty #shortest distance to decision boundary
            g = torch.zeros_like(X).cuda() #direction to the closest decision boundary, unit norm
            g_2 = torch.zeros_like(X).cuda() #direction required for beta computation, g_2 = g if p=2
            for j in range(1,topk):
                out_j = out_top_k[:,j]
                out_j.backward(torch.ones_like(out_j),retain_graph=True)
                del_j = delta.grad.detach() #compute [grad f_i(X+delta)]_j for logit index j
                with torch.no_grad():
                    zeta_j = torch.abs(out_m-out_j)
                    w_j = del_y-del_j
                    del_norm = torch.norm(w_j.view(w_j.shape[0],-1),p=1,dim=1) #q =  p/p-1 = 1
                    #del_norm = (w_j).norm(dim=1)
                    zeta_j = zeta_j/(del_norm+1e-10)
                    #g_j = -w_j/(del_norm.view(-1,1,1,1)+1e-10)
                    g_j = -torch.sign(w_j)
                    g_2_j = -w_j/(del_norm.view(-1,1,1,1)+1e-10)
                    update_indx = zeta>zeta_j
                    zeta[update_indx]=zeta_j[update_indx]
                    g[update_indx] = g_j[update_indx]
                    g_2[update_indx] = g_2_j[update_indx]
                delta.grad.zero_()
            ## now we have g and zeta needed for our algorithm
            ## we compute beta for every image in the batch
            beta = torch.ones_like(y).float().cuda()
            beta = beta*beta_m
            #print(beta)

            if i!=I[0]: #not the first classifier in the ensemble
                vec_dot = (g_2.view(g_2.size(0),-1)*delta_local.view(delta_local.size(0),-1)).sum(dim=1)
                #print(vec_dot)

                beta_c = (beta_m)/(beta_m-zeta+1e-10)*(torch.abs(-vec_dot+zeta)) + rho
                #print(beta_c)

                beta[zeta<beta_m]=beta_c[zeta<beta_m]
                #print(beta)
                #assert(1==2)
            #print(beta)
            #print(beta)
            #assert(i==0)
            delta_local_hat = delta_local + beta.view(-1,1,1,1)*g
            delta_local_hat = beta_m*delta_local_hat/(torch.norm(delta_local_hat.view(w_j.shape[0],-1),p=float('inf'),dim=1).view(-1,1,1,1)+1e-10)
            #delta_local_hat = torch.clamp(delta_local_hat, min=-beta_m, max=beta_m)

            ## delta_local_hat is now a beta_m norm peturbation away from X+delta_min

            d = torch.clamp(delta_min + delta_local_hat, min=-epsilon, max=epsilon)
            ## now we have an epsilon norm perturbation away from X
            d = clamp(d, lower_limit - X, upper_limit - X)
            ## now it is also clipped to satisfy phyiscal constraints (X_adv \in [lower_limit, upper_limit])
            new_loss =  expected_acc(models,alpha,X+d,y,normalize=normalize,reduction='none')
            delta_local[new_loss <= loss_min_local] = delta_local_hat.detach()[new_loss <= loss_min_local]
            loss_min_local = torch.min(loss_min_local, new_loss)

        ## after going over the M classifiers, update delta_min and loss_min
        d = torch.clamp(delta_min + delta_local, min=-epsilon, max=epsilon)
        #d = (delta_min + delta_local).view(delta_min.size(0),-1).renorm(p=float('inf'),dim=0,maxnorm=epsilon).view_as(delta_min)
        ## now we have an epsilon norm perturbation away from X
        d = clamp(d, lower_limit - X, upper_limit - X)
        ## now it is also clipped to satisfy phyiscal constraints (X_adv \in [lower_limit, upper_limit])
        new_loss =  expected_acc(models,alpha,X+d,y,normalize=normalize,reduction='none')
        delta_min[new_loss <= loss_min] = d.detach()[new_loss <= loss_min]
        loss_min = torch.min(loss_min, new_loss)

    return delta_min



###
def attack_arc(models, X, y, epsilon, alpha, attack_iters, step_size, norm, num_classes, g=-1, normalize=None, rand_init=True):
    if g ==-1:
        if norm == 'l_2':
            return attack_ARC_l2(models, X, y, epsilon, alpha, attack_iters, step_size, norm, num_classes, normalize=normalize, rand_init=rand_init)
        elif norm == 'l_inf':
            return attack_ARC_linf(models, X, y, epsilon, alpha, attack_iters, step_size, norm, num_classes, normalize=normalize, rand_init=rand_init)
        else:
            raise ValueError('Invalid norm choice')
    else:
        topk = g+1
        if norm == 'l_2':
            return attack_ARC_l2_lite(models, X, y, epsilon, alpha, attack_iters, step_size, norm, num_classes, normalize=normalize, rand_init=rand_init, topk=topk)
        elif norm == 'l_inf':
            return attack_ARC_linf_lite(models, X, y, epsilon, alpha, attack_iters, step_size, norm, num_classes, normalize=normalize, rand_init=rand_init, topk=topk)
        else:
            raise ValueError('Invalid norm choice')
