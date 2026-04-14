##
# @file   NesterovAcceleratedGradientOptimizer.py
# @author Yibo Lin
# @date   Aug 2018
# @brief  Nesterov's accelerated gradient method proposed by e-place.
#

import os
import sys
import time
import math
import pickle
import numpy as np
import torch
from collections import deque
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn
import pdb

class NesterovAcceleratedGradientOptimizer(Optimizer):
    """
    @brief Follow the Nesterov's implementation of e-place algorithm 2
    http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf
    Supports Blockwise Adaptive Nesterov-accelerated Gradient Descent (BNAG).
    """
    def __init__(self, params, lr=required, obj_and_grad_fn=required, constraint_fn=None, use_bb=True,
                 num_movable_nodes=0, num_nodes=0, num_filler_nodes=0, K_max=1000, eta_0=1.0):
        """
        @brief initialization
        @param params variable to optimize
        @param lr learning rate
        @param obj_and_grad_fn a callable function to get objective and gradient
        @param constraint_fn a callable function to force variables to satisfy all the constraints
        @param num_movable_nodes number of movable nodes (for blockwise step sizes)
        @param num_nodes total number of nodes (movable + fixed + filler)
        @param num_filler_nodes number of filler nodes
        @param K_max maximum number of iterations (for cosine annealing)
        @param eta_0 initial learning rate multiplier
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # u_k is major solution
        # v_k is reference solution
        # obj_k is the objective at v_k
        # a_k is optimization parameter
        # alpha_k is the step size (per-block list for BNAG)
        # v_k_1 is previous reference solution
        # g_k_1 is gradient to v_k_1
        # obj_k_1 is the objective at v_k_1
        defaults = dict(lr=lr,
                u_k=[], v_k=[], g_k=[], obj_k=[], a_k=[], alpha_k=[],
                v_k_1=[], g_k_1=[], obj_k_1=[],
                v_kp1 = [None],
                obj_eval_count=0)
        super(NesterovAcceleratedGradientOptimizer, self).__init__(params, defaults)
        self.obj_and_grad_fn = obj_and_grad_fn
        self.constraint_fn = constraint_fn
        self.use_bb = use_bb

        # BNAG parameters
        self.num_movable_nodes = num_movable_nodes
        self.num_nodes = num_nodes
        self.num_filler_nodes = num_filler_nodes
        self.K_max = K_max
        self.eta_0 = eta_0
        self.eta_k = eta_0
        self.eta_history = deque(maxlen=5)

        # Build block slices: x_movable, x_filler, y_movable, y_filler
        self.blocks = []
        if num_nodes > 0:
            num_fixed = num_nodes - num_movable_nodes - num_filler_nodes
            self.blocks.append(slice(0, num_movable_nodes))                                          # x_movable
            self.blocks.append(slice(num_movable_nodes + num_fixed, num_nodes))                      # x_filler
            self.blocks.append(slice(num_nodes, num_nodes + num_movable_nodes))                      # y_movable
            self.blocks.append(slice(num_nodes + num_movable_nodes + num_fixed, 2 * num_nodes))      # y_filler
            # Remove empty blocks (e.g., no fillers)
            self.blocks = [b for b in self.blocks if b.stop > b.start]

        # I do not know how to get generator's length
        if len(self.param_groups) != 1:
            raise ValueError("Only parameters with single tensor is supported")

    def __setstate__(self, state):
        super(NesterovAcceleratedGradientOptimizer, self).__setstate__(state)

    def step(self, closure=None, iteration=0):
        if self.use_bb:
            self.step_bb(closure, iteration)
        else:
            self.step_nobb(closure)

    def step_nobb(self, closure=None):
        """
        @brief Performs a single optimization step.
        @param closure A callable closure function that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if not group['u_k']:
                    group['u_k'].append(p.data.clone())
                    # directly use p as v_k to save memory
                    #group['v_k'].append(torch.autograd.Variable(p.data, requires_grad=True))
                    group['v_k'].append(p)
                    obj, grad = obj_and_grad_fn(group['v_k'][i])
                    group['g_k'].append(grad.data.clone()) # must clone
                    group['obj_k'].append(obj.data.clone())
                u_k = group['u_k'][i]
                v_k = group['v_k'][i]
                g_k = group['g_k'][i]
                obj_k = group['obj_k'][i]
                if not group['a_k']:
                    group['a_k'].append(torch.ones(1, dtype=g_k.dtype, device=g_k.device))
                    group['v_k_1'].append(torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True))
                    group['v_k_1'][i].data.copy_(group['v_k'][i]-group['lr']*g_k)
                    obj, grad = obj_and_grad_fn(group['v_k_1'][i])
                    group['g_k_1'].append(grad.data)
                    group['obj_k_1'].append(obj.data.clone())
                a_k = group['a_k'][i]
                v_k_1 = group['v_k_1'][i]
                g_k_1 = group['g_k_1'][i]
                obj_k_1 = group['obj_k_1'][i]
                if not group['alpha_k']:
                    group['alpha_k'].append((v_k-v_k_1).norm(p=2) / (g_k-g_k_1).norm(p=2))
                alpha_k = group['alpha_k'][i]

                if group['v_kp1'][i] is None:
                    group['v_kp1'][i] = torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True)
                v_kp1 = group['v_kp1'][i]

                # line search with alpha_k as hint
                a_kp1 = (1 + (4*a_k.pow(2)+1).sqrt()) / 2
                coef = (a_k-1) / a_kp1
                alpha_kp1 = 0
                backtrack_cnt = 0
                max_backtrack_cnt = 10

                #ttt = time.time()
                while True:
                    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    u_kp1 = v_k - alpha_k*g_k
                    #constraint_fn(u_kp1)
                    v_kp1.data.copy_(u_kp1 + coef*(u_kp1-u_k))
                    # make sure v_kp1 subjects to constraints
                    # g_kp1 must correspond to v_kp1
                    constraint_fn(v_kp1)

                    f_kp1, g_kp1 = obj_and_grad_fn(v_kp1)

                    #tt = time.time()
                    alpha_kp1 = torch.sqrt(torch.sum((v_kp1.data-v_k.data)**2) / torch.sum((g_kp1.data-g_k.data)**2))
                    # alpha_kp1 = torch.dist(v_kp1.data, v_k.data, p=2) / torch.dist(g_kp1.data, g_k.data, p=2)
                    backtrack_cnt += 1
                    group['obj_eval_count'] += 1
                    #logging.debug("\t\talpha_kp1 %.3f ms" % ((time.time()-tt)*1000))
                    #torch.cuda.synchronize()
                    #logging.debug(prof)

                    #logging.debug("alpha_kp1 = %g, line_search_count = %d, obj_eval_count = %d" % (alpha_kp1, backtrack_cnt, group['obj_eval_count']))
                    #logging.debug("|g_k| = %.6E, |g_kp1| = %.6E" % (g_k.norm(p=2), g_kp1.norm(p=2)))
                    if alpha_kp1 > 0.95*alpha_k or backtrack_cnt >= max_backtrack_cnt:
                        alpha_k.data.copy_(alpha_kp1.data)
                        break
                    else:
                        alpha_k.data.copy_(alpha_kp1.data)
                #if v_k.is_cuda:
                #    torch.cuda.synchronize()
                #logging.debug("\tline search %.3f ms" % ((time.time()-ttt)*1000))

                v_k_1.data.copy_(v_k.data)
                g_k_1.data.copy_(g_k.data)
                obj_k_1.data.copy_(obj_k.data)

                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                g_k.data.copy_(g_kp1.data)
                obj_k.data.copy_(f_kp1.data)
                a_k.data.copy_(a_kp1.data)

                # although the solution should be u_k
                # we need the gradient of v_k
                # the update of density weight also requires v_k
                # I do not know how to copy u_k back to p when exit yet
                #p.data.copy_(v_k.data)

        return loss

    def step_bb(self, closure=None, iteration=0):
        """
        @brief Blockwise Adaptive Nesterov-accelerated Gradient Descent (BNAG).
        Each block (x_movable, x_filler, y_movable, y_filler) gets its own
        BB step size, scaled by a cosine-annealed learning rate eta_k.
        @param iteration current iteration index k (for cosine annealing schedule).
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                # --- First-call initialization ---
                if not group['u_k']:
                    group['u_k'].append(p.data.clone())
                    group['v_k'].append(p)
                u_k = group['u_k'][i]
                v_k = group['v_k'][i]

                obj_k, g_k = obj_and_grad_fn(v_k)
                if not group['obj_k']:
                    group['obj_k'].append(None)
                group['obj_k'][i] = obj_k.data.clone()

                if not group['a_k']:
                    group['a_k'].append(torch.ones(1, dtype=g_k.dtype, device=g_k.device))
                    group['v_k_1'].append(torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True))
                    group['v_k_1'][i].data.copy_(group['v_k'][i] - group['lr'] * g_k)
                a_k = group['a_k'][i]
                v_k_1 = group['v_k_1'][i]

                obj_k_1, g_k_1 = obj_and_grad_fn(v_k_1)
                if not group['obj_k_1']:
                    group['obj_k_1'].append(None)
                group['obj_k_1'][i] = obj_k_1.data.clone()

                if group['v_kp1'][i] is None:
                    group['v_kp1'][i] = torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True)
                v_kp1 = group['v_kp1'][i]

                # Initialize per-block alpha_k on first call
                num_blocks = len(self.blocks)
                if not group['alpha_k']:
                    alpha_list = []
                    for blk in self.blocks:
                        s_blk = v_k.data[blk] - v_k_1.data[blk]
                        y_blk = g_k.data[blk] - g_k_1.data[blk]
                        alpha_list.append((s_blk.norm(p=2) / y_blk.norm(p=2)).clone())
                    group['alpha_k'].append(alpha_list)

                alpha_k_blocks = group['alpha_k'][i]  # list of per-block step sizes

                # --- Nesterov momentum parameters ---
                a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
                coef = (a_k - 1) / a_kp1

                # --- Compute per-block BB step sizes (Algorithm lines 10-23) ---
                u_kp1 = v_k.data.clone()
                with torch.no_grad():
                    for j, blk in enumerate(self.blocks):
                        s_j = v_k.data[blk] - v_k_1.data[blk]
                        y_j = g_k.data[blk] - g_k_1.data[blk]

                        y_dot_y = y_j.dot(y_j)
                        s_dot_y = s_j.dot(y_j)

                        bb_short = s_dot_y / y_dot_y   # alpha_bb^(j)
                        lip = s_j.norm(p=2) / y_j.norm(p=2)  # alpha_lip^(j)

                        if bb_short > 0:
                            step_j = bb_short
                        else:
                            step_j = min(lip.item(), alpha_k_blocks[j].item())

                        step_j = step_j * self.eta_k   # scale by eta_k
                        alpha_k_blocks[j].fill_(step_j)

                        # Gradient step for this block: u_{k+1}[B_j] = v_k[B_j] - alpha_k^(j) * grad[B_j]
                        u_kp1[blk] = v_k.data[blk] - step_j * g_k.data[blk]

                # --- Nesterov momentum: v_{k+1} = u_{k+1} + coef * (u_{k+1} - u_k) ---
                v_kp1.data.copy_(u_kp1 + coef * (u_kp1 - u_k.data))

                # --- Enforce constraints ---
                constraint_fn(v_kp1)
                group['obj_eval_count'] += 1

                # --- Update cosine-annealed eta for next iteration ---
                self.eta_history.append(self.eta_k)
                eta_min = min(self.eta_history)
                self.eta_k = eta_min + 0.5 * (self.eta_0 - eta_min) * (1 + math.cos(math.pi * iteration / max(self.K_max, 1)))

                # --- Shift state for next iteration ---
                v_k_1.data.copy_(v_k.data)
                u_k.data.copy_(u_kp1)
                v_k.data.copy_(v_kp1.data)
                a_k.data.copy_(a_kp1.data)

        return loss