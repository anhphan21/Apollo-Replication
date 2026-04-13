##
# @file   NesterovAcceleratedGradientOptimizer.py
# @author Yibo Lin
# @date   Aug 2018
# @brief  Nesterov's accelerated gradient method proposed by e-place.
#

import os
import sys
import time
import pickle
import math  # new: cosine annealing for BNAG
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn
import pdb

class NesterovAcceleratedGradientOptimizer(Optimizer):
    """
    @brief Follow the Nesterov's implementation of e-place algorithm 2
    http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf
    """
    def __init__(self, params, lr=required, obj_and_grad_fn=required, constraint_fn=None, use_bb=True,
                 block_indices=None, eta_0=1.0, eta_min=0.05, k_max=1500,  # new: BNAG params
                 projection_fn=None, proj_s0=0.0, proj_sT=1.0):  # new: PIC projection params
        """
        @brief initialization
        @param params variable to optimize
        @param lr learning rate
        @param obj_and_grad_fn a callable function to get objective and gradient
        @param constraint_fn a callable function to force variables to satisfy all the constraints
        @param block_indices list of 4 LongTensors selecting (x-movable, y-movable, x-filler, y-filler) entries
        @param eta_0 initial cosine annealing scale for BNAG step sizes
        @param eta_min minimum cosine annealing scale for BNAG step sizes
        @param k_max cosine annealing horizon (number of steps)
        @param projection_fn callable(pos, s_t) that projects pos onto the
                             feasible PIC constraint region in-place; called
                             after constraint_fn (move_boundary)
        @param proj_s0 initial blending factor for the projection cosine schedule
        @param proj_sT final blending factor for the projection cosine schedule
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # u_k is major solution
        # v_k is reference solution
        # obj_k is the objective at v_k
        # a_k is optimization parameter
        # alpha_k is the step size
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
        # self.block_indices = block_indices  # new: per-block index selectors for BNAG
        self.eta_0 = eta_0  # new: initial cosine annealing scale
        self.eta_min = eta_min  # new: minimum cosine annealing scale
        self.k_max = k_max  # new: cosine annealing horizon
        self.k = 0  # new: BNAG step counter
        self.eta_k = eta_0  # new: current cosine annealing factor; updated AFTER each step (Alg.1 line 17)
        self.projection_fn = projection_fn  # new: PIC constraint projection callable
        self.proj_s0 = proj_s0  # new: initial projection blending factor
        self.proj_sT = proj_sT  # new: final projection blending factor
        self.s_t = proj_s0  # new: current projection blending factor (cos schedule, k=0 -> s_0)

        # I do not know how to get generator's length
        if len(self.param_groups) != 1:
            raise ValueError("Only parameters with single tensor is supported")

    def __setstate__(self, state):
        super(NesterovAcceleratedGradientOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        if self.use_bb:
            self.step_bb(closure)
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

    def step_bb(self, closure=None):
        """
        @brief Performs a single optimization step.
        @param closure A callable closure function that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        eta_k = self.eta_k  # new: snapshot the current cosine annealing factor for use during this step
        s_t = self.s_t  # new: snapshot the current projection blending factor for use during this step

        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
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
                    group['v_k_1'][i].data.copy_(group['v_k'][i]-group['lr']*g_k)
                a_k = group['a_k'][i]
                v_k_1 = group['v_k_1'][i]
                obj_k_1, g_k_1 = obj_and_grad_fn(v_k_1)
                if not group['obj_k_1']:
                    group['obj_k_1'].append(None)
                group['obj_k_1'][i] = obj_k_1.data.clone()
                if group['v_kp1'][i] is None:
                    group['v_kp1'][i] = torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True)
                v_kp1 = group['v_kp1'][i]
                # Lazy fallback: non-PIC runs pass block_indices=None, so treat
                # the whole parameter as a single global block (recovers vanilla BB).
                # if self.block_indices is None:
                #     self.block_indices = [torch.arange(0, v_k.numel(), dtype=torch.long, device=v_k.device)]
                # num_blocks = len(self.block_indices)  # 1 for LEF/DEF, 4 for PIC
                if not group['alpha_k']:
                    init_alpha = []  # new: one alpha per logical block
                    with torch.no_grad():  # new
                        s0 = v_k - v_k_1  # new
                        y0 = g_k - g_k_1  # new
                        for j in range(num_blocks):  # new: initialize each block alpha separately
                            idx_j = self.block_indices[j]  # new
                            if idx_j.numel() == 0:  # new: empty block (e.g. no fillers) -> dummy 0 step
                                init_alpha.append(torch.zeros(1, dtype=g_k.dtype, device=g_k.device))  # new
                            else:  # new
                                init_alpha.append((s0[idx_j].norm(p=2) / y0[idx_j].norm(p=2)).detach().clone())  # new
                    group['alpha_k'].append(init_alpha)  # changed: store list-of-N instead of single scalar
                alpha_k = group['alpha_k'][i]  # changed: alpha_k is now a list of N scalar tensors
                # line search with alpha_k as hint
                a_kp1 = (1 + (4*a_k.pow(2)+1).sqrt()) / 2
                coef = (a_k-1) / a_kp1
                with torch.no_grad():
                    s_k = (v_k - v_k_1)
                    y_k = (g_k - g_k_1)
                    for j in range(num_blocks):  # new: blockwise BB step computation
                        idx_j = self.block_indices[j]  # new
                        if idx_j.numel() == 0:  # new: empty block -> nothing to update
                            continue  # new
                        s_j = s_k[idx_j]  # new
                        y_j = y_k[idx_j]  # new
                        bb_short = (s_j.dot(y_j) / y_j.dot(y_j)).data  # new
                        lip = (s_j.norm(p=2) / y_j.norm(p=2)).data  # new
                        step_j = bb_short if bb_short > 0 else torch.min(lip, alpha_k[j])  # new
                        alpha_k[j] = (step_j * eta_k).detach().clone()  # new: scale by cosine annealing factor

                # new: per-block gradient descent update
                u_kp1 = v_k.clone()  # new: start from v_k, then patch each block
                for j in range(num_blocks):  # new
                    idx_j = self.block_indices[j]  # new
                    if idx_j.numel() == 0:  # new: skip empty blocks
                        continue  # new
                    u_kp1[idx_j] = v_k[idx_j] - alpha_k[j] * g_k[idx_j]  # new

                v_kp1.data.copy_(u_kp1 + coef*(u_kp1-u_k))
                # Alg.1 line 18: projection onto the feasible placement region.
                # 1) clamp to the die box (move_boundary_op)
                constraint_fn(v_kp1)
                # 2) PIC alignment / uniform-spacing projection blended with
                #    the cosine schedule s_t (no-op if projection_fn is None
                #    or there are no constraints).
                if self.projection_fn is not None:
                    self.projection_fn(v_kp1.data, s_t)
                group['obj_eval_count'] += 1

                v_k_1.data.copy_(v_k.data)
                #g_k_1.data.copy_(g_k.data)
                #obj_k_1.data.copy_(obj_k.data)
                # alpha_k state copy removed: alpha_k is now a list-of-4 updated in-place above (changed)
                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                #g_k.data.copy_(g_kp1.data)
                #obj_k.data.copy_(f_kp1.data)
                a_k.data.copy_(a_kp1.data)

                # new (Alg.1 line 17): update cosine annealing factor AFTER the step,
                # using the current k, then advance k for the next call.
                self.eta_k = self.eta_min + 0.5 * (self.eta_0 - self.eta_min) * (1.0 + math.cos(math.pi * self.k / self.k_max))
                # new: update the projection blending factor on the same schedule.
                # s_t = s_0 + (s_T - s_0) * (1 - cos(pi * k / k_max)) / 2  -- monotonic s_0 -> s_T
                self.s_t = self.proj_s0 + (self.proj_sT - self.proj_s0) * 0.5 * (1.0 - math.cos(math.pi * self.k / self.k_max))
                self.k += 1

                # although the solution should be u_k
                # we need the gradient of v_k
                # the update of density weight also requires v_k
                # I do not know how to copy u_k back to p when exit yet
                #p.data.copy_(v_k.data)
        return loss