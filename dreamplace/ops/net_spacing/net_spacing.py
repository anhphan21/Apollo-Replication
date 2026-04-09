##
# @file   cos_weighted_average_wirelength.py
# @author Anh Phan
# @date   Apr 2026
# @brief  Compute net spacing model.
#

import time
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import logging

import dreamplace.ops.net_spacing.net_spacing as net_spacing

logger = logging.getLogger(__name__)


class NetSpacingFunction(Function):
    """
    @brief compute net spacing model (merged forward+grad).
    """

    @staticmethod
    def forward(
        ctx,
        pos,
        pin_dir,
        pin_side,
        pin2net_map,
        pin2node_map,
        flat_netpin,
        netpin_start,
        net_weights,
        net_mask,
        pin_mask,
        node_num_ports,
        update_crossing,
        net_crossing_cnt,
        bend_radii,
        cross_size,
    ):
        """
        @param pos pin location (x array, y array), not cell location
        @param flat_netpin flat netpin map, length of #pins
        @param netpin_start starting index in netpin map for each net, length of #nets+1
        @param pin2net_map pin2net map
        @param net_weights weight of nets
        @param net_mask whether to compute wirelength, 1 means to compute, 0 means to ignore
        @param pin_mask whether compute gradient for a pin, 1 means to fill with zero, 0 means to compute
        @param inv_gamma 1/gamma, the larger, the closer to HPWL
        @param pin_dir_x x-component of pin direction unit vectors (from port orientation)
        @param pin_dir_y y-component of pin direction unit vectors (from port orientation)
        """
        tt = time.time()
        func = net_spacing.forward
        output = func(
            pos.view(pos.numel()),
            pin_dir.view(pin_dir.numel()),

            flat_netpin,
            netpin_start,
            pin2net_map,
            net_weights,
            net_mask
        )
        ctx.pin2net_map = pin2net_map
        ctx.flat_netpin = flat_netpin
        ctx.netpin_start = netpin_start
        ctx.net_weights = net_weights
        ctx.net_mask = net_mask
        ctx.pin_mask = pin_mask
        ctx.grad_intermediate = output[1]
        ctx.pos = pos

        logger.debug("Net spacing forward %.3f ms" % ((time.time() - tt) * 1000))
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        tt = time.time()
        func = net_spacing.backward
        output = func(
            grad_pos,
            ctx.pos,
            ctx.grad_intermediate,
            ctx.flat_netpin,
            ctx.netpin_start,
            ctx.pin2net_map,
            ctx.net_weights,
            ctx.net_mask,
            ctx.inv_gamma,
        )
        output[: int(output.numel() // 2)].masked_fill_(ctx.pin_mask, 0.0)
        output[int(output.numel() // 2) :].masked_fill_(ctx.pin_mask, 0.0)
        if grad_pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("cos wirelength backward %.3f ms" % ((time.time() - tt) * 1000))
        # return None for each forward argument: pos, flat_netpin, netpin_start,
        # pin2net_map, net_weights, net_mask, pin_mask, inv_gamma,
        # pin_dir_x, pin_dir_y, c, alpha
        return output, None, None, None, None, None, None, None, None, None, None, None


class NetSpacing(nn.Module):
    """
    @brief Compute cosine-weighted average wirelength for PIC placement.
    cosWA_e = (1 + W_theta) * (WA_x + WA_y)^alpha
    where W_theta penalizes wire directions that deviate from pin port orientations.
    """

    def __init__(
        self,
        flat_netpin=None,
        netpin_start=None,
        pin2net_map=None,
        net_weights=None,
        net_mask=None,
        pin_mask=None,
        gamma=None,
        pin_dir_x=None,
        pin_dir_y=None,
        c=0.5,
        alpha=1.0,
    ):
        """
        @brief initialization
        @param flat_netpin flat netpin map, length of #pins
        @param netpin_start starting index in netpin map for each net, length of #nets+1, the last entry is #pins
        @param pin2net_map pin2net map
        @param net_weights weight of nets
        @param net_mask whether to compute wirelength, 1 means to compute, 0 means to ignore
        @param pin_mask whether compute gradient for a pin, 1 means to fill with zero, 0 means to compute
        @param gamma the smaller, the closer to HPWL
        @param pin_dir_x 1D tensor, x-component of pin direction unit vectors
        @param pin_dir_y 1D tensor, y-component of pin direction unit vectors
        @param c cosine threshold for penalty (scalar)
        @param alpha power exponent on WA wirelength (scalar)
        """
        super(NetSpacing, self).__init__()
        assert (
            net_weights is not None
            and net_mask is not None
            and pin_mask is not None
            and gamma is not None
        ), "net_weights, net_mask, pin_mask, gamma are required parameters"
        assert (
            flat_netpin is not None
            and netpin_start is not None
            and pin2net_map is not None
        ), "flat_netpin, netpin_start, pin2net_map are required parameters"
        assert pin_dir_x is not None and pin_dir_y is not None, (
            "pin_dir_x and pin_dir_y are required parameters"
        )

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2net_map = pin2net_map
        self.net_weights = net_weights
        self.net_mask = net_mask
        self.pin_mask = pin_mask
        self.gamma = gamma
        self.pin_dir_x = pin_dir_x
        self.pin_dir_y = pin_dir_y
        # store c and alpha as scalar tensors (same dtype as pin_dir_x)
        dtype = pin_dir_x.dtype
        device = pin_dir_x.device
        self.c = (
            torch.tensor(c, dtype=dtype, device=device)
            if not isinstance(c, torch.Tensor)
            else c
        )
        self.alpha = (
            torch.tensor(alpha, dtype=dtype, device=device)
            if not isinstance(alpha, torch.Tensor)
            else alpha
        )

    def forward(self, pos):
        return NetSpacingFunction.apply(
            pos,
            self.flat_netpin,
            self.netpin_start,
            self.pin2net_map,
            self.net_weights,
            self.net_mask,
            self.pin_mask,
            self.pin_dir_x,
            self.pin_dir_y,
            self.c,
            self.alpha,
        )
