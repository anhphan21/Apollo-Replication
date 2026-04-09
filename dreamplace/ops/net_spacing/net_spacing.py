##
# @file   net_spacing.py
# @author Anh Phan
# @date   Apr 2026
# @brief  Compute net spacing model.
#         NS = relu(s_i - |dx|)^2 + relu(s_i - |dy|)^2
#         where s_i = max(si_0, si_1) + s_crs * crossings
#               si_k = r_bend + 0.5 * node_num_ports_on_pin_side * s_crs
#

import time
import torch
from torch import nn
from torch.autograd import Function
import logging

import dreamplace.ops.net_spacing.net_spacing_cpp as net_spacing_cpp

logger = logging.getLogger(__name__)


class NetSpacingFunction(Function):
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
        tt = time.time()
        output = net_spacing_cpp.forward(
            pos.view(pos.numel()),
            pin_dir.view(pin_dir.numel()),
            pin_side,
            pin2net_map,
            pin2node_map,
            flat_netpin,
            netpin_start,
            net_weights,
            net_mask,
            node_num_ports,
            update_crossing,
            net_crossing_cnt,
            bend_radii,
            cross_size,
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
        output = net_spacing_cpp.backward(
            grad_pos,
            ctx.pos,
            ctx.grad_intermediate,
            ctx.flat_netpin,
            ctx.netpin_start,
            ctx.pin2net_map,
            ctx.net_weights,
            ctx.net_mask,
        )
        output[: int(output.numel() // 2)].masked_fill_(ctx.pin_mask, 0.0)
        output[int(output.numel() // 2) :].masked_fill_(ctx.pin_mask, 0.0)
        logger.debug("Net spacing backward %.3f ms" % ((time.time() - tt) * 1000))
        # return None for each forward argument except pos:
        # pos, pin_dir, pin_side, pin2net_map, pin2node_map, flat_netpin,
        # netpin_start, net_weights, net_mask, pin_mask, node_num_ports,
        # update_crossing, net_crossing_cnt, bend_radii, cross_size
        return output, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class NetSpacing(nn.Module):
    def __init__(
        self,
        flat_netpin=None,
        netpin_start=None,
        pin2net_map=None,
        pin2node_map=None,
        net_weights=None,
        net_mask=None,
        pin_mask=None,
        pin_dir=None,
        pin_side=None,
        node_num_ports=None,
        bend_radii=None,
        cross_size=None,
    ):
        super(NetSpacing, self).__init__()
        assert all(v is not None for v in [
            flat_netpin, netpin_start, pin2net_map, pin2node_map,
            net_weights, net_mask, pin_mask, pin_dir, pin_side,
            node_num_ports, bend_radii, cross_size,
        ]), "All parameters are required"

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.pin2net_map = pin2net_map
        self.pin2node_map = pin2node_map
        self.net_weights = net_weights
        self.net_mask = net_mask
        self.pin_mask = pin_mask
        self.pin_dir = pin_dir
        self.pin_side = pin_side
        self.node_num_ports = node_num_ports
        self.bend_radii = bend_radii
        self.cross_size = cross_size

        # Crossing state: updated periodically during placement
        num_nets = netpin_start.numel() - 1
        self.update_crossing = torch.tensor([False], dtype=torch.bool)
        self.net_crossing_cnt = torch.zeros(num_nets, dtype=torch.int32)

    def forward(self, pos):
        return NetSpacingFunction.apply(
            pos,
            self.pin_dir,
            self.pin_side,
            self.pin2net_map,
            self.pin2node_map,
            self.flat_netpin,
            self.netpin_start,
            self.net_weights,
            self.net_mask,
            self.pin_mask,
            self.node_num_ports,
            self.update_crossing,
            self.net_crossing_cnt,
            self.bend_radii,
            self.cross_size,
        )
