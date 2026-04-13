##
# @file   cos_weighted_average_wirelength_unittest.py
# @author Anh Phan
# @date   Apr 2026
# @brief  Unit test for cosine-weighted average wirelength for PIC placement.
#
# cosWA_e = (1 + W_theta) * H^alpha
# where H = WA_x + WA_y (standard weighted-average wirelength),
#       W_theta = ReLU(c - cos(theta_1))^2 + ReLU(c - cos(theta_2))^2
#       (only for 2-pin nets; 0 otherwise).

import os
import sys
import numpy as np
import unittest

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
from dreamplace.ops.cos_weighted_average_wirelength import cos_weighted_average_wirelength
sys.path.pop()

import torch
from torch.autograd import Variable


def unsorted_segment_sum(pin_x, pin2net_map, num_nets):
    result = np.zeros(num_nets, dtype=pin_x.dtype)
    for i in range(len(pin2net_map)):
        result[pin2net_map[i]] += pin_x[i]
    return result


def build_cos_wirelength(pin_x, pin_y, pin2net_map, net2pin_map,
                         gamma, net_weights, pin_dir_x, pin_dir_y, c, alpha):
    """
    Numpy reference implementation of cosine-weighted average wirelength.
    Only supports 2-pin nets.
    """
    num_nets = len(net2pin_map)
    eps = 1e-6

    per_net_wl = np.zeros(num_nets, dtype=pin_x.dtype)

    for i in range(num_nets):
        pins = net2pin_map[i]
        assert len(pins) == 2, "cos_weighted_average_wirelength only supports 2-pin nets"

        # Step 1: standard WA wirelength
        net_pin_x = pin_x[pins]
        net_pin_y = pin_y[pins]

        x_max = np.max(net_pin_x)
        x_min = np.min(net_pin_x)
        y_max = np.max(net_pin_y)
        y_min = np.min(net_pin_y)

        inv_gamma = 1.0 / gamma

        exp_x = np.exp((net_pin_x - x_max) * inv_gamma)
        exp_nx = np.exp((x_min - net_pin_x) * inv_gamma)
        exp_y = np.exp((net_pin_y - y_max) * inv_gamma)
        exp_ny = np.exp((y_min - net_pin_y) * inv_gamma)

        WA_x = np.sum(net_pin_x * exp_x) / np.sum(exp_x) - np.sum(net_pin_x * exp_nx) / np.sum(exp_nx)
        WA_y = np.sum(net_pin_y * exp_y) / np.sum(exp_y) - np.sum(net_pin_y * exp_ny) / np.sum(exp_ny)
        H = WA_x + WA_y

        # Step 2: cosine penalty for 2-pin nets
        p1, p2 = pins[0], pins[1]
        dx = pin_x[p2] - pin_x[p1]
        dy = pin_y[p2] - pin_y[p1]
        r = np.sqrt(dx * dx + dy * dy)

        W_theta = 0.0
        if r > eps:
            inv_r = 1.0 / r
            cos1 = (dx * pin_dir_x[p1] + dy * pin_dir_y[p1]) * inv_r
            relu1 = max(0.0, c - cos1)

            cos2 = (-dx * pin_dir_x[p2] - dy * pin_dir_y[p2]) * inv_r
            relu2 = max(0.0, c - cos2)

            W_theta = relu1 * relu1 + relu2 * relu2

        # Step 3: combine
        H_safe = max(H, eps)
        H_alpha = H_safe ** alpha
        per_net_wl[i] = (1.0 + W_theta) * H_alpha

    per_net_wl *= net_weights
    return np.sum(per_net_wl)


class CosWeightedAverageWirelengthOpTest(unittest.TestCase):

    def _make_test_data(self, dtype=np.float32):
        """Create a small test case with 2-pin nets."""
        # 4 pins, 2 nets (each net has exactly 2 pins)
        pin_pos = np.array(
            [[0.0, 0.0], [3.0, 4.0], [1.0, 1.0], [5.0, 2.0]],
            dtype=dtype)

        net2pin_map = [np.array([0, 1]), np.array([2, 3])]
        pin2net_map = np.zeros(len(pin_pos), dtype=np.int32)
        for net_id, pins in enumerate(net2pin_map):
            for pin in pins:
                pin2net_map[pin] = net_id

        net_weights = np.array([1.0, 1.5], dtype=dtype)

        # pin direction unit vectors (port orientations)
        # net 0: pin 0 points right (+x), pin 1 points left (-x)
        # net 1: pin 2 points up (+y), pin 3 points down (-y)
        pin_dir_x = np.array([1.0, -1.0, 0.0, 0.0], dtype=dtype)
        pin_dir_y = np.array([0.0, 0.0, 1.0, -1.0], dtype=dtype)

        gamma = 0.5
        c = 0.5
        alpha = 1.0

        # net mask: all nets active
        net_mask = np.ones(len(net2pin_map), dtype=np.uint8)
        # pin mask: no pins masked
        pin_mask = np.zeros(len(pin2net_map), dtype=np.uint8)

        # construct flat_net2pin_map and flat_net2pin_start_map
        flat_net2pin_map = np.zeros(len(pin_pos), dtype=np.int32)
        flat_net2pin_start_map = np.zeros(len(net2pin_map) + 1, dtype=np.int32)
        count = 0
        for i in range(len(net2pin_map)):
            flat_net2pin_map[count:count + len(net2pin_map[i])] = net2pin_map[i]
            flat_net2pin_start_map[i] = count
            count += len(net2pin_map[i])
        flat_net2pin_start_map[len(net2pin_map)] = len(pin_pos)

        return (pin_pos, net2pin_map, pin2net_map, net_weights,
                pin_dir_x, pin_dir_y, gamma, c, alpha,
                net_mask, pin_mask, flat_net2pin_map, flat_net2pin_start_map)

    def test_value_alpha1(self):
        """Test forward value with alpha=1.0 against numpy reference."""
        dtype = torch.float64  # use float64 for tighter tolerance
        (pin_pos, net2pin_map, pin2net_map, net_weights,
         pin_dir_x, pin_dir_y, gamma, c, alpha,
         net_mask, pin_mask, flat_net2pin_map, flat_net2pin_start_map
         ) = self._make_test_data(dtype=np.float64)

        pin_x = pin_pos[:, 0]
        pin_y = pin_pos[:, 1]

        golden = build_cos_wirelength(
            pin_x, pin_y, pin2net_map, net2pin_map,
            gamma, net_weights, pin_dir_x, pin_dir_y, c, alpha)
        print("golden_value = ", golden)

        pin_pos_var = Variable(
            torch.tensor(np.transpose(pin_pos), dtype=dtype).reshape([-1]),
            requires_grad=True)

        custom = cos_weighted_average_wirelength.CosWeightedAverageWirelength(
            flat_netpin=torch.from_numpy(flat_net2pin_map),
            netpin_start=torch.from_numpy(flat_net2pin_start_map),
            pin2net_map=torch.from_numpy(pin2net_map),
            net_weights=torch.from_numpy(net_weights),
            net_mask=torch.from_numpy(net_mask),
            pin_mask=torch.from_numpy(pin_mask),
            gamma=torch.tensor(gamma, dtype=dtype),
            pin_dir_x=torch.from_numpy(pin_dir_x),
            pin_dir_y=torch.from_numpy(pin_dir_y),
            c=c,
            alpha=alpha)

        result = custom.forward(pin_pos_var)
        print("custom_result = ", result.item())

        np.testing.assert_allclose(result.item(), golden, rtol=1e-5, atol=1e-6)

    def test_value_alpha2(self):
        """Test forward value with alpha=2.0 against numpy reference."""
        dtype = torch.float64
        (pin_pos, net2pin_map, pin2net_map, net_weights,
         pin_dir_x, pin_dir_y, gamma, c, _,
         net_mask, pin_mask, flat_net2pin_map, flat_net2pin_start_map
         ) = self._make_test_data(dtype=np.float64)

        alpha = 2.0
        pin_x = pin_pos[:, 0]
        pin_y = pin_pos[:, 1]

        golden = build_cos_wirelength(
            pin_x, pin_y, pin2net_map, net2pin_map,
            gamma, net_weights, pin_dir_x, pin_dir_y, c, alpha)
        print("golden_value (alpha=2) = ", golden)

        pin_pos_var = Variable(
            torch.tensor(np.transpose(pin_pos), dtype=dtype).reshape([-1]),
            requires_grad=True)

        custom = cos_weighted_average_wirelength.CosWeightedAverageWirelength(
            flat_netpin=torch.from_numpy(flat_net2pin_map),
            netpin_start=torch.from_numpy(flat_net2pin_start_map),
            pin2net_map=torch.from_numpy(pin2net_map),
            net_weights=torch.from_numpy(net_weights),
            net_mask=torch.from_numpy(net_mask),
            pin_mask=torch.from_numpy(pin_mask),
            gamma=torch.tensor(gamma, dtype=dtype),
            pin_dir_x=torch.from_numpy(pin_dir_x),
            pin_dir_y=torch.from_numpy(pin_dir_y),
            c=c,
            alpha=alpha)

        result = custom.forward(pin_pos_var)
        print("custom_result (alpha=2) = ", result.item())

        np.testing.assert_allclose(result.item(), golden, rtol=1e-5, atol=1e-6)

    def test_gradient_finite_difference(self):
        """Verify analytical gradient matches finite-difference gradient."""
        dtype = torch.float64
        (pin_pos, net2pin_map, pin2net_map, net_weights,
         pin_dir_x, pin_dir_y, gamma, c, alpha,
         net_mask, pin_mask, flat_net2pin_map, flat_net2pin_start_map
         ) = self._make_test_data(dtype=np.float64)

        pin_pos_var = Variable(
            torch.tensor(np.transpose(pin_pos), dtype=dtype).reshape([-1]),
            requires_grad=True)

        custom = cos_weighted_average_wirelength.CosWeightedAverageWirelength(
            flat_netpin=torch.from_numpy(flat_net2pin_map),
            netpin_start=torch.from_numpy(flat_net2pin_start_map),
            pin2net_map=torch.from_numpy(pin2net_map),
            net_weights=torch.from_numpy(net_weights),
            net_mask=torch.from_numpy(net_mask),
            pin_mask=torch.from_numpy(pin_mask),
            gamma=torch.tensor(gamma, dtype=dtype),
            pin_dir_x=torch.from_numpy(pin_dir_x),
            pin_dir_y=torch.from_numpy(pin_dir_y),
            c=c,
            alpha=alpha)

        # analytical gradient
        result = custom.forward(pin_pos_var)
        result.backward()
        analytical_grad = pin_pos_var.grad.clone().numpy()

        # finite-difference gradient
        epsilon = 1e-5
        pos_np = pin_pos_var.data.numpy().copy()
        num_elements = len(pos_np)
        numerical_grad = np.zeros(num_elements, dtype=np.float64)

        for i in range(num_elements):
            pos_plus = pos_np.copy()
            pos_plus[i] += epsilon
            pos_minus = pos_np.copy()
            pos_minus[i] -= epsilon

            wl_plus = custom.forward(
                torch.tensor(pos_plus, dtype=dtype)).item()
            wl_minus = custom.forward(
                torch.tensor(pos_minus, dtype=dtype)).item()
            numerical_grad[i] = (wl_plus - wl_minus) / (2.0 * epsilon)

        print("\nGradient comparison (analytical vs numerical):")
        print("%6s %14s %14s %14s" % ("idx", "analytical", "numerical", "rel_error"))
        for i in range(num_elements):
            denom = max(abs(analytical_grad[i]), abs(numerical_grad[i]), 1e-20)
            rel_err = abs(analytical_grad[i] - numerical_grad[i]) / denom
            print("%6d %14.6E %14.6E %14.6E" % (
                i, analytical_grad[i], numerical_grad[i], rel_err))

        np.testing.assert_allclose(analytical_grad, numerical_grad,
                                   rtol=1e-3, atol=1e-5)

    def test_no_cosine_penalty(self):
        """When pin directions are aligned with wire, W_theta should be 0
        and result should equal standard WA wirelength."""
        dtype = torch.float64
        # 2 pins along +x axis
        pin_pos = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        net2pin_map = [np.array([0, 1])]
        pin2net_map = np.array([0, 0], dtype=np.int32)
        net_weights = np.array([1.0], dtype=np.float64)
        net_mask = np.ones(1, dtype=np.uint8)
        pin_mask = np.zeros(2, dtype=np.uint8)
        flat_net2pin_map = np.array([0, 1], dtype=np.int32)
        flat_net2pin_start_map = np.array([0, 2], dtype=np.int32)

        # pin directions: pin 0 points +x, pin 1 points -x (perfectly aligned)
        pin_dir_x = np.array([1.0, -1.0], dtype=np.float64)
        pin_dir_y = np.array([0.0, 0.0], dtype=np.float64)
        gamma = 0.5
        c = 0.5    # cos(theta) should be 1.0 >> c, so relu=0
        alpha = 1.0

        pin_pos_var = torch.tensor(
            np.transpose(pin_pos).reshape([-1]), dtype=dtype, requires_grad=True)

        custom = cos_weighted_average_wirelength.CosWeightedAverageWirelength(
            flat_netpin=torch.from_numpy(flat_net2pin_map),
            netpin_start=torch.from_numpy(flat_net2pin_start_map),
            pin2net_map=torch.from_numpy(pin2net_map),
            net_weights=torch.from_numpy(net_weights),
            net_mask=torch.from_numpy(net_mask),
            pin_mask=torch.from_numpy(pin_mask),
            gamma=torch.tensor(gamma, dtype=dtype),
            pin_dir_x=torch.from_numpy(pin_dir_x),
            pin_dir_y=torch.from_numpy(pin_dir_y),
            c=c,
            alpha=alpha)

        result = custom.forward(pin_pos_var)

        # compute standard WA wirelength for comparison
        pin_x = pin_pos[:, 0]
        pin_y = pin_pos[:, 1]
        golden = build_cos_wirelength(
            pin_x, pin_y, pin2net_map, net2pin_map,
            gamma, net_weights, pin_dir_x, pin_dir_y, c, alpha)

        print("aligned result = ", result.item(), " golden = ", golden)
        np.testing.assert_allclose(result.item(), golden, rtol=1e-5)

    def test_perpendicular_penalty(self):
        """When wire is perpendicular to pin directions, penalty should be large."""
        dtype = torch.float64
        # wire goes in +x, but pins point in +y and -y (perpendicular)
        pin_pos = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        net2pin_map = [np.array([0, 1])]
        pin2net_map = np.array([0, 0], dtype=np.int32)
        net_weights = np.array([1.0], dtype=np.float64)
        net_mask = np.ones(1, dtype=np.uint8)
        pin_mask = np.zeros(2, dtype=np.uint8)
        flat_net2pin_map = np.array([0, 1], dtype=np.int32)
        flat_net2pin_start_map = np.array([0, 2], dtype=np.int32)

        # pin directions: both point in y (perpendicular to wire in x)
        pin_dir_x = np.array([0.0, 0.0], dtype=np.float64)
        pin_dir_y = np.array([1.0, -1.0], dtype=np.float64)
        gamma = 0.5
        c = 0.5
        alpha = 1.0

        pin_x = pin_pos[:, 0]
        pin_y = pin_pos[:, 1]

        # with perpendicular directions
        wl_perp = build_cos_wirelength(
            pin_x, pin_y, pin2net_map, net2pin_map,
            gamma, net_weights, pin_dir_x, pin_dir_y, c, alpha)

        # with aligned directions (for comparison)
        pin_dir_x_aligned = np.array([1.0, -1.0], dtype=np.float64)
        pin_dir_y_aligned = np.array([0.0, 0.0], dtype=np.float64)
        wl_aligned = build_cos_wirelength(
            pin_x, pin_y, pin2net_map, net2pin_map,
            gamma, net_weights, pin_dir_x_aligned, pin_dir_y_aligned, c, alpha)

        print("perpendicular wl = ", wl_perp, " aligned wl = ", wl_aligned)
        # perpendicular case should have higher wirelength due to cosine penalty
        self.assertGreater(wl_perp, wl_aligned)

        # verify C++ matches numpy for perpendicular case
        pin_pos_var = torch.tensor(
            np.transpose(pin_pos).reshape([-1]), dtype=dtype)
        custom = cos_weighted_average_wirelength.CosWeightedAverageWirelength(
            flat_netpin=torch.from_numpy(flat_net2pin_map),
            netpin_start=torch.from_numpy(flat_net2pin_start_map),
            pin2net_map=torch.from_numpy(pin2net_map),
            net_weights=torch.from_numpy(net_weights),
            net_mask=torch.from_numpy(net_mask),
            pin_mask=torch.from_numpy(pin_mask),
            gamma=torch.tensor(gamma, dtype=dtype),
            pin_dir_x=torch.from_numpy(pin_dir_x),
            pin_dir_y=torch.from_numpy(pin_dir_y),
            c=c,
            alpha=alpha)

        result = custom.forward(pin_pos_var)
        np.testing.assert_allclose(result.item(), wl_perp, rtol=1e-5)

    def test_net_weights(self):
        """Test that net weights are correctly applied."""
        dtype = torch.float64
        (pin_pos, net2pin_map, pin2net_map, net_weights,
         pin_dir_x, pin_dir_y, gamma, c, alpha,
         net_mask, pin_mask, flat_net2pin_map, flat_net2pin_start_map
         ) = self._make_test_data(dtype=np.float64)

        pin_pos_var = torch.tensor(
            np.transpose(pin_pos).reshape([-1]), dtype=dtype)

        # with weights [1.0, 1.5]
        custom = cos_weighted_average_wirelength.CosWeightedAverageWirelength(
            flat_netpin=torch.from_numpy(flat_net2pin_map),
            netpin_start=torch.from_numpy(flat_net2pin_start_map),
            pin2net_map=torch.from_numpy(pin2net_map),
            net_weights=torch.from_numpy(net_weights),
            net_mask=torch.from_numpy(net_mask),
            pin_mask=torch.from_numpy(pin_mask),
            gamma=torch.tensor(gamma, dtype=dtype),
            pin_dir_x=torch.from_numpy(pin_dir_x),
            pin_dir_y=torch.from_numpy(pin_dir_y),
            c=c,
            alpha=alpha)
        wl_weighted = custom.forward(pin_pos_var).item()

        # with uniform weights [1.0, 1.0]
        uniform_weights = np.array([1.0, 1.0], dtype=np.float64)
        custom_uniform = cos_weighted_average_wirelength.CosWeightedAverageWirelength(
            flat_netpin=torch.from_numpy(flat_net2pin_map),
            netpin_start=torch.from_numpy(flat_net2pin_start_map),
            pin2net_map=torch.from_numpy(pin2net_map),
            net_weights=torch.from_numpy(uniform_weights),
            net_mask=torch.from_numpy(net_mask),
            pin_mask=torch.from_numpy(pin_mask),
            gamma=torch.tensor(gamma, dtype=dtype),
            pin_dir_x=torch.from_numpy(pin_dir_x),
            pin_dir_y=torch.from_numpy(pin_dir_y),
            c=c,
            alpha=alpha)
        wl_uniform = custom_uniform.forward(pin_pos_var).item()

        print("weighted wl = ", wl_weighted, " uniform wl = ", wl_uniform)
        # weighted should differ from uniform (net 1 has weight 1.5 vs 1.0)
        self.assertNotAlmostEqual(wl_weighted, wl_uniform, places=3)

    def test_gradient_finite_difference_alpha2(self):
        """Verify gradient with alpha=2.0 against finite differences."""
        dtype = torch.float64
        (pin_pos, net2pin_map, pin2net_map, net_weights,
         pin_dir_x, pin_dir_y, gamma, c, _,
         net_mask, pin_mask, flat_net2pin_map, flat_net2pin_start_map
         ) = self._make_test_data(dtype=np.float64)

        alpha = 2.0

        pin_pos_var = Variable(
            torch.tensor(np.transpose(pin_pos), dtype=dtype).reshape([-1]),
            requires_grad=True)

        custom = cos_weighted_average_wirelength.CosWeightedAverageWirelength(
            flat_netpin=torch.from_numpy(flat_net2pin_map),
            netpin_start=torch.from_numpy(flat_net2pin_start_map),
            pin2net_map=torch.from_numpy(pin2net_map),
            net_weights=torch.from_numpy(net_weights),
            net_mask=torch.from_numpy(net_mask),
            pin_mask=torch.from_numpy(pin_mask),
            gamma=torch.tensor(gamma, dtype=dtype),
            pin_dir_x=torch.from_numpy(pin_dir_x),
            pin_dir_y=torch.from_numpy(pin_dir_y),
            c=c,
            alpha=alpha)

        # analytical gradient
        result = custom.forward(pin_pos_var)
        result.backward()
        analytical_grad = pin_pos_var.grad.clone().numpy()

        # finite-difference gradient
        epsilon = 1e-5
        pos_np = pin_pos_var.data.numpy().copy()
        num_elements = len(pos_np)
        numerical_grad = np.zeros(num_elements, dtype=np.float64)

        for i in range(num_elements):
            pos_plus = pos_np.copy()
            pos_plus[i] += epsilon
            pos_minus = pos_np.copy()
            pos_minus[i] -= epsilon

            wl_plus = custom.forward(
                torch.tensor(pos_plus, dtype=dtype)).item()
            wl_minus = custom.forward(
                torch.tensor(pos_minus, dtype=dtype)).item()
            numerical_grad[i] = (wl_plus - wl_minus) / (2.0 * epsilon)

        print("\nGradient comparison alpha=2 (analytical vs numerical):")
        print("%6s %14s %14s %14s" % ("idx", "analytical", "numerical", "rel_error"))
        for i in range(num_elements):
            denom = max(abs(analytical_grad[i]), abs(numerical_grad[i]), 1e-20)
            rel_err = abs(analytical_grad[i] - numerical_grad[i]) / denom
            print("%6d %14.6E %14.6E %14.6E" % (
                i, analytical_grad[i], numerical_grad[i], rel_err))

        np.testing.assert_allclose(analytical_grad, numerical_grad,
                                   rtol=1e-3, atol=1e-5)

    def test_random_positions(self):
        """Test with random pin positions and directions."""
        dtype = torch.float64
        np.random.seed(42)

        num_nets = 5
        num_pins = num_nets * 2  # all 2-pin nets

        pin_pos = np.random.uniform(0, 10, size=(num_pins, 2)).astype(np.float64)
        net2pin_map = [np.array([2 * i, 2 * i + 1]) for i in range(num_nets)]
        pin2net_map = np.zeros(num_pins, dtype=np.int32)
        for net_id, pins in enumerate(net2pin_map):
            for pin in pins:
                pin2net_map[pin] = net_id

        net_weights = np.random.uniform(0.5, 2.0, size=num_nets).astype(np.float64)

        # random unit direction vectors
        angles = np.random.uniform(0, 2 * np.pi, size=num_pins)
        pin_dir_x = np.cos(angles).astype(np.float64)
        pin_dir_y = np.sin(angles).astype(np.float64)

        gamma = 0.5
        c = 0.5
        alpha = 1.0
        net_mask = np.ones(num_nets, dtype=np.uint8)
        pin_mask = np.zeros(num_pins, dtype=np.uint8)

        flat_net2pin_map = np.zeros(num_pins, dtype=np.int32)
        flat_net2pin_start_map = np.zeros(num_nets + 1, dtype=np.int32)
        count = 0
        for i in range(num_nets):
            flat_net2pin_map[count:count + len(net2pin_map[i])] = net2pin_map[i]
            flat_net2pin_start_map[i] = count
            count += len(net2pin_map[i])
        flat_net2pin_start_map[num_nets] = num_pins

        # numpy reference
        golden = build_cos_wirelength(
            pin_pos[:, 0], pin_pos[:, 1], pin2net_map, net2pin_map,
            gamma, net_weights, pin_dir_x, pin_dir_y, c, alpha)

        # C++ op
        pin_pos_var = Variable(
            torch.tensor(np.transpose(pin_pos), dtype=dtype).reshape([-1]),
            requires_grad=True)
        custom = cos_weighted_average_wirelength.CosWeightedAverageWirelength(
            flat_netpin=torch.from_numpy(flat_net2pin_map),
            netpin_start=torch.from_numpy(flat_net2pin_start_map),
            pin2net_map=torch.from_numpy(pin2net_map),
            net_weights=torch.from_numpy(net_weights),
            net_mask=torch.from_numpy(net_mask),
            pin_mask=torch.from_numpy(pin_mask),
            gamma=torch.tensor(gamma, dtype=dtype),
            pin_dir_x=torch.from_numpy(pin_dir_x),
            pin_dir_y=torch.from_numpy(pin_dir_y),
            c=c,
            alpha=alpha)

        result = custom.forward(pin_pos_var)
        print("random test: custom = ", result.item(), " golden = ", golden)
        np.testing.assert_allclose(result.item(), golden, rtol=1e-5, atol=1e-6)

        # gradient check
        result.backward()
        analytical_grad = pin_pos_var.grad.clone().numpy()

        epsilon = 1e-5
        pos_np = pin_pos_var.data.numpy().copy()
        numerical_grad = np.zeros(len(pos_np), dtype=np.float64)
        for i in range(len(pos_np)):
            pos_plus = pos_np.copy()
            pos_plus[i] += epsilon
            pos_minus = pos_np.copy()
            pos_minus[i] -= epsilon
            wl_plus = custom.forward(torch.tensor(pos_plus, dtype=dtype)).item()
            wl_minus = custom.forward(torch.tensor(pos_minus, dtype=dtype)).item()
            numerical_grad[i] = (wl_plus - wl_minus) / (2.0 * epsilon)

        np.testing.assert_allclose(analytical_grad, numerical_grad,
                                   rtol=1e-3, atol=1e-5)
        print("random test gradient check passed")


if __name__ == '__main__':
    unittest.main()
