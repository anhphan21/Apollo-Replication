##
# @file   test_net_crossing.py
# @brief  Test the net crossing count via the net_spacing module.
#         Creates several 2-pin nets with known intersection patterns
#         and verifies the sweep-line crossing detection.
#

import sys
import os
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import dreamplace.ops.net_spacing.net_spacing_cpp as net_spacing_cpp

dtype = torch.float64


def make_tensors(pin_positions, netpin_start_list):
    """Build pos, flat_netpin, netpin_start, net_mask from pin coordinates and net structure."""
    num_pins = len(pin_positions)
    num_nets = len(netpin_start_list) - 1

    xs = [p[0] for p in pin_positions]
    ys = [p[1] for p in pin_positions]
    # pos = [x0, x1, ..., y0, y1, ...]
    pos = torch.tensor(xs + ys, dtype=dtype)

    flat_netpin = torch.tensor(list(range(num_pins)), dtype=torch.int32)
    netpin_start = torch.tensor(netpin_start_list, dtype=torch.int32)
    net_mask = torch.ones(num_nets, dtype=torch.uint8)

    return pos, flat_netpin, netpin_start, net_mask


def test_no_crossing():
    """Two parallel horizontal nets that don't cross."""
    # Net 0: (0,0) -> (10,0)
    # Net 1: (0,5) -> (10,5)
    pin_positions = [(0, 0), (10, 0), (0, 5), (10, 5)]
    netpin_start = [0, 2, 4]

    pos, flat_netpin, netpin_start_t, net_mask = make_tensors(pin_positions, netpin_start)
    crossing_cnt = net_spacing_cpp.compute_net_crossing(pos, flat_netpin, netpin_start_t, net_mask)

    print("Test: No crossing (parallel horizontal)")
    print("  Net 0: (0,0)->(10,0), Net 1: (0,5)->(10,5)")
    print("  Crossing counts:", crossing_cnt.numpy())
    assert crossing_cnt[0].item() == 0, f"Expected 0, got {crossing_cnt[0].item()}"
    assert crossing_cnt[1].item() == 0, f"Expected 0, got {crossing_cnt[1].item()}"
    print("  PASSED\n")


def test_simple_crossing():
    """Two nets that form an X — one crossing each."""
    # Net 0: (0,0) -> (10,10)
    # Net 1: (0,10) -> (10,0)
    pin_positions = [(0, 0), (10, 10), (0, 10), (10, 0)]
    netpin_start = [0, 2, 4]

    pos, flat_netpin, netpin_start_t, net_mask = make_tensors(pin_positions, netpin_start)
    crossing_cnt = net_spacing_cpp.compute_net_crossing(pos, flat_netpin, netpin_start_t, net_mask)

    print("Test: Simple X crossing")
    print("  Net 0: (0,0)->(10,10), Net 1: (0,10)->(10,0)")
    print("  Crossing counts:", crossing_cnt.numpy())
    assert crossing_cnt[0].item() == 1, f"Expected 1, got {crossing_cnt[0].item()}"
    assert crossing_cnt[1].item() == 1, f"Expected 1, got {crossing_cnt[1].item()}"
    print("  PASSED\n")


def test_three_nets_star():
    """Three nets all crossing each other (star pattern)."""
    # Net 0: (0,0) -> (10,10)    — diagonal up-right
    # Net 1: (0,10) -> (10,0)    — diagonal down-right
    # Net 2: (5,-5) -> (5,15)    — vertical through center
    pin_positions = [(0, 0), (10, 10), (0, 10), (10, 0), (5, -5), (5, 15)]
    netpin_start = [0, 2, 4, 6]

    pos, flat_netpin, netpin_start_t, net_mask = make_tensors(pin_positions, netpin_start)
    crossing_cnt = net_spacing_cpp.compute_net_crossing(pos, flat_netpin, netpin_start_t, net_mask)

    print("Test: Three nets star (all cross each other)")
    print("  Net 0: (0,0)->(10,10), Net 1: (0,10)->(10,0), Net 2: (5,-5)->(5,15)")
    print("  Crossing counts:", crossing_cnt.numpy())
    assert crossing_cnt[0].item() == 2, f"Net 0: expected 2, got {crossing_cnt[0].item()}"
    assert crossing_cnt[1].item() == 2, f"Net 1: expected 2, got {crossing_cnt[1].item()}"
    assert crossing_cnt[2].item() == 2, f"Net 2: expected 2, got {crossing_cnt[2].item()}"
    print("  PASSED\n")


def test_partial_crossing():
    """Four nets where only some pairs cross."""
    # Net 0: (0,0) -> (10,10)   — diagonal
    # Net 1: (0,10) -> (10,0)   — crosses Net 0
    # Net 2: (20,0) -> (30,0)   — far away, crosses nothing
    # Net 3: (20,5) -> (30,5)   — far away, crosses nothing
    pin_positions = [(0, 0), (10, 10), (0, 10), (10, 0),
                     (20, 0), (30, 0), (20, 5), (30, 5)]
    netpin_start = [0, 2, 4, 6, 8]

    pos, flat_netpin, netpin_start_t, net_mask = make_tensors(pin_positions, netpin_start)
    crossing_cnt = net_spacing_cpp.compute_net_crossing(pos, flat_netpin, netpin_start_t, net_mask)

    print("Test: Partial crossing (2 cross, 2 don't)")
    print("  Net 0: (0,0)->(10,10), Net 1: (0,10)->(10,0)")
    print("  Net 2: (20,0)->(30,0), Net 3: (20,5)->(30,5)")
    print("  Crossing counts:", crossing_cnt.numpy())
    assert crossing_cnt[0].item() == 1, f"Net 0: expected 1, got {crossing_cnt[0].item()}"
    assert crossing_cnt[1].item() == 1, f"Net 1: expected 1, got {crossing_cnt[1].item()}"
    assert crossing_cnt[2].item() == 0, f"Net 2: expected 0, got {crossing_cnt[2].item()}"
    assert crossing_cnt[3].item() == 0, f"Net 3: expected 0, got {crossing_cnt[3].item()}"
    print("  PASSED\n")


def test_net_mask():
    """Verify that masked nets get crossing count = 0."""
    # Same X pattern but mask out net 1
    pin_positions = [(0, 0), (10, 10), (0, 10), (10, 0)]
    netpin_start = [0, 2, 4]

    pos, flat_netpin, netpin_start_t, net_mask = make_tensors(pin_positions, netpin_start)
    net_mask[1] = 0  # mask out net 1

    crossing_cnt = net_spacing_cpp.compute_net_crossing(pos, flat_netpin, netpin_start_t, net_mask)

    print("Test: Net mask (net 1 masked out)")
    print("  Net 0: (0,0)->(10,10), Net 1: (0,10)->(10,0) [masked]")
    print("  Crossing counts:", crossing_cnt.numpy())
    # Net 1 is masked so its count should remain 0
    assert crossing_cnt[1].item() == 0, f"Net 1 (masked): expected 0, got {crossing_cnt[1].item()}"
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Net Crossing Count Tests")
    print("=" * 60 + "\n")

    test_no_crossing()
    test_simple_crossing()
    test_three_nets_star()
    test_partial_crossing()
    test_net_mask()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
