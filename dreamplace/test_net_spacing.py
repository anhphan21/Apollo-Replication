##
# @file   test_net_spacing.py
# @brief  Test the net_spacing module for a 2-pin net with 2 nodes.
#         Pin 0 (fixed) at (0,0) with pin_dir (1, 0) — right-facing.
#         Pin 1 (movable) sweeps x from -40 to 40 at y=0, pin_dir (-1, 0) — left-facing.
#         Fixed pin belongs to node with 2 ports on the same side (RIGHT).
#         Movable pin belongs to node with 3 ports on the same side (LEFT).
#         crossing_size = 5, bend_radii = 5.
#
#         Expected S_i:
#           si_0 = r_bend + 0.5 * num_ports_0 * s_crs = 5 + 0.5*2*5 = 10
#           si_1 = r_bend + 0.5 * num_ports_1 * s_crs = 5 + 0.5*3*5 = 12.5
#           s_i  = max(si_0, si_1) + s_crs * crossings = 12.5 + 0 = 12.5
#
#         NS = relu(s_i - |dx|)^2 + relu(s_i - |dy|)^2
#

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dreamplace.ops.net_spacing.net_spacing import NetSpacing

dtype = torch.float64

# --- Setup: 1 net, 2 pins, 2 nodes ---
# Pin 0: fixed at (0,0), dir (1,0)  -> pin_side = RIGHT = 1
# Pin 1: movable at (x,0), dir (-1,0) -> pin_side = LEFT = 0

flat_netpin  = torch.tensor([0, 1], dtype=torch.int32)
netpin_start = torch.tensor([0, 2], dtype=torch.int32)
pin2net_map  = torch.tensor([0, 0], dtype=torch.int32)

# pin2node_map gives base offset into node_num_ports (node_id * 4)
# Pin 0 -> node 0 (base 0), Pin 1 -> node 1 (base 4)
pin2node_map = torch.tensor([0, 4], dtype=torch.int32)

# pin_dir: [dir_x_pin0, dir_x_pin1, dir_y_pin0, dir_y_pin1]
pin_dir = torch.tensor([1.0, -1.0, 0.0, 0.0], dtype=dtype)

# pin_side: RIGHT=1 for pin 0, LEFT=0 for pin 1
pin_side = torch.tensor([1, 0], dtype=torch.int32)

net_weights = torch.tensor([1.0], dtype=dtype)
net_mask    = torch.tensor([1], dtype=torch.uint8)
pin_mask    = torch.tensor([False, False], dtype=torch.bool)

# node_num_ports: [node0_left, node0_right, node0_lower, node0_upper,
#                  node1_left, node1_right, node1_lower, node1_upper]
# Node 0: 2 ports on RIGHT side; Node 1: 3 ports on LEFT side
node_num_ports = torch.tensor([0, 2, 0, 0,
                                3, 0, 0, 0], dtype=torch.int32)

bend_radii = torch.tensor([5.0], dtype=dtype)
cross_size = torch.tensor([5.0], dtype=dtype)

# --- Build NetSpacing module ---
ns_module = NetSpacing(
    flat_netpin=flat_netpin,
    netpin_start=netpin_start,
    pin2net_map=pin2net_map,
    pin2node_map=pin2node_map,
    net_weights=net_weights,
    net_mask=net_mask,
    pin_mask=pin_mask,
    pin_dir=pin_dir,
    pin_side=pin_side,
    node_num_ports=node_num_ports,
    bend_radii=bend_radii,
    cross_size=cross_size,
)

# --- Sweep movable pin x from -40 to 40 ---
x_range = np.linspace(-40, 40, 801)
ns_values    = np.zeros_like(x_range)
grad_x_pin0  = np.zeros_like(x_range)
grad_x_pin1  = np.zeros_like(x_range)
grad_y_pin0  = np.zeros_like(x_range)
grad_y_pin1  = np.zeros_like(x_range)

for idx, mx in enumerate(x_range):
    # pos = [x_pin0, x_pin1, y_pin0, y_pin1]
    pos = torch.tensor([0.0, mx, 0.0, 0.0], dtype=dtype, requires_grad=True)

    ns = ns_module(pos)
    ns.backward()

    ns_values[idx] = ns.item()
    grad = pos.grad
    # grad = [dx_pin0, dx_pin1, dy_pin0, dy_pin1]
    grad_x_pin0[idx] = grad[0].item()
    grad_x_pin1[idx] = grad[1].item()
    grad_y_pin0[idx] = grad[2].item()
    grad_y_pin1[idx] = grad[3].item()

# --- Compute expected values analytically ---
s_i = 12.5  # max(10, 12.5) + 5*0
expected_ns = np.zeros_like(x_range)
for idx, mx in enumerate(x_range):
    dx = abs(mx)
    dy = 0.0
    relu_x = max(0, s_i - dx)
    relu_y = max(0, s_i - dy)
    expected_ns[idx] = relu_x**2 + relu_y**2

# --- Print summary ---
print("=" * 90)
print("Net Spacing Test")
print("  Pin 0 (fixed):   pos=(0, 0),  dir=(1, 0),  node has 2 ports on RIGHT")
print("  Pin 1 (movable): pos=(x, 0),  dir=(-1, 0), node has 3 ports on LEFT")
print("  bend_radii=%.1f, cross_size=%.1f, crossings=0" % (bend_radii.item(), cross_size.item()))
print("  si_0 = 5 + 0.5*2*5 = 10.0")
print("  si_1 = 5 + 0.5*3*5 = 12.5")
print("  s_i  = max(10.0, 12.5) = 12.5")
print("=" * 90)
print("%10s %12s %12s %12s %12s %12s %12s" %
      ("x_mov", "NS", "expected", "dNS/dx0", "dNS/dx1", "dNS/dy0", "dNS/dy1"))
print("-" * 90)
# Print every 80 steps
for idx in range(0, len(x_range), 80):
    print("%10.2f %12.4f %12.4f %12.6f %12.6f %12.6f %12.6f" %
          (x_range[idx], ns_values[idx], expected_ns[idx],
           grad_x_pin0[idx], grad_x_pin1[idx],
           grad_y_pin0[idx], grad_y_pin1[idx]))

# --- Verify ---
max_err = np.max(np.abs(ns_values - expected_ns))
print("\nMax error vs expected: %.6e" % max_err)
if max_err < 1e-6:
    print("PASSED: net spacing values match expected.\n")
else:
    print("FAILED: net spacing values do NOT match expected.\n")

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Net spacing value
axes[0].plot(x_range, ns_values, 'b-', linewidth=1.5, label='C++ result')
axes[0].plot(x_range, expected_ns, 'r--', linewidth=1.0, label='Expected')
axes[0].set_ylabel('Net Spacing')
axes[0].set_title('Net Spacing (2-pin net)\n'
                  'Pin0=(0,0) dir=(1,0) 2ports | Pin1=(x,0) dir=(-1,0) 3ports | '
                  'r_bend=5, s_crs=5, s_i=12.5')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Gradient x-direction
axes[1].plot(x_range, grad_x_pin0, 'r-', linewidth=1.5, label='dNS/dx (pin0, fixed)')
axes[1].plot(x_range, grad_x_pin1, 'g-', linewidth=1.5, label='dNS/dx (pin1, movable)')
axes[1].set_ylabel('Gradient (x)')
axes[1].set_title('Gradient of Net Spacing w.r.t. pin x positions')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Gradient y-direction
axes[2].plot(x_range, grad_y_pin0, 'r-', linewidth=1.5, label='dNS/dy (pin0, fixed)')
axes[2].plot(x_range, grad_y_pin1, 'g-', linewidth=1.5, label='dNS/dy (pin1, movable)')
axes[2].set_xlabel('Movable pin x position')
axes[2].set_ylabel('Gradient (y)')
axes[2].set_title('Gradient of Net Spacing w.r.t. pin y positions')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'net_spacing_test.png')
plt.savefig(out_path, dpi=150)
print("Plot saved to: %s" % out_path)
