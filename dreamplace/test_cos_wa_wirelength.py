##
# @file   test_cos_wa_wirelength.py
# @brief  Test the cos_weighted_average_wirelength module for a 2-pin net.
#         Pin 0 (fixed) at (0, 0) with direction (-1, 0).
#         Pin 1 (movable) sweeps x from -50 to 50 at y=0, direction (1, 0).
#         Plots wirelength and gradient vs movable pin x position.
#

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import dreamplace.ops.cos_weighted_average_wirelength.cos_weighted_average_wirelength_cpp_merged as cpp_merged

dtype = torch.float64

# --- Setup: 1 net, 2 pins ---
# Pin 0: fixed at (0, 0),  direction (-1, 0)
# Pin 1: movable,          direction ( 1, 0)
flat_netpin  = torch.tensor([0, 1], dtype=torch.int32)
netpin_start = torch.tensor([0, 2], dtype=torch.int32)  # 1 net, 2 pins
pin2net_map  = torch.tensor([0, 0], dtype=torch.int32)
net_weights  = torch.tensor([1.0], dtype=dtype)
net_mask     = torch.tensor([1], dtype=torch.uint8)

pin_dir_x = torch.tensor([-1.0, 1.0], dtype=dtype)  # pin0: (-1,0), pin1: (1,0)
pin_dir_y = torch.tensor([ 0.0, 0.0], dtype=dtype)

gamma = 100.0
inv_gamma = torch.tensor([1.0 / gamma], dtype=dtype)
c_val     = torch.tensor([0], dtype=dtype)
alpha_val = torch.tensor([1.4], dtype=dtype)

# --- Sweep movable pin x from -50 to 50 ---
x_range = np.linspace(-50, 50, 501)
wl_values   = np.zeros_like(x_range)
grad_x_pin0 = np.zeros_like(x_range)
grad_x_pin1 = np.zeros_like(x_range)
grad_y_pin0 = np.zeros_like(x_range)
grad_y_pin1 = np.zeros_like(x_range)

for idx, mx in enumerate(x_range):
    # pos = [x0, x1, y0, y1]  (x array followed by y array)
    pos = torch.tensor([0.0, mx, 0.0, 0.0], dtype=dtype)

    # Forward
    outputs = cpp_merged.forward(pos, flat_netpin, netpin_start, pin2net_map,
                                 net_weights, net_mask, inv_gamma,
                                 pin_dir_x, pin_dir_y, c_val, alpha_val)
    wl = outputs[0]
    grad_intermediate = outputs[1]

    wl_values[idx] = wl.item()

    # Backward with unit upstream gradient
    grad_pos = torch.tensor([1.0], dtype=dtype)
    grad_out = cpp_merged.backward(grad_pos, pos, grad_intermediate,
                                   flat_netpin, netpin_start, pin2net_map,
                                   net_weights, net_mask, inv_gamma)
    # grad_out = [dx0, dx1, dy0, dy1]
    grad_x_pin0[idx] = grad_out[0].item()
    grad_x_pin1[idx] = grad_out[1].item()
    grad_y_pin0[idx] = grad_out[2].item()
    grad_y_pin1[idx] = grad_out[3].item()

# --- Print summary ---
print("=" * 70)
print("Cos-Weighted Average Wirelength Test")
print("  Pin 0 (fixed):   pos=(0, 0),  dir=(-1, 0)")
print("  Pin 1 (movable): pos=(x, 0),  dir=( 1, 0)")
print("  gamma=%.1f, c=%.2f, alpha=%.2f" % (gamma, c_val.item(), alpha_val.item()))
print("=" * 70)
print("%10s %12s %12s %12s %12s %12s" %
      ("x_mov", "WL", "dWL/dx0", "dWL/dx1", "dWL/dy0", "dWL/dy1"))
print("-" * 70)
# Print every 50 steps
for idx in range(0, len(x_range), 50):
    print("%10.2f %12.6f %12.6f %12.6f %12.6f %12.6f" %
          (x_range[idx], wl_values[idx],
           grad_x_pin0[idx], grad_x_pin1[idx],
           grad_y_pin0[idx], grad_y_pin1[idx]))

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Wirelength
axes[0].plot(x_range, wl_values, 'b-', linewidth=1.5)
axes[0].set_ylabel('Wirelength')
axes[0].set_title('Cos-Weighted Average Wirelength (2-pin net)\n'
                  'Pin0=(0,0) dir=(-1,0) | Pin1=(x,0) dir=(1,0) | '
                  'gamma=%.1f, c=%.2f, alpha=%.2f' % (gamma, c_val.item(), alpha_val.item()))
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Gradients (x-direction)
axes[1].plot(x_range, grad_x_pin0, 'r-', linewidth=1.5, label='dWL/dx (pin0, fixed)')
axes[1].plot(x_range, grad_x_pin1, 'g-', linewidth=1.5, label='dWL/dx (pin1, movable)')
axes[1].set_xlabel('Movable pin x position')
axes[1].set_ylabel('Gradient (x)')
axes[1].set_title('Gradient of Wirelength w.r.t. pin x positions')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cos_wa_wirelength_test.png')
plt.savefig(out_path, dpi=150)
print("\nPlot saved to: %s" % out_path)
