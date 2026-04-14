##
# @file    net_spacing_unittest.py
# @author  Anh Phan
# @date    Apr 2026
# @brief   Unit test for net spacing model for PIC placement.

import os
import sys
import numpy as np
import unittest
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from torch.autograd import Variable

# Setup path to import dreamplace modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
from dreamplace.ops.net_spacing import net_spacing
sys.path.pop()


def build_net_spacing_numpy(pin_x, pin_y, net2pin_map, pin2node_map, pin_side,
                            node_num_ports, net_weights, bend_radii, cross_size,
                            net_crossing_cnt):
    """Numpy reference implementation of net spacing model.
    NS_i = w_i * (relu(s_i - |dx|)^2 + relu(s_i - |dy|)^2)
    where s_i = max(si_0, si_1) + s_crs * crossings
          si_k = r_bend + 0.5 * node_num_ports[pin2node_map[pk] + pin_side[pk]] * s_crs
    """
    num_nets = len(net2pin_map)
    per_net_spacing = np.zeros(num_nets, dtype=pin_x.dtype)

    for i in range(num_nets):
        pins = net2pin_map[i]
        p0, p1 = pins[0], pins[1]

        si_0 = bend_radii + 0.5 * node_num_ports[pin2node_map[p0] + pin_side[p0]] * cross_size
        si_1 = bend_radii + 0.5 * node_num_ports[pin2node_map[p1] + pin_side[p1]] * cross_size

        s_i = max(si_0, si_1) + cross_size * net_crossing_cnt[i]
        dx = pin_x[p0] - pin_x[p1]
        dy = pin_y[p0] - pin_y[p1]

        relu_x = max(0.0, s_i - abs(dx))
        relu_y = max(0.0, s_i - abs(dy))
        per_net_spacing[i] = (relu_x**2 + relu_y**2) * net_weights[i]

    return np.sum(per_net_spacing), per_net_spacing


def draw_nets_and_pins(pin_pos, net2pin_map, pin_dir_x=None, pin_dir_y=None, filename="net_visualization.png"):
    """Standalone utility to visualize pins and nets."""
    plt.figure(figsize=(10, 8))
    num_nets = len(net2pin_map)
    colors = cm.rainbow(np.linspace(0, 1, num_nets))

    for i, pins in enumerate(net2pin_map):
        p0, p1 = pins[0], pins[1]
        x_coords = [pin_pos[p0][0], pin_pos[p1][0]]
        y_coords = [pin_pos[p0][1], pin_pos[p1][1]]

        plt.plot(x_coords, y_coords, color=colors[i], linestyle='-', linewidth=2, alpha=0.7, label=f'Net {i}')
        plt.scatter(x_coords, y_coords, color=colors[i], s=80, zorder=5, edgecolor='black')

        plt.text(x_coords[0], y_coords[0] + 0.4, f'P{p0}', fontsize=10, ha='center', weight='bold')
        plt.text(x_coords[1], y_coords[1] + 0.4, f'P{p1}', fontsize=10, ha='center', weight='bold')

        if pin_dir_x is not None and pin_dir_y is not None:
            plt.arrow(x_coords[0], y_coords[0], pin_dir_x[p0], pin_dir_y[p0],
                      head_width=0.25, head_length=0.35, fc='black', ec='black', zorder=4)
            plt.arrow(x_coords[1], y_coords[1], pin_dir_x[p1], pin_dir_y[p1],
                      head_width=0.25, head_length=0.35, fc='black', ec='black', zorder=4)

    plt.title("Net and Pin Placement Visualization", fontsize=14, weight='bold')
    plt.xlabel("X Coordinate"); plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.5); plt.axis('equal')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Nets")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n[VISUALIZATION] Plot saved to: {filename}")
    plt.close()


class NetSpacingOpTest(unittest.TestCase):

    def _make_test_data(self, dtype=np.float64):
        """Create test data for 10 nets (20 pins, 5 nodes).

        Data model:
          - pin2node_map[pin] = node_id * 4 (base offset into node_num_ports)
          - node_num_ports = flat array [n0_left, n0_right, n0_lower, n0_upper,
                                         n1_left, n1_right, n1_lower, n1_upper, ...]
          - pin_side: 0=LEFT, 1=RIGHT, 2=LOWER, 3=UPPER
          - Access: node_num_ports[pin2node_map[pin] + pin_side[pin]]
        """
        np.random.seed(100)

        num_nets = 10
        num_pins = num_nets * 2
        num_nodes = 5

        # Pin positions
        pin_pos = np.random.uniform(0, 20, size=(num_pins, 2)).astype(dtype)

        # Net-to-pin mapping: each net has exactly 2 pins
        net2pin_map = [np.array([2 * i, 2 * i + 1]) for i in range(num_nets)]
        pin2net_map = (np.arange(num_pins) // 2).astype(np.int32)

        # Node assignment: 4 pins per node (5 nodes for 20 pins)
        # pin2node_map stores node_id * 4 (base offset), NOT raw node_id
        raw_node_ids = (np.arange(num_pins) // 4).astype(np.int32)
        pin2node_map = (raw_node_ids * 4).astype(np.int32)

        # Pin side: which side of the node each pin is on
        pin_side = np.random.randint(0, 4, size=num_pins).astype(np.int32)

        # Node num ports: 4 entries per node [left, right, lower, upper]
        # Populate based on pin assignments to ensure consistency
        node_num_ports = np.zeros(num_nodes * 4, dtype=np.int32)
        for p in range(num_pins):
            node_id = raw_node_ids[p]
            side = pin_side[p]
            node_num_ports[node_id * 4 + side] += 1

        # Pin directions: cardinal directions matching pin_side
        # LEFT->(-1,0), RIGHT->(1,0), LOWER->(0,-1), UPPER->(0,1)
        side_to_dir = {0: (-1.0, 0.0), 1: (1.0, 0.0), 2: (0.0, -1.0), 3: (0.0, 1.0)}
        pin_dir_x = np.array([side_to_dir[s][0] for s in pin_side], dtype=dtype)
        pin_dir_y = np.array([side_to_dir[s][1] for s in pin_side], dtype=dtype)

        # Net weights
        net_weights = np.ones(num_nets, dtype=dtype)

        # Masks
        net_mask = np.ones(num_nets, dtype=np.uint8)
        pin_mask = np.zeros(num_pins, dtype=np.uint8)

        # Crossing counts (start at zero)
        net_crossing_cnt = np.zeros(num_nets, dtype=np.int32)

        # Physical parameters
        bend_radii = 5.0
        cross_size = 5.0

        # Flat net-to-pin structures
        flat_net2pin_map = np.arange(num_pins, dtype=np.int32)
        flat_net2pin_start_map = np.arange(0, (num_nets + 1) * 2, 2, dtype=np.int32)

        return {
            'pin_pos': pin_pos,
            'net2pin_map': net2pin_map,
            'pin2net_map': pin2net_map,
            'pin2node_map': pin2node_map,
            'raw_node_ids': raw_node_ids,
            'pin_side': pin_side,
            'node_num_ports': node_num_ports,
            'pin_dir_x': pin_dir_x,
            'pin_dir_y': pin_dir_y,
            'net_weights': net_weights,
            'net_mask': net_mask,
            'pin_mask': pin_mask,
            'net_crossing_cnt': net_crossing_cnt,
            'flat_net2pin_map': flat_net2pin_map,
            'flat_net2pin_start_map': flat_net2pin_start_map,
            'bend_radii': bend_radii,
            'cross_size': cross_size,
            'num_nets': num_nets,
            'num_pins': num_pins,
            'num_nodes': num_nodes,
        }

    def _create_pytorch_module(self, dtype, data):
        """Create NetSpacing PyTorch module from test data dict."""
        pin_dir_concat = np.concatenate([data['pin_dir_x'], data['pin_dir_y']])
        module = net_spacing.NetSpacing(
            flat_netpin=torch.from_numpy(data['flat_net2pin_map']),
            netpin_start=torch.from_numpy(data['flat_net2pin_start_map']),
            pin2net_map=torch.from_numpy(data['pin2net_map']),
            pin2node_map=torch.from_numpy(data['pin2node_map']),
            net_weights=torch.from_numpy(data['net_weights']),
            net_mask=torch.from_numpy(data['net_mask']),
            pin_mask=torch.tensor(data['pin_mask'], dtype=torch.bool),
            pin_dir=torch.tensor(pin_dir_concat, dtype=dtype),
            pin_side=torch.from_numpy(data['pin_side']),
            node_num_ports=torch.from_numpy(data['node_num_ports']),
            bend_radii=torch.tensor(data['bend_radii'], dtype=dtype),
            cross_size=torch.tensor(data['cross_size'], dtype=dtype),
        )
        return module

    def test_10_nets_full_inspection(self):
        """Test with 10 nets: verify forward value against numpy reference,
        check gradient with finite difference, and print all details."""
        dtype = torch.float64
        data = self._make_test_data(np.float64)

        pin_pos = data['pin_pos']
        num_pins = data['num_pins']
        num_nets = data['num_nets']
        num_nodes = data['num_nodes']

        # --- Print Node Information ---
        side_names = {0: 'LEFT', 1: 'RIGHT', 2: 'LOWER', 3: 'UPPER'}
        print("\n" + "=" * 95)
        print(f"{'NET SPACING MODEL FULL INSPECTION':^95}")
        print("=" * 95)
        print(f"Config | bend_radii={data['bend_radii']} | cross_size={data['cross_size']}")
        print(f"       | num_nets={num_nets} | num_pins={num_pins} | num_nodes={num_nodes}")

        print("\n[SECTION 1: NODE PORT COUNTS]")
        print(f"{'Node':<6} | {'LEFT':<6} | {'RIGHT':<6} | {'LOWER':<6} | {'UPPER':<6}")
        print("-" * 45)
        for n in range(num_nodes):
            base = n * 4
            print(f"{n:<6} | {data['node_num_ports'][base]:<6} | {data['node_num_ports'][base+1]:<6} | "
                  f"{data['node_num_ports'][base+2]:<6} | {data['node_num_ports'][base+3]:<6}")

        print("\n[SECTION 2: PIN CONFIGURATION]")
        header = f"{'Pin':<4} | {'Node':<4} | {'NodeBase':<8} | {'Side':<6} | {'Ports':<5} | {'Pos (x, y)':<18} | {'Dir (x, y)':<14}"
        print(header)
        print("-" * 80)
        for p in range(num_pins):
            node_base = data['pin2node_map'][p]
            node_id = data['raw_node_ids'][p]
            side = data['pin_side'][p]
            ports = data['node_num_ports'][node_base + side]
            pos_str = f"({pin_pos[p][0]:>6.2f}, {pin_pos[p][1]:>6.2f})"
            dir_str = f"({data['pin_dir_x'][p]:>2.0f}, {data['pin_dir_y'][p]:>2.0f})"
            print(f"{p:<4} | {node_id:<4} | {node_base:<8} | {side_names[side]:<6} | {ports:<5} | {pos_str:<18} | {dir_str:<14}")

        # --- Forward pass with C++ module (run first to get sweep-line crossings) ---
        module = self._create_pytorch_module(dtype, data)
        module.update_crossing.fill_(True)  # Force sweep-line crossing detection

        pos_flat = np.concatenate([pin_pos[:, 0], pin_pos[:, 1]])
        pin_pos_var = Variable(torch.tensor(pos_flat, dtype=dtype), requires_grad=True)
        result = module.forward(pin_pos_var)
        result.backward()

        cpp_total = result.item()
        detected_crossings = module.net_crossing_cnt.numpy().copy()
        analytical_grad = pin_pos_var.grad.detach().numpy()
        grad_x = analytical_grad[:num_pins]
        grad_y = analytical_grad[num_pins:]

        print(f"\nCrossings (sweep-line): {detected_crossings.tolist()}")

        # --- Compute numpy reference using the same crossing counts from sweep-line ---
        np_total, np_per_net = build_net_spacing_numpy(
            pin_pos[:, 0], pin_pos[:, 1],
            data['net2pin_map'], data['pin2node_map'], data['pin_side'],
            data['node_num_ports'], data['net_weights'],
            data['bend_radii'], data['cross_size'], detected_crossings)

        print("\n[SECTION 3: PER-NET SPACING DETAILS (Numpy Reference)]")
        print(f"{'Net':<4} | {'Pins':<10} | {'Weight':<6} | {'Cross':<5} | "
              f"{'si_0':<8} | {'si_1':<8} | {'s_i':<8} | {'|dx|':<8} | {'|dy|':<8} | {'NS_i':<12}")
        print("-" * 105)
        r_bend = data['bend_radii']
        s_crs = data['cross_size']
        for i in range(num_nets):
            p0, p1 = data['net2pin_map'][i]
            si_0 = r_bend + 0.5 * data['node_num_ports'][data['pin2node_map'][p0] + data['pin_side'][p0]] * s_crs
            si_1 = r_bend + 0.5 * data['node_num_ports'][data['pin2node_map'][p1] + data['pin_side'][p1]] * s_crs
            s_i = max(si_0, si_1) + s_crs * detected_crossings[i]
            dx = abs(pin_pos[p0, 0] - pin_pos[p1, 0])
            dy = abs(pin_pos[p0, 1] - pin_pos[p1, 1])
            print(f"{i:<4} | {str([p0,p1]):<10} | {data['net_weights'][i]:<6.2f} | {detected_crossings[i]:<5} | "
                  f"{si_0:<8.2f} | {si_1:<8.2f} | {s_i:<8.2f} | {dx:<8.2f} | {dy:<8.2f} | {np_per_net[i]:<12.4f}")

        print(f"\nNumpy Total NS = {np_total:.6f}")
        print(f"C++ Total NS   = {cpp_total:.6f}")

        # --- Forward value check ---
        fwd_err = abs(cpp_total - np_total)
        print(f"\n[FORWARD CHECK] |C++ - Numpy| = {fwd_err:.6e}")
        self.assertAlmostEqual(cpp_total, np_total, places=6,
                               msg=f"Forward mismatch: C++={cpp_total}, Numpy={np_total}")

        # --- Finite difference gradient check ---
        eps = 1e-6
        fd_grad = np.zeros_like(pos_flat)
        for j in range(len(pos_flat)):
            pos_p = pos_flat.copy(); pos_p[j] += eps
            pos_m = pos_flat.copy(); pos_m[j] -= eps
            fp = module.forward(torch.tensor(pos_p, dtype=dtype)).item()
            fm = module.forward(torch.tensor(pos_m, dtype=dtype)).item()
            fd_grad[j] = (fp - fm) / (2 * eps)

        fd_grad_x = fd_grad[:num_pins]
        fd_grad_y = fd_grad[num_pins:]

        print("\n[SECTION 4: GRADIENT COMPARISON (Analytical vs Finite-Difference)]")
        header = f"{'Pin':<4} | {'Grad_X (ana)':<14} | {'Grad_X (fd)':<14} | {'err_x':<12} | {'Grad_Y (ana)':<14} | {'Grad_Y (fd)':<14} | {'err_y':<12}"
        print(header)
        print("-" * 105)
        max_grad_err = 0.0
        for p in range(num_pins):
            err_x = abs(grad_x[p] - fd_grad_x[p])
            err_y = abs(grad_y[p] - fd_grad_y[p])
            max_grad_err = max(max_grad_err, err_x, err_y)
            print(f"{p:<4} | {grad_x[p]:<14.6E} | {fd_grad_x[p]:<14.6E} | {err_x:<12.3E} | "
                  f"{grad_y[p]:<14.6E} | {fd_grad_y[p]:<14.6E} | {err_y:<12.3E}")

        print(f"\n[GRADIENT CHECK] Max |analytical - fd| = {max_grad_err:.6e}")
        self.assertLess(max_grad_err, 1e-4,
                        msg=f"Gradient error too large: {max_grad_err:.6e}")

        print("=" * 95 + "\n")

        # Visual Verification
        draw_nets_and_pins(pin_pos, data['net2pin_map'], data['pin_dir_x'], data['pin_dir_y'],
                           filename="10_nets_full_report.png")


if __name__ == '__main__':
    unittest.main()
