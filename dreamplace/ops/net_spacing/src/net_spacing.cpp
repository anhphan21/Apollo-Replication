#include "functional.h"
#include "sweep_line.h"
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
class NetCrossingCntWrapper : public NetIntersector<double> {
 public:
  NetCrossingCntWrapper(int num_thread) : num_threads(num_thread) {};

  void initializeNets(const T* x, const T* y, const int* netpin_start, int num_nets) {
    nets.clear();
    nets.reserve(num_nets);

    // Sequential — emplace_back is not thread-safe
    for (int i = 0; i < num_nets; ++i)
      nets.emplace_back(x[netpin_start[i]], y[netpin_start[i]], x[netpin_start[i] + 1], y[netpin_start[i] + 1], i);
  }

  void updateNetCrossing(int numNets, int* net_crossing_cnt, const unsigned char* net_mask) {
    assert(numNets == static_cast<int>(nets.size()));

    // Run sweep-line intersection detection
    this->calculateIntersections();

    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(numNets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < numNets; ++i) {
      if (!net_mask[i]) continue;
      net_crossing_cnt[i] = nets[i].crossingCount;
    }
  }

 private:
  int num_threads = 1;
};

/// @brief Compute cosine-weighted average wirelength and gradient.
/// cosWA_e = (1 + W_theta) * H^alpha
/// where H = WA_x + WA_y (standard weighted-average wirelength),
///       W_theta = ReLU(c - cos(theta_1))^2 + ReLU(c - cos(theta_2))^2
///                 (only for 2-pin nets; 0 otherwise),
///       cos(theta_k) = dot(wire_dir, pin_dir_k) / |wire_dir|.
///
/// @param x x location of pins.
/// @param y y location of pins.
/// @param flat_netpin consists pins of each net, pins belonging to the same net
/// are abutting to each other.
/// @param netpin_start bookmark for the starting index of each net in
/// flat_netpin.
/// @param net_mask whether compute the wirelength for a net or not
/// @param num_nets number of nets.
/// @param pin_dir_x x-component of pin direction unit vectors.
/// @param pin_dir_y y-component of pin direction unit vectors.
/// @param c cosine threshold for the penalty.
/// @param alpha power exponent on the WA wirelength.
/// @param partial_net_spacing output per-net wirelength.
/// @param grad_intermediate_x output per-pin gradient intermediate (x).
/// @param grad_intermediate_y output per-pin gradient intermediate (y).
/// @param num_threads number of threads for OpenMP.
/// @return 0 if successfully done.
template <typename T>
int computeNetSpacingLauncher(const T* x,
                              const T* y,
                              const T* pin_dir_x,
                              const T* pin_dir_y,
                              const int* pin_side,

                              const int* pin2net_map,
                              const int* pin2node_map,

                              const int* flat_netpin,
                              const int* netpin_start,
                              const unsigned char* net_mask,
                              const bool* update_crossing,
                              int* net_crossing_cnt,
                              int num_nets,

                              const int* node_num_ports,

                              T* bend_radii,
                              T* cross_size,

                              T* partial_net_spacing,
                              T* grad_intermediate_x,
                              T* grad_intermediate_y,
                              int num_threads) {
  const T r_bend = *bend_radii;
  const T s_crs  = *cross_size;

  if (*update_crossing) {
    auto engine = NetCrossingCntWrapper<T>(num_threads);
    engine.initializeNets(x, y, netpin_start, num_nets);
    engine.updateNetCrossing(num_nets, net_crossing_cnt, net_mask);
  }

  int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int i = 0; i < num_nets; ++i) {
    if (!net_mask[i]) continue;

    int degree = netpin_start[i + 1] - netpin_start[i];
    assert(degree == 2);

    // Calculate S_i for each port
    int p0_idx = flat_netpin[netpin_start[i]];
    int p1_idx = flat_netpin[netpin_start[i] + 1];

    T si_0 = r_bend + 0.5 * node_num_ports[pin2node_map[p0_idx] + pin_side[p0_idx]] * s_crs;
    T si_1 = r_bend + 0.5 * node_num_ports[pin2node_map[p1_idx] + pin_side[p1_idx]] * s_crs;
    T s_i  = ((si_0 > si_1) ? si_0 : si_1) + s_crs * net_crossing_cnt[i];

    T dx = x[p0_idx] - x[p1_idx];
    T dy = y[p0_idx] - y[p1_idx];

    // Cal the NS
    T relu_x = DREAMPLACE_STD_NAMESPACE::max(T(0), s_i - DREAMPLACE_STD_NAMESPACE::fabs(dx));
    T relu_y = DREAMPLACE_STD_NAMESPACE::max(T(0), s_i - DREAMPLACE_STD_NAMESPACE::fabs(dy));

    if (relu_x > T(0)) {
      T sign                      = (dx > T(0) ? -1.0 : 1.0);
      grad_intermediate_x[p0_idx] = 2.0 * relu_x * pin_dir_x[p0_idx] * sign;
      grad_intermediate_x[p1_idx] = 2.0 * relu_x * pin_dir_x[p1_idx] * sign;
    }

    if (relu_y > T(0)) {
      T sign                      = (dy > T(0) ? -1.0 : 1.0);
      grad_intermediate_y[p0_idx] = 2.0 * relu_y * pin_dir_y[p0_idx] * sign;
      grad_intermediate_y[p1_idx] = 2.0 * relu_y * pin_dir_y[p1_idx] * sign;
    }

    partial_net_spacing[i] = relu_x * relu_x + relu_y * relu_y;
  }

  return 0;
}

/// @brief Forward: compute cosine-weighted average wirelength.
/// @param node_num_ports number of port on each side of the node
/// @param pos pin locations (x array, y array), length 2*num_pins.
/// @param flat_netpin flat netpin map.
/// @param netpin_start starting index per net, length num_nets+1.

/// @param net_weights weight of nets.
/// @param net_mask whether to compute wirelength per net.

/// @param pin2net_map pin to net map.
/// @param pin2node_map pin to node map.

/// @param pin_dir_x x-component of pin direction unit vectors.
/// @param pin_dir_y y-component of pin direction unit vectors.
/// @param pin_side position of pin corresponding to the node.
/// @param bend_radii
/// @param cross_size
/// @param crossing_num
/// @param update_crossing bool variable to update the number of crossing for each net
/// @return {total_wl, grad_intermediate}.
std::vector<at::Tensor> net_spacing_forward(at::Tensor pos,
                                            at::Tensor pin_dir,
                                            at::Tensor pin_side,

                                            at::Tensor pin2net_map,
                                            at::Tensor pin2node_map,

                                            at::Tensor flat_netpin,
                                            at::Tensor netpin_start,

                                            at::Tensor net_weights,
                                            at::Tensor net_mask,

                                            at::Tensor node_num_ports,

                                            at::Tensor update_crossing,
                                            at::Tensor net_crossing_cnt,

                                            at::Tensor bend_radii,
                                            at::Tensor cross_size) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(pin_dir);
  CHECK_EVEN(pin_dir);
  CHECK_CONTIGUOUS(pin_dir);
  CHECK_FLAT_CPU(pin_side);
  CHECK_CONTIGUOUS(pin_side);

  CHECK_FLAT_CPU(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);
  CHECK_FLAT_CPU(pin2node_map);
  CHECK_CONTIGUOUS(pin2node_map);

  CHECK_FLAT_CPU(node_num_ports);
  CHECK_CONTIGUOUS(node_num_ports);

  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);

  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);

  CHECK_FLAT_CPU(net_crossing_cnt);
  CHECK_CONTIGUOUS(net_crossing_cnt);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pos.numel() / 2;

  at::Tensor partial_net_spacing = at::zeros({num_nets}, pos.options());
  at::Tensor grad_intermediate   = at::zeros_like(pos);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeNetSpacingLauncher", [&] {
    computeNetSpacingLauncher<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
                                        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
                                        DREAMPLACE_TENSOR_DATA_PTR(pin_dir, scalar_t),
                                        DREAMPLACE_TENSOR_DATA_PTR(pin_dir, scalar_t) + num_pins,
                                        DREAMPLACE_TENSOR_DATA_PTR(pin_side, int),

                                        DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int),
                                        DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),

                                        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
                                        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
                                        DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char),
                                        DREAMPLACE_TENSOR_DATA_PTR(update_crossing, bool),
                                        DREAMPLACE_TENSOR_DATA_PTR(net_crossing_cnt, int),
                                        num_nets,
                                        DREAMPLACE_TENSOR_DATA_PTR(node_num_ports, int),
                                        DREAMPLACE_TENSOR_DATA_PTR(bend_radii, scalar_t),
                                        DREAMPLACE_TENSOR_DATA_PTR(cross_size, scalar_t),

                                        DREAMPLACE_TENSOR_DATA_PTR(partial_net_spacing, scalar_t),
                                        DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t),
                                        DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t) + num_pins,
                                        at::get_num_threads());
    if (net_weights.numel()) {
      partial_net_spacing.mul_(net_weights);
    }
  });

  auto wl = partial_net_spacing.sum();
  return {wl, grad_intermediate};
}

/// @brief Backward: multiply precomputed gradient intermediate by upstream
/// gradient and integrate net weights.
at::Tensor net_spacing_backward(at::Tensor grad_pos,
                                at::Tensor pos,
                                at::Tensor grad_intermediate,
                                at::Tensor flat_netpin,
                                at::Tensor netpin_start,
                                at::Tensor pin2net_map,
                                at::Tensor net_weights,
                                at::Tensor net_mask) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);
  CHECK_FLAT_CPU(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);
  CHECK_FLAT_CPU(grad_intermediate);
  CHECK_EVEN(grad_intermediate);
  CHECK_CONTIGUOUS(grad_intermediate);

  at::Tensor grad_out = grad_intermediate.mul_(grad_pos);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "integrateNetWeightsLauncher", [&] {
    if (net_weights.numel()) {
      integrateNetWeightsLauncher<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
                                            DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
                                            DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char),
                                            DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
                                            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
                                            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + pos.numel() / 2,
                                            netpin_start.numel() - 1,
                                            at::get_num_threads());
    }
  });
  return grad_out;
}

/// @brief Compute net crossing counts using sweep-line algorithm.
/// @param pos pin locations (x array, y array), length 2*num_pins.
/// @param flat_netpin flat netpin map.
/// @param netpin_start starting index per net, length num_nets+1.
/// @param net_mask whether to compute crossing per net.
/// @return per-net crossing count tensor (int32).
at::Tensor compute_net_crossing(at::Tensor pos, at::Tensor flat_netpin, at::Tensor netpin_start, at::Tensor net_mask) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pos.numel() / 2;

  at::Tensor net_crossing_cnt = at::zeros({num_nets}, at::TensorOptions().dtype(at::kInt));

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "compute_net_crossing", [&] {
    auto engine = NetCrossingCntWrapper<scalar_t>(at::get_num_threads());
    engine.initializeNets(DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
                          DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
                          DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
                          num_nets);
    engine.updateNetCrossing(num_nets, DREAMPLACE_TENSOR_DATA_PTR(net_crossing_cnt, int), DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char));
  });

  return net_crossing_cnt;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::net_spacing_forward, "NetSpacing forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::net_spacing_backward, "NetSpacing backward");
  m.def("compute_net_crossing", &DREAMPLACE_NAMESPACE::compute_net_crossing, "Compute net crossing counts");
}
