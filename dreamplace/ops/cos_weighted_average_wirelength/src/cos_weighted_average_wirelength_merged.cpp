#include "cos_weighted_average_wirelength/src/functional.h"
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

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
/// @param inv_gamma the inverse number of gamma coefficient.
/// @param pin_dir_x x-component of pin direction unit vectors.
/// @param pin_dir_y y-component of pin direction unit vectors.
/// @param c cosine threshold for the penalty.
/// @param alpha power exponent on the WA wirelength.
/// @param partial_wl output per-net wirelength.
/// @param grad_intermediate_x output per-pin gradient intermediate (x).
/// @param grad_intermediate_y output per-pin gradient intermediate (y).
/// @param num_threads number of threads for OpenMP.
/// @return 0 if successfully done.
template <typename T>
int computeCosWeightedAverageWirelengthMergedLauncher(const T* x,
                                                      const T* y,
                                                      const int* flat_netpin,
                                                      const int* netpin_start,
                                                      const unsigned char* net_mask,
                                                      int num_nets,
                                                      const T* inv_gamma,
                                                      const T* pin_dir_x,
                                                      const T* pin_dir_y,
                                                      const T* c_ptr,
                                                      const T* alpha_ptr,
                                                      T* partial_wl,
                                                      T* grad_intermediate_x,
                                                      T* grad_intermediate_y,
                                                      int num_threads) {
  const T c     = *c_ptr;
  const T alpha = *alpha_ptr;
  const T eps   = static_cast<T>(1e-6);

  int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int i = 0; i < num_nets; ++i) {
    if (!net_mask[i]) continue;

    int degree = netpin_start[i + 1] - netpin_start[i];
    assert(degree == 2);

    // ---- Step 1: standard WA wirelength computation ----
    T x_max = -std::numeric_limits<T>::max();
    T x_min = std::numeric_limits<T>::max();
    T y_max = -std::numeric_limits<T>::max();
    T y_min = std::numeric_limits<T>::max();

    for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
      T xx  = x[flat_netpin[j]];
      x_max = DREAMPLACE_STD_NAMESPACE::max(xx, x_max);
      x_min = DREAMPLACE_STD_NAMESPACE::min(xx, x_min);
      T yy  = y[flat_netpin[j]];
      y_max = DREAMPLACE_STD_NAMESPACE::max(yy, y_max);
      y_min = DREAMPLACE_STD_NAMESPACE::min(yy, y_min);
    }

    T xexp_x_sum = 0, xexp_nx_sum = 0, exp_x_sum = 0, exp_nx_sum = 0;
    T yexp_y_sum = 0, yexp_ny_sum = 0, exp_y_sum = 0, exp_ny_sum = 0;

    for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
      T xx     = x[flat_netpin[j]];
      T exp_x  = exp((xx - x_max) * (*inv_gamma));
      T exp_nx = exp((x_min - xx) * (*inv_gamma));
      xexp_x_sum += xx * exp_x;
      xexp_nx_sum += xx * exp_nx;
      exp_x_sum += exp_x;
      exp_nx_sum += exp_nx;

      T yy     = y[flat_netpin[j]];
      T exp_y  = exp((yy - y_max) * (*inv_gamma));
      T exp_ny = exp((y_min - yy) * (*inv_gamma));
      yexp_y_sum += yy * exp_y;
      yexp_ny_sum += yy * exp_ny;
      exp_y_sum += exp_y;
      exp_ny_sum += exp_ny;
    }

    T WA_x = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum;
    T WA_y = yexp_y_sum / exp_y_sum - yexp_ny_sum / exp_ny_sum;
    T H    = WA_x + WA_y;

    // ---- Step 2: compute per-pin standard WA gradient ----
    T b_x  = (*inv_gamma) / exp_x_sum;
    T a_x  = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
    T b_nx = -(*inv_gamma) / exp_nx_sum;
    T a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

    T b_y  = (*inv_gamma) / exp_y_sum;
    T a_y  = (1.0 - b_y * yexp_y_sum) / exp_y_sum;
    T b_ny = -(*inv_gamma) / exp_ny_sum;
    T a_ny = (1.0 - b_ny * yexp_ny_sum) / exp_ny_sum;

    for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
      int pin  = flat_netpin[j];
      T xx     = x[pin];
      T exp_x  = exp((xx - x_max) * (*inv_gamma));
      T exp_nx = exp((x_min - xx) * (*inv_gamma));
      // dH/dx_pin  (WA_y does not depend on x)
      grad_intermediate_x[pin] = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;

      T yy     = y[pin];
      T exp_y  = exp((yy - y_max) * (*inv_gamma));
      T exp_ny = exp((y_min - yy) * (*inv_gamma));
      // dH/dy_pin  (WA_x does not depend on y)
      grad_intermediate_y[pin] = (a_y + b_y * yy) * exp_y - (a_ny + b_ny * yy) * exp_ny;
    }

    // ---- Step 3: cosine penalty for 2-pin nets ----
    T W_theta = 0;
    // W_theta gradient w.r.t. pin positions (2-pin nets only)
    T dWt_dx_p1 = 0, dWt_dy_p1 = 0;
    T dWt_dx_p2 = 0, dWt_dy_p2 = 0;

    int p1 = flat_netpin[netpin_start[i]];
    int p2 = flat_netpin[netpin_start[i] + 1];
    T dx   = x[p2] - x[p1];
    T dy   = y[p2] - y[p1];
    T r2   = dx * dx + dy * dy;
    T r    = sqrt(r2);

    if (r > eps) {
      T inv_r   = 1.0 / r;
      T dx_ov_r = dx * inv_r;  // unit wire direction x
      T dy_ov_r = dy * inv_r;  // unit wire direction y

      // cos(theta_k) = dot(wire_dir, pin_dir_k) / |wire_dir|
      T cos1  = (dx * pin_dir_x[p1] + dy * pin_dir_y[p1]) * inv_r;
      T relu1 = DREAMPLACE_STD_NAMESPACE::max(T(0), c - cos1);

      T cos2  = (dx * pin_dir_x[p2] + dy * pin_dir_y[p2]) * inv_r;
      T relu2 = DREAMPLACE_STD_NAMESPACE::max(T(0), c - cos2);

      W_theta = relu1 * relu1 + relu2 * relu2;

      // Gradient of cos(theta_k) w.r.t. pin positions:
      // d cos_k / d x[p1] = (-vx_k + cos_k * dx/r) / r
      // d cos_k / d x[p2] = ( vx_k - cos_k * dx/r) / r
      // d cos_k / d y[p1] = (-vy_k + cos_k * dy/r) / r
      // d cos_k / d y[p2] = ( vy_k - cos_k * dy/r) / r
      //
      // d W_theta / d x = sum_k -2 * relu_k * d cos_k / d x  (when relu_k > 0)

      if (relu1 > T(0)) {
        T dcos1_dx_p1 = (-pin_dir_x[p1] + cos1 * dx_ov_r) * inv_r;
        T dcos1_dy_p1 = (-pin_dir_y[p1] + cos1 * dy_ov_r) * inv_r;
        T dcos1_dx_p2 = (pin_dir_x[p1] - cos1 * dx_ov_r) * inv_r;
        T dcos1_dy_p2 = (pin_dir_y[p1] - cos1 * dy_ov_r) * inv_r;

        T coeff1 = -2 * relu1;
        dWt_dx_p1 += coeff1 * dcos1_dx_p1;
        dWt_dy_p1 += coeff1 * dcos1_dy_p1;
        dWt_dx_p2 += coeff1 * dcos1_dx_p2;
        dWt_dy_p2 += coeff1 * dcos1_dy_p2;
      }

      if (relu2 > T(0)) {
        T dcos2_dx_p1 = (-pin_dir_x[p2] + cos2 * dx_ov_r) * inv_r;
        T dcos2_dy_p1 = (-pin_dir_y[p2] + cos2 * dy_ov_r) * inv_r;
        T dcos2_dx_p2 = (pin_dir_x[p2] - cos2 * dx_ov_r) * inv_r;
        T dcos2_dy_p2 = (pin_dir_y[p2] - cos2 * dy_ov_r) * inv_r;

        T coeff2 = -2 * relu2;
        dWt_dx_p1 += coeff2 * dcos2_dx_p1;
        dWt_dy_p1 += coeff2 * dcos2_dy_p1;
        dWt_dx_p2 += coeff2 * dcos2_dx_p2;
        dWt_dy_p2 += coeff2 * dcos2_dy_p2;
      }
    }

    // ---- Step 4: apply alpha power and cosine weight to gradient ----
    // Total: cosWA_e = (1 + W_theta) * H^alpha
    // Gradient: d cosWA_e / d x_j = (1+W_theta)*alpha*H^(alpha-1) * dH/dx_j
    //                              + H^alpha * dW_theta/dx_j
    T H_safe     = DREAMPLACE_STD_NAMESPACE::max(H, eps);
    T H_alpha    = pow(H_safe, alpha);
    T H_alpha_m1 = (alpha != T(1)) ? pow(H_safe, alpha - T(1)) : T(1);
    T wa_scale   = (1 + W_theta) * alpha * H_alpha_m1;

    for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
      int pin = flat_netpin[j];
      grad_intermediate_x[pin] *= wa_scale;
      grad_intermediate_y[pin] *= wa_scale;
    }

    // Add W_theta gradient contribution for 2-pin nets
    grad_intermediate_x[p1] += H_alpha * dWt_dx_p1;
    grad_intermediate_y[p1] += H_alpha * dWt_dy_p1;
    grad_intermediate_x[p2] += H_alpha * dWt_dx_p2;
    grad_intermediate_y[p2] += H_alpha * dWt_dy_p2;

    // ---- Step 5: store per-net wirelength ----
    partial_wl[i] = (1 + W_theta) * H_alpha;
  }

  return 0;
}

/// @brief Forward: compute cosine-weighted average wirelength.
/// @param pos pin locations (x array, y array), length 2*num_pins.
/// @param flat_netpin flat netpin map.
/// @param netpin_start starting index per net, length num_nets+1.
/// @param pin2net_map pin to net map.
/// @param net_weights weight of nets.
/// @param net_mask whether to compute wirelength per net.
/// @param inv_gamma 1/gamma scalar tensor.
/// @param pin_dir_x x-component of pin direction unit vectors.
/// @param pin_dir_y y-component of pin direction unit vectors.
/// @param c cosine threshold scalar tensor.
/// @param alpha power exponent scalar tensor.
/// @return {total_wl, grad_intermediate}.
std::vector<at::Tensor> cos_weighted_average_wirelength_forward(at::Tensor pos,
                                                                at::Tensor flat_netpin,
                                                                at::Tensor netpin_start,
                                                                at::Tensor pin2net_map,
                                                                at::Tensor net_weights,
                                                                at::Tensor net_mask,
                                                                at::Tensor inv_gamma,
                                                                at::Tensor pin_dir_x,
                                                                at::Tensor pin_dir_y,
                                                                at::Tensor c,
                                                                at::Tensor alpha) {
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
  CHECK_FLAT_CPU(pin_dir_x);
  CHECK_CONTIGUOUS(pin_dir_x);
  CHECK_FLAT_CPU(pin_dir_y);
  CHECK_CONTIGUOUS(pin_dir_y);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pos.numel() / 2;

  at::Tensor partial_wl        = at::zeros({num_nets}, pos.options());
  at::Tensor grad_intermediate = at::zeros_like(pos);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeCosWeightedAverageWirelengthMergedLauncher", [&] {
    computeCosWeightedAverageWirelengthMergedLauncher<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
                                                                DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
                                                                DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
                                                                DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
                                                                DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char),
                                                                num_nets,
                                                                DREAMPLACE_TENSOR_DATA_PTR(inv_gamma, scalar_t),
                                                                DREAMPLACE_TENSOR_DATA_PTR(pin_dir_x, scalar_t),
                                                                DREAMPLACE_TENSOR_DATA_PTR(pin_dir_y, scalar_t),
                                                                DREAMPLACE_TENSOR_DATA_PTR(c, scalar_t),
                                                                DREAMPLACE_TENSOR_DATA_PTR(alpha, scalar_t),
                                                                DREAMPLACE_TENSOR_DATA_PTR(partial_wl, scalar_t),
                                                                DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t),
                                                                DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t) + num_pins,
                                                                at::get_num_threads());
    if (net_weights.numel()) {
      partial_wl.mul_(net_weights);
    }
  });

  auto wl = partial_wl.sum();
  return {wl, grad_intermediate};
}

/// @brief Backward: multiply precomputed gradient intermediate by upstream
/// gradient and integrate net weights.
at::Tensor cos_weighted_average_wirelength_backward(at::Tensor grad_pos,
                                                    at::Tensor pos,
                                                    at::Tensor grad_intermediate,
                                                    at::Tensor flat_netpin,
                                                    at::Tensor netpin_start,
                                                    at::Tensor pin2net_map,
                                                    at::Tensor net_weights,
                                                    at::Tensor net_mask,
                                                    at::Tensor inv_gamma) {
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

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::cos_weighted_average_wirelength_forward, "CosWeightedAverageWirelength forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::cos_weighted_average_wirelength_backward, "CosWeightedAverageWirelength backward");
}
