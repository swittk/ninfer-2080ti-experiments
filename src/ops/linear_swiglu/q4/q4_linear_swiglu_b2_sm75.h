#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops::detail {

bool q4_linear_swiglu_b2_sm75_select_layer(int layer) noexcept;
std::size_t q4_linear_swiglu_b2_sm75_reservation_bytes() noexcept;
void q4_linear_swiglu_b2_sm75_prepare(const Weight& gate_up, cudaStream_t stream);
bool q4_linear_swiglu_b2_sm75_ready(const Weight& gate_up) noexcept;
void q4_linear_swiglu_b2_sm75_launch(const Tensor& normalized, const Weight& gate_up,
                                     Tensor& output_q, Tensor& output_scales,
                                     cudaStream_t stream);

} // namespace ninfer::ops::detail
