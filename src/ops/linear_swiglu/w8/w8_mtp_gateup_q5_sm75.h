#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace ninfer::ops::detail {

std::size_t w8_mtp_gateup_q5_sm75_reservation_bytes() noexcept;
void w8_mtp_gateup_q5_sm75_prepare(const Weight& gate_up, cudaStream_t stream);
bool w8_mtp_gateup_q5_sm75_use(const Weight& gate_up, int tokens) noexcept;
void w8_mtp_gateup_q5_sm75_launch(const Tensor& hidden, const Weight& gate_up,
                                  Tensor& gate_up_output, Tensor& activation,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
