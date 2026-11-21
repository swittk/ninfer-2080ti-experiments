#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace ninfer::ops::detail {

std::size_t w8_mtp_down_w4a8_sm75_reservation_bytes() noexcept;
void w8_mtp_down_w4a8_sm75_prepare(const Weight& down, cudaStream_t stream);
bool w8_mtp_down_w4a8_sm75_use(const Weight& down, int tokens) noexcept;
void w8_mtp_down_w4a8_sm75_launch(const Tensor& activation, const Weight& down,
                                  Tensor& residual, cudaStream_t stream);

} // namespace ninfer::ops::detail
