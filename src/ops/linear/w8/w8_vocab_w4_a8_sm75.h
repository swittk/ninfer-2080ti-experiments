#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstddef>
namespace ninfer::ops::detail {
std::size_t w8_vocab_w4_a8_reservation_bytes() noexcept;
void w8_vocab_w4_a8_prepare(const Weight& weight, cudaStream_t stream);
bool w8_vocab_w4_a8_ready(const Weight& weight) noexcept;
void w8_vocab_w4_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                            Tensor& tokens, cudaStream_t stream);
}
