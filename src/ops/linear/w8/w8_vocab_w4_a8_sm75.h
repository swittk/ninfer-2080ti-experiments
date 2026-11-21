#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstddef>
namespace ninfer::ops::detail {
std::size_t w8_vocab_w4_a8_reservation_bytes() noexcept;
void w8_vocab_w4_a8_prepare(const Weight& weight, cudaStream_t stream);
// Release only after the owning execution stream has stopped and synchronized.
void w8_vocab_w4_a8_release(const Weight& weight) noexcept;
bool w8_vocab_w4_a8_ready(const Weight& weight) noexcept;
void w8_vocab_w4_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                            Tensor& tokens, cudaStream_t stream);
void w8_vocab_w4_a8_launch_t1(const Tensor& x, const Weight& weight, Tensor& out,
                               Tensor& token, cudaStream_t stream);
void w8_vocab_w4_a8_include_drafts(const Tensor& x,const Weight& weight,
                                   const Tensor& drafts,const Tensor& extents,
                                   Tensor& metadata,Tensor& tokens,cudaStream_t stream);
void w8_vocab_w4_a8_score_mandatory_drafts(const Tensor& x, const Weight& weight,
                                                 const Tensor& drafts, Tensor& metadata,
                                                 cudaStream_t stream);
}
