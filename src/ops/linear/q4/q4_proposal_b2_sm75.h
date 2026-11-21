#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops::detail {

std::size_t q4_proposal_b2_reservation_bytes() noexcept;
void q4_proposal_b2_prepare(const Weight& proposal_head, cudaStream_t stream);
// Release only after the owning execution stream has stopped and synchronized.
void q4_proposal_b2_release(const Weight& weight) noexcept;
bool q4_proposal_b2_ready(const Weight& proposal_head) noexcept;
void q4_proposal_b2_launch(const Tensor& hidden, const Weight& proposal_head,
                           Tensor& proposal_tokens, cudaStream_t stream);

} // namespace ninfer::ops::detail
