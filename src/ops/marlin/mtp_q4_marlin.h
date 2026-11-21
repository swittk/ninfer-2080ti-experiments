#pragma once

#include "core/tensor.h"

#include <cuda_runtime_api.h>

namespace ninfer::ops::detail {

// Environment-gated native-MTP Q4 Marlin route. Preparation is explicit and
// must happen before any CUDA Graph capture. Launch returns false for an
// unprepared weight or unsupported token width so callers retain the canonical
// Q4 route outside the qualified domain.
[[nodiscard]] bool mtp_q4_marlin_enabled() noexcept;
void mtp_q4_marlin_prepare(const Weight& weight, cudaStream_t stream);
// The owner must stop all launches before release. No process-exit cleanup is required.
void mtp_q4_marlin_release(const Weight& weight) noexcept;
[[nodiscard]] bool mtp_q4_marlin_ready(const Weight& weight) noexcept;
[[nodiscard]] bool mtp_q4_marlin_launch(const Tensor& input, const Weight& weight,
                                        Tensor& output, cudaStream_t stream);

// Explicit retained draft-only numeric profile, bounded to an exact prepared
// parent's base/scale addresses. Normal weight lookup remains shape-exact.
void mtp_q4_marlin_retained_prefix_launch(const Tensor& input, const Weight& owner,
                                         const Weight& view, Tensor& output, cudaStream_t stream);

} // namespace ninfer::ops::detail
