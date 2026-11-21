#pragma once

#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

class MlpExecutionOverride {
public:
    virtual ~MlpExecutionOverride() = default;

    // Return true after writing the complete post-MLP residual update. Returning false delegates
    // to the canonical local implementation. Decode and prefill hooks are separate: a decode
    // override is invoked only for its qualified Verify widths; a prefill override only for Prefill.
    virtual bool execute(std::int32_t global_layer, const Tensor& normalized_hidden,
                         const Weight& canonical_gate_up,
                         const Weight& canonical_down, Tensor& residual,
                         cudaStream_t owner_stream) = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
