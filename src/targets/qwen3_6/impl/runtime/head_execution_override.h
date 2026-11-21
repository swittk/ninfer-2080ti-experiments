#pragma once
#include "core/tensor.h"
#include <cuda_runtime_api.h>

namespace ninfer::targets::qwen3_6::detail {
// Borrowed by TextContext. Source weights stay immutable until the coordinator
// is detached and drained. Re-preparation is required after any source change.
class HeadExecutionOverride {
public:
    virtual ~HeadExecutionOverride() = default;
    virtual void prepare(const Weight& canonical, cudaStream_t owner_stream) = 0;
    // A successful call writes every native output logit, in the original order.
    // Unsupported widths return false without writing, preserving native fallback.
    virtual bool execute(const Tensor& hidden, const Weight& canonical,
                         Tensor& logits, cudaStream_t owner_stream) = 0;
    // Separate result contract: writes only exact greedy IDs over valid_rows.
    // No caller may treat a successful result as materialized logits.
    virtual bool execute_greedy(const Tensor&, const Weight&, Tensor&,
                                std::int32_t, cudaStream_t) { return false; }

};
}
