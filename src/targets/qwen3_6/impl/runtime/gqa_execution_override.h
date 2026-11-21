#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_6::detail {

// Optional execution seam for product-owned multi-device attention engines. The canonical
// TextContext retains projection, normalization, RoPE, layer scheduling, and all state authority.
// An override receives the exact A1 operands and may execute the cache append + attention result.
// Returning false delegates the unchanged call to the canonical GQA wrapper.
class GqaExecutionOverride {
public:
    virtual ~GqaExecutionOverride() = default;
    // An override may derive immutable serving state from an already restored
    // prefix. This phase runs before decode timing or token production, and may
    // not append tokens or retain ownership of the canonical cache allocation.
    virtual bool requires_prefix_preparation() const noexcept { return false; }
    virtual void prepare_prefix_layer(std::int32_t, PagedKVLayerView,
                                      std::uint32_t, cudaStream_t) {}

    virtual bool proposal_page_guide(Tensor& log_mass,Tensor& anchor_position) const {
        return false;
    }


    virtual bool execute(std::int32_t global_layer, const Tensor& q, const Tensor& k,
                         const Tensor& v, const Tensor& positions,
                         const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                         PagedKVBatchLayerView local_cache, ops::GqaExecutionEnvelope envelope,
                         WorkspaceArena& workspace, Tensor& out,
                         cudaStream_t owner_stream) = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
