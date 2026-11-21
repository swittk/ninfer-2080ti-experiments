#pragma once
#include "targets/qwen3_6/impl/runtime/mlp_execution_override.h"
namespace ninfer::targets::qwen3_6::detail {
// Optional Program-owned prefill leaf implementation. Request state, KV, recurrent
// state updates and scheduling remain in the family runtime. Returning false must
// leave outputs untouched and selects the existing leaf.
class PrefillDenseOverride : public MlpExecutionOverride {
public:
    // The backend's actual lower bound, not a scheduler tuning threshold.
    // Cold cohorts are useful when individual prompts cannot use this backend
    // but their concatenated real columns can. Larger prompts run immediately.
    [[nodiscard]] virtual std::uint32_t minimum_batch_tokens() const noexcept = 0;
    virtual bool attention_input(const Tensor& hidden, const Weight& query_key,
                                 const Weight& gate_value, Tensor& query, Tensor& gate,
                                 Tensor& key, Tensor& value, cudaStream_t stream) = 0;
    virtual bool gdn_input(const Tensor& hidden, const Weight& query_key,
                           const Weight& value_z, Tensor& qkv, Tensor& output_gate,
                           cudaStream_t stream) = 0;
    virtual bool residual(const Tensor& hidden, const Weight& weight,
                          Tensor& output, cudaStream_t stream) = 0;
};
}
