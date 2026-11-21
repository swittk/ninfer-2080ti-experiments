#pragma once
#include "core/arena.h"
#include "targets/qwen3_6/impl/runtime/prefill_dense_override.h"
#include <cstddef>
#include <cstdint>
#include <memory>
namespace ninfer::targets::qwen3_6_27b::detail {
// Program-owned prefill-only executor. Neither weights nor mutable library state are
// shared between Programs. All matrix and library scratch lives in the planned arena.
class Sm75PrefillDense final : public qwen3_6::detail::PrefillDenseOverride {
public:
    Sm75PrefillDense(WorkspaceArena& workspace, cudaStream_t stream);
    ~Sm75PrefillDense() override;
    Sm75PrefillDense(const Sm75PrefillDense&) = delete;
    Sm75PrefillDense& operator=(const Sm75PrefillDense&) = delete;
    static constexpr std::int32_t kMinimumTokens = 64;
    [[nodiscard]] std::uint32_t minimum_batch_tokens() const noexcept override {
        return kMinimumTokens;
    }
    static bool enabled() noexcept;
    static std::size_t workspace_bytes(std::int32_t tokens);
    static std::size_t projection_workspace_bytes(int rows, int columns, int tokens);
    bool attention_input(const Tensor& hidden, const Weight& query_key,
                         const Weight& gate_value, Tensor& query, Tensor& gate,
                         Tensor& key, Tensor& value, cudaStream_t stream) override;
    bool gdn_input(const Tensor& hidden, const Weight& query_key, const Weight& value_z,
                   Tensor& qkv, Tensor& output_gate, cudaStream_t stream) override;
    bool residual(const Tensor& hidden, const Weight& weight, Tensor& output,
                  cudaStream_t stream) override;
    bool execute(std::int32_t layer, const Tensor& normalized_hidden,
                 const Weight& gate_up, const Weight& down, Tensor& residual,
                 cudaStream_t stream) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
