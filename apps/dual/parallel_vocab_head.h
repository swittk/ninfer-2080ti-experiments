#pragma once
#include "targets/qwen3_6/impl/runtime/head_execution_override.h"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::apps::dual {
struct ParallelVocabHeadStats {
    std::uint64_t calls = 0;
    std::size_t helper_weight_bytes = 0;
    std::size_t owner_workspace_bytes = 0;
    std::size_t helper_workspace_bytes = 0;
    std::size_t pinned_bytes = 0;
};
// Complete native W8G32 T4 logits, partitioned by output row on two devices.
class ParallelVocabHead final : public targets::qwen3_6::detail::HeadExecutionOverride {
public:
    explicit ParallelVocabHead(int helper_device);
    ~ParallelVocabHead() override;
    ParallelVocabHead(const ParallelVocabHead&) = delete;
    ParallelVocabHead& operator=(const ParallelVocabHead&) = delete;
    void prepare(const Weight& canonical, cudaStream_t owner_stream) override;
    bool execute(const Tensor& hidden, const Weight& canonical,
                 Tensor& logits, cudaStream_t owner_stream) override;
    bool execute_greedy(const Tensor& hidden, const Weight& canonical, Tensor& tokens,
                        std::int32_t valid_rows, cudaStream_t owner_stream) override;

    [[nodiscard]] ParallelVocabHeadStats stats() const noexcept;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}
