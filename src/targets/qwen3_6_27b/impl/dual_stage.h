#pragma once
#include "targets/qwen3_6/impl/runtime/long_context_policy.h"

#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6/impl/runtime/gqa_execution_override.h"
#include "targets/qwen3_6/impl/runtime/mlp_execution_override.h"
#include "targets/qwen3_6/impl/runtime/head_execution_override.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/materializer.h"
#include <ninfer/types.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {

inline constexpr std::size_t kDualMtpMaximumDrafts = qwen3_6::kMtpDecodeMaximumDrafts;
inline constexpr std::size_t kDualMtpMaximumWidth = qwen3_6::kMtpDecodeMaximumWidth;

[[nodiscard]] ArtifactLoadPlan bind_text_shard_artifact(artifact::Binder& binder,
                                                        const artifact::Reader& reader,
                                                        WeightsProfile weights_profile,
                                                        TextShardSpec shard,
                                                        qwen3_6::StartupFeatures features = {});

struct TextPromptSequence {
    std::vector<TokenId> token_ids;
    std::int32_t rope_delta = 0;
    std::shared_ptr<qwen3_6::PreparedPromptData> prepared;

    [[nodiscard]] bool has_media() const noexcept {
        return prepared != nullptr && prepared->has_media();
    }
};

struct Mtp3SessionState {
    TokenId anchor = 0;
    std::array<TokenId, kDualMtpMaximumDrafts> drafts{};
    std::int32_t draft_extent = 3;
    std::int32_t frontier = 0;
    std::int32_t remaining_budget = 0;
    std::int32_t rope_delta = 0;
};

struct Mtp3VerifyInput {
    std::array<TokenId, kDualMtpMaximumWidth> ids{};
    std::array<std::int32_t, kDualMtpMaximumWidth> positions{};
    std::array<std::int32_t, kDualMtpMaximumWidth> rope_positions{};
    std::int32_t valid_columns = 4;
};

struct Mtp3RoundResult {
    std::array<TokenId, kDualMtpMaximumWidth> licensed_tokens{};
    std::array<TokenId, kDualMtpMaximumDrafts> next_drafts{};
    std::int32_t licensed_count = 0;
    std::int32_t accepted_drafts = 0;
    std::int32_t next_extent = 0;
    TokenId next_anchor = 0;
    std::int32_t next_frontier = 0;
};

struct TextKvPageOwnership {
    bool enabled = false;
    std::uint32_t logical_page_begin = 0;
    std::uint32_t logical_page_count = 0;
};

class TextShardStage {
public:
    TextShardStage(std::string artifact_path, int device_id, std::uint32_t capacity,
                   TextShardSpec shard, artifact::LoadProgress* progress = nullptr,
                   std::uint32_t prefill_chunk = 128,
                   KvCacheStorage kv_cache = KvCacheStorage::BFloat16,
                   std::uint32_t sessions = 1, std::uint32_t verify_width = 0,
                   bool enable_mtp = false,
                   TextKvPageOwnership text_kv_ownership = {},
                   bool enable_vision = false,
                   qwen3_6::LongContextComputation computation =
                       qwen3_6::LongContextComputation::RetainedThroughput,
                   std::uint32_t mtp_attention_window=0, bool full_vocab_draft=false);
    ~TextShardStage();

    TextShardStage(const TextShardStage&)            = delete;
    TextShardStage& operator=(const TextShardStage&) = delete;
    TextShardStage(TextShardStage&&)                 = delete;
    TextShardStage& operator=(TextShardStage&&)      = delete;

    [[nodiscard]] TextPromptSequence prepare_text_prompt(std::string_view text,
                                                         bool enable_thinking = false) const;
    [[nodiscard]] TextPromptSequence prepare_prompt(PromptInput input) const;
    void forward_token(TokenId token, std::int32_t position, std::int32_t rope_position,
                       std::uint32_t session = 0);
    void forward_hidden(std::int32_t position, std::int32_t rope_position,
                        std::uint32_t session = 0);
    void prefill_tokens(std::span<const TokenId> tokens, std::int32_t position,
                        std::int32_t rope_position, std::uint32_t session = 0);
    void copy_prefill_residual_to_host(void* destination, std::size_t tokens);
    void copy_prefill_residual_from_host(const void* source, std::size_t tokens);
    void prefill_hidden(std::size_t tokens, std::int32_t position,
                        std::int32_t rope_position, std::uint32_t session = 0);
    [[nodiscard]] std::size_t prefill_multimodal_tokens(
        const TextPromptSequence& prompt, std::uint32_t begin, std::uint32_t nominal_length,
        std::uint32_t session = 0);
    void prefill_multimodal_hidden(const TextPromptSequence& prompt, std::uint32_t begin,
                                   std::size_t tokens, std::uint32_t session = 0);
    void mtp_prefill_prompt_chunk(std::span<const TokenId> full_prompt,
                                  std::uint32_t chunk_begin, std::size_t tokens,
                                  bool final_chunk, TokenId generated_token,
                                  std::uint32_t session = 0);
    void mtp_prefill_multimodal_prompt_chunk(
        const TextPromptSequence& prompt, std::uint32_t chunk_begin, std::size_t tokens,
        bool final_chunk, TokenId generated_token, std::uint32_t session = 0);
    void prepare_sm75_a8_decode_layout();
    void set_mlp_execution_override(qwen3_6::detail::MlpExecutionOverride* override);
    void set_head_execution_override(qwen3_6::detail::HeadExecutionOverride* value,
                                     bool greedy_tokens_only = false);
    void set_gqa_execution_override(qwen3_6::detail::GqaExecutionOverride* override);
    void prepare_gqa_prefix(qwen3_6::detail::GqaExecutionOverride& executor,
                            std::uint32_t populated_tokens);
    void set_mtp_greedy_margin(float delta);
    [[nodiscard]] std::vector<TokenId> mtp_initial_drafts(std::uint32_t session = 0);
    [[nodiscard]] Mtp3VerifyInput mtp_prepare_verify(const Mtp3SessionState& state,
                                                     std::uint32_t session = 0);
    [[nodiscard]] Mtp3RoundResult mtp_accept_round(const Mtp3SessionState& state,
                                                   std::uint32_t session = 0);
    [[nodiscard]] std::array<TokenId, kDualMtpMaximumDrafts>
    mtp_propose_round(const Mtp3SessionState& state, const Mtp3RoundResult& accepted,
                      const void* target_hidden_bf16, std::size_t target_hidden_bytes,
                      std::uint32_t session = 0);
    [[nodiscard]] Mtp3RoundResult mtp_finish_round(const Mtp3SessionState& state,
                                                   std::uint32_t session = 0);
    void copy_residual_to_host_async(void* destination, std::size_t bytes,
                                     std::uint32_t session = 0);
    void copy_residual_to_host(void* destination, std::size_t bytes,
                               std::uint32_t session = 0);
    void copy_residual_from_host(const void* source, std::size_t bytes,
                                 std::uint32_t session = 0);
    [[nodiscard]] TokenId project_token(std::uint32_t session = 0);
    void target_verify_tokens(std::span<const TokenId> tokens, std::int32_t position,
                              std::int32_t rope_position, std::uint32_t session = 0);
    void target_verify_tokens(const Mtp3VerifyInput& input, std::uint32_t session = 0);
    void target_verify_tokens_batch(std::span<const Mtp3VerifyInput> inputs,
                                    std::uint32_t session_begin = 0);
    void copy_verify_residual_to_host(void* destination, std::size_t bytes,
                                      std::uint32_t session = 0);
    void copy_verify_residual_from_host(const void* source, std::size_t bytes,
                                        std::uint32_t session = 0);
    void copy_verify_residual_batch_to_host(void* destination, std::size_t bytes,
                                            std::uint32_t session_begin = 0);
    void copy_verify_residual_batch_from_host(const void* source, std::size_t bytes,
                                              std::uint32_t session_begin = 0);
    [[nodiscard]] std::vector<TokenId> target_verify_hidden(std::int32_t position,
                                                            std::int32_t rope_position,
                                                            std::uint32_t session = 0);
    void launch_target_verify_hidden(std::int32_t position, std::int32_t rope_position,
                                     std::uint32_t session = 0);
    [[nodiscard]] std::vector<TokenId> target_verify_hidden(const Mtp3VerifyInput& input,
                                                            std::uint32_t session = 0);
    void launch_target_verify_hidden(const Mtp3VerifyInput& input,
                                     std::uint32_t session = 0);
    [[nodiscard]] std::vector<TokenId> collect_verify_tokens(std::uint32_t session = 0);
    [[nodiscard]] std::vector<TokenId> project_verify_tokens(std::uint32_t session = 0);
    [[nodiscard]] std::vector<std::array<TokenId, 4>>
    target_verify_hidden_batch(std::span<const Mtp3VerifyInput> inputs,
                               std::uint32_t session_begin = 0);
    void copy_verify_hidden_to_host_async(void* destination, std::size_t bytes,
                                          std::uint32_t session = 0);
    void copy_verify_hidden_to_host(void* destination, std::size_t bytes,
                                    std::uint32_t session = 0);
    void copy_verify_logits_to_host(void* destination, std::size_t bytes,
                                    std::uint32_t session = 0);
    [[nodiscard]] std::vector<std::uint64_t>
    verify_record_layer_digests(std::uint32_t session = 0);
    void fold_verify_records(std::int32_t commit_columns, std::uint32_t session = 0);
    void fold_verify_records_batch(std::span<const std::int32_t> commit_columns,
                                   std::uint32_t session_begin = 0);
    void synchronize();
    [[nodiscard]] std::size_t prefix_snapshot_bytes() const noexcept;
    void copy_prefix_snapshot_to_host(void* destination, std::size_t bytes);
    void copy_prefix_snapshot_from_host(const void* source, std::size_t bytes);
    [[nodiscard]] std::size_t legacy_full_prefix_snapshot_bytes() const noexcept;
    void copy_legacy_full_prefix_snapshot_from_host(const void* source, std::size_t bytes);
    [[nodiscard]] std::size_t legacy_text_kv_range_bytes(
        std::uint32_t logical_page_begin, std::uint32_t logical_page_count) const;
    [[nodiscard]] std::vector<std::uint8_t> pack_legacy_text_kv_range(
        const void* source, std::size_t bytes, std::uint32_t logical_page_begin,
        std::uint32_t logical_page_count) const;
    [[nodiscard]] TextKvPageOwnership text_kv_ownership() const noexcept;

    [[nodiscard]] int device_id() const noexcept;
    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::size_t weights_bytes() const noexcept;
    [[nodiscard]] std::size_t state_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_bytes() const noexcept;
    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
    [[nodiscard]] KvCacheStorage kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t session_count() const noexcept;
    [[nodiscard]] std::uint32_t verify_width() const noexcept;
    [[nodiscard]] std::uint32_t prefill_chunk() const noexcept;
    [[nodiscard]] static constexpr std::size_t residual_bytes() noexcept {
        return 5120 * sizeof(std::uint16_t);
    }
    [[nodiscard]] static constexpr std::size_t residual_bytes(std::size_t tokens) noexcept {
        return tokens * residual_bytes();
    }
    [[nodiscard]] std::size_t verify_residual_bytes() const noexcept {
        return residual_bytes(verify_width());
    }
    [[nodiscard]] std::size_t verify_logits_bytes() const noexcept {
        return static_cast<std::size_t>(248320) * verify_width() * sizeof(std::uint16_t);
    }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::qwen3_6_27b::detail
