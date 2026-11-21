#pragma once

#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/materializer.h"
#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {

[[nodiscard]] ArtifactLoadPlan bind_text_shard_artifact(artifact::Binder& binder,
                                                        const artifact::Reader& reader,
                                                        WeightsProfile weights_profile,
                                                        TextShardSpec shard);

struct TextPromptSequence {
    std::vector<TokenId> token_ids;
    std::int32_t rope_delta = 0;
};

class TextShardStage {
public:
    TextShardStage(std::string artifact_path, int device_id, std::uint32_t capacity,
                   TextShardSpec shard, artifact::LoadProgress* progress = nullptr,
                   std::uint32_t prefill_chunk = 128);
    ~TextShardStage();

    TextShardStage(const TextShardStage&)            = delete;
    TextShardStage& operator=(const TextShardStage&) = delete;
    TextShardStage(TextShardStage&&)                 = delete;
    TextShardStage& operator=(TextShardStage&&)      = delete;

    [[nodiscard]] TextPromptSequence prepare_text_prompt(std::string_view text,
                                                         bool enable_thinking = false) const;
    void forward_token(TokenId token, std::int32_t position, std::int32_t rope_position);
    void forward_hidden(std::int32_t position, std::int32_t rope_position);
    void prefill_tokens(std::span<const TokenId> tokens, std::int32_t position,
                        std::int32_t rope_position);
    void copy_prefill_residual_to_host(void* destination, std::size_t tokens);
    void copy_prefill_residual_from_host(const void* source, std::size_t tokens);
    void prefill_hidden(std::size_t tokens, std::int32_t position,
                        std::int32_t rope_position);
    void copy_residual_to_host(void* destination, std::size_t bytes);
    void copy_residual_from_host(const void* source, std::size_t bytes);
    [[nodiscard]] TokenId project_token();
    void synchronize();

    [[nodiscard]] int device_id() const noexcept;
    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::size_t weights_bytes() const noexcept;
    [[nodiscard]] std::size_t state_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_bytes() const noexcept;
    [[nodiscard]] std::uint32_t prefill_chunk() const noexcept;
    [[nodiscard]] static constexpr std::size_t residual_bytes() noexcept {
        return 5120 * sizeof(std::uint16_t);
    }
    [[nodiscard]] static constexpr std::size_t residual_bytes(std::size_t tokens) noexcept {
        return tokens * residual_bytes();
    }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::qwen3_6_27b::detail
