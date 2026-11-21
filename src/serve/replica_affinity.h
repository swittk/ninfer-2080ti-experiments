#pragma once

#include "serve/request.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::serve {
// Routing hints only. Engine token/media/position validation still decides reuse.
// Keep the most recent turn boundaries so appending assistant/user/tool messages
// can find the owner of the preceding request without storing prompt contents.
struct ReplicaAffinity {
    std::array<std::uint64_t, 64> turn_keys{};
    std::size_t count = 0;
    [[nodiscard]] std::span<const std::uint64_t> prefixes() const noexcept {
        return {turn_keys.data(), count};
    }
};

inline ReplicaAffinity request_replica_affinity(const GenerationRequest& request) {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto integer = [&](std::uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            hash = (hash ^ static_cast<unsigned char>(value)) * 1099511628211ULL;
            value >>= 8;
        }
    };
    const auto bytes = [&](std::span<const std::uint8_t> value) {
        integer(value.size());
        for (const auto c : value) { hash = (hash ^ c) * 1099511628211ULL; }
    };
    const auto text = [&](std::string_view value) {
        bytes({reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
    };
    std::array<std::uint64_t, 64> ring{};
    std::size_t total = 0;
    for (const ChatTurn& turn : request.messages) {
        text(turn.role);
        integer(turn.content.size());
        for (const ContentPart& part : turn.content) {
            integer(static_cast<std::uint64_t>(part.kind));
            if (part.kind == ContentKind::Text) { text(part.text); }
            else {
                integer(static_cast<std::uint64_t>(part.source.kind));
                text(part.source.value);
                text(part.source.media_type);
                bytes(part.source.bytes);
            }
        }
        text(turn.reasoning_content);
        text(turn.tool_call_id);
        integer(turn.tool_calls.size());
        for (const ToolCall& call : turn.tool_calls) {
            text(call.id); text(call.name); text(call.arguments_json);
        }
        ring[total++ % ring.size()] = hash;
    }
    ReplicaAffinity result;
    result.count = std::min(total, ring.size());
    for (std::size_t i = 0; i < result.count; ++i) {
        result.turn_keys[i] = ring[(total - result.count + i) % ring.size()];
    }
    return result;
}
} // namespace ninfer::serve
