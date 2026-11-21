#pragma once

#include "targets/qwen3_6/impl/frontend/processor.h"
#include <algorithm>
#include <list>
#include <memory>
#include <mutex>
#include <span>

namespace ninfer::targets::qwen3_6::frontend_internal {
struct PreparedMedia {
    VisionItem item;
    // Exact resized RGB8 samples in merger-friendly patch order. Normalization
    // is deterministic and performed into each request's original FP32 payload.
    // This retains source pixels, not a lossy quantization of normalized values.
    std::vector<std::uint8_t> pixels;
};

// Per-Frontend CPU cache. It holds only validated, immutable preprocessing results;
// neither generated output nor GPU/KV state is cached here. Each request still
// rebuilds token spans/positions and enforces its aggregate media/prompt budgets.
class MediaPreparationCache {
    struct Entry {
        ChatPartKind kind;
        ProcessorOptions options;
        std::vector<std::uint8_t> bytes;
        std::shared_ptr<const PreparedMedia> value;
        std::size_t charge;
    };
    static bool matches(const Entry& entry, ChatPartKind kind,
                        std::span<const std::uint8_t> bytes,
                        const std::array<std::uint8_t, 32>& digest,
                        const ProcessorOptions& options) {
        return entry.kind == kind && entry.options == options &&
               entry.value->item.content_digest == digest &&
               entry.bytes.size() == bytes.size() &&
               std::equal(bytes.begin(), bytes.end(), entry.bytes.begin());
    }
public:
    explicit MediaPreparationCache(std::size_t maximum_bytes) : limit_(maximum_bytes) {}

    std::shared_ptr<const PreparedMedia> find(
        ChatPartKind kind, std::span<const std::uint8_t> bytes,
        const std::array<std::uint8_t, 32>& digest, const ProcessorOptions& options) {
        std::lock_guard lock(mutex_);
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (matches(*it, kind, bytes, digest, options)) {
                auto value = it->value;
                entries_.splice(entries_.begin(), entries_, it);
                return value;
            }
        }
        return {};
    }

    void store(ChatPartKind kind, std::span<const std::uint8_t> bytes,
               const ProcessorOptions& options, std::shared_ptr<const PreparedMedia> value) {
        if (!value || bytes.size() > limit_ || value->pixels.capacity() > limit_) {
            return;
        }
        // Encoded identity plus actual vector capacity, not just visible elements.
        const std::size_t payload = sizeof(Entry) + sizeof(PreparedMedia) +
            value->pixels.capacity() +
            value->item.timestamps.capacity() * sizeof(double) +
            value->item.token_spans.capacity() * sizeof(TokenSpan) + 4 * sizeof(void*);
        if (payload > limit_ || bytes.size() > limit_ - payload) { return; }
        Entry entry{kind, options, std::vector<std::uint8_t>(bytes.begin(), bytes.end()), value, 0};
        entry.charge = payload + entry.bytes.capacity();
        if (entry.charge > limit_) { return; }
        std::lock_guard lock(mutex_);
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (matches(*it, kind, bytes, value->item.content_digest, options)) {
                entries_.splice(entries_.begin(), entries_, it);
                return; // Concurrent equivalent insertion need not duplicate storage.
            }
        }
        while (!entries_.empty() && (used_ > limit_ - entry.charge || entries_.size() >= 64)) {
            used_ -= entries_.back().charge;
            entries_.pop_back();
        }
        const auto charge = entry.charge;
        entries_.push_front(std::move(entry));
        used_ += charge;
    }

    [[nodiscard]] std::size_t retained_bytes() const {
        std::lock_guard lock(mutex_);
        return used_;
    }
private:
    const std::size_t limit_;
    mutable std::mutex mutex_;
    std::list<Entry> entries_;
    std::size_t used_ = 0;
};
} // namespace ninfer::targets::qwen3_6::frontend_internal
