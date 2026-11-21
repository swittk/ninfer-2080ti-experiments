#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace ninfer::runtime {
// Per-worker scheduling only; no request identity or GPU state is cached here.
// Cancellation/deadline checks remain outside this policy on every work unit.
class PrefillDecodeBudget {
public:
    explicit PrefillDecodeBudget(std::uint32_t rounds) : limit_(rounds), remaining_(rounds) {
        if (rounds == 0 || rounds > 64) {
            throw std::invalid_argument("prefill decode rounds must be in [1,64]");
        }
    }
    [[nodiscard]] bool should_decode(bool decode_ready) const noexcept {
        return decode_ready && remaining_ != 0;
    }
    // A burst protects service which was already streaming when this request
    // arrived, not decoders created later from the same all-cold ingress cohort.
    static bool protects_existing_stream(
        bool decode_ready,
        const std::optional<std::chrono::steady_clock::time_point>& first_token,
        std::chrono::steady_clock::time_point arrival) noexcept {
        return decode_ready && first_token && *first_token <= arrival;
    }
    // Classify the original uncached suffix, not the remaining tail. Otherwise
    // the final chunks of a cold image lose protection and stall established
    // streams, while genuinely short warm suffixes should still finish promptly.
    static bool has_substantial_suffix(std::uint32_t computed, std::uint32_t remaining,
                                       std::uint32_t chunk) noexcept {
        return remaining != 0 && static_cast<std::uint64_t>(computed) + remaining > chunk;
    }
    void limit_to_one() noexcept { if (remaining_ > 1) { remaining_ = 1; } }
    void decoded() noexcept { if (remaining_ != 0) { --remaining_; } }
    // Do not delay a genuinely short warm suffix behind a burst.
    void prefilled(bool substantial_prefill = true, std::uint32_t measured_rounds = 0) noexcept {
        remaining_ = substantial_prefill ? limit_ : 1U;
        if (substantial_prefill && measured_rounds != 0 && measured_rounds < remaining_) {
            remaining_ = measured_rounds;
        }
    }
private:
    const std::uint32_t limit_;
    std::uint32_t remaining_;
};
} // namespace ninfer::runtime
