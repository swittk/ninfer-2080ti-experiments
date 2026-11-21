#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::runtime {
/** Recent per-request progress, never aggregate token throughput across users. */
class DecodeProgressWindow {
public:
    void observe(std::uint32_t tokens, double seconds, std::size_t membership) noexcept {
        if (!(seconds > 0.0) || !std::isfinite(seconds) || tokens == 0) { return; }
        // Different batch sizes have different per-round cost. Do not spend
        // a new C4 round using the throughput measured in a previous C1 batch.
        if (membership != membership_) { *this = {}; membership_ = membership; }
        if (count_ == samples_.size()) {
            tokens_ -= samples_[next_].tokens;
            seconds_ -= samples_[next_].seconds;
        } else { ++count_; }
        samples_[next_] = {tokens, seconds};
        next_ = (next_ + 1) % samples_.size();
        tokens_ += tokens;
        seconds_ += seconds;
    }

    /** Bounded round count needed to amortize one observed prefill interruption. */
    [[nodiscard]] std::uint32_t rounds(double prefill_seconds, double target_tps,
                                        std::uint32_t maximum) const noexcept {
        if (count_ < 4 || !(prefill_seconds > 0.0) || !std::isfinite(prefill_seconds) ||
            !(target_tps > 0.0) || !std::isfinite(target_tps)) { return maximum; }
        const double spare_tokens = (static_cast<double>(tokens_) - target_tps * seconds_) /
                                    static_cast<double>(count_);
        // When the target is above the measured decoder rate, keep the existing
        // maximum burst rather than starving the incoming request indefinitely.
        if (!(spare_tokens > 0.0)) { return maximum; }
        const double needed = std::ceil(target_tps * prefill_seconds / spare_tokens);
        if (!std::isfinite(needed) || needed >= maximum) { return maximum; }
        return static_cast<std::uint32_t>(std::max(1.0, needed));
    }
private:
    struct Sample { std::uint32_t tokens = 0; double seconds = 0.0; };
    std::array<Sample, 8> samples_{};
    std::size_t count_ = 0, next_ = 0, membership_ = 0;
    std::uint64_t tokens_ = 0;
    double seconds_ = 0.0;
};

/** Lane-owned prefill samples avoid carrying a cheap old prompt's timing forward. */
class PrefillDurationWindow {
public:
    void observe(double seconds) noexcept {
        if (seconds > 0.0 && std::isfinite(seconds)) {
            samples_[next_] = seconds;
            next_ = (next_ + 1) % samples_.size();
        }
    }
    [[nodiscard]] double peak() const noexcept {
        double maximum = 0.0;
        for (const double sample : samples_) { maximum = std::max(maximum, sample); }
        return maximum;
    }
private:
    std::array<double, 4> samples_{};
    std::size_t next_ = 0;
};

inline void validate_prefill_target_tps(double value) {
    if (!std::isfinite(value) || value < 0.0 || value > 1000.0) {
        throw std::invalid_argument("prefill target decode rate must be finite in [0,1000]");
    }
}
} // namespace ninfer::runtime
