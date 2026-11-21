#pragma once
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <span>
namespace ninfer::runtime {
// Nonfinal fragments retain one real prompt token for the ordinary finalizer.
// Large full work units remain whole; never shrink one to squeeze in more owners.
inline std::uint32_t packed_prefill_fragment(std::uint32_t remaining,
                                            std::uint32_t work_limit) noexcept {
    return remaining > 1 ? std::min(work_limit, remaining - 1) : 0;
}
inline std::uint32_t packed_prefill_ragged_total(std::span<const std::uint32_t> widths,
                                                std::uint32_t capacity) noexcept {
    if (widths.size() < 2 || widths.size() > 8) return 0;
    std::uint64_t total = 0;
    for (const auto width : widths) {
        if (width == 0) return 0;
        total += width;
    }
    return total > 0 && total <= capacity ? static_cast<std::uint32_t>(total) : 0;
}
} // namespace ninfer::runtime
