#pragma once

#include <cstdint>

namespace ninfer::targets::qwen3_6 {
// Existing public frontend limit, shared with runtime workspace planning.
// No individual attention segment admitted under this total pair budget can
// exceed its integer square root. Keep the spatial 2x2 merger boundary intact.
inline constexpr std::uint64_t kVisionAttentionPairBudget = 128ULL * 1024ULL * 1024ULL;
inline constexpr std::uint32_t kVisionBatchPatchCapacity = [] {
    std::uint64_t low = 0, high = kVisionAttentionPairBudget;
    while (low < high) {
        const auto mid = low + (high - low + 1) / 2;
        if (mid <= kVisionAttentionPairBudget / mid) low = mid;
        else high = mid - 1;
    }
    return static_cast<std::uint32_t>(low / 4 * 4);
}();
static_assert(kVisionBatchPatchCapacity == 11584);
} // namespace ninfer::targets::qwen3_6
