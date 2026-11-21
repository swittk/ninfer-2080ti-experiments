#pragma once
#include <cstdint>
namespace ninfer::targets::qwen3_6 {
// Retain historical throughput semantics separately from complete computation.
// This choice is fixed before any graph capture or prefix-snapshot operation.
enum class LongContextComputation : std::uint32_t { RetainedThroughput=0, Complete=1 };
constexpr bool use_retained_long_context(LongContextComputation policy,std::uint32_t visible) noexcept {
    return policy==LongContextComputation::RetainedThroughput && visible>=32768;
}
// The retained 1%/5% ranges need local subdivision to fill the device. Full
// attention uses every existing split and therefore the native full-range plan.
constexpr std::uint32_t long_context_split_subdivision(LongContextComputation policy,
                                                        std::uint32_t capacity) noexcept {
    return use_retained_long_context(policy,capacity) ? 8U : 1U;
}
constexpr bool license_retained_d4(LongContextComputation policy,std::uint32_t width,
                                   std::uint32_t extent) noexcept {
    return policy==LongContextComputation::RetainedThroughput && width==5 && extent==4;
}
constexpr bool complete_proposal_window_allowed(LongContextComputation policy,bool mtp,
    unsigned sessions,unsigned depth,bool proposal_stage0,unsigned window) noexcept {
    return window==0 || (policy==LongContextComputation::Complete && mtp && sessions==1 &&
        depth==3 && !proposal_stage0 && window>=128 && window<=262144 && window%64==0);
}
constexpr bool stop_at_message_end(LongContextComputation policy,bool media) noexcept {
    // Preserve historical text timing fixtures, while complete computation must
    // never continue inventing turns after the model completes its answer.
    return media || policy==LongContextComputation::Complete;
}
} // namespace ninfer::targets::qwen3_6
