#include "ops/linear_swiglu/w8/w8_mtp_gateup_q5_sm75.h"

#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/silu_mul.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kN = 34816;
constexpr int kK = 5120;
constexpr int kSourceGroup = 32;
constexpr int kSourceGroups = kK / kSourceGroup;
constexpr int kQ5Group = 64;
constexpr int kQ5Groups = kK / kQ5Group;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr std::size_t kLowBytes = static_cast<std::size_t>(kN) * kQ5Groups * 32;
constexpr std::size_t kHighBytes = static_cast<std::size_t>(kN) * kQ5Groups * 8;
constexpr std::size_t kScaleBytes = static_cast<std::size_t>(kN) * kQ5Groups * sizeof(__half);

struct State {
    const void* source = nullptr;
    std::uint8_t* low = nullptr;
    std::uint8_t* high = nullptr;
    __half* scales = nullptr;
    Weight weight{};
};

State& state() {
    static State value;
    return value;
}

int mode() {
    static int value = [] {
        const char* env = std::getenv("NINFER_SM75_MTP_Q5_MODE");
        const int parsed = env ? std::atoi(env) : 0;
        return std::clamp(parsed, 0, 3);
    }();
    return value;
}

__global__ void convert_w8_to_q5(const std::int8_t* source_codes,
                                 const std::uint16_t* source_scales,
                                 std::uint8_t* low,
                                 std::uint8_t* high,
                                 __half* output_scales) {
    __shared__ std::int8_t codes[kWarps][kQ5Group];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int unit = static_cast<int>(blockIdx.x) * kWarps + warp;
    const int total = kN * kQ5Groups;
    if (unit >= total) { return; }
    const int row = unit / kQ5Groups;
    const int group = unit - row * kQ5Groups;
    const int index0 = lane * 2;
    const int index1 = index0 + 1;
    const int source_group0 = group * 2 + index0 / kSourceGroup;
    const int source_group1 = group * 2 + index1 / kSourceGroup;
    const float scale0 = __half2float(__ushort_as_half(
        source_scales[static_cast<std::int64_t>(row) * kSourceGroups + source_group0]));
    const float scale1 = __half2float(__ushort_as_half(
        source_scales[static_cast<std::int64_t>(row) * kSourceGroups + source_group1]));
    const float value0 = static_cast<float>(
        source_codes[static_cast<std::int64_t>(row) * kK + group * kQ5Group + index0]) *
                         scale0;
    const float value1 = static_cast<float>(
        source_codes[static_cast<std::int64_t>(row) * kK + group * kQ5Group + index1]) *
                         scale1;

    float max_abs = fmaxf(fabsf(value0), fabsf(value1));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffffu, max_abs, offset));
    }
    float scale = max_abs > 0.0f ? max_abs / 15.0f : 0.0f;
#pragma unroll
    for (int iteration = 0; iteration < 2; ++iteration) {
        const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
        int q0 = __float2int_rn(value0 * inverse);
        int q1 = __float2int_rn(value1 * inverse);
        q0 = max(-16, min(15, q0));
        q1 = max(-16, min(15, q1));
        float numerator = value0 * static_cast<float>(q0) + value1 * static_cast<float>(q1);
        float denominator = static_cast<float>(q0 * q0 + q1 * q1);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            numerator += __shfl_xor_sync(0xffffffffu, numerator, offset);
            denominator += __shfl_xor_sync(0xffffffffu, denominator, offset);
        }
        scale = denominator > 0.0f ? numerator / denominator : 0.0f;
    }

    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    int q0 = __float2int_rn(value0 * inverse);
    int q1 = __float2int_rn(value1 * inverse);
    q0 = max(-16, min(15, q0));
    q1 = max(-16, min(15, q1));
    codes[warp][index0] = static_cast<std::int8_t>(q0);
    codes[warp][index1] = static_cast<std::int8_t>(q1);
    low[static_cast<std::int64_t>(unit) * 32 + lane] =
        static_cast<std::uint8_t>((q0 & 15) | ((q1 & 15) << 4));
    if (lane == 0) { output_scales[unit] = __float2half_rn(scale); }
    __syncwarp();
    if (lane < 8) {
        std::uint8_t bits = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            bits |= static_cast<std::uint8_t>((static_cast<unsigned>(codes[warp][lane * 8 + i]) >> 4) & 1u)
                    << i;
        }
        high[static_cast<std::int64_t>(unit) * 8 + lane] = bits;
    }
}

} // namespace
