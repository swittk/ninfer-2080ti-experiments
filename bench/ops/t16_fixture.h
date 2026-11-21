#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::bench {
// Deterministic, nonuniform data: distinguish every row, group and token column.
inline std::uint32_t t16_next(std::uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
inline void t16_fill_input(void* device, std::size_t elements, std::uint32_t seed) {
    std::vector<std::uint16_t> values(elements);
    for (auto& v : values) {
        const auto r = t16_next(seed);
        v = static_cast<std::uint16_t>((r & 0x8000u) | 0x3e00u | ((r >> 16) & 0x1ffu));
    }
    CUDA_CHECK(cudaMemcpy(device, values.data(), values.size()*sizeof(values[0]), cudaMemcpyHostToDevice));
}
inline void t16_fill_weight(const Weight& weight, std::size_t rows, std::size_t columns,
                            bool q5, std::uint32_t seed) {
    std::vector<std::uint8_t> codes(rows*columns/2);
    for (auto& v : codes) v = static_cast<std::uint8_t>(t16_next(seed));
    CUDA_CHECK(cudaMemcpy(const_cast<void*>(weight.qdata), codes.data(), codes.size(), cudaMemcpyHostToDevice));
    if (q5) {
        codes.resize(rows*columns/8);
        for (auto& v : codes) v = static_cast<std::uint8_t>(t16_next(seed));
        CUDA_CHECK(cudaMemcpy(const_cast<void*>(weight.qhigh), codes.data(), codes.size(), cudaMemcpyHostToDevice));
    }
    std::vector<std::uint16_t> scales(rows*columns/64);
    for (auto& v : scales) v = static_cast<std::uint16_t>(0x3000u + t16_next(seed)%0xc00u);
    CUDA_CHECK(cudaMemcpy(const_cast<void*>(weight.scales), scales.data(), scales.size()*sizeof(scales[0]), cudaMemcpyHostToDevice));
}
} // namespace ninfer::bench
