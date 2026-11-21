#include "ninfer/ops/gqa_tree.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQueryHeads = 24;
constexpr std::int32_t kKvHeads = 4;
constexpr std::int32_t kMaximumNodes = 32;

void require(const Tensor& tensor, DType dtype, const char* label) {
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("GQA tree: invalid ") + label);
    }
}

std::int32_t validate_tail(const Tensor& q, const Tensor& k, const Tensor& v,
                           const Tensor& parent_ids, const Tensor& state_acc,
                           const Tensor& state_m, const Tensor& state_l) {
    require(q, DType::BF16, "q");
    require(k, DType::BF16, "k");
    require(v, DType::BF16, "v");
    require(parent_ids, DType::I32, "parent_ids");
    require(state_acc, DType::FP32, "state_acc");
    require(state_m, DType::FP32, "state_m");
    require(state_l, DType::FP32, "state_l");
    const std::int32_t nodes = q.ne[2];
    if (nodes <= 0 || nodes > kMaximumNodes || q.ne[0] != kHeadDim ||
        q.ne[1] != kQueryHeads || q.ne[3] != 1 || k.ne[0] != kHeadDim ||
        k.ne[1] != kKvHeads || k.ne[2] != nodes || k.ne[3] != 1 ||
        v.ne[0] != kHeadDim || v.ne[1] != kKvHeads || v.ne[2] != nodes ||
        v.ne[3] != 1 || parent_ids.ne[0] != nodes || parent_ids.ne[1] != 1 ||
        parent_ids.ne[2] != 1 || parent_ids.ne[3] != 1 ||
        state_acc.ne[0] != kHeadDim || state_acc.ne[1] != kQueryHeads ||
        state_acc.ne[2] != nodes || state_acc.ne[3] != 1 ||
        state_m.ne[0] != kQueryHeads || state_m.ne[1] != nodes ||
        state_m.ne[2] != 1 || state_m.ne[3] != 1 ||
        state_l.ne[0] != kQueryHeads || state_l.ne[1] != nodes ||
        state_l.ne[2] != 1 || state_l.ne[3] != 1) {
        throw std::invalid_argument("GQA tree: tail-state shape mismatch");
    }
    return nodes;
}

__global__ void tree_tail_state_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ parent_ids,
    float scale,
    float* __restrict__ state_acc,
    float* __restrict__ state_m,
    float* __restrict__ state_l,
    std::int32_t nodes) {
    const std::int32_t query_head = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t node = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t dim = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t kv_head = query_head / (kQueryHeads / kKvHeads);

    __shared__ std::int32_t ancestry[kMaximumNodes];
    __shared__ std::int32_t ancestry_length;
    __shared__ float reduction[kHeadDim];
    __shared__ float factors[4]; // alpha, probability, m, l

    if (dim == 0) {
        std::int32_t length = 0;
        std::int32_t current = node;
        while (current >= 0 && length < kMaximumNodes) {
            ancestry[length++] = current;
            const std::int32_t parent = parent_ids[current];
            if (parent >= current) {
                length = 0;
                break;
            }
            current = parent;
        }
        ancestry_length = length;
    }
    __syncthreads();

    const std::int64_t query_base =
        (static_cast<std::int64_t>(node) * kQueryHeads + query_head) * kHeadDim;
    const float query = __bfloat162float(q[query_base + dim]);
    float accumulator = 0.0F;
    float maximum = -CUDART_INF_F;
    float denominator = 0.0F;

    for (std::int32_t reverse = ancestry_length - 1; reverse >= 0; --reverse) {
        const std::int32_t source = ancestry[reverse];
        const std::int64_t key_base =
            (static_cast<std::int64_t>(source) * kKvHeads + kv_head) * kHeadDim;
        reduction[dim] = query * __bfloat162float(k[key_base + dim]);
        __syncthreads();
        for (std::int32_t stride = kHeadDim / 2; stride > 0; stride >>= 1) {
            if (dim < stride) { reduction[dim] += reduction[dim + stride]; }
            __syncthreads();
        }
        if (dim == 0) {
            const float score = reduction[0] * scale;
            const float next_maximum = fmaxf(maximum, score);
            const float alpha = isinf(maximum)
                ? 0.0F
                : expf(maximum - next_maximum);
            const float probability = expf(score - next_maximum);
            const float next_denominator = denominator * alpha + probability;
            factors[0] = alpha;
            factors[1] = probability;
            factors[2] = next_maximum;
            factors[3] = next_denominator;
        }
        __syncthreads();
        const std::int64_t value_base =
            (static_cast<std::int64_t>(source) * kKvHeads + kv_head) * kHeadDim;
        accumulator = fmaf(factors[1], __bfloat162float(v[value_base + dim]),
                           accumulator * factors[0]);
        maximum = factors[2];
        denominator = factors[3];
        __syncthreads();
    }

    const std::int64_t state_base =
        (static_cast<std::int64_t>(node) * kQueryHeads + query_head) * kHeadDim;
    state_acc[state_base + dim] = accumulator;
    if (dim == 0) {
        const std::int64_t scalar =
            static_cast<std::int64_t>(node) * kQueryHeads + query_head;
        state_m[scalar] = maximum;
        state_l[scalar] = denominator;
    }
}

__global__ void merge_prefix_tree_kernel(
    const float* __restrict__ prefix_acc,
    const float* __restrict__ prefix_m,
    const float* __restrict__ prefix_l,
    const float* __restrict__ tail_acc,
    const float* __restrict__ tail_m,
    const float* __restrict__ tail_l,
    __nv_bfloat16* __restrict__ out,
    std::int32_t nodes) {
    const std::int32_t query_head = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t node = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t dim = static_cast<std::int32_t>(threadIdx.x);
    const std::int64_t scalar =
        static_cast<std::int64_t>(node) * kQueryHeads + query_head;
    const float m0 = prefix_m[scalar];
    const float m1 = tail_m[scalar];
    const float l0 = prefix_l[scalar];
    const float l1 = tail_l[scalar];
    const float maximum = fmaxf(m0, m1);
    const float alpha0 = l0 > 0.0F ? expf(m0 - maximum) : 0.0F;
    const float alpha1 = l1 > 0.0F ? expf(m1 - maximum) : 0.0F;
    const float denominator = l0 * alpha0 + l1 * alpha1;
    const std::int64_t base = scalar * kHeadDim;
    const float numerator = prefix_acc[base + dim] * alpha0 + tail_acc[base + dim] * alpha1;
    out[base + dim] = __float2bfloat16_rn(numerator / denominator);
}

std::int32_t validate_merge(const Tensor& prefix_acc, const Tensor& prefix_m,
                            const Tensor& prefix_l, const Tensor& tail_acc,
                            const Tensor& tail_m, const Tensor& tail_l,
                            const Tensor& out) {
    require(prefix_acc, DType::FP32, "prefix_acc");
    require(prefix_m, DType::FP32, "prefix_m");
    require(prefix_l, DType::FP32, "prefix_l");
    require(tail_acc, DType::FP32, "tail_acc");
    require(tail_m, DType::FP32, "tail_m");
    require(tail_l, DType::FP32, "tail_l");
    require(out, DType::BF16, "out");
    const std::int32_t nodes = prefix_acc.ne[2];
    const auto valid_acc = [&](const Tensor& tensor) {
        return tensor.ne[0] == kHeadDim && tensor.ne[1] == kQueryHeads &&
               tensor.ne[2] == nodes && tensor.ne[3] == 1;
    };
    const auto valid_scalar = [&](const Tensor& tensor) {
        return tensor.ne[0] == kQueryHeads && tensor.ne[1] == nodes &&
               tensor.ne[2] == 1 && tensor.ne[3] == 1;
    };
    if (nodes <= 0 || nodes > kMaximumNodes || !valid_acc(prefix_acc) ||
        !valid_acc(tail_acc) || !valid_acc(out) || !valid_scalar(prefix_m) ||
        !valid_scalar(prefix_l) || !valid_scalar(tail_m) || !valid_scalar(tail_l)) {
        throw std::invalid_argument("GQA tree: prefix merge shape mismatch");
    }
    return nodes;
}

} // namespace

void gqa_attention_tree_tail_state(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& parent_ids, float scale, Tensor& state_acc,
                                   Tensor& state_m, Tensor& state_l, cudaStream_t stream) {
    if (!(scale > 0.0F)) { throw std::invalid_argument("GQA tree: invalid scale"); }
    const std::int32_t nodes =
        validate_tail(q, k, v, parent_ids, state_acc, state_m, state_l);
    const dim3 grid(kQueryHeads, static_cast<unsigned>(nodes), 1U);
    tree_tail_state_kernel<<<grid, kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(parent_ids.data), scale,
        static_cast<float*>(state_acc.data), static_cast<float*>(state_m.data),
        static_cast<float*>(state_l.data), nodes);
    CUDA_CHECK(cudaGetLastError());
}

void gqa_attention_tree_merge_prefix(const Tensor& prefix_acc, const Tensor& prefix_m,
                                     const Tensor& prefix_l, const Tensor& tail_acc,
                                     const Tensor& tail_m, const Tensor& tail_l, Tensor& out,
                                     cudaStream_t stream) {
    const std::int32_t nodes = validate_merge(prefix_acc, prefix_m, prefix_l, tail_acc,
                                              tail_m, tail_l, out);
    const dim3 grid(kQueryHeads, static_cast<unsigned>(nodes), 1U);
    merge_prefix_tree_kernel<<<grid, kHeadDim, 0, stream>>>(
        static_cast<const float*>(prefix_acc.data),
        static_cast<const float*>(prefix_m.data),
        static_cast<const float*>(prefix_l.data),
        static_cast<const float*>(tail_acc.data),
        static_cast<const float*>(tail_m.data),
        static_cast<const float*>(tail_l.data),
        static_cast<__nv_bfloat16*>(out.data), nodes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
