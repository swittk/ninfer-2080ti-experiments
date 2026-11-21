/*
 * The fixed SM75 Marlin kernel body under ops/marlin/upstream is adapted from
 * vLLM/Marlin under Apache-2.0. This file owns NInfer-specific preparation,
 * source-weight-owned prepared storage, BF16 boundaries, and MTP admission.
 */
#include "ops/marlin/mtp_q4_marlin.h"

#include "core/device.h"
#include "ops/marlin/upstream/kernel.h"
#include "ops/marlin/upstream/marlin_template.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace ninfer::ops::detail {
namespace {

constexpr int kMaximumTokens = 8;
constexpr int kThreads = 256;

struct PreparedWeight {
    const void* source_qdata = nullptr;
    const void* source_scales = nullptr;
    cudaStream_t source_stream = nullptr;
    int device = -1;
    int n = 0;
    int k = 0;
    int sms = 0;
    int shared_bytes = 0;
    std::uint32_t* packed = nullptr;
    __half* scales = nullptr;
    __half* input = nullptr;
    __half* output = nullptr;
    float* c_tmp = nullptr;
    int* locks = nullptr;
    ~PreparedWeight() noexcept {
        int previous = -1;
        (void)cudaGetDevice(&previous);
        if (device >= 0 && previous != device) { (void)cudaSetDevice(device); }
        if (source_stream != nullptr) { (void)cudaStreamSynchronize(source_stream); }
        if (locks) { (void)cudaFree(locks); }
        if (c_tmp) { (void)cudaFree(c_tmp); }
        if (output) { (void)cudaFree(output); }
        if (input) { (void)cudaFree(input); }
        if (scales) { (void)cudaFree(scales); }
        if (packed) { (void)cudaFree(packed); }
        if (previous >= 0 && previous != device) { (void)cudaSetDevice(previous); }
    }
};

std::mutex& cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<const void*, std::unique_ptr<PreparedWeight>>& cache() {
    // The registry itself survives static teardown; allocations do not. Owning
    // Programs release their weight entries before CUDA streams/weights die.
    // This also permits loading a new model at a reused device address safely.
    static auto* values =
        new std::unordered_map<const void*, std::unique_ptr<PreparedWeight>>();
    return *values;
}

bool env_enabled() noexcept {
    const char* value = std::getenv("NINFER_MTP_MARLIN_Q4");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void validate_weight(const Weight& weight) {
    if (weight.qtype != QType::Q4G64_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 64 || weight.group != 64 ||
        weight.n <= 0 || weight.k <= 0 || weight.n % 64 != 0 || weight.k % 64 != 0 ||
        weight.padded_shape[0] != weight.n || weight.padded_shape[1] != weight.k ||
        weight.qdata == nullptr || weight.qhigh != nullptr || weight.scales == nullptr) {
        throw std::invalid_argument(
            "MTP Q4 Marlin requires canonical Q4G64_F16S row-split weight");
    }
}

__device__ __forceinline__ std::uint32_t q4_value(const std::uint8_t* codes, int k,
                                                   int n, int size_k) {
    const std::uint8_t packed =
        codes[static_cast<std::int64_t>(n) * (size_k / 2) + k / 2];
    return ((packed >> ((k & 1) * 4)) & 0x0fu) ^ 0x08u;
}

__global__ void q4_rowsplit_repack_kernel(const std::uint8_t* __restrict__ codes,
                                           std::uint32_t* __restrict__ out,
                                           int size_k, int size_n,
                                           std::int64_t words) {
    constexpr int kTileWords = 128;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
         index < words;
         index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const std::int64_t tile = index / kTileWords;
        const int local = static_cast<int>(index % kTileWords);
        const int th = local / 4;
        const int warp = local & 3;
        const int k_base = static_cast<int>(tile / (size_n / 64)) * 16;
        const int n_base = static_cast<int>(tile % (size_n / 64)) * 64;
        const int tc_col = th / 4;
        const int tc_row = (th & 3) * 2;
        const int n0 = n_base + warp * 16 + tc_col;
        constexpr int offsets[4] = {0, 1, 8, 9};
        std::uint32_t values[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int k = k_base + tc_row + offsets[i];
            values[i] = q4_value(codes, k, n0, size_k);
            values[4 + i] = q4_value(codes, k, n0 + 8, size_k);
        }
        constexpr int order[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        std::uint32_t result = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) { result |= values[order[i]] << (i * 4); }
        out[index] = result;
    }
}

__device__ __forceinline__ int scale_permutation(int lane) {
    return (lane & 7) * 8 + (lane >> 3);
}

__global__ void permute_scales_kernel(const __half* __restrict__ source,
                                      __half* __restrict__ destination,
                                      int groups, int n) {
    const std::int64_t count = static_cast<std::int64_t>(groups) * n;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
         index < count;
         index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const int group = static_cast<int>(index / n);
        const int column = static_cast<int>(index % n);
        const int source_column = (column / 64) * 64 + scale_permutation(column & 63);
        destination[index] =
            source[static_cast<std::int64_t>(source_column) * groups + group];
    }
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ source,
                                    __half* __restrict__ destination,
                                    std::int64_t elements) {
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
         index < elements;
         index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        destination[index] = __float2half_rn(__bfloat162float(source[index]));
    }
}

__global__ void fp16_to_bf16_kernel(const __half* __restrict__ source,
                                    __nv_bfloat16* __restrict__ destination,
                                    std::int64_t elements) {
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
         index < elements;
         index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        destination[index] = __float2bfloat16_rn(__half2float(source[index]));
    }
}

using SmallKernel = decltype(&ninfer_marlin::Marlin<
    vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 4, false>);

SmallKernel small_kernel() {
    return &ninfer_marlin::Marlin<vllm::kFloat16.id(), vllm::kU4B8.id(),
                                  vllm::kFloat16.id(), vllm::kFloat16.id(),
                                  256, 1, 8, 8, true, 2, 4, false>;
}

PreparedWeight* lookup(const Weight& weight) noexcept {
    std::lock_guard lock(cache_mutex());
    const auto it = cache().find(weight.qdata);
    if (it == cache().end()) { return nullptr; }
    PreparedWeight* value = it->second.get();
    return value->source_scales == weight.scales && value->n == weight.n && value->k == weight.k
               ? value : nullptr;
}

int launch_blocks(std::int64_t work) {
    return static_cast<int>(
        std::min<std::int64_t>((work + kThreads - 1) / kThreads, 272));
}

} // namespace

bool mtp_q4_marlin_enabled() noexcept { return env_enabled(); }

void mtp_q4_marlin_prepare(const Weight& weight, cudaStream_t stream) {
    if (!env_enabled() || weight.qtype != QType::Q4G64_F16S) { return; }
    validate_weight(weight);
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    // Serialize publication and allocation together: concurrent preparation of
    // the same immutable source must not allocate a losing duplicate entry.
    std::lock_guard lock(cache_mutex());
    if (const auto it = cache().find(weight.qdata); it != cache().end()) {
        const auto& prior = *it->second;
        if (prior.device != device || prior.source_scales != weight.scales ||
            prior.n != weight.n || prior.k != weight.k) {
            throw std::logic_error("MTP Marlin source identity changed before owner release");
        }
        return;
    }
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int shared_bytes = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_bytes,
                                      cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      device));
    if (properties.major != 7 || properties.minor != 5 || shared_bytes <= 0) {
        throw std::invalid_argument("MTP Q4 Marlin is qualified only on SM75");
    }

    auto prepared = std::make_unique<PreparedWeight>();
    prepared->source_qdata = weight.qdata;
    prepared->source_scales = weight.scales;
    prepared->source_stream = stream;
    prepared->device = device;
    prepared->n = weight.n;
    prepared->k = weight.k;
    prepared->sms = properties.multiProcessorCount;
    prepared->shared_bytes = shared_bytes;

    const std::size_t packed_bytes =
        static_cast<std::size_t>(weight.n) * weight.k / 2U;
    const std::size_t scale_bytes = static_cast<std::size_t>(weight.n) *
                                    (weight.k / 64) * sizeof(__half);
    const std::size_t input_bytes = static_cast<std::size_t>(kMaximumTokens) *
                                    weight.k * sizeof(__half);
    const std::size_t output_bytes = static_cast<std::size_t>(kMaximumTokens) *
                                     weight.n * sizeof(__half);
    // Marlin's FP32 global-reduction route always materializes a temporary
    // sms x ceil(M,16) x max_thread_n buffer, even for M<=8.
    constexpr int kMaximumMBlock = 16;
    constexpr int kMaximumThreadN = 256;
    const std::size_t c_tmp_bytes = static_cast<std::size_t>(properties.multiProcessorCount) *
                                    kMaximumMBlock * kMaximumThreadN * sizeof(float);
    const std::size_t lock_bytes =
        static_cast<std::size_t>(weight.n / 64) * 16U * sizeof(int);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&prepared->packed), packed_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&prepared->scales), scale_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&prepared->input), input_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&prepared->output), output_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&prepared->c_tmp), c_tmp_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&prepared->locks), lock_bytes));

    const bool trace = std::getenv("NINFER_MTP_MARLIN_TRACE") != nullptr;
    if (trace) {
        std::fprintf(stderr, "MTP Marlin prepare N=%d K=%d packed=%zu scales=%zu\n",
                     weight.n, weight.k, packed_bytes, scale_bytes);
    }
    const std::int64_t words = packed_bytes / sizeof(std::uint32_t);
    q4_rowsplit_repack_kernel<<<launch_blocks(words), kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata), prepared->packed,
        weight.k, weight.n, words);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (trace) { std::fprintf(stderr, "MTP Marlin repack OK N=%d K=%d\n", weight.n, weight.k); }
    const std::int64_t scale_count =
        static_cast<std::int64_t>(weight.n) * (weight.k / 64);
    permute_scales_kernel<<<launch_blocks(scale_count), kThreads, 0, stream>>>(
        static_cast<const __half*>(weight.scales), prepared->scales,
        weight.k / 64, weight.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (trace) { std::fprintf(stderr, "MTP Marlin scales OK N=%d K=%d\n", weight.n, weight.k); }
    CUDA_CHECK(cudaMemsetAsync(prepared->locks, 0, lock_bytes, stream));
    CUDA_CHECK(cudaFuncSetAttribute(
        small_kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    cache().emplace(weight.qdata, std::move(prepared));
}

void mtp_q4_marlin_release(const Weight& weight) noexcept {
    std::unique_ptr<PreparedWeight> owned;
    {
        std::lock_guard lock(cache_mutex());
        const auto it = cache().find(weight.qdata);
        if (it == cache().end()) { return; }
        owned = std::move(it->second);
        cache().erase(it);
    }
    // Free on the entry's CUDA device, outside the registry lock.
}

bool mtp_q4_marlin_ready(const Weight& weight) noexcept {
    return env_enabled() && lookup(weight) != nullptr;
}

static bool launch_prepared(const Tensor& input, const Weight& weight,
                            Tensor& output, cudaStream_t stream, PreparedWeight* prepared) {
    if (prepared == nullptr) { return false; }
    const int tokens = input.ne[1];
    if (tokens <= 0 || tokens > kMaximumTokens || input.dtype != DType::BF16 ||
        output.dtype != DType::BF16 || input.ne[0] != weight.k ||
        output.ne[0] != weight.n || output.ne[1] != tokens ||
        !input.is_contiguous() || !output.is_contiguous() ||
        input.data == nullptr || output.data == nullptr) {
        return false;
    }

    const bool trace = std::getenv("NINFER_MTP_MARLIN_TRACE") != nullptr;
    if (trace) {
        std::fprintf(stderr, "MTP Marlin launch begin N=%d K=%d T=%d\n",
                     weight.n, weight.k, tokens);
    }
    const std::int64_t input_elements =
        static_cast<std::int64_t>(tokens) * weight.k;
    const std::int64_t output_elements =
        static_cast<std::int64_t>(tokens) * weight.n;
    bf16_to_fp16_kernel<<<launch_blocks(input_elements), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), prepared->input,
        input_elements);
    CUDA_CHECK(cudaGetLastError());
    if (trace) { std::fprintf(stderr, "MTP Marlin input cast submitted N=%d K=%d T=%d\n", weight.n, weight.k, tokens); }
    // Every successful slice releases its barrier with `last=true`, which
    // resets the lock to zero before this ordered stream can launch the next
    // projection. Preparation zeroes the array for the first launch, so a
    // per-launch memset is redundant and only adds a graph node.
    if (trace) { std::fprintf(stderr, "MTP Marlin kernel submit N=%d K=%d T=%d shared=%d\n", weight.n, weight.k, tokens, prepared->shared_bytes); }
    small_kernel()<<<prepared->sms, 256, prepared->shared_bytes, stream>>>(
        reinterpret_cast<const int4*>(prepared->input),
        reinterpret_cast<const int4*>(prepared->packed),
        reinterpret_cast<int4*>(prepared->output),
        reinterpret_cast<int4*>(prepared->c_tmp), nullptr, nullptr,
        reinterpret_cast<const int4*>(prepared->scales), nullptr, nullptr,
        nullptr, weight.k / 64, tokens, weight.n, weight.k, weight.k,
        prepared->locks, false, false, true, prepared->shared_bytes);
    CUDA_CHECK(cudaGetLastError());
    if (trace) { std::fprintf(stderr, "MTP Marlin kernel submitted N=%d K=%d T=%d\n", weight.n, weight.k, tokens); }

    fp16_to_bf16_kernel<<<launch_blocks(output_elements), kThreads, 0, stream>>>(
        prepared->output, static_cast<__nv_bfloat16*>(output.data),
        output_elements);
    CUDA_CHECK(cudaGetLastError());
    if (trace) { std::fprintf(stderr, "MTP Marlin launch end N=%d K=%d T=%d\n", weight.n, weight.k, tokens); }
    return true;
}

bool mtp_q4_marlin_launch(const Tensor& input, const Weight& weight,
                          Tensor& output, cudaStream_t stream) {
    return launch_prepared(input, weight, output, stream, lookup(weight));
}

void mtp_q4_marlin_retained_prefix_launch(const Tensor& input, const Weight& owner,
                                         const Weight& view, Tensor& output, cudaStream_t stream) {
    // Historical C1 interpreted the packed parent's base with a narrower N.
    // Preserve that accepted draft-only numeric policy, not a logical row slice.
    PreparedWeight* prepared = lookup(owner);
    if (!prepared || owner.qtype != QType::Q4G64_F16S ||
        view.qtype != owner.qtype || view.layout != owner.layout ||
        view.qdata != owner.qdata || view.scales != owner.scales ||
        view.k != owner.k || view.n <= 0 || view.n > owner.n || view.n % 64 != 0 ||
        input.ne[1] != 1) {
        throw std::invalid_argument("retained packed-prefix draft requires a T1 view of its exact prepared owner");
    }
    if (!launch_prepared(input, view, output, stream, prepared)) {
        throw std::invalid_argument("retained packed-prefix draft projection shape rejected");
    }
}

} // namespace ninfer::ops::detail
