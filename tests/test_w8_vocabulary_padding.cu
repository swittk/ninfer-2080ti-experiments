#include "core/device.h"
#include "core/arena.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace ninfer;
namespace detail = ninfer::ops::detail;
using Stage = detail::W8SmallTMmaActivationStage;
template<int T> using Production = typename detail::W8LinearSmallTProductionSchedule<detail::W8VocabularyProjectionGeometry,T>::Type;
#if defined(NINFER_CUDA_SM75)
static_assert(Production<4>::kActivationStage == Stage::ActiveOnly);
static_assert(Production<5>::kActivationStage == Stage::PaddedZero);
static_assert(Production<6>::kActivationStage == Stage::PaddedZero);
static_assert(Production<7>::kActivationStage == Stage::PaddedZero);
static_assert(Production<8>::kActivationStage == Stage::ActiveOnly);
static_assert(Production<9>::kActivationStage == Stage::ActiveOnly);
#else
static_assert(Production<5>::kActivationStage == Stage::ActiveOnly);
#endif
namespace {
constexpr int N = 512, K = 5120, Stride = N + 16;
__device__ unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}
__global__ void initialize(std::uint8_t* codes, std::uint16_t* scales,
                           __nv_bfloat16* input, int tokens, unsigned seed) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto step = gridDim.x * blockDim.x;
    for (int i = index; i < N * K; i += step) codes[i] = static_cast<std::uint8_t>(mix(i ^ seed));
    for (int i = index; i < N * (K / 32); i += step)
        scales[i] = static_cast<std::uint16_t>(0x2000U + mix(i + seed) % 0xc00U);
    for (int i = index; i < K * tokens; i += step) {
        float x = static_cast<float>(static_cast<int>(mix(i ^ seed ^ 43U) % 2049) - 1024) / 1024.0F;
        input[i] = __float2bfloat16_rn(seed == 0 ? 0.0F : x);
    }
}
template<int T>
void check_width(cudaStream_t stream) {
    using G = detail::W8LinearGeometry<N, K>;
    using Old = detail::W8SmallTMmaSchedule<8,8,2,
        (T > 4 ? detail::W8SmallTMmaScaleAccess::Shared : detail::W8SmallTMmaScaleAccess::Direct)>;
    DeviceBuffer codes(N*K), scales(N*(K/32)*2), input(K*T*2), reference((Stride*T+16)*2), output(reference.bytes);
    auto old = [&] {
        const detail::W8ContiguousOutput out{static_cast<__nv_bfloat16*>(reference.p), Stride};
        detail::w8_small_t_mma_kernel<G,T,Old><<<N/16,256,0,stream>>>(
            static_cast<const __nv_bfloat16*>(input.p), static_cast<const std::uint8_t*>(codes.p),
            static_cast<const std::uint8_t*>(scales.p), out);
        CUDA_CHECK(cudaGetLastError());
    };
    auto current = [&] {
        const detail::W8ContiguousOutput out{static_cast<__nv_bfloat16*>(output.p), Stride};
        detail::w8_small_t_mma_kernel<G,T,Production<T>><<<N/16,256,0,stream>>>(
            static_cast<const __nv_bfloat16*>(input.p), static_cast<const std::uint8_t*>(codes.p),
            static_cast<const std::uint8_t*>(scales.p), out);
        CUDA_CHECK(cudaGetLastError());
    };
    cudaGraph_t graph{}; cudaGraphExec_t execution{};
    CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    current();
    CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
    CUDA_CHECK(cudaGraphInstantiate(&execution,graph,nullptr,nullptr,0));
    try {
        for (unsigned seed : {20260918U, 912871U, 0U}) {
            // Change the same source allocations between graph replays. No
            // side cache or stale prepared input may hide that new content.
            initialize<<<128,256,0,stream>>>(static_cast<std::uint8_t*>(codes.p),
                static_cast<std::uint16_t*>(scales.p), static_cast<__nv_bfloat16*>(input.p), T, seed);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemsetAsync(reference.p,0x55,reference.bytes,stream));
            CUDA_CHECK(cudaMemsetAsync(output.p,0x55,output.bytes,stream));
            old();
            CUDA_CHECK(cudaGraphLaunch(execution,stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<std::uint16_t> a(reference.bytes/2), b(a.size());
            reference.copy_to_host(a.data(),reference.bytes); output.copy_to_host(b.data(),output.bytes);
            if (a != b) throw std::runtime_error("padded vocabulary output differs from original");
            for (int t = 0; t < T; ++t) {
                for (int r = 0; r < N; ++r)
                    if ((b[t*Stride+r] & 0x7f80U) == 0x7f80U) throw std::runtime_error("nonfinite output");
                for (int r = N; r < Stride; ++r)
                    if (b[t*Stride+r] != 0x5555U) throw std::runtime_error("output stride guard overwritten");
            }
            for (std::size_t i = Stride*T; i < b.size(); ++i)
                if (b[i] != 0x5555U) throw std::runtime_error("output tail overwritten");
            CUDA_CHECK(cudaGraphLaunch(execution,stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            output.copy_to_host(b.data(),output.bytes);
            if (b != a) throw std::runtime_error("repeat output differs");
        }
    } catch (...) {
        (void)cudaGraphExecDestroy(execution); (void)cudaGraphDestroy(graph); throw;
    }
    CUDA_CHECK(cudaGraphExecDestroy(execution)); CUDA_CHECK(cudaGraphDestroy(graph));
    std::printf("PASS W8 T=%d source-reuse, zero inputs, strided guards and graph-repeat parity\n",T);
}
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
        if (prop.major != 7 || prop.minor != 5) return 77;
        cudaStream_t stream{}; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        try { check_width<4>(stream); check_width<5>(stream); check_width<6>(stream); check_width<7>(stream); check_width<8>(stream); }
        catch (...) { (void)cudaStreamDestroy(stream); throw; }
        CUDA_CHECK(cudaStreamDestroy(stream)); return 0;
    } catch (const std::exception& error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
}
