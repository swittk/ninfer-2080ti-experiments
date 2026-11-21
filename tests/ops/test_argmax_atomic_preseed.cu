#include "ops/kernel/argmax.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

constexpr int kPhysicalRows = 4096;
constexpr int kValidRows    = 4093;
constexpr int kTokens       = 4;

__global__ void fill_logits(__nv_bfloat16* logits) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= kPhysicalRows * kTokens) { return; }
    const int token = i / kPhysicalRows;
    const int row   = i - token * kPhysicalRows;
    float value     = -8.0f - static_cast<float>((row * 17 + token * 31) & 255) / 256.0f;
    if (token == 0 && (row == 0 || row == 777)) { value = 32.0f; }
    if (token == 0 && row == kValidRows - 2) { value = CUDART_NAN_F; }
    if (token == 1 && row == kValidRows - 1) { value = 33.0f; }
    if (token == 2 && row == 123) { value = 34.0f; }
    if (token == 3 && row == 2048) { value = 35.0f; }
    if (row >= kValidRows) { value = 96.0f; } // Padding must never participate.
    logits[i] = __float2bfloat16(value);
}

__global__ void seed_stale_winners(std::int32_t* out) {
    if (threadIdx.x == 0) {
        out[0] = kValidRows - 2; // Valid winner from an unrelated previous replay.
        out[1] = INT_MIN;       // Arbitrary invalid/uninitialized state.
        out[2] = kValidRows;    // Physical padding, outside the logical token domain.
        out[3] = 321;           // Valid finite winner from an unrelated previous replay.
    }
}

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) { return true; }
    std::cerr << "argmax preseed: " << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    __nv_bfloat16* logits = nullptr;
    std::int32_t* out     = nullptr;
    cudaStream_t stream   = nullptr;
    cudaGraph_t graph     = nullptr;
    cudaGraphExec_t exec  = nullptr;
    int failures          = 0;
    constexpr std::int32_t expected[kTokens] = {0, kValidRows - 1, 123, 2048};

    if (!cuda_ok(cudaMalloc(&logits, static_cast<std::size_t>(kPhysicalRows) * kTokens *
                                        sizeof(__nv_bfloat16)),
                 "allocate logits") ||
        !cuda_ok(cudaMalloc(&out, kTokens * sizeof(std::int32_t)), "allocate output") ||
        !cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream")) {
        failures = 1;
        goto cleanup;
    }

    fill_logits<<<(kPhysicalRows * kTokens + 255) / 256, 256, 0, stream>>>(logits);
    if (!cuda_ok(cudaGetLastError(), "launch fill") ||
        !cuda_ok(cudaStreamSynchronize(stream), "synchronize fill")) {
        failures = 1;
        goto cleanup;
    }

    if (!cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture")) {
        failures = 1;
        goto cleanup;
    }
    seed_stale_winners<<<1, 1, 0, stream>>>(out);
    {
        constexpr int kTiles =
            (kValidRows + ninfer::ops::kArgmaxBlock - 1) / ninfer::ops::kArgmaxBlock;
        const dim3 grid(kTiles, kTokens);
        ninfer::ops::argmax_tiled_atomic_kernel<<<grid, ninfer::ops::kArgmaxBlock, 0, stream>>>(
            logits, out, kValidRows, kPhysicalRows);
    }
    if (!cuda_ok(cudaGetLastError(), "capture launches") ||
        !cuda_ok(cudaStreamEndCapture(stream, &graph), "end capture") ||
        !cuda_ok(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate graph")) {
        failures = 1;
        goto cleanup;
    }

    for (int replay = 0; replay < 64; ++replay) {
        if (!cuda_ok(cudaGraphLaunch(exec, stream), "launch graph") ||
            !cuda_ok(cudaStreamSynchronize(stream), "synchronize graph")) {
            failures = 1;
            break;
        }
        std::int32_t actual[kTokens]{};
        if (!cuda_ok(cudaMemcpy(actual, out, sizeof(actual), cudaMemcpyDeviceToHost),
                     "copy result")) {
            failures = 1;
            break;
        }
        for (int token = 0; token < kTokens; ++token) {
            if (actual[token] != expected[token]) {
                std::cerr << "argmax preseed replay " << replay << " token " << token
                          << ": expected " << expected[token] << ", got " << actual[token]
                          << '\n';
                ++failures;
            }
        }
        if (failures != 0) { break; }
    }

cleanup:
    if (exec != nullptr) { cudaGraphExecDestroy(exec); }
    if (graph != nullptr) { cudaGraphDestroy(graph); }
    if (stream != nullptr) { cudaStreamDestroy(stream); }
    if (out != nullptr) { cudaFree(out); }
    if (logits != nullptr) { cudaFree(logits); }
    std::cout << (failures ? "FAIL" : "OK") << " argmax atomic preseed\n";
    return failures;
}
