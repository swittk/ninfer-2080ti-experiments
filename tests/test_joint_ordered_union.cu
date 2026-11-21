#include "core/device.h"
#include "core/arena.h"
#include "../apps/dual/joint_kv/ordered_union.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace ninfer;
namespace op = ninfer::ops::ordered_union;
static_assert(op::workspace_words(1) * sizeof(int) <= 96 * sizeof(unsigned));
static_assert(op::workspace_words(op::kMaximumWords) < 96U * op::kMaximumWords);
unsigned random_word(unsigned& x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }
std::vector<int> read(DeviceBuffer& buffer) {
    std::vector<int> result(buffer.bytes / sizeof(int));
    buffer.copy_to_host(result.data(), buffer.bytes);
    return result;
}
void check(int blocks, cudaStream_t stream) {
    constexpr int Guard = 32;
    const int tiles = op::tiles_for(blocks);
    DeviceBuffer masks(4 * blocks * sizeof(unsigned));
    DeviceBuffer scratch((op::workspace_words(blocks) + Guard) * sizeof(int));
    DeviceBuffer indices((4 * op::kUnionLimit + Guard) * sizeof(int)), counts(4 * sizeof(int)), failed(4 * sizeof(int));
    auto launch = [&] {
        auto* offsets = static_cast<int*>(scratch.p);
        auto* tile_counts = offsets + op::kHeads * blocks;
        op::tile_prefix<<<dim3(tiles,4),256,0,stream>>>(static_cast<unsigned*>(masks.p), blocks, offsets, tile_counts, static_cast<int*>(failed.p));
        op::head_prefix<<<4,32,0,stream>>>(blocks, tile_counts, static_cast<int*>(counts.p), static_cast<int*>(failed.p));
        op::scatter<<<dim3(tiles,4),256,0,stream>>>(static_cast<unsigned*>(masks.p), blocks, offsets, tile_counts, static_cast<int*>(indices.p), static_cast<int*>(failed.p));
        CUDA_CHECK(cudaGetLastError());
    };
    cudaGraph_t graph{}; cudaGraphExec_t execution{};
    CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal)); launch();
    CUDA_CHECK(cudaStreamEndCapture(stream,&graph)); CUDA_CHECK(cudaGraphInstantiate(&execution,graph,nullptr,nullptr,0));
    try {
        unsigned seed = 20260919U;
        for (int pattern = 0; pattern < 7; ++pattern) {
            std::vector<unsigned> words(4 * blocks, 0);
            int original_failed[4] = {0,0,0,pattern == 6 ? 2 : 0};
            std::vector<std::vector<int>> expected(4);
            for (int h = 0; h < 4; ++h) for (int b = 0; b < blocks; ++b) {
                unsigned word = 0;
                if (pattern == 1) word = 0xffffffffU;
                else if (pattern == 2) word = 0x80000001U;
                else if (pattern == 3) word = b < 2048 ? 0xffffffffU : 0U;
                else if (pattern == 4) word = b <= 2048 ? 0xffffffffU : 0U;
                else if (pattern >= 5) for (int bit = 0; bit < 32; ++bit) if (random_word(seed) % 1000 < 45) word |= 1U << bit;
                words[h * blocks + b] = word;
                for (int bit = 0; bit < 32; ++bit) if (word & (1U << bit)) expected[h].push_back(b * 32 + bit);
            }
            masks.copy_from_host(words.data(), words.size() * sizeof(unsigned));
            CUDA_CHECK(cudaDeviceSynchronize());
            for (int replay = 0; replay < 2; ++replay) {
                failed.copy_from_host(original_failed, sizeof(original_failed));
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemsetAsync(indices.p,0x55,indices.bytes,stream));
                CUDA_CHECK(cudaMemsetAsync(scratch.p,0x55,scratch.bytes,stream));
                CUDA_CHECK(cudaGraphLaunch(execution,stream)); CUDA_CHECK(cudaStreamSynchronize(stream));
                const auto out = read(indices), count = read(counts), failure = read(failed), work = read(scratch);
                for (int h = 0; h < 4; ++h) {
                    const bool overflow = expected[h].size() > op::kUnionLimit;
                    const int expected_count = original_failed[h] ? 0 : int(expected[h].size());
                    const int expected_failure = original_failed[h] ? original_failed[h] : int(overflow);
                    if (count[h] != expected_count || failure[h] != expected_failure) throw std::runtime_error("union count or fallback flag differs");
                    const int kept = expected_failure ? 0 : expected_count;
                    for (int i = 0; i < op::kUnionLimit; ++i) {
                        const int value = i < kept ? expected[h][i] : 0x55555555;
                        if (out[h * op::kUnionLimit + i] != value) throw std::runtime_error("ordered union output or untouched guard differs");
                    }
                }
                for (std::size_t i = 4 * op::kUnionLimit; i < out.size(); ++i) if (out[i] != 0x55555555) throw std::runtime_error("union tail overwritten");
                for (std::size_t i = op::workspace_words(blocks); i < work.size(); ++i) if (work[i] != 0x55555555) throw std::runtime_error("scratch capacity exceeded");
            }
        }
    } catch (...) { (void)cudaGraphExecDestroy(execution); (void)cudaGraphDestroy(graph); throw; }
    CUDA_CHECK(cudaGraphExecDestroy(execution)); CUDA_CHECK(cudaGraphDestroy(graph));
    printf("PASS words=%d empty/dense/ties/capacity/failure/source-reuse/ordered-graphs\n",blocks);
}
int main() {
    try {
        int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
        cudaStream_t stream{}; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        try { for(int blocks : {1,31,256,257,2048,8191,8192}) check(blocks,stream); }
        catch (...) { (void)cudaStreamDestroy(stream); throw; }
        CUDA_CHECK(cudaStreamDestroy(stream)); return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"ordered union: %s\n",e.what()); return 1; }
}
