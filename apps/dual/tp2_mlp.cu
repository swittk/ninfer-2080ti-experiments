#include "dual/tp2_packet_layout.h"
#include "dual/tp2_mlp.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
std::size_t tp2_fast_q4_code_bytes();
std::size_t tp2_fast_q4_scale_bytes();
std::size_t tp2_fast_q4_scratch_bytes();
int tp2_fast_q4_rows();
int tp2_fast_q4_output_groups();
void tp2_fast_q4_transform(std::uint8_t*, std::uint8_t*, void*, cudaStream_t);
void tp2_fast_q4_run(const std::uint8_t*, const std::uint8_t*, const std::int8_t*,
                     const __half*, std::int8_t*, __half*, cudaStream_t);
std::size_t tp2_fast_q5_scratch_bytes();
int tp2_fast_q5_rows();
int tp2_fast_q5_k();
void tp2_fast_q5_transform(std::uint8_t*, std::uint8_t*, std::uint8_t*, void*,
                           cudaStream_t);
void tp2_fast_q5_run(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                     const std::int8_t*, const __half*, __nv_bfloat16*, cudaStream_t);
void tp2_fast_q5_run_canonical(
    const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
    const std::int8_t*, const __half*, __nv_bfloat16*, cudaStream_t);
void tp2_t5_fast_q4_run(const std::uint8_t*, const std::uint8_t*, const std::int8_t*,
                        const __half*, std::int8_t*, __half*, cudaStream_t);
void tp2_t5_fast_q5_run(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                        const std::int8_t*, const __half*, __nv_bfloat16*, cudaStream_t);
void tp2_fast_q5_run_canonical_t5(
    const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
    const std::int8_t*, const __half*, __nv_bfloat16*, cudaStream_t);

std::size_t tp2_slow_q4_code_bytes();
std::size_t tp2_slow_q4_scale_bytes();
std::size_t tp2_slow_q4_scratch_bytes();
int tp2_slow_q4_rows();
int tp2_slow_q4_output_groups();
void tp2_slow_q4_transform(std::uint8_t*, std::uint8_t*, void*, cudaStream_t);
void tp2_slow_q4_run(const std::uint8_t*, const std::uint8_t*, const std::int8_t*,
                     const __half*, std::int8_t*, __half*, cudaStream_t);
std::size_t tp2_slow_q5_scratch_bytes();
int tp2_slow_q5_rows();
int tp2_slow_q5_k();
void tp2_slow_q5_transform(std::uint8_t*, std::uint8_t*, std::uint8_t*, void*,
                           cudaStream_t);
void tp2_slow_q5_run(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                     const std::int8_t*, const __half*, __nv_bfloat16*, cudaStream_t);
void tp2_slow_q5_run_canonical(
    const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
    const std::int8_t*, const __half*, __nv_bfloat16*, cudaStream_t);
void tp2_t5_slow_q4_run(const std::uint8_t*, const std::uint8_t*, const std::int8_t*,
                        const __half*, std::int8_t*, __half*, cudaStream_t);
void tp2_t5_slow_q5_run(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                        const std::int8_t*, const __half*, __nv_bfloat16*, cudaStream_t);
void tp2_slow_q5_run_canonical_t5(
    const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
    const std::int8_t*, const __half*, __nv_bfloat16*, cudaStream_t);
}

namespace ninfer::apps::dual {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMaximumTokens = detail::kTp2MaximumTokens;
constexpr int kHidden = detail::kTp2Hidden;
constexpr int kInputGroups = kHidden / 64;
constexpr int kLayers = 64;
constexpr auto kMaximumInputLayout=detail::kTp2MaximumInputPacket;
constexpr std::size_t kInputPacketBytes=kMaximumInputLayout.packet_bytes;
constexpr std::size_t kMaximumPartialElements=static_cast<std::size_t>(kMaximumTokens)*kHidden;
constexpr std::size_t kMaximumPartialBytes=kMaximumPartialElements*sizeof(__nv_bfloat16);
static_assert(sizeof(__half)==detail::kTp2ScaleBytes);
std::size_t partial_elements(int tokens) {
    return static_cast<std::size_t>(tokens)*kHidden;
}
std::size_t partial_bf16_bytes(int tokens) {
    return partial_elements(tokens)*sizeof(__nv_bfloat16);
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

#pragma pack(push, 1)
struct WeightHeader {
    char magic[8];
    std::uint32_t n;
    std::uint32_t k;
    std::uint32_t group;
    std::uint64_t low_bytes;
    std::uint64_t high_bytes;
    std::uint64_t scale_bytes;
};
#pragma pack(pop)

struct HostWeight {
    WeightHeader header{};
    std::vector<std::uint8_t> low;
    std::vector<std::uint8_t> high;
    std::vector<std::uint8_t> scales;
};

HostWeight load_weight(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open TP2 weight: " + path.string());
    }
    HostWeight weight;
    input.read(reinterpret_cast<char*>(&weight.header), sizeof(weight.header));
    if (!input || weight.header.group != 64) {
        throw std::runtime_error("invalid TP2 weight header: " + path.string());
    }
    weight.low.resize(static_cast<std::size_t>(weight.header.low_bytes));
    weight.high.resize(static_cast<std::size_t>(weight.header.high_bytes));
    weight.scales.resize(static_cast<std::size_t>(weight.header.scale_bytes));
    input.read(reinterpret_cast<char*>(weight.low.data()),
               static_cast<std::streamsize>(weight.low.size()));
    input.read(reinterpret_cast<char*>(weight.high.data()),
               static_cast<std::streamsize>(weight.high.size()));
    input.read(reinterpret_cast<char*>(weight.scales.data()),
               static_cast<std::streamsize>(weight.scales.size()));
    if (!input) {
        throw std::runtime_error("truncated TP2 weight: " + path.string());
    }
    return weight;
}

template <class T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t count) : count_(count) {
        check(cudaHostAlloc(reinterpret_cast<void**>(&data_), count * sizeof(T),
                            cudaHostAllocPortable),
              "cudaHostAlloc TP2");
    }
    ~PinnedBuffer() {
        if (data_ != nullptr) {
            (void)cudaFreeHost(data_);
        }
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <int Tokens>
__global__ void quantize_a8(const __nv_bfloat16* input, std::int8_t* codes,
                            __half* scales) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int unit = static_cast<int>(blockIdx.x) * 8 + warp;
    if (unit >= Tokens * kInputGroups) {
        return;
    }
    const int token = unit / kInputGroups;
    const int group = unit - token * kInputGroups;
    const int d0 = group * 64 + lane;
    const int d1 = d0 + 32;
    const float x0 =
        __bfloat162float(input[static_cast<std::int64_t>(token) * kHidden + d0]);
    const float x1 =
        __bfloat162float(input[static_cast<std::int64_t>(token) * kHidden + d1]);
    float maximum = fmaxf(fabsf(x0), fabsf(x1));
#pragma unroll
    for (int offset = 16; offset != 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, offset));
    }
    const float scale = maximum != 0.0F ? maximum / 127.0F : 0.0F;
    const float reciprocal = scale != 0.0F ? 1.0F / scale : 0.0F;
    int q0 = __float2int_rn(x0 * reciprocal);
    int q1 = __float2int_rn(x1 * reciprocal);
    q0 = max(-127, min(127, q0));
    q1 = max(-127, min(127, q1));
    codes[static_cast<std::int64_t>(token) * kHidden + d0] =
        static_cast<std::int8_t>(q0);
    codes[static_cast<std::int64_t>(token) * kHidden + d1] =
        static_cast<std::int8_t>(q1);
    if (lane == 0) {
        scales[token * kInputGroups + group] = __float2half_rn(scale);
    }
}

__global__ void add_two_bf16_residual(const __nv_bfloat16* local_update,
                                      const __nv_bfloat16* remote_update,
                                      __nv_bfloat16* residual, int count) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float combined = __bfloat162float(local_update[index]) +
                               __bfloat162float(remote_update[index]);
        residual[index] = __float2bfloat16_rn(
            __bfloat162float(residual[index]) + combined);
    }
}

enum class EngineKind { Fast, Slow };

struct LayerWeight {
    std::uint8_t* q4_codes = nullptr;
    std::uint8_t* q4_scales = nullptr;
    std::uint8_t* q5_low = nullptr;
    std::uint8_t* q5_high = nullptr;
    std::uint8_t* q5_scales = nullptr;
};

struct Engine {
    int device = -1;
    EngineKind kind = EngineKind::Fast;
    int intermediate = 0;
    int intermediate_groups = 0;
    int canonical_begin = 0;
    int canonical_end = 0;
    std::size_t canonical_alias_bytes = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t quant_ready = nullptr;
    cudaEvent_t input_ready = nullptr;
    std::vector<LayerWeight> layers;
    std::uint8_t* input_packet = nullptr;
    std::int8_t* input_codes = nullptr;
    __half* input_scales = nullptr;
    std::int8_t* intermediate_codes = nullptr;
    __half* intermediate_scales = nullptr;
    __nv_bfloat16* output_bf16 = nullptr;
    __nv_bfloat16* owner_sum = nullptr;
    std::size_t device_bytes = 0;

    ~Engine() { release(); }
    Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    const char* tag() const noexcept {
        return kind == EngineKind::Fast ? "fast" : "slow";
    }

    std::size_t q4_code_bytes() const {
        return kind == EngineKind::Fast ? tp2_fast_q4_code_bytes()
                                        : tp2_slow_q4_code_bytes();
    }
    std::size_t q4_scale_bytes() const {
        return kind == EngineKind::Fast ? tp2_fast_q4_scale_bytes()
                                        : tp2_slow_q4_scale_bytes();
    }
    bool uses_canonical(int layer) const noexcept {
        return layer >= canonical_begin && layer < canonical_end;
    }
    std::size_t q5_low_bytes() const noexcept {
        return static_cast<std::size_t>(kHidden) * intermediate / 2U;
    }
    std::size_t q5_high_bytes() const noexcept {
        return static_cast<std::size_t>(kHidden) * intermediate / 8U;
    }
    std::size_t q5_scale_bytes() const noexcept {
        return static_cast<std::size_t>(kHidden) * intermediate_groups * sizeof(__half);
    }
    std::size_t shard_weight_bytes() const {
        return q4_code_bytes() + q4_scale_bytes() + q5_low_bytes() +
               q5_high_bytes() + q5_scale_bytes();
    }
    bool canonical_weights_valid(const Weight& gate_up, const Weight& down) const noexcept {
        return gate_up.qtype == QType::Q4G64_F16S && gate_up.group == 64 &&
               gate_up.n == 34816 && gate_up.k == kHidden && gate_up.qdata != nullptr &&
               gate_up.scales != nullptr &&
               gate_up.layout == QuantLayout::RowSplitSm75Fragment &&
               down.qtype == QType::Q5G64_F16S && down.group == 64 &&
               down.n == kHidden && down.k == 17408 && down.qdata != nullptr &&
               down.qhigh != nullptr && down.scales != nullptr &&
               down.layout == QuantLayout::RowSplitSm75Fragment;
    }
    std::size_t scratch_bytes() const {
        const std::size_t q4 = kind == EngineKind::Fast
                                   ? tp2_fast_q4_scratch_bytes()
                                   : tp2_slow_q4_scratch_bytes();
        const std::size_t q5 = kind == EngineKind::Fast
                                   ? tp2_fast_q5_scratch_bytes()
                                   : tp2_slow_q5_scratch_bytes();
        return std::max(q4, q5);
    }

    void allocate(void** pointer, std::size_t bytes) {
        check(cudaMalloc(pointer, bytes), "cudaMalloc TP2");
        device_bytes += bytes;
    }

    void upload(std::uint8_t*& destination,
                const std::vector<std::uint8_t>& source) {
        if (source.empty()) {
            destination = nullptr;
            return;
        }
        allocate(reinterpret_cast<void**>(&destination), source.size());
        check(cudaMemcpyAsync(destination, source.data(), source.size(),
                              cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync TP2 weight");
    }

    void transform(LayerWeight& weight, void* scratch) {
        if (kind == EngineKind::Fast) {
            tp2_fast_q4_transform(weight.q4_codes, weight.q4_scales, scratch,
                                  stream);
            tp2_fast_q5_transform(weight.q5_low, weight.q5_high, weight.q5_scales,
                                  scratch, stream);
        } else {
            tp2_slow_q4_transform(weight.q4_codes, weight.q4_scales, scratch,
                                  stream);
            tp2_slow_q5_transform(weight.q5_low, weight.q5_high, weight.q5_scales,
                                  scratch, stream);
        }
    }

    void initialize(int target_device, EngineKind target_kind,
                    const std::filesystem::path& root, int target_canonical_begin,
                    int target_canonical_end) {
        if (target_canonical_begin < 0 || target_canonical_end > kLayers ||
            target_canonical_begin > target_canonical_end) {
            throw std::invalid_argument("TP2 canonical layer range is invalid");
        }
        device = target_device;
        kind = target_kind;
        canonical_begin = target_canonical_begin;
        canonical_end = target_canonical_end;
        check(cudaSetDevice(device), "cudaSetDevice TP2 init");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreate TP2");
        check(cudaEventCreateWithFlags(&quant_ready, cudaEventDisableTiming),
              "cudaEventCreate TP2 quant");
        check(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming),
              "cudaEventCreate TP2 input");
        intermediate =
            kind == EngineKind::Fast ? tp2_fast_q4_rows() : tp2_slow_q4_rows();
        intermediate_groups = kind == EngineKind::Fast
                                  ? tp2_fast_q4_output_groups()
                                  : tp2_slow_q4_output_groups();
        const int q5_k =
            kind == EngineKind::Fast ? tp2_fast_q5_k() : tp2_slow_q5_k();
        const int q5_rows =
            kind == EngineKind::Fast ? tp2_fast_q5_rows() : tp2_slow_q5_rows();
        if (q5_k != intermediate || q5_rows != kHidden) {
            throw std::runtime_error("TP2 generated kernel dimensions disagree");
        }

        allocate(reinterpret_cast<void**>(&input_packet), kInputPacketBytes);
        input_codes = reinterpret_cast<std::int8_t*>(input_packet);
        input_scales = reinterpret_cast<__half*>(input_packet + kMaximumInputLayout.scale_offset);
        allocate(reinterpret_cast<void**>(&intermediate_codes),
                 static_cast<std::size_t>(kMaximumTokens) * intermediate);
        allocate(reinterpret_cast<void**>(&intermediate_scales),
                 static_cast<std::size_t>(kMaximumTokens) * intermediate_groups *
                     sizeof(__half));
        allocate(reinterpret_cast<void**>(&output_bf16),
                 kMaximumPartialElements * sizeof(__nv_bfloat16));
        allocate(reinterpret_cast<void**>(&owner_sum), kMaximumPartialBytes);

        void* scratch = nullptr;
        const std::size_t scratch_size = scratch_bytes();
        allocate(&scratch, scratch_size);
        layers.resize(kLayers);
        canonical_alias_bytes = shard_weight_bytes() *
            static_cast<std::size_t>(canonical_end - canonical_begin);
        for (int layer = 0; layer < kLayers; ++layer) {
            if (uses_canonical(layer)) { continue; }
            char layer_name[32];
            std::snprintf(layer_name, sizeof(layer_name), "layer%02d", layer);
            const std::filesystem::path directory =
                root / layer_name / "weights";
            const HostWeight q4 =
                load_weight(directory / (std::string("q4_") + tag() + ".bin"));
            const HostWeight q5 =
                load_weight(directory / (std::string("q5_") + tag() + ".bin"));
            if (q4.header.n != static_cast<std::uint32_t>(2 * intermediate) ||
                q4.header.k != kHidden || q4.low.size() != q4_code_bytes() ||
                q4.scales.size() != q4_scale_bytes() || !q4.high.empty()) {
                throw std::runtime_error("TP2 Q4 shard mismatch at layer " +
                                         std::to_string(layer));
            }
            if (q5.header.n != kHidden ||
                q5.header.k != static_cast<std::uint32_t>(intermediate) ||
                q5.high.empty()) {
                throw std::runtime_error("TP2 Q5 shard mismatch at layer " +
                                         std::to_string(layer));
            }
            LayerWeight& destination = layers[static_cast<std::size_t>(layer)];
            upload(destination.q4_codes, q4.low);
            upload(destination.q4_scales, q4.scales);
            upload(destination.q5_low, q5.low);
            upload(destination.q5_high, q5.high);
            upload(destination.q5_scales, q5.scales);
            transform(destination, scratch);
            check(cudaStreamSynchronize(stream),
                  "cudaStreamSynchronize TP2 transform");
        }
        check(cudaFree(scratch), "cudaFree TP2 scratch");
        device_bytes -= scratch_size;
    }

    void run(int layer, const Weight* canonical_gate_up,
             const Weight* canonical_down, int tokens, void* host_output) {
        check(cudaSetDevice(device), "cudaSetDevice TP2 run");
        const std::size_t active_elements = partial_elements(tokens);
        const std::size_t active_bf16_bytes = partial_bf16_bytes(tokens);
        check(cudaMemsetAsync(output_bf16, 0,
                              active_elements * sizeof(__nv_bfloat16), stream),
              "cudaMemsetAsync TP2 output");
        const bool canonical = uses_canonical(layer);
        const std::uint8_t* q4_codes = nullptr;
        const std::uint8_t* q4_scales = nullptr;
        const std::uint8_t* q5_low = nullptr;
        const std::uint8_t* q5_high = nullptr;
        const std::uint8_t* q5_scales = nullptr;
        if (canonical) {
            if (canonical_gate_up == nullptr || canonical_down == nullptr ||
                !canonical_weights_valid(*canonical_gate_up, *canonical_down)) {
                throw std::runtime_error(
                    "TP2 canonical alias received an incompatible local MLP layout");
            }
            q4_codes = static_cast<const std::uint8_t*>(canonical_gate_up->qdata);
            q4_scales = static_cast<const std::uint8_t*>(canonical_gate_up->scales);
            if (kind == EngineKind::Slow) {
                q4_codes += tp2_fast_q4_code_bytes();
                q4_scales += tp2_fast_q4_scale_bytes();
            }
            q5_low = static_cast<const std::uint8_t*>(canonical_down->qdata);
            q5_high = static_cast<const std::uint8_t*>(canonical_down->qhigh);
            q5_scales = static_cast<const std::uint8_t*>(canonical_down->scales);
        } else {
            LayerWeight& weight = layers.at(static_cast<std::size_t>(layer));
            if (weight.q4_codes == nullptr || weight.q4_scales == nullptr ||
                weight.q5_low == nullptr || weight.q5_high == nullptr ||
                weight.q5_scales == nullptr) {
                throw std::logic_error("TP2 nonlocal shard was not loaded");
            }
            q4_codes = weight.q4_codes;
            q4_scales = weight.q4_scales;
            q5_low = weight.q5_low;
            q5_high = weight.q5_high;
            q5_scales = weight.q5_scales;
        }
        if (kind == EngineKind::Fast) {
            if (tokens == 4) {
                tp2_fast_q4_run(q4_codes, q4_scales, input_codes, input_scales,
                                intermediate_codes, intermediate_scales, stream);
                if (canonical) {
                    tp2_fast_q5_run_canonical(q5_low, q5_high, q5_scales,
                                               intermediate_codes, intermediate_scales,
                                               output_bf16, stream);
                } else {
                    tp2_fast_q5_run(q5_low, q5_high, q5_scales, intermediate_codes,
                                    intermediate_scales, output_bf16, stream);
                }
            } else {
                tp2_t5_fast_q4_run(q4_codes, q4_scales, input_codes, input_scales,
                                   intermediate_codes, intermediate_scales, stream);
                if (canonical) {
                    tp2_fast_q5_run_canonical_t5(q5_low, q5_high, q5_scales,
                                                  intermediate_codes, intermediate_scales,
                                                  output_bf16, stream);
                } else {
                    tp2_t5_fast_q5_run(q5_low, q5_high, q5_scales, intermediate_codes,
                                       intermediate_scales, output_bf16, stream);
                }
            }
        } else {
            if (tokens == 4) {
                tp2_slow_q4_run(q4_codes, q4_scales, input_codes, input_scales,
                                intermediate_codes, intermediate_scales, stream);
                if (canonical) {
                    tp2_slow_q5_run_canonical(q5_low, q5_high, q5_scales,
                                               intermediate_codes, intermediate_scales,
                                               output_bf16, stream);
                } else {
                    tp2_slow_q5_run(q5_low, q5_high, q5_scales, intermediate_codes,
                                    intermediate_scales, output_bf16, stream);
                }
            } else {
                tp2_t5_slow_q4_run(q4_codes, q4_scales, input_codes, input_scales,
                                   intermediate_codes, intermediate_scales, stream);
                if (canonical) {
                    tp2_slow_q5_run_canonical_t5(q5_low, q5_high, q5_scales,
                                                  intermediate_codes, intermediate_scales,
                                                  output_bf16, stream);
                } else {
                    tp2_t5_slow_q5_run(q5_low, q5_high, q5_scales, intermediate_codes,
                                       intermediate_scales, output_bf16, stream);
                }
            }
        }
        // The caller consumes BF16 directly for both supported widths. Only the
        // nonlocal shard needs a host copy; no unused FP32 expansion is produced.
        if(host_output!=nullptr) {
            check(cudaMemcpyAsync(host_output,output_bf16,active_bf16_bytes,
                                  cudaMemcpyDeviceToHost,stream),
                  "cudaMemcpyAsync TP2 remote exact BF16 partial");
        }
    }

    void synchronize() {
        check(cudaSetDevice(device), "cudaSetDevice TP2 sync");
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize TP2 engine");
    }

    void release() noexcept {
        if (device < 0) {
            return;
        }
        (void)cudaSetDevice(device);
        for (LayerWeight& weight : layers) {
            (void)cudaFree(weight.q4_codes);
            (void)cudaFree(weight.q4_scales);
            (void)cudaFree(weight.q5_low);
            (void)cudaFree(weight.q5_high);
            (void)cudaFree(weight.q5_scales);
        }
        (void)cudaFree(input_packet);
        (void)cudaFree(intermediate_codes);
        (void)cudaFree(intermediate_scales);
        (void)cudaFree(output_bf16);
        (void)cudaFree(owner_sum);
        if (quant_ready != nullptr) {
            (void)cudaEventDestroy(quant_ready);
        }
        if (input_ready != nullptr) {
            (void)cudaEventDestroy(input_ready);
        }
        if (stream != nullptr) {
            (void)cudaStreamDestroy(stream);
        }
        device = -1;
    }
};

} // namespace

class Tp2MlpCoordinator::Impl {
public:
    Impl(std::filesystem::path root, int fast_device, int slow_device,
         std::int32_t split_layer)
        : host_input_(kInputPacketBytes),
          host_remote_(kMaximumPartialBytes) {
        if (fast_device == slow_device || split_layer <= 0 || split_layer >= kLayers) {
            throw std::invalid_argument("TP2 MLP device/split configuration is invalid");
        }
        fast_.initialize(fast_device, EngineKind::Fast, root, 0, split_layer);
        slow_.initialize(slow_device, EngineKind::Slow, root, split_layer, kLayers);
        stats_.fast_device_bytes = fast_.device_bytes;
        stats_.slow_device_bytes = slow_.device_bytes;
        stats_.fast_canonical_alias_bytes = fast_.canonical_alias_bytes;
        stats_.slow_canonical_alias_bytes = slow_.canonical_alias_bytes;
    }

    bool execute(std::int32_t layer, const Tensor& normalized,
                 const Weight& canonical_gate_up, const Weight& canonical_down,
                 Tensor& residual, cudaStream_t owner_stream) {
        std::lock_guard lock(mutex_);
        if (layer < 0 || layer >= kLayers || normalized.dtype != DType::BF16 ||
            residual.dtype != DType::BF16 || normalized.ne[0] != kHidden ||
            (normalized.ne[1] != 4 && normalized.ne[1] != 5) ||
            residual.ne[0] != kHidden || residual.ne[1] != normalized.ne[1] ||
            !normalized.is_contiguous() || !residual.is_contiguous()) {
            return false;
        }

        int owner_device = -1;
        check(cudaGetDevice(&owner_device), "cudaGetDevice TP2 owner");
        Engine* owner = nullptr;
        if (owner_device == fast_.device) {
            owner = &fast_;
        } else if (owner_device == slow_.device) {
            owner = &slow_;
        } else {
            return false;
        }

        const int tokens = normalized.ne[1];
        const auto input_layout=detail::tp2_input_packet_layout(tokens);
        // execute() is serialized and the prior call drains both engine streams.
        // These views therefore cannot race a previous-width kernel or DMA.
        for(Engine* engine:{&fast_,&slow_})
            engine->input_scales=reinterpret_cast<__half*>(engine->input_packet+input_layout.scale_offset);
        const std::size_t active_elements = partial_elements(tokens);
        const auto input_started = Clock::now();
        if (tokens == 4) {
            quantize_a8<4><<<static_cast<unsigned>((4 * kInputGroups + 7) / 8),
                             256, 0, owner_stream>>>(
                static_cast<const __nv_bfloat16*>(normalized.data),
                owner->input_codes, owner->input_scales);
        } else {
            quantize_a8<5><<<static_cast<unsigned>((5 * kInputGroups + 7) / 8),
                             256, 0, owner_stream>>>(
                static_cast<const __nv_bfloat16*>(normalized.data),
                owner->input_codes, owner->input_scales);
        }
        check(cudaGetLastError(), "TP2 quantize_a8 launch");
        check(cudaEventRecord(owner->quant_ready, owner_stream),
              "cudaEventRecord TP2 quant");
        check(cudaMemcpyAsync(host_input_.data(), owner->input_packet,
                              input_layout.packet_bytes, cudaMemcpyDeviceToHost,
                              owner_stream),
              "cudaMemcpyAsync TP2 input packet D2H");
        check(cudaEventRecord(owner->input_ready, owner_stream),
              "cudaEventRecord TP2 input DMA");
        stats_.input_seconds +=
            std::chrono::duration<double>(Clock::now() - input_started).count();

        const auto engine_started = Clock::now();
        const auto stage_input = [&](Engine& engine) {
            check(cudaSetDevice(engine.device),
                  "cudaSetDevice TP2 stage input");
            if (engine.device == owner_device) {
                check(cudaStreamWaitEvent(engine.stream, owner->quant_ready, 0),
                      "cudaStreamWaitEvent TP2 local quant");
            } else {
                check(cudaStreamWaitEvent(engine.stream, owner->input_ready, 0),
                      "cudaStreamWaitEvent TP2 remote input");
                check(cudaMemcpyAsync(engine.input_packet, host_input_.data(),
                                      input_layout.packet_bytes, cudaMemcpyHostToDevice,
                                      engine.stream),
                      "cudaMemcpyAsync TP2 remote input packet");
            }
        };
        stage_input(fast_);
        stage_input(slow_);
        Engine* remote = owner == &fast_ ? &slow_ : &fast_;
        if (!owner->uses_canonical(layer) || remote->uses_canonical(layer)) {
            throw std::logic_error("TP2 canonical/nonlocal layer ownership is inconsistent");
        }
        void* remote_host = host_remote_.data();
        owner->run(layer, &canonical_gate_up, &canonical_down, tokens, nullptr);
        remote->run(layer, nullptr, nullptr, tokens, remote_host);
        owner->synchronize();
        remote->synchronize();
        stats_.engine_seconds +=
            std::chrono::duration<double>(Clock::now() - engine_started).count();

        const auto reduction_started = Clock::now();
        check(cudaSetDevice(owner_device), "cudaSetDevice TP2 residual");
        const std::size_t active_bf16_bytes=partial_bf16_bytes(tokens);
        check(cudaMemcpyAsync(owner->owner_sum,remote_host,active_bf16_bytes,
                              cudaMemcpyHostToDevice,owner_stream),
              "cudaMemcpyAsync TP2 remote exact BF16 partial");
        add_two_bf16_residual<<<static_cast<unsigned>((active_elements+255)/256),256,0,owner_stream>>>(
            owner->output_bf16,owner->owner_sum,static_cast<__nv_bfloat16*>(residual.data),
            static_cast<int>(active_elements));
        check(cudaGetLastError(),"TP2 owner-device exact BF16 reduction launch");
        stats_.input_transfer_bytes+=2*input_layout.packet_bytes;
        stats_.partial_transfer_bytes+=2*active_bf16_bytes;
        stats_.reduction_seconds +=
            std::chrono::duration<double>(Clock::now() - reduction_started)
                .count();
        stats_.calls += 1;
        if (tokens == 4) {
            stats_.width4_calls += 1;
        } else {
            stats_.width5_calls += 1;
        }
        return true;
    }

    Tp2MlpStats stats() const noexcept {
        std::lock_guard lock(mutex_);
        return stats_;
    }

private:
    mutable std::mutex mutex_;
    Engine fast_;
    Engine slow_;
    PinnedBuffer<std::uint8_t> host_input_;
    PinnedBuffer<std::uint8_t> host_remote_;
    Tp2MlpStats stats_;
};

Tp2MlpCoordinator::Tp2MlpCoordinator(std::filesystem::path weights_root,
                                     int fast_device, int slow_device,
                                     std::int32_t split_layer)
    : impl_(std::make_unique<Impl>(std::move(weights_root), fast_device,
                                   slow_device, split_layer)) {}

Tp2MlpCoordinator::~Tp2MlpCoordinator() = default;

bool Tp2MlpCoordinator::execute(std::int32_t global_layer,
                                const Tensor& normalized_hidden,
                                const Weight& canonical_gate_up,
                                const Weight& canonical_down, Tensor& residual,
                                cudaStream_t owner_stream) {
    return impl_->execute(global_layer, normalized_hidden, canonical_gate_up,
                          canonical_down, residual, owner_stream);
}

Tp2MlpStats Tp2MlpCoordinator::stats() const noexcept {
    return impl_->stats();
}

} // namespace ninfer::apps::dual
