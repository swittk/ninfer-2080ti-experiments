#include "ninfer_bench_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/launcher/gqa_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;
using Clock = std::chrono::steady_clock;

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQueryHeads = 24;
constexpr std::int32_t kKvHeads = 4;
constexpr std::int32_t kQuantGroup = 64;
constexpr std::int32_t kScaleRows = kHeadDim / kQuantGroup;
constexpr std::int32_t kTokens = 4;
constexpr std::int32_t kAppendChunk = 128;
constexpr std::int32_t kKeyTile = 32;
constexpr float kAttentionScale = 0.0625F;
constexpr double kRelativeL2Tolerance = 3.15e-3;

struct Options {
    int fast_device = 0;
    int slow_device = 1;
    std::int32_t window = 262144;
    std::int32_t capacity = 0;
    DType kv_dtype       = DType::I8;
    double fast_fraction = 0.60;
    int warmup = 5;
    int repeat = 20;
};

[[noreturn]] void usage(const char* argv0, const char* error = nullptr) {
    if (error != nullptr) std::fprintf(stderr, "error: %s\n\n", error);
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "  --fast-device N       fast CUDA device (default 0)\n"
                 "  --slow-device N       slow CUDA device (default 1)\n"
                 "  --window N            visible keys, up to 262144\n"
                 "  --capacity N          fixed physical capacity, optionally larger than visible keys\n"
                 "  --kv-dtype int8|bf16  cache representation (default int8)\n"
                 "  --fast-fraction F     global split fraction on fast GPU\n"
                 "  --warmup N            alternating warmup pairs\n"
                 "  --repeat N            alternating measured pairs\n",
                 argv0);
    std::exit(error == nullptr ? 0 : 2);
}

Options parse_options(int argc, char** argv) {
    Options out;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg(argv[index]);
        const auto take = [&]() -> std::string {
            if (++index >= argc) usage(argv[0], "missing option value");
            return argv[index];
        };
        if (arg == "--fast-device") out.fast_device = std::stoi(take());
        else if (arg == "--slow-device") out.slow_device = std::stoi(take());
        else if (arg == "--window") out.window = std::stoi(take());
        else if (arg == "--capacity") out.capacity = std::stoi(take());
        else if (arg == "--kv-dtype") {
            const std::string value = take();
            if (value == "int8")
                out.kv_dtype = DType::I8;
            else if (value == "bf16")
                out.kv_dtype = DType::BF16;
            else
                usage(argv[0], "--kv-dtype expects int8 or bf16");
        } else if (arg == "--fast-fraction") out.fast_fraction = std::stod(take());
        else if (arg == "--warmup") out.warmup = std::stoi(take());
        else if (arg == "--repeat") out.repeat = std::stoi(take());
        else if (arg == "--help" || arg == "-h") usage(argv[0]);
        else usage(argv[0], "unknown option");
    }
    if (out.fast_device == out.slow_device || out.window < kTokens || out.window > 262144 ||
        !(out.fast_fraction > 0.0 && out.fast_fraction < 1.0) || out.warmup < 0 ||
        out.repeat <= 0) {
        usage(argv[0], "options are outside the supported domain");
    }
    if(out.capacity!=0 && (out.capacity<out.window || out.capacity>262144 || out.capacity%64))
        usage(argv[0],"capacity must be page-aligned and cover the visible prefix");
    return out;
}

struct SplitGeometry {
    std::int32_t total_splits = 0;
    std::int32_t fast_splits = 0;
    std::int32_t slow_splits = 0;
    std::int32_t split_unit_keys = 0;
    std::int32_t boundary_key = 0;
};

SplitGeometry split_geometry(std::int32_t window, double fast_fraction, DType kv_dtype) {
    SplitGeometry out;
    out.total_splits = ops::detail::gqa_attention_split_capacity(
        kQueryHeads, kTokens, kv_dtype,
        {static_cast<std::uint32_t>(window), static_cast<std::uint32_t>(window)});
    if (out.total_splits < 2) throw std::logic_error("sequence sharding needs two splits");
    out.fast_splits = std::clamp(
        static_cast<std::int32_t>(std::llround(out.total_splits * fast_fraction)), 1,
        out.total_splits - 1);
    out.slow_splits = out.total_splits - out.fast_splits;
    const std::int32_t logical_tiles = (window + kKeyTile - 1) / kKeyTile;
    const bool tile_split = logical_tiles >= out.total_splits;
    const std::int32_t units = tile_split
        ? (logical_tiles + out.total_splits - 1) / out.total_splits
        : (window + out.total_splits - 1) / out.total_splits;
    out.split_unit_keys = units * (tile_split ? kKeyTile : 1);
    out.boundary_key = std::min(window, out.fast_splits * out.split_unit_keys);
    if (out.boundary_key <= 0 || out.boundary_key >= window) {
        throw std::logic_error("sequence-sharded boundary is empty");
    }
    return out;
}

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

float source_value(std::uint32_t position, std::uint32_t index, std::uint32_t salt) {
    const std::uint32_t hash = mix(position * 0x9e3779b9U + index * 0x85ebca6bU + salt);
    return (static_cast<float>(hash & 2047U) / 1023.5F - 1.0F) * 0.75F;
}

std::uint16_t to_bf16(float value) { return ninfer::bench::f32_to_bf16(value); }

float from_bf16(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float output = 0.0F;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

struct Comparison {
    std::size_t mismatches = 0;
    std::size_t first = std::numeric_limits<std::size_t>::max();
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double relative_l2 = 0.0;
    double cosine = 1.0;
};

Comparison compare_bf16(const std::vector<std::uint16_t>& left,
                        const std::vector<std::uint16_t>& right) {
    if (left.size() != right.size()) throw std::invalid_argument("BF16 compare size mismatch");
    Comparison result;
    double absolute_sum = 0.0;
    double delta_sq = 0.0;
    double left_sq = 0.0;
    double right_sq = 0.0;
    double dot = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index] != right[index]) {
            if (result.first == std::numeric_limits<std::size_t>::max()) result.first = index;
            ++result.mismatches;
        }
        const double a = from_bf16(left[index]);
        const double b = from_bf16(right[index]);
        const double delta = b - a;
        result.max_abs = std::max(result.max_abs, std::abs(delta));
        absolute_sum += std::abs(delta);
        delta_sq += delta * delta;
        left_sq += a * a;
        right_sq += b * b;
        dot += a * b;
    }
    result.mean_abs = left.empty() ? 0.0 : absolute_sum / left.size();
    result.relative_l2 = left_sq > 0.0 ? std::sqrt(delta_sq / left_sq) : 0.0;
    result.cosine = left_sq > 0.0 && right_sq > 0.0
                        ? dot / std::sqrt(left_sq * right_sq)
                        : 1.0;
    return result;
}

Comparison compare_fp32(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size()) throw std::invalid_argument("FP32 compare size mismatch");
    Comparison result;
    double absolute_sum = 0.0;
    double delta_sq = 0.0;
    double left_sq = 0.0;
    double right_sq = 0.0;
    double dot = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (std::memcmp(&left[index], &right[index], sizeof(float)) != 0) {
            if (result.first == std::numeric_limits<std::size_t>::max()) result.first = index;
            ++result.mismatches;
        }
        const double a = left[index];
        const double b = right[index];
        const double delta = b - a;
        result.max_abs = std::max(result.max_abs, std::abs(delta));
        absolute_sum += std::abs(delta);
        delta_sq += delta * delta;
        left_sq += a * a;
        right_sq += b * b;
        dot += a * b;
    }
    result.mean_abs = left.empty() ? 0.0 : absolute_sum / left.size();
    result.relative_l2 = left_sq > 0.0 ? std::sqrt(delta_sq / left_sq) : 0.0;
    result.cosine = left_sq > 0.0 && right_sq > 0.0
                        ? dot / std::sqrt(left_sq * right_sq)
                        : 1.0;
    return result;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct RangeState {
    static int activate_device(int device) {
        CUDA_CHECK(cudaSetDevice(device));
        return device;
    }

    int device;
    DeviceBuffer acc;
    DeviceBuffer maximum;
    DeviceBuffer denominator;

    explicit RangeState(int device_id)
        : device(activate_device(device_id)),
          acc(static_cast<std::size_t>(kHeadDim) * kQueryHeads * kTokens * sizeof(float)),
          maximum(static_cast<std::size_t>(kQueryHeads) * kTokens * sizeof(float)),
          denominator(static_cast<std::size_t>(kQueryHeads) * kTokens * sizeof(float)) {}

    void activate() const { CUDA_CHECK(cudaSetDevice(device)); }
    std::size_t bytes() const noexcept { return acc.bytes + maximum.bytes + denominator.bytes; }
    Tensor acc_tensor() { return Tensor(acc.p, DType::FP32, {kHeadDim, kQueryHeads, kTokens}); }
    Tensor maximum_tensor() { return Tensor(maximum.p, DType::FP32, {kQueryHeads, kTokens}); }
    Tensor denominator_tensor() {
        return Tensor(denominator.p, DType::FP32, {kQueryHeads, kTokens});
    }

    std::vector<float> download() {
        activate();
        std::vector<float> host(bytes() / sizeof(float));
        std::size_t offset = 0;
        acc.copy_to_host(host.data() + offset, acc.bytes);
        offset += acc.bytes / sizeof(float);
        maximum.copy_to_host(host.data() + offset, maximum.bytes);
        offset += maximum.bytes / sizeof(float);
        denominator.copy_to_host(host.data() + offset, denominator.bytes);
        return host;
    }

    void download_async(PinnedHostBuffer& host, cudaStream_t stream) {
        activate();
        if (host.size() != bytes()) throw std::invalid_argument("state packet size mismatch");
        auto* destination = static_cast<std::uint8_t*>(host.data());
        CUDA_CHECK(cudaMemcpyAsync(destination, acc.p, acc.bytes, cudaMemcpyDeviceToHost, stream));
        destination += acc.bytes;
        CUDA_CHECK(cudaMemcpyAsync(destination, maximum.p, maximum.bytes,
                                   cudaMemcpyDeviceToHost, stream));
        destination += maximum.bytes;
        CUDA_CHECK(cudaMemcpyAsync(destination, denominator.p, denominator.bytes,
                                   cudaMemcpyDeviceToHost, stream));
    }

    void upload_async(const PinnedHostBuffer& host, cudaStream_t stream) {
        activate();
        if (host.size() != bytes()) throw std::invalid_argument("state packet size mismatch");
        const auto* source = static_cast<const std::uint8_t*>(host.data());
        CUDA_CHECK(cudaMemcpyAsync(acc.p, source, acc.bytes, cudaMemcpyHostToDevice, stream));
        source += acc.bytes;
        CUDA_CHECK(cudaMemcpyAsync(maximum.p, source, maximum.bytes,
                                   cudaMemcpyHostToDevice, stream));
        source += maximum.bytes;
        CUDA_CHECK(cudaMemcpyAsync(denominator.p, source, denominator.bytes,
                                   cudaMemcpyHostToDevice, stream));
    }
};

class CacheFixture {
public:
    CacheFixture(int device, std::int32_t window, std::int32_t key_begin,
                 std::int32_t key_end, std::int32_t total_splits, DType kv_dtype, bool fixed_ranges=false)
        : device_(device), context_(device), window_(window), key_begin_(key_begin),
          kv_dtype_(kv_dtype), quantized_(kv_dtype == DType::I8), fixed_ranges_(fixed_ranges),
          key_end_(key_end), logical_pages_((window + kPagedKVPageSize - 1) / kPagedKVPageSize),
          first_page_(key_begin / kPagedKVPageSize),
          last_page_((key_end - 1) / kPagedKVPageSize),
          physical_pages_(last_page_ - first_page_ + 1), splits_(total_splits),
          code_elements_(static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize * kKvHeads *
                         physical_pages_),
          scale_elements_(static_cast<std::size_t>(kScaleRows) * kPagedKVPageSize * kKvHeads *
                          physical_pages_),
          k_cache_(code_elements_ * (quantized_ ? 1U : sizeof(std::uint16_t))),
          v_cache_(code_elements_ * (quantized_ ? 1U : sizeof(std::uint16_t))),
          k_scale_(quantized_ ? scale_elements_ * sizeof(std::uint16_t) : 1U),
          v_scale_(quantized_ ? scale_elements_ * sizeof(std::uint16_t) : 1U),
          block_table_(static_cast<std::size_t>(logical_pages_) * sizeof(std::int32_t)),
          input_k_(static_cast<std::size_t>(kHeadDim) * kKvHeads * kAppendChunk *
                   sizeof(std::uint16_t)),
          input_v_(static_cast<std::size_t>(kHeadDim) * kKvHeads * kAppendChunk *
                   sizeof(std::uint16_t)),
          input_positions_(static_cast<std::size_t>(kAppendChunk) * sizeof(std::int32_t)),
          q_(static_cast<std::size_t>(kHeadDim) * kQueryHeads * kTokens *
             sizeof(std::uint16_t)),
          positions_(static_cast<std::size_t>(kTokens) * sizeof(std::int32_t)),
          partial_acc_(static_cast<std::size_t>(kHeadDim) * kQueryHeads * kTokens * splits_ *
                       sizeof(std::uint16_t)),
          partial_m_(static_cast<std::size_t>(kQueryHeads) * kTokens * splits_ * sizeof(float)),
          partial_l_(static_cast<std::size_t>(kQueryHeads) * kTokens * splits_ * sizeof(float)),
          output_(static_cast<std::size_t>(kHeadDim) * kQueryHeads * kTokens *
                  sizeof(std::uint16_t)) {
        if (key_begin_ < 0 || key_end_ <= key_begin_ || key_end_ > window_) {
            throw std::invalid_argument("cache ownership range is invalid");
        }
        activate();
        mapping_.assign(static_cast<std::size_t>(logical_pages_), -1);
        for (std::int32_t logical = first_page_; logical <= last_page_; ++logical) {
            mapping_[static_cast<std::size_t>(logical)] = logical - first_page_;
        }
        block_table_.copy_from_host(mapping_.data(), block_table_.bytes);
        k_cache_.fill();
        v_cache_.fill();
        if (quantized_) {
            k_scale_.fill();
            v_scale_.fill();
        }
        const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
            kQueryHeads, kv_dtype_,
            {static_cast<std::uint32_t>(window_), static_cast<std::uint32_t>(window_)},
            1, kTokens, kTokens);
        workspace_ = std::make_unique<WorkspaceArena>(
            std::max<std::size_t>(workspace_bytes, static_cast<std::size_t>(256)));
        context_.synchronize();
    }

    void activate() const { CUDA_CHECK(cudaSetDevice(device_)); }
    void synchronize() const { context_.synchronize(); }
    cudaStream_t stream() const noexcept { return context_.stream; }
    bool owns(std::int32_t position) const noexcept {
        return position >= key_begin_ && position < key_end_;
    }
    std::int32_t key_begin() const noexcept { return key_begin_; }
    std::int32_t key_end() const noexcept { return key_end_; }
    std::int32_t logical_pages() const noexcept { return logical_pages_; }
    std::int32_t physical_pages() const noexcept { return physical_pages_; }
    std::size_t payload_bytes() const noexcept {
        return k_cache_.bytes + v_cache_.bytes +
               (quantized_ ? k_scale_.bytes + v_scale_.bytes : 0U);
    }

    PagedKVLayerView cache_view() {
        return {
            .k_pages = Tensor(k_cache_.p, kv_dtype_,
                              {kHeadDim, kPagedKVPageSize, kKvHeads, physical_pages_}),
            .v_pages = Tensor(v_cache_.p, kv_dtype_,
                              {kHeadDim, kPagedKVPageSize, kKvHeads, physical_pages_}),
            .k_scale_pages = quantized_
                ? Tensor(k_scale_.p, DType::FP16,
                         {kScaleRows, kPagedKVPageSize, kKvHeads, physical_pages_})
                : Tensor(),
            .v_scale_pages = quantized_
                ? Tensor(v_scale_.p, DType::FP16,
                         {kScaleRows, kPagedKVPageSize, kKvHeads, physical_pages_})
                : Tensor(),
            .block_table = Tensor(block_table_.p, DType::I32, {logical_pages_}),
            .head_dim = kHeadDim,
            .num_kv_heads = kKvHeads,
            .dtype = kv_dtype_,
            .quant_group = quantized_ ? kQuantGroup : 0,
        };
    }

    void append_slice(const std::vector<std::uint16_t>& k,
                      const std::vector<std::uint16_t>& v,
                      const std::vector<std::int32_t>& positions,
                      std::int32_t offset, std::int32_t tokens) {
        if (tokens <= 0) return;
        if (offset < 0 || offset + tokens > static_cast<std::int32_t>(positions.size())) {
            throw std::invalid_argument("append slice is outside host packet");
        }
        for (std::int32_t index = 0; index < tokens; ++index) {
            if (!owns(positions[static_cast<std::size_t>(offset + index)])) {
                throw std::logic_error("append slice crosses cache ownership");
            }
        }
        activate();
        const std::size_t token_elements = static_cast<std::size_t>(kHeadDim) * kKvHeads;
        const std::size_t kv_bytes = token_elements * tokens * sizeof(std::uint16_t);
        input_k_.copy_from_host(k.data() + static_cast<std::size_t>(offset) * token_elements,
                                kv_bytes);
        input_v_.copy_from_host(v.data() + static_cast<std::size_t>(offset) * token_elements,
                                kv_bytes);
        input_positions_.copy_from_host(positions.data() + offset,
                                        static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
        Tensor k_tensor(input_k_.p, DType::BF16, {kHeadDim, kKvHeads, tokens});
        Tensor v_tensor(input_v_.p, DType::BF16, {kHeadDim, kKvHeads, tokens});
        Tensor position_tensor(input_positions_.p, DType::I32, {tokens});
        ops::gqa_kv_append(k_tensor, v_tensor, position_tensor, cache_view(), context_.stream);
    }

    void set_q_positions(const std::vector<std::uint16_t>& q,
                         const std::array<std::int32_t, kTokens>& positions) {
        activate();
        q_.copy_from_host(q.data(), q_.bytes);
        positions_.copy_from_host(positions.data(), positions_.bytes);
        context_.synchronize();
    }

    void download_q(PinnedHostBuffer& host) {
        activate();
        if (host.size() != q_.bytes) throw std::invalid_argument("Q packet size mismatch");
        CUDA_CHECK(cudaMemcpyAsync(host.data(), q_.p, q_.bytes, cudaMemcpyDeviceToHost,
                                   context_.stream));
    }

    void upload_q(const PinnedHostBuffer& host) {
        activate();
        if (host.size() != q_.bytes) throw std::invalid_argument("Q packet size mismatch");
        CUDA_CHECK(cudaMemcpyAsync(q_.p, host.data(), q_.bytes, cudaMemcpyHostToDevice,
                                   context_.stream));
    }

    void full_attention() {
        activate();
        workspace_->reset();
        Tensor output = output_tensor();
        Tensor q = q_tensor();
        Tensor positions = position_tensor();
        ops::gqa_attention_cached(q, positions, kAttentionScale, cache_view(), envelope(),
                                  *workspace_, output, context_.stream);
        workspace_->reset();
    }

    void range(std::int32_t begin, std::int32_t count, RangeState& state) {
        activate();
        Tensor q = q_tensor();
        Tensor positions = position_tensor();
        Tensor partial_acc(partial_acc_.p, DType::BF16,
                           {kHeadDim, kQueryHeads, kTokens, splits_});
        Tensor partial_m(partial_m_.p, DType::FP32, {kQueryHeads, kTokens, splits_});
        Tensor partial_l(partial_l_.p, DType::FP32, {kQueryHeads, kTokens, splits_});
        ops::detail::gqa_attention_cached_small_t_partial_range_launch(
            q, positions, kAttentionScale, cache_view(), envelope(), begin, count,
            partial_acc, partial_m, partial_l, context_.stream,1,fixed_ranges_?window_:0);
        Tensor state_acc = state.acc_tensor();
        Tensor state_m = state.maximum_tensor();
        Tensor state_l = state.denominator_tensor();
        ops::detail::gqa_attention_small_t_reduce_range_state_launch(
            kQueryHeads, kTokens, splits_, begin, count, partial_acc, partial_m, partial_l,
            state_acc, state_m, state_l, context_.stream);
    }

    void merge(RangeState& left, RangeState& right) {
        activate();
        Tensor left_acc = left.acc_tensor();
        Tensor left_m = left.maximum_tensor();
        Tensor left_l = left.denominator_tensor();
        Tensor right_acc = right.acc_tensor();
        Tensor right_m = right.maximum_tensor();
        Tensor right_l = right.denominator_tensor();
        Tensor output = output_tensor();
        ops::detail::gqa_attention_small_t_merge_two_states_launch(
            kQueryHeads, kTokens, left_acc, left_m, left_l, right_acc, right_m, right_l,
            output, context_.stream);
    }

    std::vector<std::uint16_t> output() {
        activate();
        std::vector<std::uint16_t> host(output_.bytes / sizeof(std::uint16_t));
        output_.copy_to_host(host.data(), output_.bytes);
        return host;
    }

private:
    ops::GqaExecutionEnvelope envelope() const {
        return {static_cast<std::uint32_t>(window_), static_cast<std::uint32_t>(window_)};
    }
    Tensor q_tensor() { return Tensor(q_.p, DType::BF16, {kHeadDim, kQueryHeads, kTokens}); }
    Tensor position_tensor() { return Tensor(positions_.p, DType::I32, {kTokens}); }
    Tensor output_tensor() {
        return Tensor(output_.p, DType::BF16, {kHeadDim, kQueryHeads, kTokens});
    }

    int device_ = 0;
    DeviceContext context_;
    std::int32_t window_ = 0;
    std::int32_t key_begin_ = 0;
    DType kv_dtype_              = DType::I8;
    bool quantized_              = true;
    bool fixed_ranges_=false;
    std::int32_t key_end_ = 0;
    std::int32_t logical_pages_ = 0;
    std::int32_t first_page_ = 0;
    std::int32_t last_page_ = 0;
    std::int32_t physical_pages_ = 0;
    std::int32_t splits_ = 0;
    std::size_t code_elements_ = 0;
    std::size_t scale_elements_ = 0;
    DeviceBuffer k_cache_;
    DeviceBuffer v_cache_;
    DeviceBuffer k_scale_;
    DeviceBuffer v_scale_;
    DeviceBuffer block_table_;
    DeviceBuffer input_k_;
    DeviceBuffer input_v_;
    DeviceBuffer input_positions_;
    DeviceBuffer q_;
    DeviceBuffer positions_;
    DeviceBuffer partial_acc_;
    DeviceBuffer partial_m_;
    DeviceBuffer partial_l_;
    DeviceBuffer output_;
    std::unique_ptr<WorkspaceArena> workspace_;
    std::vector<std::int32_t> mapping_;
};

void make_chunk(std::int32_t begin, std::int32_t tokens,
                std::vector<std::uint16_t>& k, std::vector<std::uint16_t>& v,
                std::vector<std::int32_t>& positions) {
    k.resize(static_cast<std::size_t>(kHeadDim) * kKvHeads * tokens);
    v.resize(k.size());
    positions.resize(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::uint32_t position = static_cast<std::uint32_t>(begin + token);
        positions[static_cast<std::size_t>(token)] = static_cast<std::int32_t>(position);
        for (std::int32_t index = 0; index < kHeadDim * kKvHeads; ++index) {
            const std::size_t offset = static_cast<std::size_t>(token) * kHeadDim * kKvHeads +
                                       static_cast<std::size_t>(index);
            k[offset] = to_bf16(source_value(position, static_cast<std::uint32_t>(index),
                                             0x1234U));
            v[offset] = to_bf16(source_value(position, static_cast<std::uint32_t>(index),
                                             0x5678U));
        }
    }
}

std::vector<std::uint16_t> make_q() {
    std::vector<std::uint16_t> q(
        static_cast<std::size_t>(kHeadDim) * kQueryHeads * kTokens);
    for (std::int32_t token = 0; token < kTokens; ++token) {
        for (std::int32_t index = 0; index < kHeadDim * kQueryHeads; ++index) {
            const std::size_t offset = static_cast<std::size_t>(token) * kHeadDim * kQueryHeads +
                                       static_cast<std::size_t>(index);
            q[offset] = to_bf16(source_value(0x10000U + static_cast<std::uint32_t>(token),
                                             static_cast<std::uint32_t>(index), 0x9abcU));
        }
    }
    return q;
}

void append_intersection(CacheFixture& fixture, std::int32_t chunk_begin,
                         std::int32_t chunk_tokens,
                         const std::vector<std::uint16_t>& k,
                         const std::vector<std::uint16_t>& v,
                         const std::vector<std::int32_t>& positions) {
    const std::int32_t begin = std::max(chunk_begin, fixture.key_begin());
    const std::int32_t end = std::min(chunk_begin + chunk_tokens, fixture.key_end());
    if (begin >= end) return;
    fixture.append_slice(k, v, positions, begin - chunk_begin, end - begin);
}

struct DualTiming {
    double total_us = 0.0;
    double q_stage_us = 0.0;
    double engines_us = 0.0;
    double reduction_us = 0.0;
};

double time_full(CacheFixture& reference,
                 const std::vector<std::uint16_t>& tail_k,
                 const std::vector<std::uint16_t>& tail_v,
                 const std::vector<std::int32_t>& tail_positions) {
    const auto started = Clock::now();
    reference.append_slice(tail_k, tail_v, tail_positions, 0, kTokens);
    reference.full_attention();
    reference.synchronize();
    return std::chrono::duration<double, std::micro>(Clock::now() - started).count();
}

DualTiming time_dual(CacheFixture& fast, CacheFixture& slow,
                     const SplitGeometry& geometry,
                     const std::vector<std::uint16_t>& tail_k,
                     const std::vector<std::uint16_t>& tail_v,
                     const std::vector<std::int32_t>& tail_positions,
                     RangeState& fast_state, RangeState& slow_state,
                     RangeState& transferred_state,
                     PinnedHostBuffer& q_packet, PinnedHostBuffer& state_packet) {
    DualTiming timing;
    const auto started = Clock::now();
    auto phase = started;
    fast.download_q(q_packet);
    fast.synchronize();
    slow.upload_q(q_packet);
    timing.q_stage_us =
        std::chrono::duration<double, std::micro>(Clock::now() - phase).count();

    phase = Clock::now();
    append_intersection(fast, tail_positions.front(), kTokens, tail_k, tail_v, tail_positions);
    fast.range(0, geometry.fast_splits, fast_state);
    append_intersection(slow, tail_positions.front(), kTokens, tail_k, tail_v, tail_positions);
    slow.range(geometry.fast_splits, geometry.slow_splits, slow_state);
    slow_state.download_async(state_packet, slow.stream());
    slow.synchronize();
    timing.engines_us =
        std::chrono::duration<double, std::micro>(Clock::now() - phase).count();

    phase = Clock::now();
    transferred_state.upload_async(state_packet, fast.stream());
    fast.merge(fast_state, transferred_state);
    fast.synchronize();
    timing.reduction_us =
        std::chrono::duration<double, std::micro>(Clock::now() - phase).count();
    timing.total_us =
        std::chrono::duration<double, std::micro>(Clock::now() - started).count();
    return timing;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const int capacity=options.capacity?options.capacity:options.window;
        const bool fixed=options.capacity!=0;
        const SplitGeometry geometry = split_geometry(capacity, options.fast_fraction, options.kv_dtype);
        CacheFixture reference(options.fast_device, capacity, 0, capacity,
                               geometry.total_splits, options.kv_dtype,fixed);
        CacheFixture fast(options.fast_device, capacity, 0, geometry.boundary_key,
                          geometry.total_splits, options.kv_dtype,fixed);
        CacheFixture slow(options.slow_device, capacity, geometry.boundary_key,
                          capacity, geometry.total_splits, options.kv_dtype,fixed);
        std::printf("PARTITION visible=%d capacity=%d fixed=%d\n",options.window,capacity,int(fixed));

        std::vector<std::uint16_t> k;
        std::vector<std::uint16_t> v;
        std::vector<std::int32_t> positions;
        const std::int32_t prefix = options.window - kTokens;
        for (std::int32_t begin = 0; begin < prefix; begin += kAppendChunk) {
            const std::int32_t tokens = std::min(kAppendChunk, prefix - begin);
            make_chunk(begin, tokens, k, v, positions);
            reference.append_slice(k, v, positions, 0, tokens);
            append_intersection(fast, begin, tokens, k, v, positions);
            append_intersection(slow, begin, tokens, k, v, positions);
        }
        reference.synchronize();
        fast.synchronize();
        slow.synchronize();

        const std::array<std::int32_t, kTokens> tail_positions_array{
            prefix, prefix + 1, prefix + 2, prefix + 3};
        const std::vector<std::uint16_t> q = make_q();
        reference.set_q_positions(q, tail_positions_array);
        fast.set_q_positions(q, tail_positions_array);
        slow.set_q_positions(q, tail_positions_array);
        make_chunk(prefix, kTokens, k, v, positions);
        const std::vector<std::uint16_t> tail_k = k;
        const std::vector<std::uint16_t> tail_v = v;
        const std::vector<std::int32_t> tail_positions = positions;

        reference.append_slice(tail_k, tail_v, tail_positions, 0, kTokens);
        reference.full_attention();
        reference.synchronize();
        const std::vector<std::uint16_t> full_output = reference.output();

        RangeState canonical_fast(options.fast_device);
        RangeState canonical_slow(options.fast_device);
        reference.range(0, geometry.fast_splits, canonical_fast);
        reference.range(geometry.fast_splits, geometry.slow_splits, canonical_slow);
        reference.synchronize();
        append_intersection(fast, prefix, kTokens, tail_k, tail_v, tail_positions);
        append_intersection(slow, prefix, kTokens, tail_k, tail_v, tail_positions);
        RangeState sharded_fast(options.fast_device);
        RangeState sharded_slow(options.slow_device);
        RangeState sharded_remote_on_fast(options.fast_device);
        fast.range(0, geometry.fast_splits, sharded_fast);
        slow.range(geometry.fast_splits, geometry.slow_splits, sharded_slow);
        fast.synchronize();
        slow.synchronize();

        const std::vector<float> canonical_fast_host = canonical_fast.download();
        const std::vector<float> canonical_slow_host = canonical_slow.download();
        const std::vector<float> sharded_fast_host = sharded_fast.download();
        const std::vector<float> sharded_slow_host = sharded_slow.download();
        PinnedHostBuffer correctness_packet(sharded_slow.bytes());
        sharded_slow.download_async(correctness_packet, slow.stream());
        slow.synchronize();
        sharded_remote_on_fast.upload_async(correctness_packet, fast.stream());
        fast.merge(sharded_fast, sharded_remote_on_fast);
        fast.synchronize();
        const std::vector<std::uint16_t> sharded_output = fast.output();

        RangeState canonical_slow_on_fast(options.fast_device);
        PinnedHostBuffer canonical_packet(canonical_slow.bytes());
        canonical_slow.download_async(canonical_packet, reference.stream());
        reference.synchronize();
        canonical_slow_on_fast.upload_async(canonical_packet, fast.stream());
        fast.merge(canonical_fast, canonical_slow_on_fast);
        fast.synchronize();
        const std::vector<std::uint16_t> canonical_split_output = fast.output();

        const Comparison fast_state_comparison =
            compare_fp32(canonical_fast_host, sharded_fast_host);
        const Comparison slow_state_comparison =
            compare_fp32(canonical_slow_host, sharded_slow_host);
        const Comparison canonical_output_comparison =
            compare_bf16(full_output, canonical_split_output);
        const Comparison sharded_output_comparison =
            compare_bf16(canonical_split_output, sharded_output);
        if (fast_state_comparison.mismatches != 0 || slow_state_comparison.mismatches != 0 ||
            !std::isfinite(sharded_output_comparison.relative_l2) ||
            sharded_output_comparison.relative_l2 > kRelativeL2Tolerance) {
            throw std::runtime_error("sequence-sharded correctness gate failed");
        }

        RangeState timed_fast(options.fast_device);
        RangeState timed_slow(options.slow_device);
        RangeState timed_remote(options.fast_device);
        PinnedHostBuffer q_packet(static_cast<std::size_t>(kHeadDim) * kQueryHeads * kTokens *
                                  sizeof(std::uint16_t));
        PinnedHostBuffer state_packet(timed_slow.bytes());
        for (int index = 0; index < options.warmup; ++index) {
            if ((index & 1) == 0) {
                (void)time_full(reference, tail_k, tail_v, tail_positions);
                (void)time_dual(fast, slow, geometry, tail_k, tail_v, tail_positions,
                                timed_fast, timed_slow, timed_remote, q_packet, state_packet);
            } else {
                (void)time_dual(fast, slow, geometry, tail_k, tail_v, tail_positions,
                                timed_fast, timed_slow, timed_remote, q_packet, state_packet);
                (void)time_full(reference, tail_k, tail_v, tail_positions);
            }
        }
        std::vector<double> full_us;
        std::vector<double> dual_us;
        std::vector<double> q_us;
        std::vector<double> engines_us;
        std::vector<double> reduction_us;
        full_us.reserve(options.repeat);
        dual_us.reserve(options.repeat);
        for (int index = 0; index < options.repeat; ++index) {
            DualTiming timing;
            if ((index & 1) == 0) {
                full_us.push_back(time_full(reference, tail_k, tail_v, tail_positions));
                timing = time_dual(fast, slow, geometry, tail_k, tail_v, tail_positions,
                                   timed_fast, timed_slow, timed_remote, q_packet, state_packet);
            } else {
                timing = time_dual(fast, slow, geometry, tail_k, tail_v, tail_positions,
                                   timed_fast, timed_slow, timed_remote, q_packet, state_packet);
                full_us.push_back(time_full(reference, tail_k, tail_v, tail_positions));
            }
            dual_us.push_back(timing.total_us);
            q_us.push_back(timing.q_stage_us);
            engines_us.push_back(timing.engines_us);
            reduction_us.push_back(timing.reduction_us);
        }

        const std::size_t reference_payload = reference.payload_bytes();
        const std::size_t physical_payload = fast.payload_bytes() + slow.payload_bytes();
        const std::int32_t duplicate_pages =
            fast.physical_pages() + slow.physical_pages() - reference.logical_pages();
        std::printf(
            "gqa_sequence_sharded kv_dtype=%s window=%d total_splits=%d fast_splits=%d "
            "slow_splits=%d "
            "split_unit_keys=%d boundary_key=%d boundary_page=%d boundary_offset=%d "
            "logical_pages=%d fast_pages=%d slow_pages=%d duplicate_pages=%d "
            "reference_payload=%zu sharded_payload=%zu payload_ratio=%.9g "
            "fast_state_mismatches=%zu slow_state_mismatches=%zu "
            "canonical_output_mismatches=%zu canonical_output_max_abs=%.9g "
            "canonical_output_rel_l2=%.9g sharded_output_mismatches=%zu "
            "sharded_output_max_abs=%.9g sharded_output_rel_l2=%.9g "
            "full_median_us=%.3f dual_median_us=%.3f speedup=%.6f "
            "q_stage_median_us=%.3f engines_median_us=%.3f reduction_median_us=%.3f "
            "q_bytes=%zu state_bytes=%zu current_kv_bytes=%zu\n",
            options.kv_dtype == DType::BF16 ? "bf16" : "int8",
            options.window, geometry.total_splits, geometry.fast_splits, geometry.slow_splits,
            geometry.split_unit_keys, geometry.boundary_key,
            geometry.boundary_key / kPagedKVPageSize,
            geometry.boundary_key % kPagedKVPageSize, reference.logical_pages(),
            fast.physical_pages(), slow.physical_pages(), duplicate_pages,
            reference_payload, physical_payload,
            static_cast<double>(physical_payload) / static_cast<double>(reference_payload),
            fast_state_comparison.mismatches, slow_state_comparison.mismatches,
            canonical_output_comparison.mismatches, canonical_output_comparison.max_abs,
            canonical_output_comparison.relative_l2, sharded_output_comparison.mismatches,
            sharded_output_comparison.max_abs, sharded_output_comparison.relative_l2,
            median(full_us), median(dual_us), median(full_us) / median(dual_us),
            median(q_us), median(engines_us), median(reduction_us), q_packet.size(),
            state_packet.size(),
            static_cast<std::size_t>(2) * kHeadDim * kKvHeads * kTokens *
                sizeof(std::uint16_t));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
