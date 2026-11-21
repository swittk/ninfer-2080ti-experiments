#include "ninfer_bench_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQueryHeads = 24;
constexpr std::int32_t kKvHeads = 4;
constexpr std::int32_t kQuantGroup = 64;
constexpr std::int32_t kScaleRows = kHeadDim / kQuantGroup;
constexpr float kAttentionScale = 0.0625F;

enum class CacheDType { Int8, Bf16 };
enum class Mode { Append, Cached };
enum class Execution { Eager, Graph };

struct Options {
    int device = 0;
    std::int32_t tokens = 3;
    std::int32_t window = 8192;
    CacheDType dtype = CacheDType::Int8;
    Mode mode = Mode::Append;
    Execution execution = Execution::Eager;
    int warmup = 20;
    int repeat = 100;
    std::uint64_t flush_mib = 0;
};

std::uint64_t parse_u64(std::string_view text, const char* label) {
    const std::string value(text);
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + value);
    }
    return static_cast<std::uint64_t>(parsed);
}

int parse_int(std::string_view text, const char* label, int minimum, int maximum) {
    const std::uint64_t value = parse_u64(text, label);
    if (value < static_cast<std::uint64_t>(minimum) ||
        value > static_cast<std::uint64_t>(maximum)) {
        throw std::invalid_argument(std::string(label) + " is outside the supported range");
    }
    return static_cast<int>(value);
}

[[noreturn]] void usage(const char* argv0, const char* error = nullptr) {
    if (error != nullptr) std::fprintf(stderr, "error: %s\n\n", error);
    std::fprintf(stderr,
                 "Usage: %s [options]\n\n"
                 "  --device N              CUDA device (default 0)\n"
                 "  --tokens N              query width T1..T6 (default 3)\n"
                 "  --window N              visible keys 1..262144 (default 8192)\n"
                 "  --dtype int8|bf16       cache storage (default int8)\n"
                 "  --mode append|cached    production append or cache-only (default append)\n"
                 "  --execution eager|graph launch mode (default eager)\n"
                 "  --warmup N              warmup iterations (default 20)\n"
                 "  --repeat N              measured iterations (default 100)\n"
                 "  --flush-mib N           cold-cache flush allocation (default 0)\n",
                 argv0);
    std::exit(error == nullptr ? 0 : 2);
}

Options parse_options(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto take = [&](const char* label) -> std::string_view {
            if (++i >= argc) usage(argv[0], (std::string("missing ") + label).c_str());
            return argv[i];
        };
        if (arg == "--device") {
            out.device = parse_int(take("device"), "device", 0, 63);
        } else if (arg == "--tokens") {
            out.tokens = parse_int(take("tokens"), "tokens", 1, 6);
        } else if (arg == "--window") {
            out.window = parse_int(take("window"), "window", 1,
                                   static_cast<int>(ops::kGqaAttentionMaximumVisibleKeys));
        } else if (arg == "--dtype") {
            const auto value = take("dtype");
            if (value == "int8") out.dtype = CacheDType::Int8;
            else if (value == "bf16") out.dtype = CacheDType::Bf16;
            else usage(argv[0], "--dtype must be int8 or bf16");
        } else if (arg == "--mode") {
            const auto value = take("mode");
            if (value == "append") out.mode = Mode::Append;
            else if (value == "cached") out.mode = Mode::Cached;
            else usage(argv[0], "--mode must be append or cached");
        } else if (arg == "--execution") {
            const auto value = take("execution");
            if (value == "eager") out.execution = Execution::Eager;
            else if (value == "graph") out.execution = Execution::Graph;
            else usage(argv[0], "--execution must be eager or graph");
        } else if (arg == "--warmup") {
            out.warmup = parse_int(take("warmup"), "warmup", 0, 1000000);
        } else if (arg == "--repeat") {
            out.repeat = parse_int(take("repeat"), "repeat", 1, 1000000);
        } else if (arg == "--flush-mib") {
            out.flush_mib = parse_u64(take("flush-mib"), "flush-mib");
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else {
            usage(argv[0], (std::string("unknown argument: ") + std::string(arg)).c_str());
        }
    }
    if (out.window < out.tokens) usage(argv[0], "window must be >= tokens");
    if (out.flush_mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
        usage(argv[0], "flush allocation is too large");
    }
    return out;
}

std::size_t checked_elements(std::initializer_list<std::uint64_t> dimensions) {
    std::uint64_t elements = 1;
    for (const std::uint64_t dimension : dimensions) {
        if (dimension == 0 || elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
            throw std::overflow_error("benchmark tensor size overflow");
        }
        elements *= dimension;
    }
    if (elements > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("benchmark tensor does not fit size_t");
    }
    return static_cast<std::size_t>(elements);
}

std::vector<std::uint16_t> bf16_ramp(std::size_t elements, std::uint32_t salt) {
    std::vector<std::uint16_t> out(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        const std::uint32_t mixed = static_cast<std::uint32_t>(i) * 1664525U + 1013904223U + salt;
        const float value = (static_cast<float>(mixed & 1023U) / 1023.0F - 0.5F) * 0.5F;
        out[i] = ninfer::bench::f32_to_bf16(value);
    }
    return out;
}

class Fixture {
public:
    explicit Fixture(const Options& options)
        : options_(options), device_(options.device),
          pages_((options.window + kPagedKVPageSize - 1) / kPagedKVPageSize),
          code_elements_(checked_elements({static_cast<std::uint64_t>(kHeadDim),
                                           static_cast<std::uint64_t>(kPagedKVPageSize),
                                           static_cast<std::uint64_t>(kKvHeads),
                                           static_cast<std::uint64_t>(pages_)})),
          scale_elements_(checked_elements({static_cast<std::uint64_t>(kScaleRows),
                                            static_cast<std::uint64_t>(kPagedKVPageSize),
                                            static_cast<std::uint64_t>(kKvHeads),
                                            static_cast<std::uint64_t>(pages_)})),
          cache_dtype_(options.dtype == CacheDType::Int8 ? DType::I8 : DType::BF16),
          k_cache_(code_elements_ * (cache_dtype_ == DType::I8 ? 1 : 2)),
          v_cache_(code_elements_ * (cache_dtype_ == DType::I8 ? 1 : 2)),
          k_scale_(cache_dtype_ == DType::I8 ? scale_elements_ * 2 : 1),
          v_scale_(cache_dtype_ == DType::I8 ? scale_elements_ * 2 : 1),
          block_table_(static_cast<std::size_t>(pages_) * sizeof(std::int32_t)),
          q_(checked_elements({kHeadDim, kQueryHeads,
                               static_cast<std::uint64_t>(options.tokens)}) * 2),
          k_(checked_elements({kHeadDim, kKvHeads,
                               static_cast<std::uint64_t>(options.tokens)}) * 2),
          v_(checked_elements({kHeadDim, kKvHeads,
                               static_cast<std::uint64_t>(options.tokens)}) * 2),
          positions_(static_cast<std::size_t>(options.tokens) * sizeof(std::int32_t)),
          table_row_(sizeof(std::int32_t)),
          out_(checked_elements({kHeadDim, kQueryHeads,
                                 static_cast<std::uint64_t>(options.tokens)}) * 2),
          workspace_bytes_(ops::gqa_attention_workspace_capacity_bytes(
              kQueryHeads, cache_dtype_,
              ops::GqaExecutionEnvelope{static_cast<std::uint32_t>(options.window),
                                        static_cast<std::uint32_t>(options.window)},
              1, options.tokens, options.tokens)),
          workspace_storage_(std::max<std::size_t>(workspace_bytes_, 256)),
          workspace_(DeviceSpan{workspace_storage_.p, workspace_storage_.bytes}),
          flush_(options.flush_mib == 0 ? 1 : static_cast<std::size_t>(options.flush_mib << 20)) {
        initialise();
    }

    void launch(cudaStream_t stream) {
        workspace_.reset();
        const auto envelope = ops::GqaExecutionEnvelope{
            static_cast<std::uint32_t>(options_.window),
            static_cast<std::uint32_t>(options_.window)};
        Tensor out = out_tensor();
        if (options_.mode == Mode::Append) {
            Tensor no_valid;
            ops::gqa_attention(q_tensor(), k_tensor(), v_tensor(), positions_tensor(), no_valid,
                               table_row_tensor(), kAttentionScale, batch_cache(), envelope,
                               workspace_, out, stream);
        } else {
            auto cache = direct_cache();
            ops::gqa_attention_cached(q_tensor(), positions_tensor(), kAttentionScale, cache,
                                      envelope, workspace_, out, stream);
        }
        workspace_.reset();
    }

    [[nodiscard]] cudaStream_t stream() const noexcept { return device_.stream; }
    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }
    [[nodiscard]] std::size_t cache_bytes() const noexcept {
        const std::size_t code_bytes = code_elements_ * (cache_dtype_ == DType::I8 ? 1 : 2);
        const std::size_t scale_bytes = cache_dtype_ == DType::I8 ? scale_elements_ * 2 : 0;
        return 2 * (code_bytes + scale_bytes);
    }
    [[nodiscard]] DeviceBuffer& flush() noexcept { return flush_; }

private:
    void initialise() {
        k_cache_.fill(0x11);
        v_cache_.fill(0x2d);
        if (cache_dtype_ == DType::I8) {
            std::vector<std::uint16_t> scales(scale_elements_, 0x3c00U);
            k_scale_.copy_from_host(scales.data(), scales.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(scales.data(), scales.size() * sizeof(std::uint16_t));
        }

        std::vector<std::int32_t> table(static_cast<std::size_t>(pages_));
        std::iota(table.begin(), table.end(), 0);
        block_table_.copy_from_host(table.data(), table.size() * sizeof(std::int32_t));

        const auto q_host = bf16_ramp(checked_elements(
                                           {kHeadDim, kQueryHeads,
                                            static_cast<std::uint64_t>(options_.tokens)}),
                                       17U);
        const auto k_host = bf16_ramp(checked_elements(
                                           {kHeadDim, kKvHeads,
                                            static_cast<std::uint64_t>(options_.tokens)}),
                                       29U);
        const auto v_host = bf16_ramp(checked_elements(
                                           {kHeadDim, kKvHeads,
                                            static_cast<std::uint64_t>(options_.tokens)}),
                                       43U);
        q_.copy_from_host(q_host.data(), q_host.size() * sizeof(std::uint16_t));
        k_.copy_from_host(k_host.data(), k_host.size() * sizeof(std::uint16_t));
        v_.copy_from_host(v_host.data(), v_host.size() * sizeof(std::uint16_t));

        std::vector<std::int32_t> positions(static_cast<std::size_t>(options_.tokens));
        const std::int32_t begin = options_.window - options_.tokens;
        for (std::int32_t token = 0; token < options_.tokens; ++token) {
            positions[static_cast<std::size_t>(token)] = begin + token;
        }
        positions_.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
        const std::int32_t row = 0;
        table_row_.copy_from_host(&row, sizeof(row));
        out_.fill(0);
        flush_.fill(0xa5);
        device_.synchronize();
    }

    [[nodiscard]] Tensor q_tensor() {
        return Tensor(q_.p, DType::BF16, {kHeadDim, kQueryHeads, options_.tokens});
    }
    [[nodiscard]] Tensor k_tensor() {
        return Tensor(k_.p, DType::BF16, {kHeadDim, kKvHeads, options_.tokens});
    }
    [[nodiscard]] Tensor v_tensor() {
        return Tensor(v_.p, DType::BF16, {kHeadDim, kKvHeads, options_.tokens});
    }
    [[nodiscard]] Tensor positions_tensor() {
        return Tensor(positions_.p, DType::I32, {options_.tokens});
    }
    [[nodiscard]] Tensor table_row_tensor() { return Tensor(table_row_.p, DType::I32, {1}); }
    [[nodiscard]] Tensor out_tensor() {
        return Tensor(out_.p, DType::BF16, {kHeadDim, kQueryHeads, options_.tokens});
    }

    [[nodiscard]] PagedKVLayerView direct_cache() {
        return {
            .k_pages = Tensor(k_cache_.p, cache_dtype_,
                              {kHeadDim, kPagedKVPageSize, kKvHeads, pages_}),
            .v_pages = Tensor(v_cache_.p, cache_dtype_,
                              {kHeadDim, kPagedKVPageSize, kKvHeads, pages_}),
            .k_scale_pages = cache_dtype_ == DType::I8
                                 ? Tensor(k_scale_.p, DType::FP16,
                                          {kScaleRows, kPagedKVPageSize, kKvHeads, pages_})
                                 : Tensor{},
            .v_scale_pages = cache_dtype_ == DType::I8
                                 ? Tensor(v_scale_.p, DType::FP16,
                                          {kScaleRows, kPagedKVPageSize, kKvHeads, pages_})
                                 : Tensor{},
            .block_table = Tensor(block_table_.p, DType::I32, {pages_}),
            .head_dim = kHeadDim,
            .num_kv_heads = kKvHeads,
            .dtype = cache_dtype_,
            .quant_group = cache_dtype_ == DType::I8 ? kQuantGroup : 0,
        };
    }

    [[nodiscard]] PagedKVBatchLayerView batch_cache() {
        const auto cache = direct_cache();
        return {
            .k_pages = cache.k_pages,
            .v_pages = cache.v_pages,
            .k_scale_pages = cache.k_scale_pages,
            .v_scale_pages = cache.v_scale_pages,
            .block_tables = cache.block_table.view({pages_, 1}),
            .head_dim = cache.head_dim,
            .num_kv_heads = cache.num_kv_heads,
            .dtype = cache.dtype,
            .quant_group = cache.quant_group,
        };
    }

    Options options_;
    DeviceContext device_;
    std::int32_t pages_;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    DType cache_dtype_;
    DeviceBuffer k_cache_;
    DeviceBuffer v_cache_;
    DeviceBuffer k_scale_;
    DeviceBuffer v_scale_;
    DeviceBuffer block_table_;
    DeviceBuffer q_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer positions_;
    DeviceBuffer table_row_;
    DeviceBuffer out_;
    std::size_t workspace_bytes_;
    DeviceBuffer workspace_storage_;
    WorkspaceArena workspace_;
    DeviceBuffer flush_;
};

const char* dtype_name(CacheDType dtype) { return dtype == CacheDType::Int8 ? "int8" : "bf16"; }
const char* mode_name(Mode mode) { return mode == Mode::Append ? "append" : "cached"; }
const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        CUDA_CHECK(cudaSetDevice(options.device));
        Fixture fixture(options);
        const auto launch = [&](cudaStream_t stream) { fixture.launch(stream); };

        ninfer::bench::ColdTiming timing;
        if (options.execution == Execution::Graph) {
            fixture.launch(fixture.stream());
            CUDA_CHECK(cudaStreamSynchronize(fixture.stream()));
            ninfer::bench::TimedGraph graph;
            graph.capture(fixture.stream(), launch);
            timing = options.flush_mib == 0
                         ? ninfer::bench::measure_graph(graph, fixture.stream(), options.warmup,
                                                       options.repeat)
                         : ninfer::bench::measure_cold_graph(graph, fixture.flush(), fixture.stream(),
                                                            options.warmup, options.repeat);
        } else {
            timing = options.flush_mib == 0
                         ? ninfer::bench::measure_launch(launch, fixture.stream(), options.warmup,
                                                        options.repeat)
                         : ninfer::bench::measure_cold_launch(launch, fixture.flush(),
                                                             fixture.stream(), options.warmup,
                                                             options.repeat);
        }

        const double seconds = timing.median_us * 1e-6;
        const double gbs = seconds > 0.0 ? static_cast<double>(fixture.cache_bytes()) / seconds / 1e9
                                         : 0.0;
        std::printf(
            "gqa27 dtype=%s mode=%s execution=%s T=%d window=%d median_us=%.3f min_us=%.3f "
            "p95_us=%.3f cache_GB=%.6f effective_GBps=%.3f workspace_bytes=%zu\n",
            dtype_name(options.dtype), mode_name(options.mode), execution_name(options.execution),
            options.tokens, options.window, timing.median_us, timing.min_us, timing.p95_us,
            static_cast<double>(fixture.cache_bytes()) / 1e9, gbs, fixture.workspace_bytes());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
