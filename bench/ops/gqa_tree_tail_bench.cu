#include "core/device.h"
#include "ninfer/ops/gqa_tree.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQueryHeads = 24;
constexpr std::int32_t kKvHeads = 4;
constexpr std::int32_t kMaximumNodes = 32;
constexpr float kScale = 1.0F / 16.0F;
constexpr std::size_t kDefaultFlushBytes = 512ULL << 20;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct Options {
    int device = 0;
    int warmup = 5;
    int repeat = 40;
    std::size_t flush_bytes = kDefaultFlushBytes;
    std::vector<std::int32_t> budgets{4, 5, 6, 8, 12, 16, 22};
    std::filesystem::path csv_out;
};

[[noreturn]] void usage(const char* program, const char* error = nullptr) {
    if (error != nullptr) { std::fprintf(stderr, "error: %s\n\n", error); }
    std::fprintf(stderr,
                 "Usage: %s [--device N] [--budgets 4,5,6,8,12,16,22] "
                 "[--warmup N] [--repeat N] [--flush-mib N] [--csv-out PATH]\n",
                 program);
    std::exit(error == nullptr ? 0 : 2);
}

std::uint64_t parse_u64(std::string_view text, const char* label) {
    const std::string value(text);
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    return parsed;
}

int parse_int(std::string_view text, const char* label, int minimum, int maximum) {
    const std::uint64_t value = parse_u64(text, label);
    if (value < static_cast<std::uint64_t>(minimum) ||
        value > static_cast<std::uint64_t>(maximum)) {
        throw std::invalid_argument(std::string(label) + " out of range");
    }
    return static_cast<int>(value);
}

std::vector<std::int32_t> parse_budgets(std::string_view text) {
    std::vector<std::int32_t> values;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::string_view part = text.substr(
            begin, end == std::string_view::npos ? text.size() - begin : end - begin);
        values.push_back(parse_int(part, "budget", 2, kMaximumNodes));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto take = [&](const char* label) -> std::string_view {
            if (++index >= argc) { usage(argv[0], label); }
            return argv[index];
        };
        if (argument == "--device") {
            options.device = parse_int(take("device"), "device", 0, 63);
        } else if (argument == "--budgets") {
            options.budgets = parse_budgets(take("budgets"));
        } else if (argument == "--warmup") {
            options.warmup = parse_int(take("warmup"), "warmup", 0, 1000);
        } else if (argument == "--repeat") {
            options.repeat = parse_int(take("repeat"), "repeat", 1, 10000);
        } else if (argument == "--flush-mib") {
            options.flush_bytes = static_cast<std::size_t>(
                parse_u64(take("flush MiB"), "flush MiB")) << 20;
        } else if (argument == "--csv-out") {
            options.csv_out = take("CSV path");
        } else if (argument == "-h" || argument == "--help") {
            usage(argv[0]);
        } else {
            usage(argv[0], "unknown argument");
        }
    }
    if (options.budgets.empty()) { throw std::invalid_argument("empty budget set"); }
    return options;
}

std::uint16_t bf16_bits(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t bias = 0x7fffU + ((bits >> 16U) & 1U);
    bits += bias;
    return static_cast<std::uint16_t>(bits >> 16U);
}

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return value;
}

std::vector<std::uint16_t> make_bf16(std::size_t count, std::uint32_t seed,
                                     float magnitude) {
    std::vector<std::uint16_t> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float unit = static_cast<float>(mix(seed + static_cast<std::uint32_t>(index)) &
                                              0xffffU) /
                           65535.0F;
        values[index] = bf16_bits((unit * 2.0F - 1.0F) * magnitude);
    }
    return values;
}

std::vector<float> make_fp32(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float unit = static_cast<float>(mix(seed + static_cast<std::uint32_t>(index)) &
                                              0xffffU) /
                           65535.0F;
        values[index] = lo + (hi - lo) * unit;
    }
    return values;
}

std::vector<std::int32_t> make_parents(std::int32_t nodes) {
    const std::int32_t second_root = (nodes + 1) / 2;
    std::vector<std::int32_t> parents(static_cast<std::size_t>(nodes));
    for (std::int32_t node = 0; node < nodes; ++node) {
        parents[static_cast<std::size_t>(node)] =
            node == 0 || node == second_root ? -1 : node - 1;
    }
    return parents;
}

template <class T>
DeviceBuffer upload(const std::vector<T>& values) {
    DeviceBuffer buffer(values.size() * sizeof(T));
    buffer.copy_from_host(values.data(), buffer.bytes);
    return buffer;
}

struct Buffers {
    std::int32_t nodes;
    DeviceContext context;
    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer parents;
    DeviceBuffer tail_acc;
    DeviceBuffer tail_m;
    DeviceBuffer tail_l;
    DeviceBuffer prefix_acc;
    DeviceBuffer prefix_m;
    DeviceBuffer prefix_l;
    DeviceBuffer out;

    Tensor q_tensor;
    Tensor k_tensor;
    Tensor v_tensor;
    Tensor parent_tensor;
    Tensor tail_acc_tensor;
    Tensor tail_m_tensor;
    Tensor tail_l_tensor;
    Tensor prefix_acc_tensor;
    Tensor prefix_m_tensor;
    Tensor prefix_l_tensor;
    Tensor out_tensor;

    Buffers(int device, std::int32_t node_count)
        : nodes(node_count), context(device),
          q(upload(make_bf16(static_cast<std::size_t>(kHeadDim) * kQueryHeads * nodes,
                             8101U + nodes, 0.12F))),
          k(upload(make_bf16(static_cast<std::size_t>(kHeadDim) * kKvHeads * nodes,
                             8103U + nodes, 0.12F))),
          v(upload(make_bf16(static_cast<std::size_t>(kHeadDim) * kKvHeads * nodes,
                             8105U + nodes, 0.12F))),
          parents(upload(make_parents(nodes))),
          tail_acc(static_cast<std::size_t>(kHeadDim) * kQueryHeads * nodes * sizeof(float)),
          tail_m(static_cast<std::size_t>(kQueryHeads) * nodes * sizeof(float)),
          tail_l(tail_m.bytes),
          prefix_acc(upload(make_fp32(
              static_cast<std::size_t>(kHeadDim) * kQueryHeads * nodes,
              8107U + nodes, -0.8F, 0.8F))),
          prefix_m(upload(make_fp32(static_cast<std::size_t>(kQueryHeads) * nodes,
                                   8109U + nodes, -1.2F, 1.2F))),
          prefix_l(upload(make_fp32(static_cast<std::size_t>(kQueryHeads) * nodes,
                                   8111U + nodes, 0.2F, 4.0F))),
          out(static_cast<std::size_t>(kHeadDim) * kQueryHeads * nodes *
              sizeof(std::uint16_t)),
          q_tensor(q.p, DType::BF16, {kHeadDim, kQueryHeads, nodes}),
          k_tensor(k.p, DType::BF16, {kHeadDim, kKvHeads, nodes}),
          v_tensor(v.p, DType::BF16, {kHeadDim, kKvHeads, nodes}),
          parent_tensor(parents.p, DType::I32, {nodes}),
          tail_acc_tensor(tail_acc.p, DType::FP32, {kHeadDim, kQueryHeads, nodes}),
          tail_m_tensor(tail_m.p, DType::FP32, {kQueryHeads, nodes}),
          tail_l_tensor(tail_l.p, DType::FP32, {kQueryHeads, nodes}),
          prefix_acc_tensor(prefix_acc.p, DType::FP32, {kHeadDim, kQueryHeads, nodes}),
          prefix_m_tensor(prefix_m.p, DType::FP32, {kQueryHeads, nodes}),
          prefix_l_tensor(prefix_l.p, DType::FP32, {kQueryHeads, nodes}),
          out_tensor(out.p, DType::BF16, {kHeadDim, kQueryHeads, nodes}) {}
};

struct Result {
    std::int32_t nodes = 0;
    std::size_t graph_nodes = 0;
    bench::ColdTiming total{};
    float tail_ms = 0.0F;
    float merge_ms = 0.0F;
    std::size_t state_bytes = 0;
};

void launch(Buffers& buffers, cudaStream_t stream) {
    ops::gqa_attention_tree_tail_state(
        buffers.q_tensor, buffers.k_tensor, buffers.v_tensor, buffers.parent_tensor,
        kScale, buffers.tail_acc_tensor, buffers.tail_m_tensor, buffers.tail_l_tensor,
        stream);
    ops::gqa_attention_tree_merge_prefix(
        buffers.prefix_acc_tensor, buffers.prefix_m_tensor, buffers.prefix_l_tensor,
        buffers.tail_acc_tensor, buffers.tail_m_tensor, buffers.tail_l_tensor,
        buffers.out_tensor, stream);
}

Result run_budget(const Options& options, std::int32_t nodes, DeviceBuffer& flush) {
    Buffers buffers(options.device, nodes);
    const cudaStream_t stream = buffers.context.stream;
    launch(buffers, stream);
    buffers.context.synchronize();

    bench::TimedGraph graph;
    graph.capture(stream, [&](cudaStream_t capture_stream) { launch(buffers, capture_stream); });
    Result result;
    result.nodes = nodes;
    result.graph_nodes = graph.nodes();
    result.total = bench::measure_cold_graph(
        graph, flush, stream, options.warmup, options.repeat);

    cudaEvent_t begin = nullptr;
    cudaEvent_t middle = nullptr;
    cudaEvent_t end = nullptr;
    check(cudaEventCreate(&begin), "create begin event");
    check(cudaEventCreate(&middle), "create middle event");
    check(cudaEventCreate(&end), "create end event");
    check(cudaEventRecord(begin, stream), "record begin");
    ops::gqa_attention_tree_tail_state(
        buffers.q_tensor, buffers.k_tensor, buffers.v_tensor, buffers.parent_tensor,
        kScale, buffers.tail_acc_tensor, buffers.tail_m_tensor, buffers.tail_l_tensor,
        stream);
    check(cudaEventRecord(middle, stream), "record middle");
    ops::gqa_attention_tree_merge_prefix(
        buffers.prefix_acc_tensor, buffers.prefix_m_tensor, buffers.prefix_l_tensor,
        buffers.tail_acc_tensor, buffers.tail_m_tensor, buffers.tail_l_tensor,
        buffers.out_tensor, stream);
    check(cudaEventRecord(end, stream), "record end");
    check(cudaEventSynchronize(end), "wait end");
    check(cudaEventElapsedTime(&result.tail_ms, begin, middle), "tail elapsed");
    check(cudaEventElapsedTime(&result.merge_ms, middle, end), "merge elapsed");
    check(cudaEventDestroy(begin), "destroy begin");
    check(cudaEventDestroy(middle), "destroy middle");
    check(cudaEventDestroy(end), "destroy end");
    result.state_bytes = buffers.tail_acc.bytes + buffers.tail_m.bytes +
                         buffers.tail_l.bytes;
    return result;
}

void print_result(const Result& result) {
    std::printf(
        "budget=%d graph_nodes=%zu total_median_us=%.3f total_min_us=%.3f "
        "total_p95_us=%.3f tail_us=%.3f merge_us=%.3f state_bytes=%zu\n",
        result.nodes, result.graph_nodes, result.total.median_us, result.total.min_us,
        result.total.p95_us, result.tail_ms * 1000.0F,
        result.merge_ms * 1000.0F, result.state_bytes);
}

void write_csv(const Options& options, std::span<const Result> results) {
    if (options.csv_out.empty()) { return; }
    if (!options.csv_out.parent_path().empty()) {
        std::filesystem::create_directories(options.csv_out.parent_path());
    }
    std::ofstream out(options.csv_out);
    if (!out) { throw std::runtime_error("failed to open CSV"); }
    out << "budget,graph_nodes,total_median_us,total_min_us,total_p95_us,tail_us,merge_us,"
           "state_bytes\n";
    for (const Result& result : results) {
        out << result.nodes << ',' << result.graph_nodes << ',' << result.total.median_us << ','
            << result.total.min_us << ',' << result.total.p95_us << ','
            << result.tail_ms * 1000.0F << ',' << result.merge_ms * 1000.0F << ','
            << result.state_bytes << '\n';
    }
}

int run(const Options& options) {
    check(cudaSetDevice(options.device), "set benchmark device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, options.device), "device properties");
    std::printf("device=%d name=%s sm=%d memory_bytes=%zu\n", options.device,
                properties.name, properties.multiProcessorCount, properties.totalGlobalMem);
    DeviceBuffer flush(options.flush_bytes);
    std::vector<Result> results;
    for (const std::int32_t budget : options.budgets) {
        Result result = run_budget(options, budget, flush);
        print_result(result);
        results.push_back(result);
    }
    write_csv(options, results);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::fprintf(stderr, "no CUDA device available\n");
            return 77;
        }
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gqa_tree_tail_bench: %s\n", error.what());
        return 1;
    }
}
