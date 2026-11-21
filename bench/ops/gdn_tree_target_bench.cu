// Real Qwen3.8-27B 48-layer GDN tree-target budget benchmark.
//
// The benchmark loads the represented production weights from MODEL.ninfer and executes the
// complete GDN branch for every flattened tree node: input norm/control, Q4+Q5 projection,
// parent-indexed convolution, parent-indexed normalized GDN recurrence, gated RMSNorm, Q5 output
// projection/residual, compact selected-path record gather, and one all-layer ReplaySSM fold.
// It intentionally excludes full-attention and MLP layers; the emitted component costs are inputs
// to the complete target-round budget model.

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/gdn_tree.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer_bench_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace ninfer;

constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kQkHeads = 16;
constexpr std::int32_t kValueHeads = 48;
constexpr std::int32_t kHeadDim = 128;
constexpr std::int32_t kKeyRows = kQkHeads * kHeadDim;
constexpr std::int32_t kValueRows = kValueHeads * kHeadDim;
constexpr std::int32_t kConvRows = 2 * kKeyRows + kValueRows;
constexpr std::int32_t kGdnLayers = 48;
constexpr std::int32_t kMaximumNodes = 22;
constexpr float kEps = 1.0e-6F;
constexpr float kGdnScale = 0.08838834764831845F;
constexpr std::size_t kDefaultFlushBytes = 512ULL << 20;
constexpr std::size_t kWorkspaceBytes = 256ULL << 20;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct Options {
    std::filesystem::path artifact;
    std::filesystem::path csv_out;
    int device = 0;
    int warmup = 2;
    int repeat = 8;
    std::size_t flush_bytes = kDefaultFlushBytes;
    std::vector<std::int32_t> budgets{8, 12, 16, 22};
};

[[noreturn]] void usage(const char* program, const char* error = nullptr) {
    if (error != nullptr) { std::fprintf(stderr, "error: %s\n\n", error); }
    std::fprintf(stderr,
                 "Usage: %s MODEL.ninfer [--device N] [--budgets 8,12,16,22] "
                 "[--warmup N] [--repeat N] [--flush-mib N] [--csv-out PATH]\n",
                 program);
    std::exit(error == nullptr ? 0 : 2);
}

std::uint64_t parse_u64(std::string_view text, const char* label) {
    const std::string value(text);
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + value);
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
        const std::string_view part =
            text.substr(begin, end == std::string_view::npos ? text.size() - begin : end - begin);
        values.push_back(parse_int(part, "budget", 2, kMaximumNodes));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    if (values.empty()) { throw std::invalid_argument("budget set is empty"); }
    return values;
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) { usage(argv[0], "missing artifact"); }
    Options options;
    options.artifact = argv[1];
    for (int index = 2; index < argc; ++index) {
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
            options.warmup = parse_int(take("warmup"), "warmup", 0, 100);
        } else if (argument == "--repeat") {
            options.repeat = parse_int(take("repeat"), "repeat", 1, 100);
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
    if (!std::filesystem::is_regular_file(options.artifact)) {
        throw std::invalid_argument("artifact is missing");
    }
    return options;
}

std::string layer_prefix(std::int32_t layer) {
    return "text/layers/" + std::to_string(layer) + "/";
}

std::vector<std::int32_t> gdn_layer_ids() {
    std::vector<std::int32_t> values;
    for (std::int32_t layer = 0; layer < 64; ++layer) {
        if (layer % 4 != 3) { values.push_back(layer); }
    }
    if (values.size() != kGdnLayers) { throw std::logic_error("invalid GDN layer inventory"); }
    return values;
}

const artifact::TensorDescriptor& require_tensor(const artifact::Reader& reader,
                                                  std::string_view name,
                                                  artifact::NumericFormat format,
                                                  std::initializer_list<std::uint64_t> shape) {
    const auto* object = reader.find(name);
    if (object == nullptr) { throw std::invalid_argument("missing " + std::string(name)); }
    const auto* tensor = std::get_if<artifact::TensorDescriptor>(object);
    if (tensor == nullptr || tensor->format != format ||
        tensor->shape != std::vector<std::uint64_t>(shape)) {
        throw std::invalid_argument("tensor contract mismatch: " + std::string(name));
    }
    return *tensor;
}

template <class T>
std::vector<T> payload(const artifact::Reader& reader,
                       const artifact::TensorDescriptor& descriptor) {
    const auto bytes = reader.payload(descriptor);
    if (bytes.data.size() % sizeof(T) != 0) {
        throw std::invalid_argument("payload alignment mismatch: " + descriptor.name);
    }
    std::vector<T> values(bytes.data.size() / sizeof(T));
    std::memcpy(values.data(), bytes.data.data(), bytes.data.size());
    return values;
}

struct HostPacked {
    QType qtype = QType::Q4G64_F16S;
    std::int32_t n = 0;
    std::int32_t k = 0;
    std::int32_t padded_k = 0;
    std::int32_t group = 64;
    std::vector<std::byte> low;
    std::vector<std::byte> high;
    std::vector<std::byte> scales;
};

HostPacked load_rowsplit(const artifact::Reader& reader, std::string_view name,
                         artifact::NumericFormat format,
                         std::initializer_list<std::uint64_t> shape) {
    const auto& descriptor = require_tensor(reader, name, format, shape);
    if (descriptor.layout != artifact::StorageLayout::RowSplitK128V1) {
        throw std::invalid_argument("tensor is not RowSplitK128V1: " + std::string(name));
    }
    const artifact::RowSplitGeometry geometry = artifact::row_split_geometry(format, descriptor.shape);
    const auto bytes = reader.payload(descriptor);
    if (bytes.data.size() != geometry.encoded_bytes) {
        throw std::invalid_argument("encoded size mismatch: " + std::string(name));
    }
    HostPacked result;
    result.qtype = format == artifact::NumericFormat::Q4G64_F16S
        ? QType::Q4G64_F16S
        : QType::Q5G64_F16S;
    result.n = geometry.rows;
    result.k = geometry.columns;
    result.padded_k = geometry.padded_columns;
    result.group = geometry.group_size;
    result.low.assign(bytes.data.begin(), bytes.data.begin() + geometry.low_plane_bytes);
    if (geometry.high_plane_bytes != 0) {
        result.high.assign(bytes.data.begin() + geometry.high_plane_offset,
                           bytes.data.begin() + geometry.high_plane_offset +
                               geometry.high_plane_bytes);
    }
    result.scales.assign(bytes.data.begin() + geometry.scale_plane_offset,
                         bytes.data.begin() + geometry.scale_plane_offset +
                             geometry.scale_plane_bytes);
    return result;
}

struct HostLayer {
    std::vector<std::uint16_t> input_norm;
    std::vector<std::uint16_t> a;
    std::vector<std::uint16_t> b;
    std::vector<std::uint16_t> convolution;
    std::vector<std::uint16_t> output_norm;
    std::vector<float> a_log;
    std::vector<float> dt_bias;
    HostPacked qk;
    HostPacked value_z;
    HostPacked output;
};

HostLayer load_layer(const artifact::Reader& reader, std::int32_t layer) {
    const std::string prefix = layer_prefix(layer);
    HostLayer result;
    result.input_norm = payload<std::uint16_t>(
        reader, require_tensor(reader, prefix + "input_norm", artifact::NumericFormat::BF16,
                               {kHidden}));
    result.a_log = payload<float>(
        reader, require_tensor(reader, prefix + "gdn/a_log", artifact::NumericFormat::FP32,
                               {kValueHeads}));
    result.dt_bias = payload<float>(
        reader, require_tensor(reader, prefix + "gdn/dt_bias", artifact::NumericFormat::FP32,
                               {kValueHeads}));
    result.convolution = payload<std::uint16_t>(
        reader, require_tensor(reader, prefix + "gdn/convolution",
                               artifact::NumericFormat::BF16, {4, kConvRows}));
    result.a = payload<std::uint16_t>(
        reader, require_tensor(reader, prefix + "gdn/a_projection",
                               artifact::NumericFormat::BF16, {kValueHeads, kHidden}));
    result.b = payload<std::uint16_t>(
        reader, require_tensor(reader, prefix + "gdn/b_projection",
                               artifact::NumericFormat::BF16, {kValueHeads, kHidden}));
    result.qk = load_rowsplit(reader, prefix + "gdn/query_key",
                              artifact::NumericFormat::Q4G64_F16S,
                              {2 * kKeyRows, kHidden});
    result.value_z = load_rowsplit(reader, prefix + "gdn/value_z",
                                   artifact::NumericFormat::Q5G64_F16S,
                                   {2 * kValueRows, kHidden});
    result.output_norm = payload<std::uint16_t>(
        reader, require_tensor(reader, prefix + "gdn/norm", artifact::NumericFormat::BF16,
                               {kHeadDim}));
    result.output = load_rowsplit(reader, prefix + "gdn/output",
                                  artifact::NumericFormat::Q5G64_F16S,
                                  {kHidden, kValueRows});
    return result;
}

template <class T>
struct DeviceArray {
    DeviceBuffer bytes;
    explicit DeviceArray(const std::vector<T>& host)
        : bytes(host.size() * sizeof(T)) {
        bytes.copy_from_host(host.data(), bytes.bytes);
    }
    T* data() { return static_cast<T*>(bytes.p); }
};

struct DevicePacked {
    DeviceBuffer bytes;
    Weight weight{};
    explicit DevicePacked(const HostPacked& host)
        : bytes(host.low.size() + host.high.size() + host.scales.size()) {
        const std::size_t low_bytes = host.low.size();
        const std::size_t high_bytes = host.high.size();
        const std::size_t scale_bytes = host.scales.size();
        bytes.copy_from_host(host.low.data(), low_bytes);
        if (high_bytes != 0) {
            bytes.copy_from_host(host.high.data(), high_bytes, low_bytes);
        }
        bytes.copy_from_host(host.scales.data(), scale_bytes, low_bytes + high_bytes);
        auto* base = static_cast<std::byte*>(bytes.p);
        weight.payload = base;
        weight.payload_bytes = bytes.bytes;
        weight.high_plane_bytes = high_bytes;
        weight.qtype = host.qtype;
        weight.group_size = host.group;
        weight.qdata = base;
        weight.qhigh = high_bytes == 0 ? nullptr : base + low_bytes;
        weight.scales = base + low_bytes + high_bytes;
        weight.n = host.n;
        weight.k = host.k;
        weight.group = host.group;
        weight.layout = QuantLayout::RowSplit;
        weight.scale_dtype = DType::FP16;
        weight.ndim = 2;
        weight.shape[0] = host.n;
        weight.shape[1] = host.k;
        weight.padded_shape[0] = host.n;
        weight.padded_shape[1] = host.padded_k;
    }
};

Weight make_bf16_weight(DeviceArray<std::uint16_t>& storage, std::int32_t rows,
                        std::int32_t columns) {
    Weight weight{};
    weight.payload = storage.bytes.p;
    weight.qdata = storage.bytes.p;
    weight.payload_bytes = storage.bytes.bytes;
    weight.qtype = QType::BF16_CTRL;
    weight.layout = QuantLayout::Contiguous;
    weight.n = rows;
    weight.k = columns;
    weight.ndim = 2;
    weight.shape[0] = weight.padded_shape[0] = rows;
    weight.shape[1] = weight.padded_shape[1] = columns;
    return weight;
}

struct DeviceLayer {
    DeviceArray<std::uint16_t> input_norm;
    DeviceArray<std::uint16_t> a;
    DeviceArray<std::uint16_t> b;
    DeviceArray<std::uint16_t> convolution;
    DeviceArray<std::uint16_t> output_norm;
    DeviceArray<float> a_log;
    DeviceArray<float> dt_bias;
    DevicePacked qk;
    DevicePacked value_z;
    DevicePacked output;
    Weight a_weight{};
    Weight b_weight{};

    explicit DeviceLayer(const HostLayer& host)
        : input_norm(host.input_norm),
          a(host.a),
          b(host.b),
          convolution(host.convolution),
          output_norm(host.output_norm),
          a_log(host.a_log),
          dt_bias(host.dt_bias),
          qk(host.qk),
          value_z(host.value_z),
          output(host.output) {
        a_weight = make_bf16_weight(a, kValueHeads, kHidden);
        b_weight = make_bf16_weight(b, kValueHeads, kHidden);
    }

    Tensor input_norm_tensor() { return Tensor(input_norm.bytes.p, DType::BF16, {kHidden}); }
    Tensor convolution_tensor() {
        return Tensor(convolution.bytes.p, DType::BF16, {kConvRows, 4});
    }
    Tensor output_norm_tensor() {
        return Tensor(output_norm.bytes.p, DType::BF16, {kHeadDim});
    }
    Tensor a_log_tensor() { return Tensor(a_log.bytes.p, DType::FP32, {kValueHeads}); }
    Tensor dt_bias_tensor() { return Tensor(dt_bias.bytes.p, DType::FP32, {kValueHeads}); }
};

std::vector<std::unique_ptr<DeviceLayer>> load_model(const Options& options) {
    check(cudaSetDevice(options.device), "set model device");
    artifact::Reader reader(options.artifact);
    std::vector<std::unique_ptr<DeviceLayer>> layers;
    layers.reserve(kGdnLayers);
    const std::vector<std::int32_t> ids = gdn_layer_ids();
    for (std::size_t index = 0; index < ids.size(); ++index) {
        layers.push_back(std::make_unique<DeviceLayer>(load_layer(reader, ids[index])));
        std::fprintf(stderr, "load gdn_layer=%d ordinal=%zu/%zu\n", ids[index], index + 1,
                     ids.size());
    }
    return layers;
}

struct LayerEvents {
    std::array<cudaEvent_t, 7> event{};
    LayerEvents() {
        for (cudaEvent_t& value : event) { check(cudaEventCreate(&value), "create layer event"); }
    }
    ~LayerEvents() {
        for (cudaEvent_t value : event) {
            if (value != nullptr) { (void)cudaEventDestroy(value); }
        }
    }
};

std::vector<std::int32_t> make_parents(std::int32_t nodes) {
    const std::int32_t first_branch = (nodes + 1) / 2;
    std::vector<std::int32_t> parents(static_cast<std::size_t>(nodes));
    for (std::int32_t node = 0; node < nodes; ++node) {
        parents[static_cast<std::size_t>(node)] =
            node == 0 || node == first_branch ? -1 : node - 1;
    }
    return parents;
}

std::vector<std::int32_t> make_selected_path(std::int32_t nodes) {
    const std::int32_t first_branch = (nodes + 1) / 2;
    const std::int32_t second_length = nodes - first_branch;
    const std::int32_t begin = second_length >= first_branch ? first_branch : 0;
    const std::int32_t length = std::max(first_branch, second_length);
    std::vector<std::int32_t> path(static_cast<std::size_t>(length));
    std::iota(path.begin(), path.end(), begin);
    return path;
}

std::vector<std::uint16_t> make_seed(std::int32_t nodes) {
    std::vector<std::uint16_t> values(static_cast<std::size_t>(kHidden) * nodes);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint32_t bits = static_cast<std::uint32_t>(index) * 1664525U + 1013904223U;
        const float value = (static_cast<float>(bits & 2047U) / 2047.0F - 0.5F) * 0.35F;
        values[index] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
    }
    return values;
}

struct Buffers {
    std::int32_t nodes;
    std::int32_t path_length;
    std::vector<std::int32_t> selected_path;
    DeviceContext context;
    WorkspaceArena workspace;
    DeviceBuffer seed;
    DeviceBuffer residual;
    DeviceBuffer hidden;
    DeviceBuffer projected;
    DeviceBuffer z;
    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer g;
    DeviceBuffer beta;
    DeviceBuffer node_history;
    DeviceBuffer node_states;
    DeviceBuffer recurrent;
    DeviceBuffer gated;
    DeviceBuffer parents;
    DeviceBuffer path;
    LayoutBuilder state_builder;
    LinearAttentionStatePoolLayout state_layout;
    DeviceBuffer state_storage;
    LinearAttentionStatePool states;
    LayoutBuilder fold_state_builder;
    LinearAttentionStatePoolLayout fold_state_layout;
    DeviceBuffer fold_state_storage;
    LinearAttentionStatePool fold_states;
    LayoutBuilder record_builder;
    GdnReplayRecordLayout record_layout;
    DeviceBuffer record_storage;
    GdnReplayRecords records;
    std::vector<std::unique_ptr<LayerEvents>> events;

    Buffers(int device, std::int32_t node_count)
        : nodes(node_count),
          path_length(static_cast<std::int32_t>(make_selected_path(node_count).size())),
          selected_path(make_selected_path(node_count)),
          context(device),
          workspace(kWorkspaceBytes),
          seed(static_cast<std::size_t>(kHidden) * nodes * sizeof(std::uint16_t)),
          residual(seed.bytes),
          hidden(seed.bytes),
          projected(static_cast<std::size_t>(kConvRows) * nodes * sizeof(std::uint16_t)),
          z(static_cast<std::size_t>(kValueRows) * nodes * sizeof(std::uint16_t)),
          query(static_cast<std::size_t>(kKeyRows) * nodes * sizeof(std::uint16_t)),
          key(query.bytes),
          value(static_cast<std::size_t>(kValueRows) * nodes * sizeof(std::uint16_t)),
          g(static_cast<std::size_t>(kValueHeads) * nodes * sizeof(float)),
          beta(g.bytes),
          node_history(static_cast<std::size_t>(kConvRows) * 3 * nodes *
                       sizeof(std::uint16_t)),
          node_states(static_cast<std::size_t>(kHeadDim) * kHeadDim * kValueHeads * nodes *
                      sizeof(float)),
          recurrent(value.bytes),
          gated(value.bytes),
          parents(static_cast<std::size_t>(nodes) * sizeof(std::int32_t)),
          path(static_cast<std::size_t>(nodes) * sizeof(std::int32_t)),
          state_layout(plan_linear_attention_state_pool(
              state_builder, {.layers = kGdnLayers,
                              .conv_channels = kConvRows,
                              .conv_width = 3,
                              .value_heads = kValueHeads,
                              .value_head_dim = kHeadDim,
                              .key_head_dim = kHeadDim,
                              .slot_count = 1,
                              .conv_dtype = DType::BF16})),
          state_storage(state_builder.finish(256)),
          states({state_storage.p, state_storage.bytes}, state_layout),
          fold_state_layout(plan_linear_attention_state_pool(
              fold_state_builder, {.layers = kGdnLayers,
                                   .conv_channels = kConvRows,
                                   .conv_width = 3,
                                   .value_heads = kValueHeads,
                                   .value_head_dim = kHeadDim,
                                   .key_head_dim = kHeadDim,
                                   .slot_count = 1,
                                   .conv_dtype = DType::BF16})),
          fold_state_storage(fold_state_builder.finish(256)),
          fold_states({fold_state_storage.p, fold_state_storage.bytes}, fold_state_layout),
          record_layout(plan_gdn_replay_records(
              record_builder, {.layers = kGdnLayers,
                               .record_capacity = 1,
                               .width = nodes,
                               .conv_channels = kConvRows,
                               .qk_heads = kQkHeads,
                               .value_heads = kValueHeads,
                               .key_dim = kHeadDim,
                               .value_dim = kHeadDim})),
          record_storage(record_builder.finish(256)),
          records({record_storage.p, record_storage.bytes}, record_layout) {
        check(cudaSetDevice(device), "set buffer device");
        const std::vector<std::uint16_t> seed_values = make_seed(nodes);
        seed.copy_from_host(seed_values.data(), seed.bytes);
        const std::vector<std::int32_t> parent_values = make_parents(nodes);
        std::vector<std::int32_t> path_values(static_cast<std::size_t>(nodes));
        std::iota(path_values.begin(), path_values.end(), 0);
        parents.copy_from_host(parent_values.data(), parents.bytes);
        path.copy_from_host(path_values.data(), path.bytes);
        state_storage.fill(0);
        fold_state_storage.fill(0);
        record_storage.fill(0);
        events.reserve(kGdnLayers);
        for (std::int32_t layer = 0; layer < kGdnLayers; ++layer) {
            events.push_back(std::make_unique<LayerEvents>());
        }
        context.synchronize();
    }

    Tensor residual_tensor() { return Tensor(residual.p, DType::BF16, {kHidden, nodes}); }
    Tensor hidden_tensor() { return Tensor(hidden.p, DType::BF16, {kHidden, nodes}); }
    Tensor projected_tensor() {
        return Tensor(projected.p, DType::BF16, {kConvRows, nodes});
    }
    Tensor z_tensor() { return Tensor(z.p, DType::BF16, {kHeadDim, kValueHeads, nodes}); }
    Tensor query_tensor() { return Tensor(query.p, DType::BF16, {kHeadDim, kQkHeads, nodes}); }
    Tensor key_tensor() { return Tensor(key.p, DType::BF16, {kHeadDim, kQkHeads, nodes}); }
    Tensor value_tensor() { return Tensor(value.p, DType::BF16, {kHeadDim, kValueHeads, nodes}); }
    Tensor g_tensor() { return Tensor(g.p, DType::FP32, {kValueHeads, nodes}); }
    Tensor beta_tensor() { return Tensor(beta.p, DType::FP32, {kValueHeads, nodes}); }
    Tensor node_history_tensor() {
        return Tensor(node_history.p, DType::BF16, {kConvRows, 3, nodes});
    }
    Tensor node_states_tensor() {
        return Tensor(node_states.p, DType::FP32,
                      {kHeadDim, kHeadDim, kValueHeads, nodes});
    }
    Tensor recurrent_tensor() {
        return Tensor(recurrent.p, DType::BF16, {kHeadDim, kValueHeads, nodes});
    }
    Tensor gated_tensor() { return Tensor(gated.p, DType::BF16, {kHeadDim, kValueHeads, nodes}); }
    Tensor parent_tensor() { return Tensor(parents.p, DType::I32, {nodes}); }
    Tensor path_tensor() { return Tensor(path.p, DType::I32, {nodes}); }
};

struct ComponentTimes {
    double total_ms = 0.0;
    double norm_control_ms = 0.0;
    double input_projection_ms = 0.0;
    double tree_conv_ms = 0.0;
    double tree_recurrent_ms = 0.0;
    double output_ms = 0.0;
    double gather_ms = 0.0;
    double fold_ms = 0.0;
};

void launch_chain(std::vector<std::unique_ptr<DeviceLayer>>& layers, Buffers& buffers,
                  cudaStream_t stream, bool record_events) {
    check(cudaMemcpyAsync(buffers.residual.p, buffers.seed.p, buffers.seed.bytes,
                          cudaMemcpyDeviceToDevice, stream),
          "reset tree residual");
    Tensor residual = buffers.residual_tensor();
    Tensor hidden = buffers.hidden_tensor();
    Tensor projected = buffers.projected_tensor();
    Tensor z = buffers.z_tensor();
    Tensor query = buffers.query_tensor();
    Tensor key = buffers.key_tensor();
    Tensor value = buffers.value_tensor();
    Tensor g = buffers.g_tensor();
    Tensor beta = buffers.beta_tensor();
    Tensor node_history = buffers.node_history_tensor();
    Tensor node_states = buffers.node_states_tensor();
    Tensor recurrent = buffers.recurrent_tensor();
    Tensor gated = buffers.gated_tensor();
    Tensor parents = buffers.parent_tensor();
    Tensor path = buffers.path_tensor();

    for (std::int32_t layer_index = 0; layer_index < kGdnLayers; ++layer_index) {
        DeviceLayer& layer = *layers[static_cast<std::size_t>(layer_index)];
        LayerEvents& events = *buffers.events[static_cast<std::size_t>(layer_index)];
        buffers.workspace.reset();
        if (record_events) { check(cudaEventRecord(events.event[0], stream), "record stage start"); }
        ops::gdn_norm_gating_proj(residual, layer.input_norm_tensor(), kEps,
                                  layer.a_weight, layer.b_weight, layer.a_log_tensor(),
                                  layer.dt_bias_tensor(), buffers.workspace, hidden, g, beta,
                                  stream);
        if (record_events) { check(cudaEventRecord(events.event[1], stream), "record norm end"); }
        Tensor z_flat = z.view({kValueRows, buffers.nodes});
        ops::gdn_input_proj(hidden, layer.qk.weight, layer.value_z.weight,
                            projected, z_flat, stream);
        if (record_events) {
            check(cudaEventRecord(events.event[2], stream), "record projection end");
        }
        Tensor query_flat = query.view({kKeyRows, buffers.nodes});
        Tensor key_flat = key.view({kKeyRows, buffers.nodes});
        Tensor value_flat = value.view({kValueRows, buffers.nodes});
        ops::gdn_projected_conv_tree(projected, layer.convolution_tensor(),
                                     buffers.states.conv_slot(layer_index, 0), parents,
                                     node_history, query_flat, key_flat, value_flat, stream);
        if (record_events) { check(cudaEventRecord(events.event[3], stream), "record conv end"); }
        ops::gated_delta_net_tree_snapshot(
            query, key, value, g, beta, kGdnScale,
            buffers.states.recurrent_slot(layer_index, 0), parents,
            node_states, recurrent, stream);
        if (record_events) {
            check(cudaEventRecord(events.event[4], stream), "record recurrent end");
        }
        ops::gated_rmsnorm(recurrent, layer.output_norm_tensor(), z, kEps, gated, stream);
        ops::linear_add(gated.view({kValueRows, buffers.nodes}), layer.output.weight,
                        residual, buffers.workspace, stream);
        if (record_events) { check(cudaEventRecord(events.event[5], stream), "record output end"); }
        GdnReplayRecordLayer record = buffers.records.layer(layer_index, 1);
        ops::gdn_tree_gather_path_records(projected, key, value, g, beta, path,
                                          record.conv, record.key, record.value,
                                          record.gate, stream);
        if (record_events) { check(cudaEventRecord(events.event[6], stream), "record gather end"); }
    }
}

ComponentTimes collect_component_times(Buffers& buffers) {
    ComponentTimes times;
    for (std::int32_t layer = 0; layer < kGdnLayers; ++layer) {
        const LayerEvents& events = *buffers.events[static_cast<std::size_t>(layer)];
        std::array<float, 6> elapsed{};
        for (std::size_t stage = 0; stage < elapsed.size(); ++stage) {
            check(cudaEventElapsedTime(&elapsed[stage], events.event[stage],
                                       events.event[stage + 1]),
                  "collect tree stage elapsed");
        }
        times.norm_control_ms += elapsed[0];
        times.input_projection_ms += elapsed[1];
        times.tree_conv_ms += elapsed[2];
        times.tree_recurrent_ms += elapsed[3];
        times.output_ms += elapsed[4];
        times.gather_ms += elapsed[5];
    }
    return times;
}

struct Result {
    std::int32_t nodes = 0;
    std::int32_t selected_path = 0;
    std::size_t graph_nodes = 0;
    bench::ColdTiming graph{};
    bench::ColdTiming fold{};
    ComponentTimes components{};
    std::size_t dense_node_state_bytes = 0;
    std::size_t compact_record_bytes = 0;
};

Result run_budget(std::vector<std::unique_ptr<DeviceLayer>>& layers, const Options& options,
                  std::int32_t nodes, DeviceBuffer& flush) {
    Buffers buffers(options.device, nodes);
    cudaStream_t stream = buffers.context.stream;
    launch_chain(layers, buffers, stream, false);
    buffers.context.synchronize();

    bench::TimedGraph graph;
    graph.capture(stream, [&](cudaStream_t launch_stream) {
        launch_chain(layers, buffers, launch_stream, false);
    });
    const bench::ColdTiming graph_timing =
        bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);

    launch_chain(layers, buffers, stream, true);
    buffers.context.synchronize();
    ComponentTimes components = collect_component_times(buffers);

    const auto fold_launch = [&](cudaStream_t launch_stream) {
        ops::gdn_tree_fold_selected_path(
            buffers.records, buffers.fold_states.all_layers_view(), 0, 0,
            buffers.selected_path, launch_stream);
    };
    const bench::ColdTiming fold_timing =
        bench::measure_cold_launch(fold_launch, flush, stream, options.warmup, options.repeat);
    components.fold_ms = fold_timing.median_us / 1000.0;
    components.total_ms = graph_timing.median_us / 1000.0 + components.fold_ms;

    Result result;
    result.nodes = nodes;
    result.selected_path = buffers.path_length;
    result.graph_nodes = graph.nodes();
    result.graph = graph_timing;
    result.fold = fold_timing;
    result.components = components;
    result.dense_node_state_bytes = buffers.node_states.bytes;
    result.compact_record_bytes = buffers.record_storage.bytes;
    return result;
}

void print_result(const Result& result) {
    const ComponentTimes& c = result.components;
    std::printf(
        "budget=%d selected_path=%d graph_nodes=%zu graph_median_ms=%.6f "
        "fold_median_ms=%.6f total_with_fold_ms=%.6f norm_control_ms=%.6f "
        "input_projection_ms=%.6f tree_conv_ms=%.6f tree_recurrent_ms=%.6f "
        "output_ms=%.6f gather_ms=%.6f dense_node_state_bytes=%zu compact_record_bytes=%zu\n",
        result.nodes, result.selected_path, result.graph_nodes,
        result.graph.median_us / 1000.0, result.fold.median_us / 1000.0,
        c.total_ms, c.norm_control_ms, c.input_projection_ms, c.tree_conv_ms,
        c.tree_recurrent_ms, c.output_ms, c.gather_ms,
        result.dense_node_state_bytes, result.compact_record_bytes);
}

void write_csv(const Options& options, std::span<const Result> results) {
    if (options.csv_out.empty()) { return; }
    if (!options.csv_out.parent_path().empty()) {
        std::filesystem::create_directories(options.csv_out.parent_path());
    }
    std::ofstream out(options.csv_out);
    if (!out) { throw std::runtime_error("failed to open CSV"); }
    out << "budget,selected_path,graph_nodes,graph_median_ms,graph_min_ms,graph_p95_ms,"
           "fold_median_ms,total_with_fold_ms,norm_control_ms,input_projection_ms,tree_conv_ms,"
           "tree_recurrent_ms,output_ms,gather_ms,dense_node_state_bytes,compact_record_bytes\n";
    for (const Result& result : results) {
        const ComponentTimes& c = result.components;
        out << result.nodes << ',' << result.selected_path << ',' << result.graph_nodes << ','
            << result.graph.median_us / 1000.0 << ',' << result.graph.min_us / 1000.0 << ','
            << result.graph.p95_us / 1000.0 << ',' << result.fold.median_us / 1000.0 << ','
            << c.total_ms << ',' << c.norm_control_ms << ',' << c.input_projection_ms << ','
            << c.tree_conv_ms << ',' << c.tree_recurrent_ms << ',' << c.output_ms << ','
            << c.gather_ms << ',' << result.dense_node_state_bytes << ','
            << result.compact_record_bytes << '\n';
    }
    if (!out) { throw std::runtime_error("failed to write CSV"); }
}

std::string uuid_string(const cudaUUID_t& uuid) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string text;
    text.reserve(sizeof(uuid.bytes) * 2);
    for (unsigned char byte : uuid.bytes) {
        text.push_back(digits[byte >> 4U]);
        text.push_back(digits[byte & 0x0fU]);
    }
    return text;
}

int run(const Options& options) {
    check(cudaSetDevice(options.device), "set benchmark device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, options.device), "device properties");
    std::printf("device index=%d name=%s uuid=%s sm=%d memory_bytes=%zu\n",
                options.device, properties.name, uuid_string(properties.uuid).c_str(),
                properties.multiProcessorCount, properties.totalGlobalMem);
    std::vector<std::unique_ptr<DeviceLayer>> layers = load_model(options);
    DeviceBuffer flush(options.flush_bytes);
    std::vector<Result> results;
    results.reserve(options.budgets.size());
    for (const std::int32_t budget : options.budgets) {
        Result result = run_budget(layers, options, budget, flush);
        print_result(result);
        results.push_back(result);
    }
    write_csv(options, results);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::fprintf(stderr, "no CUDA device available\n");
            return 77;
        }
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gdn_tree_target_bench: %s\n", error.what());
        return 1;
    }
}
