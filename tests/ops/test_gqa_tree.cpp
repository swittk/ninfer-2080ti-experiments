#include "ninfer/ops/gqa_tree.h"

#include "core/device.h"
#include "ops/op_tester.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t D = 256;
constexpr std::int32_t Hq = 24;
constexpr std::int32_t Hkv = 4;
constexpr std::int32_t N = 6;
constexpr float Scale = 1.0F / 16.0F;

template <class T>
DeviceBuffer upload(std::span<const T> values) {
    DeviceBuffer buffer(values.size_bytes());
    buffer.copy_from_host(values.data(), values.size_bytes());
    return buffer;
}

template <class T, std::size_t S>
DeviceBuffer upload(const std::array<T, S>& values) {
    return upload<T>(std::span<const T>(values));
}

template <class T>
DeviceBuffer upload(const std::vector<T>& values) {
    return upload<T>(std::span<const T>(values));
}

template <class T>
std::vector<T> download(const Tensor& tensor) {
    std::vector<T> values(static_cast<std::size_t>(tensor.numel()));
    cuda_check(cudaMemcpy(values.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost),
               "download GQA tree tensor");
    return values;
}

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed,
                                       float lo = -0.12F, float hi = 0.12F) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t i = 0; i < count; ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

template <class T>
std::vector<T> gather_columns(const std::vector<T>& source, std::int32_t rows,
                              std::span<const std::int32_t> columns) {
    std::vector<T> out(static_cast<std::size_t>(rows) * columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i) {
        std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(rows) * columns[i], rows,
                    out.begin() + static_cast<std::ptrdiff_t>(rows) * i);
    }
    return out;
}

template <class T>
int compare(const std::string& label, std::span<const T> lhs, std::span<const T> rhs) {
    if (lhs.size() != rhs.size()) {
        std::cerr << label << ": size mismatch\n";
        return 1;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            std::cerr << label << ": mismatch at " << i << "\n";
            return 1;
        }
    }
    return 0;
}

template <class T>
int compare(const std::string& label, const std::vector<T>& lhs,
            const std::vector<T>& rhs) {
    return compare<T>(label, std::span<const T>(lhs), std::span<const T>(rhs));
}

template <class T>
std::span<const T> column(std::span<const T> values, std::int32_t rows,
                          std::int32_t index) {
    return values.subspan(static_cast<std::size_t>(rows) * index, rows);
}

struct Run {
    std::int32_t nodes;
    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer parents;
    DeviceBuffer acc;
    DeviceBuffer m;
    DeviceBuffer l;
    DeviceBuffer prefix_acc;
    DeviceBuffer prefix_m;
    DeviceBuffer prefix_l;
    DeviceBuffer out;
    Tensor q_tensor;
    Tensor k_tensor;
    Tensor v_tensor;
    Tensor parent_tensor;
    Tensor acc_tensor;
    Tensor m_tensor;
    Tensor l_tensor;
    Tensor prefix_acc_tensor;
    Tensor prefix_m_tensor;
    Tensor prefix_l_tensor;
    Tensor out_tensor;

    Run(const std::vector<std::uint16_t>& q_bits,
        const std::vector<std::uint16_t>& k_bits,
        const std::vector<std::uint16_t>& v_bits,
        const std::vector<std::int32_t>& parent_values,
        const std::vector<float>& prefix_acc_values,
        const std::vector<float>& prefix_m_values,
        const std::vector<float>& prefix_l_values)
        : nodes(static_cast<std::int32_t>(parent_values.size())),
          q(upload(q_bits)), k(upload(k_bits)), v(upload(v_bits)),
          parents(upload(parent_values)),
          acc(static_cast<std::size_t>(D) * Hq * nodes * sizeof(float)),
          m(static_cast<std::size_t>(Hq) * nodes * sizeof(float)),
          l(m.bytes), prefix_acc(upload(prefix_acc_values)),
          prefix_m(upload(prefix_m_values)), prefix_l(upload(prefix_l_values)),
          out(static_cast<std::size_t>(D) * Hq * nodes * sizeof(std::uint16_t)),
          q_tensor(q.p, DType::BF16, {D, Hq, nodes}),
          k_tensor(k.p, DType::BF16, {D, Hkv, nodes}),
          v_tensor(v.p, DType::BF16, {D, Hkv, nodes}),
          parent_tensor(parents.p, DType::I32, {nodes}),
          acc_tensor(acc.p, DType::FP32, {D, Hq, nodes}),
          m_tensor(m.p, DType::FP32, {Hq, nodes}),
          l_tensor(l.p, DType::FP32, {Hq, nodes}),
          prefix_acc_tensor(prefix_acc.p, DType::FP32, {D, Hq, nodes}),
          prefix_m_tensor(prefix_m.p, DType::FP32, {Hq, nodes}),
          prefix_l_tensor(prefix_l.p, DType::FP32, {Hq, nodes}),
          out_tensor(out.p, DType::BF16, {D, Hq, nodes}) {
        ops::gqa_attention_tree_tail_state(q_tensor, k_tensor, v_tensor, parent_tensor,
                                           Scale, acc_tensor, m_tensor, l_tensor, nullptr);
        ops::gqa_attention_tree_merge_prefix(prefix_acc_tensor, prefix_m_tensor,
                                             prefix_l_tensor, acc_tensor, m_tensor,
                                             l_tensor, out_tensor, nullptr);
        cuda_synchronize();
    }
};

int verify_path(const Run& tree, const Run& branch,
                std::span<const std::int32_t> path, const std::string& label) {
    const auto tree_acc = download<float>(tree.acc_tensor);
    const auto tree_m = download<float>(tree.m_tensor);
    const auto tree_l = download<float>(tree.l_tensor);
    const auto tree_out = download<std::uint16_t>(tree.out_tensor);
    const auto branch_acc = download<float>(branch.acc_tensor);
    const auto branch_m = download<float>(branch.m_tensor);
    const auto branch_l = download<float>(branch.l_tensor);
    const auto branch_out = download<std::uint16_t>(branch.out_tensor);
    int failures = 0;
    for (std::size_t depth = 0; depth < path.size(); ++depth) {
        failures += compare<float>(
            label + " acc depth=" + std::to_string(depth),
            column<float>(tree_acc, D * Hq, path[depth]),
            column<float>(branch_acc, D * Hq, static_cast<std::int32_t>(depth)));
        failures += compare<float>(
            label + " m depth=" + std::to_string(depth),
            column<float>(tree_m, Hq, path[depth]),
            column<float>(branch_m, Hq, static_cast<std::int32_t>(depth)));
        failures += compare<float>(
            label + " l depth=" + std::to_string(depth),
            column<float>(tree_l, Hq, path[depth]),
            column<float>(branch_l, Hq, static_cast<std::int32_t>(depth)));
        failures += compare<std::uint16_t>(
            label + " merged output depth=" + std::to_string(depth),
            column<std::uint16_t>(tree_out, D * Hq, path[depth]),
            column<std::uint16_t>(branch_out, D * Hq, static_cast<std::int32_t>(depth)));
    }
    return failures;
}

int run_test() {
    constexpr std::array<std::int32_t, N> parent_array{-1, 0, 1, -1, 3, 4};
    constexpr std::array<std::int32_t, 3> path_a{0, 1, 2};
    constexpr std::array<std::int32_t, 3> path_b{3, 4, 5};
    const std::vector<std::int32_t> parents(parent_array.begin(), parent_array.end());
    const auto q = random_bf16(static_cast<std::size_t>(D) * Hq * N, 7201U);
    const auto k = random_bf16(static_cast<std::size_t>(D) * Hkv * N, 7203U);
    const auto v = random_bf16(static_cast<std::size_t>(D) * Hkv * N, 7205U);
    std::vector<float> prefix_acc(static_cast<std::size_t>(D) * Hq * N);
    std::vector<float> prefix_m(static_cast<std::size_t>(Hq) * N);
    std::vector<float> prefix_l(prefix_m.size());
    fill_uniform(prefix_acc, 7207U, -0.8F, 0.8F);
    fill_uniform(prefix_m, 7209U, -1.2F, 1.2F);
    fill_uniform(prefix_l, 7211U, 0.2F, 3.0F);
    Run tree(q, k, v, parents, prefix_acc, prefix_m, prefix_l);

    const auto make_branch = [&](std::span<const std::int32_t> path) {
        const std::vector<std::int32_t> chain{-1, 0, 1};
        return Run(gather_columns(q, D * Hq, path),
                   gather_columns(k, D * Hkv, path),
                   gather_columns(v, D * Hkv, path), chain,
                   gather_columns(prefix_acc, D * Hq, path),
                   gather_columns(prefix_m, Hq, path),
                   gather_columns(prefix_l, Hq, path));
    };
    Run branch_a = make_branch(path_a);
    Run branch_b = make_branch(path_b);
    int failures = 0;
    failures += verify_path(tree, branch_a, path_a, "branch A");
    failures += verify_path(tree, branch_b, path_b, "branch B");

    // Modifying branch A K/V must not affect branch B states or merged outputs.
    auto changed_k = k;
    auto changed_v = v;
    std::fill_n(changed_k.begin(), static_cast<std::size_t>(D) * Hkv * 3,
                f32_to_bf16(0.75F));
    std::fill_n(changed_v.begin(), static_cast<std::size_t>(D) * Hkv * 3,
                f32_to_bf16(-0.5F));
    Run changed(q, changed_k, changed_v, parents, prefix_acc, prefix_m, prefix_l);
    const auto original_acc = download<float>(tree.acc_tensor);
    const auto changed_acc = download<float>(changed.acc_tensor);
    const auto original_out = download<std::uint16_t>(tree.out_tensor);
    const auto changed_out = download<std::uint16_t>(changed.out_tensor);
    for (std::int32_t node : path_b) {
        failures += compare<float>(
            "sibling isolation acc node=" + std::to_string(node),
            column<float>(original_acc, D * Hq, node),
            column<float>(changed_acc, D * Hq, node));
        failures += compare<std::uint16_t>(
            "sibling isolation output node=" + std::to_string(node),
            column<std::uint16_t>(original_out, D * Hq, node),
            column<std::uint16_t>(changed_out, D * Hq, node));
    }
    failures += compare<std::int32_t>("parent ids remained read-only",
                                      download<std::int32_t>(tree.parent_tensor), parents);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const int failures = run_test();
    std::cout << (failures == 0 ? "OK" : "FAIL")
              << " gqa_tree ancestor-only tail + prefix merge\n";
    return failures == 0 ? 0 : 1;
}
