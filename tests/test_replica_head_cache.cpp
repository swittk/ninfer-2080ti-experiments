#include "core/device.h"
#include "core/tensor.h"
#include "ops/linear/q4/q4_proposal_b2_sm75.h"
#include "ops/linear/w8/w8_vocab_w4_a8_sm75.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ninfer;
namespace detail = ninfer::ops::detail;
namespace {
constexpr int kWidth = 5120;
constexpr int kProposalRows = 131072;
constexpr int kVocabularyRows = 248320;

struct Replica {
    DeviceContext device;
    std::vector<void*> allocations;
    Weight proposal, vocabulary;
    Tensor hidden, logits, token;
    const int expected;

    explicit Replica(int index) : device(index), expected(70017 + index * 401) {
        proposal = make_weight(kProposalRows, QType::Q4G64_F16S);
        vocabulary = make_weight(kVocabularyRows, QType::W8G32_F16S);
        hidden = Tensor(allocate(kWidth * 2), DType::BF16, {kWidth, 1});
        logits = Tensor(allocate(kVocabularyRows * 2), DType::BF16, {kVocabularyRows, 1});
        token = Tensor(allocate(4), DType::I32, {1});
        std::vector<std::uint16_t> ones(kWidth, 0x3f80);
        CUDA_CHECK(cudaMemcpy(hidden.data, ones.data(), kWidth * 2, cudaMemcpyHostToDevice));
        device.synchronize();
    }
    ~Replica() {
        CUDA_CHECK(cudaSetDevice(device.device));
        device.synchronize();
        detail::q4_proposal_b2_release(proposal);
        detail::w8_vocab_w4_a8_release(vocabulary);
        for (void* allocation : allocations) { (void)cudaFree(allocation); }
    }
    void* allocate(std::size_t bytes) {
        void* pointer = nullptr;
        CUDA_CHECK(cudaMalloc(&pointer, bytes));
        allocations.push_back(pointer);
        return pointer;
    }
    Weight make_weight(int rows, QType type) {
        Weight weight;
        weight.qtype = type;
        weight.n = rows;
        weight.k = kWidth;
        weight.group = weight.group_size = type == QType::Q4G64_F16S ? 64 : 32;
        weight.ndim = 2;
        weight.shape[0] = weight.padded_shape[0] = rows;
        weight.shape[1] = weight.padded_shape[1] = kWidth;
        weight.scale_dtype = DType::FP16;
        const std::size_t row_bytes = type == QType::Q4G64_F16S ? kWidth / 2 : kWidth;
        void* codes = allocate(static_cast<std::size_t>(rows) * row_bytes);
        CUDA_CHECK(cudaMemset(codes, 0, static_cast<std::size_t>(rows) * row_bytes));
        CUDA_CHECK(cudaMemset(static_cast<std::uint8_t*>(codes) + expected * row_bytes,
                             type == QType::Q4G64_F16S ? 0x11 : 1, row_bytes));
        const std::size_t scale_count = static_cast<std::size_t>(rows) * kWidth / weight.group;
        void* scales = allocate(scale_count * 2);
        std::vector<std::uint16_t> ones(scale_count, 0x3c00);
        CUDA_CHECK(cudaMemcpy(scales, ones.data(), scale_count * 2, cudaMemcpyHostToDevice));
        weight.qdata = codes;
        weight.scales = scales;
        return weight;
    }
    void prepare() {
        CUDA_CHECK(cudaSetDevice(device.device));
        detail::q4_proposal_b2_prepare(proposal, device.stream);
        detail::w8_vocab_w4_a8_prepare(vocabulary, device.stream);
    }
    void verify() {
        CUDA_CHECK(cudaSetDevice(device.device));
        if (!detail::q4_proposal_b2_ready(proposal) ||
            !detail::w8_vocab_w4_a8_ready(vocabulary)) {
            throw std::runtime_error("another replica displaced this GPU's prepared cache");
        }
        std::int32_t actual = -1;
        detail::q4_proposal_b2_launch(hidden, proposal, token, device.stream);
        device.synchronize();
        CUDA_CHECK(cudaMemcpy(&actual, token.data, 4, cudaMemcpyDeviceToHost));
        if (actual != expected) { throw std::runtime_error("proposal cache crossed GPU ownership"); }
        detail::w8_vocab_w4_a8_launch_t1(hidden, vocabulary, logits, token, device.stream);
        device.synchronize();
        CUDA_CHECK(cudaMemcpy(&actual, token.data, 4, cudaMemcpyDeviceToHost));
        if (actual != expected) { throw std::runtime_error("vocabulary cache crossed GPU ownership"); }
    }
    std::size_t free_memory() const {
        CUDA_CHECK(cudaSetDevice(device.device));
        std::size_t available = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&available, &total));
        return available;
    }
};
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::cout << "SKIP: two physical CUDA devices required\n";
        return 77;
    }
    try {
        std::array<std::unique_ptr<Replica>, 2> replicas;
        for (int i = 0; i < 2; ++i) { replicas[i] = std::make_unique<Replica>(i); }
        for (auto& replica : replicas) { replica->prepare(); replica->verify(); }
        const std::array<std::size_t, 2> before =
            {replicas[0]->free_memory(), replicas[1]->free_memory()};
        for (int round = 0; round < 24; ++round) {
            for (auto& replica : replicas) { replica->prepare(); replica->verify(); }
        }
        std::array<std::exception_ptr, 2> failures;
        std::array<std::thread, 2> workers;
        for (int i = 0; i < 2; ++i) {
            workers[i] = std::thread([&, i] {
                try {
                    for (int round = 0; round < 16; ++round) {
                        replicas[i]->prepare(); replicas[i]->verify();
                    }
                } catch (...) { failures[i] = std::current_exception(); }
            });
        }
        for (auto& worker : workers) { worker.join(); }
        for (auto& failure : failures) { if (failure) { std::rethrow_exception(failure); } }
        for (int i = 0; i < 2; ++i) {
            const auto after = replicas[i]->free_memory();
            std::cout << "GPU " << i << " repeated_prepare_bytes_lost="
                      << (static_cast<std::int64_t>(before[i])-static_cast<std::int64_t>(after))
                      << " expected_token=" << replicas[i]->expected << '\n';
            if (after + 8 * 1024 * 1024 < before[i]) {
                throw std::runtime_error("repeated preparation leaked GPU memory");
            }
        }
        detail::q4_proposal_b2_release(replicas[0]->proposal);
        detail::w8_vocab_w4_a8_release(replicas[0]->vocabulary);
        if (detail::q4_proposal_b2_ready(replicas[0]->proposal) ||
            detail::w8_vocab_w4_a8_ready(replicas[0]->vocabulary)) {
            throw std::runtime_error("released cache remains visible");
        }
        replicas[1]->verify();
        replicas[0]->prepare(); replicas[0]->verify();
        std::cout << "PASS: dual GPU head isolation, repeated/concurrent preparation, release and reload\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
