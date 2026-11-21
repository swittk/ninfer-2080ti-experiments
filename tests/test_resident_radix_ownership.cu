#include "core/device.h"
#include "ops/linear/sm75_radix_projection.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace ninfer;
namespace radix = ninfer::ops::detail;
namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
struct Stream {
    cudaStream_t value{};
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { (void)cudaStreamSynchronize(value); (void)cudaStreamDestroy(value); }
};
struct Graph {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    ~Graph() {
        if (executable) { (void)cudaGraphExecDestroy(executable); }
        if (graph) { (void)cudaGraphDestroy(graph); }
    }
};
std::vector<unsigned char> read(const DeviceBuffer& buffer, cudaStream_t stream) {
    std::vector<unsigned char> bytes(buffer.bytes);
    CUDA_CHECK(cudaMemcpyAsync(bytes.data(), buffer.p, bytes.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return bytes;
}
struct Projection {
    int n, k, stride;
    bool q5;
    cudaStream_t stream;
    bench::PackedQuantizedWeight reference, resident;
    DeviceBuffer input, padded_input, expected, actual;
    std::vector<unsigned char> original;

    Projection(int rows, int columns, bool high, unsigned seed, cudaStream_t owner)
        : n(rows), k(columns), stride(rows + 16), q5(high), stream(owner),
          reference(bench::make_row_split_weight(high ? QType::Q5G64_F16S : QType::Q4G64_F16S,
                                                 rows, columns, columns)),
          resident(bench::make_row_split_weight(high ? QType::Q5G64_F16S : QType::Q4G64_F16S,
                                                rows, columns, columns)),
          input(std::size_t(columns) * 16 * 2), padded_input(input.bytes),
          expected(std::size_t(stride) * 16 * 2), actual(expected.bytes) {
        bench::t16_fill_weight(reference.weight, n, k, q5, seed);
        bench::t16_fill_weight(resident.weight, n, k, q5, seed);
        bench::t16_fill_input(input.p, std::size_t(k) * 16, seed + 1);
        // Fixture uploads use the default stream; all subsequent work is on a
        // nonblocking stream, so staging alone is not a completion guarantee.
        CUDA_CHECK(cudaDeviceSynchronize());
        original = read(resident.storage, stream);
        radix::sm75_radix_projection_prepare(reference.weight, stream, 16);
        radix::sm75_radix_resident_prepare(resident.weight, stream, 16);
    }
    ~Projection() {
        (void)cudaStreamSynchronize(stream);
        radix::sm75_radix_projection_release(reference.weight);
        radix::sm75_radix_projection_release(resident.weight);
        try {
            radix::sm75_radix_resident_transform(resident.weight, QuantLayout::RowSplit, stream);
        } catch (...) { /* Do not throw during failure cleanup. */ }
    }
    void enqueue(int active) {
        const int padded = active <= 4 ? 4 : active <= 8 ? 8 : 16;
        CUDA_CHECK(cudaMemsetAsync(padded_input.p, 0, padded_input.bytes, stream));
        CUDA_CHECK(cudaMemcpyAsync(padded_input.p, input.p, std::size_t(k) * active * 2,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(expected.p, 0x2a, expected.bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(actual.p, 0x2a, actual.bytes, stream));
        Tensor reference_x(padded_input.p, DType::BF16, {k, padded});
        Tensor actual_x(input.p, DType::BF16, {k, active});
        Tensor reference_whole(expected.p, DType::BF16, {stride, padded});
        Tensor actual_whole(actual.p, DType::BF16, {stride, active});
        Tensor reference_y = reference_whole.slice(0, 0, n);
        Tensor actual_y = actual_whole.slice(0, 0, n);
        require(radix::sm75_radix_projection_launch(reference_x, reference.weight, reference_y,
                                                    q5, stream), "independent reference refused");
        require(radix::sm75_radix_projection_launch(actual_x, resident.weight, actual_y,
                                                    q5, stream), "resident projection refused");
    }
    std::uint64_t check(int active) {
        auto a = read(expected, stream), b = read(actual, stream);
        for (int token = 0; token < 16; ++token) {
            for (int row = 0; row < stride; ++row) {
                const auto offset = (std::size_t(token) * stride + row) * 2;
                for (int byte = 0; byte < 2; ++byte) {
                    const auto target = token < active && row < n ? a[offset + byte] : 0x2a;
                    require(b[offset + byte] == target,
                            "multi-stream/mixed-shape result or inactive bytes changed");
                }
            }
        }
        return std::uint64_t(n) * active;
    }
    void canonical_roundtrip() {
        radix::sm75_radix_projection_release(resident.weight);
        radix::sm75_radix_resident_transform(resident.weight, QuantLayout::RowSplit, stream);
        require(read(resident.storage, stream) == original, "canonical allocation bytes changed");
        radix::sm75_radix_resident_prepare(resident.weight, stream, 16);
    }
};
}

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
        if (props.major != 7 || props.minor != 5) { return 77; }
        Stream first, second;
        std::array<std::unique_ptr<Projection>, 4> cases{
            std::make_unique<Projection>(128, 5120, false, 317U, first.value),
            std::make_unique<Projection>(192, 6144, true, 691U, first.value),
            std::make_unique<Projection>(256, 5120, false, 1223U, second.value),
            std::make_unique<Projection>(128, 6144, true, 2027U, second.value)};
        std::uint64_t checked = 0;
        for (int active : {1, 3, 4, 7, 8, 12, 16, 4}) {
            // Both stream graphs are live and replay concurrently, while each
            // graph serially uses Q4 and Q5 sources with different dimensions.
            std::array<Graph, 2> graphs;
            const std::array<cudaStream_t, 2> streams{first.value, second.value};
            for (int i = 0; i < 2; ++i) {
                CUDA_CHECK(cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeGlobal));
                cases[i * 2]->enqueue(active);
                cases[i * 2 + 1]->enqueue(active);
                CUDA_CHECK(cudaStreamEndCapture(streams[i], &graphs[i].graph));
                CUDA_CHECK(cudaGraphInstantiate(&graphs[i].executable, graphs[i].graph, nullptr, nullptr, 0));
            }
            for (int replay = 0; replay < 5; ++replay) {
                CUDA_CHECK(cudaGraphLaunch(graphs[0].executable, first.value));
                CUDA_CHECK(cudaGraphLaunch(graphs[1].executable, second.value));
            }
            for (auto& item : cases) { checked += item->check(active); }
        }
        // Graphs above are destroyed before any prepared source is released.
        for (auto& item : cases) { item->canonical_roundtrip(); }
        {
            Projection temporary(320, 5120, false, 9991U, first.value);
            temporary.enqueue(7);
            checked += temporary.check(7);
        }
        // Releasing a different source sharing the same scratch cannot invalidate
        // either surviving source or the other stream's prepared storage.
        for (auto& item : cases) { item->enqueue(16); }
        for (auto& item : cases) { checked += item->check(16); }
        auto& source = *cases[0];
        bool refused = false;
        try { radix::sm75_radix_resident_prepare(source.resident.weight, second.value, 16); }
        catch (const std::logic_error&) { refused = true; }
        require(refused, "resident source admitted a different owning stream");
        refused = false;
        try { radix::sm75_radix_resident_transform(source.resident.weight, QuantLayout::RowSplit, first.value); }
        catch (const std::logic_error&) { refused = true; }
        require(refused, "live prepared source was transformable");
        std::cout << "PASS two concurrent streams, mixed Q4/Q5 shapes, ragged columns, graph repeats, "
                     "source release/reprepare, canonical roundtrip; checked_values=" << checked << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
