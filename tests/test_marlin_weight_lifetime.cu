#include "core/device.h"
#include "ops/marlin/mtp_q4_marlin.h"
#include "quantized_weight.cuh"

#include <array>
#include <barrier>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ninfer;
namespace detail = ninfer::ops::detail;

float bf16_value(std::uint16_t bits) {
    union { std::uint32_t bits; float value; } result{static_cast<std::uint32_t>(bits) << 16};
    return result.value;
}

void exercise(int device, std::barrier<>& ready, std::mutex& output_mutex) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    constexpr int n = 4096, k = 5120, t = 4;
    {
        auto packed = bench::make_row_split_weight(QType::Q4G64_F16S, n, k, k);
        struct Release {
            const Weight& weight;
            ~Release() { detail::mtp_q4_marlin_release(weight); }
        } owner{packed.weight};
        DeviceBuffer input(k*t*sizeof(std::uint16_t)), result(n*t*sizeof(std::uint16_t));
        std::vector<std::uint16_t> ones(k*t, 0x3f80), actual(n*t);
        CUDA_CHECK(cudaMemcpy(input.p, ones.data(), input.bytes, cudaMemcpyHostToDevice));
        Tensor x(input.p, DType::BF16, {k,t}), y(result.p, DType::BF16, {n,t});
        const auto verify = [&](int code) {
            if (!detail::mtp_q4_marlin_launch(x, packed.weight, y, stream)) {
                throw std::runtime_error("Marlin refused prepared test shape");
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            result.copy_to_host(actual.data(), result.bytes);
            for (const auto bits : actual) {
                if (bf16_value(bits) != static_cast<float>(k*code)) {
                    throw std::runtime_error("Marlin returned a stale or wrong-replica weight result");
                }
            }
        };
        const auto prepare_concurrent = [&] {
            std::array<std::exception_ptr, 6> errors{};
            std::array<std::thread, 6> threads;
            for (std::size_t i=0; i<threads.size(); ++i) {
                threads[i] = std::thread([&,i] {
                    try {
                        CUDA_CHECK(cudaSetDevice(device));
                        detail::mtp_q4_marlin_prepare(packed.weight, stream);
                    } catch (...) { errors[i] = std::current_exception(); }
                });
            }
            for (auto& thread : threads) { thread.join(); }
            for (const auto& error : errors) { if (error) { std::rethrow_exception(error); } }
        };
        CUDA_CHECK(cudaMemset(packed.storage.p, 0x11, packed.low_bytes));
        prepare_concurrent(); verify(1);
        {
            Weight view = packed.weight;
            view.n = n / 2;
            Tensor one = x.slice(1, 0, 1);
            Tensor target = y.slice(1, 0, 1).slice(0, 0, n / 2);
            if (detail::mtp_q4_marlin_ready(view) ||
                detail::mtp_q4_marlin_launch(one, view, target, stream)) {
                throw std::runtime_error("normal Marlin lookup silently accepted a parent-sized packed view");
            }
            detail::mtp_q4_marlin_retained_prefix_launch(one, packed.weight, view, target, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            result.copy_to_host(actual.data(), static_cast<std::size_t>(n / 2) * sizeof(std::uint16_t));
            for(int i=0;i<n/2;++i) if(bf16_value(actual[i])!=static_cast<float>(k)) {
                throw std::runtime_error("retained prefix constant-weight oracle failed");
            }
            for(int invalid=0;invalid<5;++invalid) {
                Weight bad=view;
                if(invalid==0)bad.n=n+64;
                if(invalid==1)bad.k=k+64;
                if(invalid==2)bad.n=63;
                if(invalid==3)bad.scales=nullptr;
                if(invalid==4)bad.qdata=nullptr;
                bool rejected=false;
                try{detail::mtp_q4_marlin_retained_prefix_launch(one,packed.weight,bad,target,stream);}
                catch(const std::invalid_argument&){rejected=true;}
                if(!rejected)throw std::runtime_error("retained prefix escaped its exact owner bounds");
            }
        }

        detail::mtp_q4_marlin_release(packed.weight);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::size_t baseline = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&baseline, &total));
        ready.arrive_and_wait();
        for (int cycle=0; cycle<8; ++cycle) {
            const int code = 1 + device + cycle%3;
            CUDA_CHECK(cudaMemset(packed.storage.p, code | (code<<4), packed.low_bytes));
            prepare_concurrent(); verify(code);
            std::size_t before = 0, after = 0;
            CUDA_CHECK(cudaMemGetInfo(&before, &total));
            prepare_concurrent();
            CUDA_CHECK(cudaMemGetInfo(&after, &total));
            if (before != after) { throw std::runtime_error("duplicate Marlin preparation allocated extra memory"); }
            detail::mtp_q4_marlin_release(packed.weight);
            detail::mtp_q4_marlin_release(packed.weight); // Idempotent teardown.
            if (detail::mtp_q4_marlin_ready(packed.weight)) {
                throw std::runtime_error("released Marlin entry remained visible");
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemGetInfo(&after, &total));
            if (after != baseline) { throw std::runtime_error("Marlin owner release did not return all GPU bytes"); }
        }
        std::lock_guard lock(output_mutex);
        std::cout << "PASS GPU " << device
                  << " 8 changed-weight reloads at identical source address; concurrent prepare; zero bytes lost\n";
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) { return 77; }
    if (!detail::mtp_q4_marlin_enabled()) {
        std::cerr << "NINFER_MTP_MARLIN_Q4 must enable this physical regression\n";
        return 77;
    }
    std::barrier ready(2);
    std::mutex output_mutex;
    std::array<std::exception_ptr,2> errors{};
    std::array<std::thread,2> threads;
    for (int i=0;i<2;++i) {
        threads[i] = std::thread([&,i] {
            try { exercise(i, ready, output_mutex); }
            catch (...) { ready.arrive_and_drop(); errors[i] = std::current_exception(); }
        });
    }
    for (auto& thread : threads) { thread.join(); }
    for (const auto& error : errors) {
        if (error) {
            try { std::rethrow_exception(error); }
            catch (const std::exception& e) { std::cerr << e.what() << '\n'; }
            return 1;
        }
    }
    return 0;
}
