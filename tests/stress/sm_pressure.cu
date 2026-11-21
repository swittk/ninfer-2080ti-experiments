#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void pressure_kernel(unsigned long long cycles) {
    const unsigned long long start = clock64();
    unsigned value = static_cast<unsigned>(blockIdx.x) * 2654435761U + threadIdx.x + 1U;
    while (clock64() - start < cycles) {
        value = value * 1664525U + 1013904223U;
        value ^= value >> 13;
    }
    if (value == 0x12345678U) { printf("%u", value); }
}

int main(int argc, char** argv) {
    const int seconds = argc > 1 ? std::atoi(argv[1]) : 46;
    const int blocks  = argc > 2 ? std::atoi(argv[2]) : 4;
    if (seconds <= 0 || blocks <= 0) { return 2; }
    int clock_khz = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0) != cudaSuccess) { return 2; }
    const auto cycles = static_cast<unsigned long long>(clock_khz) * 1000ULL *
                        static_cast<unsigned long long>(seconds);
    pressure_kernel<<<blocks, 256>>>(cycles);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : 2;
}
