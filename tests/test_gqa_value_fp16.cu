#include "core/device.h"
#include "../apps/dual/headscale_gqa.cuh"
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
using namespace ninfer;
__global__ void all_value_conversions(unsigned* errors,unsigned* tested) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    constexpr unsigned scales=0x7c00u,groups=32u;
    if(i>=scales*groups)return;
    const unsigned scale_bits=i/groups;
    alignas(8) std::int8_t values[8];
    for(unsigned j=0;j<8;++j)values[j]=static_cast<std::int8_t>(int((i%groups)*8+j)-128);
    const float scale=__half2float(__ushort_as_half(static_cast<unsigned short>(scale_bits)));
    const int4 result=ops::headscale_qk::gqa_value_packed_fp16(values,scale);
    const unsigned words[4]={static_cast<unsigned>(result.x),static_cast<unsigned>(result.y),
                             static_cast<unsigned>(result.z),static_cast<unsigned>(result.w)};
    unsigned wrong=0;
    for(unsigned j=0;j<8;++j){
        const unsigned expected=__half_as_ushort(__float2half_rn(float(values[j])*scale));
        const unsigned actual=(words[j/2]>>(16*(j%2)))&65535u;
        wrong+=actual!=expected;
        wrong+=values[j]!=static_cast<std::int8_t>(int((i%groups)*8+j)-128);
    }
    if(wrong)atomicAdd(errors,wrong);
    if(i==0)*tested=scales*256u;
}
int main(){try{
    unsigned *errors=nullptr,*tested=nullptr;
    CUDA_CHECK(cudaMalloc(&errors,sizeof(unsigned)));CUDA_CHECK(cudaMalloc(&tested,sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(errors,0,sizeof(unsigned)));CUDA_CHECK(cudaMemset(tested,0,sizeof(unsigned)));
    all_value_conversions<<<(0x7c00u*32u+255)/256,256>>>(errors,tested);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    unsigned bad=0,total=0;
    CUDA_CHECK(cudaMemcpy(&bad,errors,sizeof(bad),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&total,tested,sizeof(total),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(errors));CUDA_CHECK(cudaFree(tested));
    std::printf("CONVERSION cases=%u errors=%u all_signed_bytes all_nonnegative_finite_FP16_scales including_zero_subnormal_overflow input_unchanged\n",total,bad);
    if(bad||total!=8126464u)return 1;
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"value_conversion: %s\n",e.what());return 1;}}
