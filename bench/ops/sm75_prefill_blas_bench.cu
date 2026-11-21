#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma_config.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
using namespace ninfer;
static void blas_check(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status=" + std::to_string(s));
}
// Match the production RowSplit codec and its BF16 weight rounding boundary.
template<bool Q5, bool Native=false> __global__ void unpack(const std::uint8_t* low, const std::uint8_t* high,
                                        const __half* scales, __half2* out, std::int64_t pairs) {
    for (std::int64_t i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<pairs;i+=static_cast<std::int64_t>(blockDim.x)*gridDim.x) {
        const int lane=i&31; const auto group=i>>5;
        int a=low[i]&15,b=low[i]>>4;
        if constexpr(Q5) { unsigned h=high[group*8+(lane>>2)]; int shift=(lane&3)*2;
            a|=((h>>shift)&1)<<4; b|=((h>>(shift+1))&1)<<4;
            a=(a^16)-16; b=(b^16)-16;
        } else { a=(a^8)-8; b=(b^8)-8; }
        const float scale=__half2float(scales[group]);
        const float wa=__bfloat162float(__float2bfloat16_rn(a*scale));
        const float wb=__bfloat162float(__float2bfloat16_rn(b*scale));
        if constexpr(Native) reinterpret_cast<__nv_bfloat162*>(out)[i]=__floats2bfloat162_rn(wa,wb);
        else out[i]=__floats2half2_rn(wa,wb);
    }
}
__global__ void input_half(const __nv_bfloat16* src, __half* dst, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) dst[i]=__float2half_rn(__bfloat162float(src[i]));
}
__global__ void output_bf16(const float* src, __nv_bfloat16* dst, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) dst[i]=__float2bfloat16_rn(src[i]);
}
template<int N,int K> static void native_gemm(const void* x,const void* w,void* out,int tokens,cudaStream_t stream) {
    using namespace ninfer::ops;
    using namespace ninfer::ops::detail;
    using G=Bf16GemvGeometry<N,K>;using S=Bf16MmaProductionSchedule<G>;
    const Bf16MmaContiguousOutput dst{static_cast<__nv_bfloat16*>(out),N};
    const int grid=N/S::kBlockRows*((tokens+S::kBlockCols-1)/S::kBlockCols);
    if(tokens%S::kBlockCols==0) bf16_gemm_mma_kernel<G,S,true><<<grid,S::kThreads,S::kSharedBytes,stream>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const __nv_bfloat16*>(w),dst,tokens);
    else bf16_gemm_mma_kernel<G,S,false><<<grid,S::kThreads,S::kSharedBytes,stream>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const __nv_bfloat16*>(w),dst,tokens);
    CUDA_CHECK(cudaGetLastError());
}
static float from_bf16(std::uint16_t x) { union {std::uint32_t u;float f;} y{static_cast<std::uint32_t>(x)<<16};return y.f; }
int main(int argc,char** argv) {
 try {
    const int tokens=argc>1?std::stoi(argv[1]):128;
    if(tokens<32||tokens>512) throw std::invalid_argument("oracle T32..T512");
    cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cublasHandle_t h{};blas_check(cublasCreate(&h));blas_check(cublasSetStream(h,stream));
    DeviceBuffer library_scratch(4ULL<<20);blas_check(cublasSetWorkspace(h,library_scratch.p,library_scratch.bytes));
    blas_check(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH));
    DeviceBuffer flush(256ULL<<20);
    const int shapes[][3]={{34816,5120,4},{5120,17408,5},{5120,6144,5},{4096,5120,4}};
    for(const auto& shape:shapes) {
        const int n=shape[0],k=shape[1]; const bool q5=shape[2]==5;
        auto packed=bench::make_row_split_weight(q5?QType::Q5G64_F16S:QType::Q4G64_F16S,n,k,k);
        bench::t16_fill_weight(packed.weight,n,k,q5,1234567u+n);
        DeviceBuffer x(static_cast<std::size_t>(k)*tokens*2),xh(x.bytes);
        bench::t16_fill_input(x.p,x.bytes/2,987654321u+k);
        DeviceBuffer w(static_cast<std::size_t>(n)*k*2),wn(w.bytes);
        DeviceBuffer y(static_cast<std::size_t>(n)*tokens*4),ref(y.bytes/2),cand(y.bytes/2);
        Tensor xt(x.p,DType::BF16,{k,tokens}),rt(ref.p,DType::BF16,{n,tokens});
        auto decode=[&](cudaStream_t s){
            const auto pairs=static_cast<std::int64_t>(n)*k/2;
            if(q5) unpack<true><<<4096,256,0,s>>>(static_cast<const std::uint8_t*>(packed.weight.qdata),static_cast<const std::uint8_t*>(packed.weight.qhigh),static_cast<const __half*>(packed.weight.scales),static_cast<__half2*>(w.p),pairs);
            else unpack<false><<<4096,256,0,s>>>(static_cast<const std::uint8_t*>(packed.weight.qdata),nullptr,static_cast<const __half*>(packed.weight.scales),static_cast<__half2*>(w.p),pairs);
            input_half<<<(k*tokens+255)/256,256,0,s>>>(static_cast<const __nv_bfloat16*>(x.p),static_cast<__half*>(xh.p),k*tokens);
            CUDA_CHECK(cudaGetLastError());
        };
        auto gemm=[&](cudaStream_t s){
            const float alpha=1.f,beta=0.f;
            blas_check(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,tokens,k,&alpha,w.p,CUDA_R_16F,k,xh.p,CUDA_R_16F,k,&beta,y.p,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            output_bf16<<<(n*tokens+255)/256,256,0,s>>>(static_cast<const float*>(y.p),static_cast<__nv_bfloat16*>(cand.p),n*tokens);
            CUDA_CHECK(cudaGetLastError());
        };
        auto native_decode=[&](cudaStream_t s){
            const auto pairs=static_cast<std::int64_t>(n)*k/2;
            if(q5) unpack<true,true><<<4096,256,0,s>>>(static_cast<const std::uint8_t*>(packed.weight.qdata),static_cast<const std::uint8_t*>(packed.weight.qhigh),static_cast<const __half*>(packed.weight.scales),static_cast<__half2*>(wn.p),pairs);
            else unpack<false,true><<<4096,256,0,s>>>(static_cast<const std::uint8_t*>(packed.weight.qdata),nullptr,static_cast<const __half*>(packed.weight.scales),static_cast<__half2*>(wn.p),pairs);
        };
        auto native=[&](cudaStream_t s){
            native_decode(s);
            if(n==34816) native_gemm<34816,5120>(x.p,wn.p,cand.p,tokens,s);
            else if(k==17408) native_gemm<5120,17408>(x.p,wn.p,cand.p,tokens,s);
            else if(k==6144) native_gemm<5120,6144>(x.p,wn.p,cand.p,tokens,s);
            else native_gemm<4096,5120>(x.p,wn.p,cand.p,tokens,s);
        };
        auto reference=[&](cudaStream_t s){ops::linear(xt,packed.weight,rt,s);};
        auto candidate=[&](cudaStream_t s){decode(s);gemm(s);};
        reference(stream);candidate(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::uint16_t> r(ref.bytes/2),c(cand.bytes/2);
        ref.copy_to_host(r.data(),ref.bytes);cand.copy_to_host(c.data(),cand.bytes);
        double err=0,norm=0;float maxdiff=0;std::size_t mismatches=0,nonfinite=0;
        for(std::size_t i=0;i<r.size();++i){float a=from_bf16(r[i]),b=from_bf16(c[i]);double d=static_cast<double>(b)-a;err+=d*d;norm+=static_cast<double>(a)*a;maxdiff=std::max(maxdiff,static_cast<float>(std::abs(d)));mismatches+=r[i]!=c[i];nonfinite+=!std::isfinite(b);}
        const auto ref_time=bench::measure_cold_launch(reference,flush,stream,3,12);
        const auto dec_time=bench::measure_cold_launch(decode,flush,stream,3,12);
        const auto gemm_time=bench::measure_cold_launch(gemm,flush,stream,3,12);
        const auto cand_time=bench::measure_cold_launch(candidate,flush,stream,3,12);
        std::printf("RESULT n=%d k=%d bits=%d tokens=%d reference_us=%.3f dequant_us=%.3f gemm_us=%.3f candidate_us=%.3f speedup=%.6f rel_l2=%.8g max_abs=%.7g mismatches=%zu nonfinite=%zu workspace=%zu\n",n,k,shape[2],tokens,ref_time.median_us,dec_time.median_us,gemm_time.median_us,cand_time.median_us,ref_time.median_us/cand_time.median_us,std::sqrt(err/std::max(norm,1e-30)),maxdiff,mismatches,nonfinite,w.bytes+xh.bytes+y.bytes+library_scratch.bytes);std::fflush(stdout);
        if(nonfinite||std::sqrt(err/std::max(norm,1e-30))>0.003) return 2;
        native(stream);CUDA_CHECK(cudaStreamSynchronize(stream));cand.copy_to_host(c.data(),cand.bytes);
        err=0;mismatches=0;nonfinite=0;
        for(std::size_t i=0;i<r.size();++i){double d=static_cast<double>(from_bf16(c[i]))-from_bf16(r[i]);err+=d*d;mismatches+=r[i]!=c[i];nonfinite+=!std::isfinite(from_bf16(c[i]));}
        const auto native_time=bench::measure_cold_launch(native,flush,stream,3,12);
        std::printf("NATIVE n=%d k=%d tokens=%d total_us=%.3f speedup=%.6f rel_l2=%.8g mismatches=%zu nonfinite=%zu\n",n,k,tokens,native_time.median_us,ref_time.median_us/native_time.median_us,std::sqrt(err/std::max(norm,1e-30)),mismatches,nonfinite);std::fflush(stdout);
        if(nonfinite||std::sqrt(err/std::max(norm,1e-30))>0.003) return 2;
    }
    blas_check(cublasDestroy(h));CUDA_CHECK(cudaStreamDestroy(stream));return 0;
 } catch(const std::exception& e){std::fprintf(stderr,"sm75_prefill_blas_bench: %s\n",e.what());return 1;}
}
