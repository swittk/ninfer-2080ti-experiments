// Bounded architecture oracle. No production dispatch is enabled here.
#include "core/device.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_a8_sm75.h"
#include "ops/marlin/mtp_q4_marlin.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
using namespace ninfer;
constexpr int M=17408,K=5120,T=4,G=M/64;
__global__ void finish_marlin(const __nv_bfloat16* gu, signed char* output, __half* scales) {
    const int unit=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(unit>=T*G)return;
    const int t=unit/G,g=unit%G;
    float v[2];
#pragma unroll
    for(int j=0;j<2;++j) {
        const int r=g*64+lane+j*32;
        const float gate=__bfloat162float(gu[t*2*M+r]);
        const float up=__bfloat162float(gu[t*2*M+M+r]);
        v[j]=__bfloat162float(__float2bfloat16_rn((gate/(1.f+expf(-gate)))*up));
    }
    float a=fmaxf(fabsf(v[0]),fabsf(v[1]));
#pragma unroll
    for(int off=16;off;off/=2)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,off));
    const float s=a?a/127.f:0.f,inv=s?1.f/s:0.f;
    if(lane==0)scales[unit]=__float2half_rn(s);
#pragma unroll
    for(int j=0;j<2;++j)output[t*M+g*64+lane+j*32]=max(-127,min(127,__float2int_rn(v[j]*inv)));
}
int main(int argc,char**argv) {
 try {
    const unsigned seed=argc>1?std::stoul(argv[1]):1234567u;
    setenv("NINFER_MTP_MARLIN_Q4","1",1);
    cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    {
      auto native=bench::make_row_split_weight(QType::Q4G64_F16S,2*M,K,K);
      auto fragment=bench::make_row_split_weight(QType::Q4G64_F16S,2*M,K,K);
      bench::t16_fill_weight(native.weight,2*M,K,false,seed);
      bench::t16_fill_weight(fragment.weight,2*M,K,false,seed);
      DeviceBuffer xbuf(K*T*2),gu(2*M*T*2),rq(M*T),cq(M*T),rs(G*T*2),cs(G*T*2),flush(256ULL<<20);
      bench::t16_fill_input(xbuf.p,K*T,seed^231u);
      ops::detail::mtp_q4_marlin_prepare(native.weight,stream);
      struct Release{const Weight&w;~Release(){ops::detail::mtp_q4_marlin_release(w);}}owner{native.weight};
      DeviceBuffer trans(ops::detail::q4_linear_swiglu_sm75_transform_scratch_bytes());
      ops::detail::q4_linear_swiglu_sm75_transform_layout(fragment.weight,QuantLayout::RowSplitSm75Fragment,trans.p,trans.bytes,stream);
      WorkspaceArena ws(ops::detail::q4_linear_swiglu_sm75_a8_workspace_bytes(T));
      Tensor x(xbuf.p,DType::BF16,{K,T}),projected(gu.p,DType::BF16,{2*M,T});
      Tensor qr(rq.p,DType::I8,{M,T}),qc(cq.p,DType::I8,{M,T});
      Tensor sr(rs.p,DType::FP16,{G,T}),sc(cs.p,DType::FP16,{G,T});
      auto reference=[&](cudaStream_t s){ops::detail::q4_linear_swiglu_sm75_a8_to_a8_launch(x,fragment.weight,qr,sr,ws,s);};
      auto candidate=[&](cudaStream_t s){
        if(!ops::detail::mtp_q4_marlin_launch(x,native.weight,projected,s))throw std::runtime_error("Marlin shape rejected");
        finish_marlin<<<(G*T+7)/8,256,0,s>>>(static_cast<const __nv_bfloat16*>(gu.p),static_cast<signed char*>(cq.p),static_cast<__half*>(cs.p));
        CUDA_CHECK(cudaGetLastError());
      };
      reference(stream);candidate(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
      std::vector<signed char>a(M*T),b(M*T);std::vector<__half>sa(G*T),sb(G*T);
      rq.copy_to_host(a.data(),rq.bytes);cq.copy_to_host(b.data(),cq.bytes);rs.copy_to_host(sa.data(),rs.bytes);cs.copy_to_host(sb.data(),cs.bytes);
      double err=0,norm=0;int mismatch=0,nonfinite=0;
      for(int i=0;i<M*T;++i){const double av=a[i]*__half2float(sa[i/64]),bv=b[i]*__half2float(sb[i/64]);err+=(av-bv)*(av-bv);norm+=av*av;mismatch+=a[i]!=b[i];nonfinite+=!std::isfinite(bv);}
      for(int pass=0;pass<2;++pass){
        auto r=bench::measure_cold_launch(reference,flush,stream,3,20);
        auto c=bench::measure_cold_launch(candidate,flush,stream,3,20);
        printf("RESULT pass=%d reference_us=%.3f marlin_us=%.3f speedup=%.6f output_rel_l2=%.8f code_mismatches=%d nonfinite=%d\n",pass,r.median_us,c.median_us,r.median_us/c.median_us,sqrt(err/std::max(norm,1e-30)),mismatch,nonfinite);
      }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));return 0;
 } catch(const std::exception&e){fprintf(stderr,"q4_c1_marlin_oracle: %s\n",e.what());return 1;}
}
