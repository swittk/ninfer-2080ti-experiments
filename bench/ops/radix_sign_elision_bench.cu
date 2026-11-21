#include "core/device.h"
#include "core/arena.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include "t16_fixture.h"
#include "../../.chatgpt/production_serve_20260914/radix_sign_elision/legacy_radix.cuh"
#include "../../.chatgpt/production_serve_20260914/radix_sign_elision/compact_radix.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdexcept>
using namespace ninfer;
__global__ void exhaust_unpack(int* failures) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=1048576)return;
    const unsigned p=i&65535u,h=i>>16;
    if(bench::legacy_radix::expand(p,h)!=ninfer::bench::compact_radix::compact_expand<true>(p,h))atomicAdd(failures,1);
    unsigned signs=0;for(int j=0;j<4;++j)signs|=((p>>(j*4+3))&1u)<<j;
    if(bench::legacy_radix::expand(p,signs)!=ninfer::bench::compact_radix::compact_expand<false>(p,0))atomicAdd(failures,1);
}
int main(int argc,char**argv){
 try {
  const unsigned seed=argc>1?std::stoul(argv[1]):1234567;
  cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  DeviceBuffer flush(256ULL<<20);
  DeviceBuffer faults(sizeof(int));CUDA_CHECK(cudaMemsetAsync(faults.p,0,sizeof(int),stream));
  exhaust_unpack<<<4096,256,0,stream>>>((int*)faults.p);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
  int fault_count=0;faults.copy_to_host(&fault_count,sizeof(int));if(fault_count)throw std::runtime_error("exhaustive sign expansion mismatch");
  printf("PASS exhaustive Q4/Q5 2097152 nibble/sign comparisons\n");
  for(bool q5:{false,true}){
   const int n=q5?5120:4096,k=q5?6144:5120,t=4;
   auto w=bench::make_row_split_weight(q5?QType::Q5G64_F16S:QType::Q4G64_F16S,n,k,k);
   bench::t16_fill_weight(w.weight,n,k,q5,seed+q5);
   DeviceBuffer x(k*t*2),initial(n*t*2),ref(n*t*2),out(n*t*2);
   DeviceBuffer packed(n*k/2),high(n*k/8),scale(n*k/64*2),aq(k*8),as(k/64*4*4),tmp(n*8*4*4);
   if(q5)bench::legacy_radix::repack<true><<<(n/16*(k/64)*32+255)/256,256,0,stream>>>((const unsigned char*)w.weight.qdata,(const unsigned char*)w.weight.qhigh,(const __half*)w.weight.scales,(uint4*)packed.p,(unsigned*)high.p,(__half*)scale.p,n,k);
   else bench::legacy_radix::repack<false><<<(n/16*(k/64)*32+255)/256,256,0,stream>>>((const unsigned char*)w.weight.qdata,nullptr,(const __half*)w.weight.scales,(uint4*)packed.p,(unsigned*)high.p,(__half*)scale.p,n,k);
   bench::t16_fill_input(x.p,k*t,seed);bench::t16_fill_input(initial.p,n*t,seed^0x1234);
   Tensor xt(x.p,DType::BF16,{k,t}),rt(ref.p,DType::BF16,{n,t}),yt(out.p,DType::BF16,{n,t});
   WorkspaceArena ws(8ULL<<20);
   auto call=[&](cudaStream_t s,bool compact,void* output) {
    bench::legacy_radix::quantize<<<(4*(k/64)*32+255)/256,256,0,s>>>((const __nv_bfloat16*)x.p,(signed char*)aq.p,(float*)as.p,k);
    if(compact) {
      if(q5)bench::compact_radix::matmul<4,true><<<dim3(n/64,4),128,0,s>>>((const uint4*)packed.p,(const unsigned*)high.p,(const __half*)scale.p,(const signed char*)aq.p,(const float*)as.p,(float*)tmp.p,n,k);
      else bench::compact_radix::matmul<4,false><<<dim3(n/64,4),128,0,s>>>((const uint4*)packed.p,nullptr,(const __half*)scale.p,(const signed char*)aq.p,(const float*)as.p,(float*)tmp.p,n,k);
    } else bench::legacy_radix::matmul<4><<<dim3(n/64,4),128,0,s>>>((const uint4*)packed.p,(const unsigned*)high.p,(const __half*)scale.p,(const signed char*)aq.p,(const float*)as.p,(float*)tmp.p,n,k);
    if(q5)bench::legacy_radix::finish<4,true><<<(n*4+255)/256,256,0,s>>>((const float*)tmp.p,(const __nv_bfloat16*)initial.p,(__nv_bfloat16*)output,n,n);
    else bench::legacy_radix::finish<4,false><<<(n*4+255)/256,256,0,s>>>((const float*)tmp.p,nullptr,(__nv_bfloat16*)output,n,n);
    CUDA_CHECK(cudaGetLastError());
   };
   const auto baseline=[&](cudaStream_t s){call(s,false,ref.p);};
   const auto candidate=[&](cudaStream_t s){call(s,true,out.p);};
   baseline(stream);candidate(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
   std::vector<__nv_bfloat16> a(n*t),b(n*t);ref.copy_to_host(a.data(),ref.bytes);out.copy_to_host(b.data(),out.bytes);
   double diff=0,norm=0;int mismatch=0,nonfinite=0;
   for(int i=0;i<n*t;++i){float av=__bfloat162float(a[i]),bv=__bfloat162float(b[i]);mismatch+=av!=bv;nonfinite+=!std::isfinite(bv);diff+=(av-bv)*(av-bv);norm+=av*av;}
   if(nonfinite || mismatch)throw std::runtime_error("exact output mismatch");
   cudaGraph_t gr{},gc{};cudaGraphExec_t er{},ec{};
   CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));baseline(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gr));CUDA_CHECK(cudaGraphInstantiate(&er,gr,nullptr,nullptr,0));
   CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));candidate(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gc));CUDA_CHECK(cudaGraphInstantiate(&ec,gc,nullptr,nullptr,0));
   const auto br=[&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(er,s));};const auto cr=[&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(ec,s));};
   for(int pass=0;pass<2;++pass){auto a=bench::measure_cold_launch(br,flush,stream,4,24);auto b=bench::measure_cold_launch(cr,flush,stream,4,24);std::printf("RESULT q%d N=%d K=%d T=%d pass=%d baseline_us=%.3f tensor_us=%.3f speedup=%.5f rel_l2=%.8f mismatches=%d nonfinite=%d\n",q5?5:4,n,k,t,pass,a.median_us,b.median_us,a.median_us/b.median_us,std::sqrt(diff/norm),mismatch,nonfinite);}
   CUDA_CHECK(cudaGraphExecDestroy(er));CUDA_CHECK(cudaGraphExecDestroy(ec));CUDA_CHECK(cudaGraphDestroy(gr));CUDA_CHECK(cudaGraphDestroy(gc));
  }
  CUDA_CHECK(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"projection_oracle: %s\n",e.what());return 1;}
}
