#include "../../.chatgpt/production_serve_20261001/group_scaled_native_half_dense/kernel.cuh"
#include "../../.chatgpt/production_serve_20260928/head_local_projection_bound/source_weights.cuh"
#include "ops/linear/sm75_radix_projection.h"
#include "ops/gdn_input_proj/q4_q5/q5_a8_sm75.h"
#include "ninfer_bench_common.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include <chrono>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
using namespace ninfer;
namespace src=coarse_gdn;
__global__ void pack_outputs(const __nv_bfloat16* qk,const __nv_bfloat16* vz,__nv_bfloat16* qkv,__nv_bfloat16* z){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<4*10240){int t=i/10240,row=i%10240;qkv[i]=row<4096?qk[t*4096+row]:vz[t*12288+row-4096];}
 if(i<4*6144){int t=i/6144,row=i%6144;z[i]=vz[t*12288+6144+row];}
}
std::vector<std::uint16_t> inputs(int k,unsigned seed){std::vector<std::uint16_t> result(k*4);for(std::size_t i=0;i<result.size();++i)result[i]=__bfloat16_as_ushort(__float2bfloat16_rn(float(int((i*1664525U+seed)%2049)-1024)*.002f));return result;}
float bf(std::uint16_t v){unsigned bits=unsigned(v)<<16;float f;std::memcpy(&f,&bits,4);return f;}
float hf(const std::byte* ptr){std::uint16_t v;std::memcpy(&v,ptr,2);return __half2float(__ushort_as_half(v));}
template<class T>std::vector<T> download(DeviceBuffer& b){std::vector<T> v(b.bytes/sizeof(T));b.copy_to_host(v.data(),b.bytes);return v;}
void reference(const src::HostPacked& w,const std::vector<std::uint16_t>& input,DeviceBuffer& result){
 const auto y=download<std::uint16_t>(result);double error=0,norm=0;unsigned nf=0;
 for(int i=0;i<128;++i){int t=i%4,row=(i*131+i/11*569)%w.n;double expected=0;
  for(int k=0;k<w.k;++k){const auto at=std::size_t(row)*w.padded_k+k;int q=(std::to_integer<unsigned>(w.low[at/2])>>(4*(at%2)))&15;
   if(w.qtype==QType::Q5G64_F16S){q|=((std::to_integer<unsigned>(w.high[at/8])>>(at%8))&1)<<4;q=(q^16)-16;}else q=(q^8)-8;
   expected+=double(q)*hf(w.scales.data()+(std::size_t(row)*(w.padded_k/64)+k/64)*2)*bf(input[t*w.k+k]);
  }
  const double got=bf(y[t*w.n+row]);nf+=!std::isfinite(got);error+=(got-expected)*(got-expected);norm+=expected*expected;
 }
 const double rel=std::sqrt(error/std::max(norm,1e-30));printf("FP64_REFERENCE n=%d k=%d relative_L2=%.9g nonfinite=%u\n",w.n,w.k,rel,nf);
 if(nf||rel>.004)throw std::runtime_error("unquantized activation projection differs from independent reference");
 for(std::size_t i=w.n*4;i<y.size();++i)if(y[i]!=0x5555)throw std::runtime_error("projection output guard overwritten");
}
struct Plan {
 DeviceBuffer low,high,scales,output;int n,k;bool q5;
 Plan(const Weight& w,cudaStream_t stream):low(std::size_t(w.n)*w.k/2),high(w.qhigh?std::size_t(w.n)*w.k/8:16),scales(std::size_t(w.n)*(w.k/64)*2),output(w.n*4*2+64),n(w.n),k(w.k),q5(w.qhigh!=nullptr){
  group_half::prepare<<<(n/16*(k/64)*32+255)/256,256,0,stream>>>((const unsigned char*)w.qdata,(const unsigned char*)w.qhigh,(const __half*)w.scales,(uint4*)low.p,q5?(unsigned*)high.p:nullptr,(__half*)scales.p,n,k);CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemsetAsync(output.p,0x55,output.bytes,stream));
 }
 void run(DeviceBuffer& input,cudaStream_t stream){
  if(q5)group_half::project<true><<<n/16,128,0,stream>>>((const uint4*)low.p,(const unsigned*)high.p,(const __half*)scales.p,(const __half*)input.p,(__nv_bfloat16*)output.p,n,k);
  else group_half::project<false><<<n/16,128,0,stream>>>((const uint4*)low.p,nullptr,(const __half*)scales.p,(const __half*)input.p,(__nv_bfloat16*)output.p,n,k);
  CUDA_CHECK(cudaGetLastError());
 }
};
struct RadixLifetime {
 Weight* q;Weight* o;
 ~RadixLifetime(){ops::detail::sm75_radix_projection_release(*q);ops::detail::sm75_radix_projection_release(*o);}
};
int main(int argc,char**argv){try{
 if(argc!=2)throw std::invalid_argument("exact artifact path required");artifact::Reader reader(argv[1]);
 cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 DeviceBuffer error(4),flush(256ULL<<20);CUDA_CHECK(cudaMemsetAsync(error.p,0,4,stream));group_half::codec_test<<<4,256,0,stream>>>((unsigned*)error.p);CUDA_CHECK(cudaStreamSynchronize(stream));
 if(download<unsigned>(error)[0])throw std::runtime_error("integer-pair decoding is not exact");
 printf("INTEGER_PAIR_REFERENCE Q4=256 Q5=1024 exact=1\n");
 for(int layer:{0,30,62}){
  auto h=src::load_layer(reader,layer);src::DevicePacked qk(0,h.qk),vz(0,h.value_z),out(0,h.output),native_vz(0,h.value_z);
  Plan pq(qk.weight,stream),pv(vz.weight,stream),po(out.weight,stream);
  DeviceBuffer scratch(ops::detail::gdn_q5_sm75_transform_scratch_bytes());
  ops::detail::gdn_q5_sm75_transform_layout(vz.weight,QuantLayout::RowSplitSm75Fragment,scratch.p,stream);
  ops::detail::sm75_radix_projection_prepare(qk.weight,stream,4);ops::detail::sm75_radix_projection_prepare(out.weight,stream,4);
  RadixLifetime lifetime{&qk.weight,&out.weight};
  DeviceBuffer x(5120*4*2),g(6144*4*2),fx(x.bytes),fg(g.bytes),original_qkv(10240*4*2+64),original_z(6144*4*2+64),original_out(5120*4*2+64),qkv(original_qkv.bytes),z(original_z.bytes);
  WorkspaceArena workspace(ops::detail::gdn_q5_sm75_a8_workspace_bytes(4));
  Tensor xt(x.p,DType::BF16,{5120,4}),gt(g.p,DType::BF16,{6144,4}),qt(original_qkv.p,DType::BF16,{10240,4}),zt(original_z.p,DType::BF16,{6144,4}),ot(original_out.p,DType::BF16,{5120,4});
  auto current=[&](cudaStream_t st){ops::detail::gdn_q5_sm75_a8_launch(xt,qk.weight,vz.weight,qt,zt,workspace,st);if(!ops::detail::sm75_radix_projection_launch(gt,out.weight,ot,false,st))throw std::runtime_error("current output radix did not dispatch");};
  DeviceBuffer nq(4096*4*2+64),nv(12288*4*2+64),no(5120*4*2+64);
  Tensor nqt(nq.p,DType::BF16,{4096,4}),nvt(nv.p,DType::BF16,{12288,4}),not_(no.p,DType::BF16,{5120,4});
  auto native=[&](cudaStream_t st){
   ops::detail::launch_q4_simt_r8_c4(xt,qk.weight,nqt,st);
   const auto& w=native_vz.weight;
   ops::detail::q5_rowsplit_gemm_simt_split4_kernel<ops::detail::Q5RowSplitSimtSchedule,4,5,5120>
    <<<12288,128,0,st>>>((const __nv_bfloat16*)x.p,(const std::uint8_t*)w.qdata,(const std::uint8_t*)w.qhigh,(const std::uint8_t*)w.scales,(__nv_bfloat16*)nv.p,nullptr,12288,12288,5120,4,5120,5);
   ops::detail::launch_q5_simt_split2_r2_t4_sm75(gt,out.weight,not_,st);
   pack_outputs<<<(10240*4+255)/256,256,0,st>>>((const __nv_bfloat16*)nq.p,(const __nv_bfloat16*)nv.p,(__nv_bfloat16*)qkv.p,(__nv_bfloat16*)z.p);CUDA_CHECK(cudaGetLastError());
  };
  auto candidate=[&](cudaStream_t st){
   group_half::convert_input<<<(5120*4+255)/256,256,0,st>>>((const __nv_bfloat16*)x.p,(__half*)fx.p,(unsigned*)error.p,5120*4);
   group_half::convert_input<<<(6144*4+255)/256,256,0,st>>>((const __nv_bfloat16*)g.p,(__half*)fg.p,(unsigned*)error.p,6144*4);
   pq.run(fx,st);pv.run(fx,st);po.run(fg,st);
   pack_outputs<<<(10240*4+255)/256,256,0,st>>>((const __nv_bfloat16*)pq.output.p,(const __nv_bfloat16*)pv.output.p,(__nv_bfloat16*)qkv.p,(__nv_bfloat16*)z.p);CUDA_CHECK(cudaGetLastError());
  };
  for(int pattern=0;pattern<3;++pattern){
   auto hx=inputs(5120,20261001+layer+pattern*193),hg=inputs(6144,718299+layer+pattern*193);
   if(pattern==1){for(std::size_t i=0;i<hx.size();i+=173)hx[i]=__bfloat16_as_ushort(__float2bfloat16_rn(bf(hx[i])*16));for(std::size_t i=0;i<hg.size();i+=181)hg[i]=__bfloat16_as_ushort(__float2bfloat16_rn(bf(hg[i])*16));}
   if(pattern==2){std::fill(hx.begin(),hx.end(),0);std::fill(hg.begin(),hg.end(),0);}
   x.copy_from_host(hx.data(),x.bytes);g.copy_from_host(hg.data(),g.bytes);CUDA_CHECK(cudaDeviceSynchronize());
   candidate(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
   if(download<unsigned>(error)[0])throw std::runtime_error("BF16 input not exactly representable in FP16");
   reference(h.qk,hx,pq.output);reference(h.value_z,hx,pv.output);reference(h.output,hg,po.output);
   printf("PATTERN_REFERENCE layer=%d pattern=%d original_weights_unchanged=1 no_model_quality_claim=1\n",layer,pattern);
  }
  auto hx=inputs(5120,20261001+layer),hg=inputs(6144,718299+layer);x.copy_from_host(hx.data(),x.bytes);g.copy_from_host(hg.data(),g.bytes);CUDA_CHECK(cudaDeviceSynchronize());candidate(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto cq=download<unsigned>(pq.output),cv=download<unsigned>(pv.output),co=download<unsigned>(po.output);
  cudaGraph_t a{},b{},c{};cudaGraphExec_t ea{},eb{},ec{};
  CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));current(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&a));CUDA_CHECK(cudaGraphInstantiate(&ea,a,nullptr,nullptr,0));
  CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));candidate(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&b));CUDA_CHECK(cudaGraphInstantiate(&eb,b,nullptr,nullptr,0));
  CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));native(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&c));CUDA_CHECK(cudaGraphInstantiate(&ec,c,nullptr,nullptr,0));
  for(int pass=0;pass<2;++pass){
   auto a0=[&](cudaStream_t st){CUDA_CHECK(cudaGraphLaunch(ea,st));};auto a1=[&](cudaStream_t st){CUDA_CHECK(cudaGraphLaunch(eb,st));};auto a2=[&](cudaStream_t st){CUDA_CHECK(cudaGraphLaunch(ec,st));};double t0,t1,t2;
   if(pass==0){t0=bench::measure_cold_launch(a0,flush,stream,4,24).median_us;t2=bench::measure_cold_launch(a2,flush,stream,4,24).median_us;t1=bench::measure_cold_launch(a1,flush,stream,4,24).median_us;}
   else{t1=bench::measure_cold_launch(a1,flush,stream,4,24).median_us;t2=bench::measure_cold_launch(a2,flush,stream,4,24).median_us;t0=bench::measure_cold_launch(a0,flush,stream,4,24).median_us;}
   printf("LIVE_NATIVE_ROLE layer=%d pass=%d current_C1_us=%.3f current_dual_native_us=%.3f candidate_us=%.3f dual_ratio=%.6f\n",layer,pass,t0,t2,t1,t2/t1);fflush(stdout);
  }
  CUDA_CHECK(cudaGraphExecDestroy(ec));CUDA_CHECK(cudaGraphDestroy(c));
  if(cq!=download<unsigned>(pq.output)||cv!=download<unsigned>(pv.output)||co!=download<unsigned>(po.output))throw std::runtime_error("same-source graph output changed");
  if(download<unsigned>(error)[0])throw std::runtime_error("range failure during cost test");
  CUDA_CHECK(cudaGraphExecDestroy(ea));CUDA_CHECK(cudaGraphExecDestroy(eb));CUDA_CHECK(cudaGraphDestroy(a));CUDA_CHECK(cudaGraphDestroy(b));
  printf("REPLAY_EXACT layer=%d prepared_weight_bytes=%zu\n",layer,pq.low.bytes+pq.high.bytes+pq.scales.bytes+pv.low.bytes+pv.high.bytes+pv.scales.bytes+po.low.bytes+po.high.bytes+po.scales.bytes);
 }
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){fprintf(stderr,"group-scale native half: %s\n",e.what());return 1;}}
