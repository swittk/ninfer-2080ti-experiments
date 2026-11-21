#include "core/device.h"
#include "core/arena.h"
#include "ops/linear/sm75_radix_projection.h"
#include "ops/linear/sm75_radix_prefill.cuh"
#include "targets/qwen3_6_27b/impl/sm75_prefill_dense.h"

#include "../bench/ops/quantized_weight.cuh"
#include "../bench/ops/t16_fixture.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>
using namespace ninfer;
using ninfer::targets::qwen3_6_27b::detail::Sm75PrefillDense;
std::vector<std::uint8_t> bytes(const DeviceBuffer& b){std::vector<std::uint8_t> v(b.bytes);b.copy_to_host(v.data(),v.size());return v;}
void equal(const DeviceBuffer& a,const DeviceBuffer& b,const char* label){if(bytes(a)!=bytes(b))throw std::runtime_error(label);}
// Both paths are warmed before capture; their BLAS plans and source pointers
// stay fixed. Different inputs are installed between replays below.
template<class A,class B,class Prepare,class Validate>
void replay_pair(A original,B revised,Prepare prepare,Validate validate,cudaStream_t stream) {
    cudaGraph_t ga{},gb{};cudaGraphExec_t ea{},eb{};
    CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));original(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream,&ga));CUDA_CHECK(cudaGraphInstantiate(&ea,ga,nullptr,nullptr,0));
    CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));revised(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream,&gb));CUDA_CHECK(cudaGraphInstantiate(&eb,gb,nullptr,nullptr,0));
    try {
        for(int repeat=0;repeat<3;++repeat) {
            prepare(repeat);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGraphLaunch(ea,stream));CUDA_CHECK(cudaGraphLaunch(eb,stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));validate();
        }
    }catch(...){(void)cudaGraphExecDestroy(ea);(void)cudaGraphExecDestroy(eb);(void)cudaGraphDestroy(ga);(void)cudaGraphDestroy(gb);throw;}
    CUDA_CHECK(cudaGraphExecDestroy(ea));CUDA_CHECK(cudaGraphExecDestroy(eb));CUDA_CHECK(cudaGraphDestroy(ga));CUDA_CHECK(cudaGraphDestroy(gb));
}
template<bool Q5,int N,int K>
__global__ void canonical_expand(const std::uint8_t* low,const std::uint8_t* high,const __half* scales,__half* out){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N*K;i+=gridDim.x*blockDim.x){
  int q=(low[i/2]>>(4*(i%2)))&15;
  if constexpr(Q5){q|=((high[i/8]>>(i%8))&1)<<4;q=(q^16)-16;}else q=(q^8)-8;
  out[i]=__float2half_rn(__bfloat162float(__float2bfloat16_rn(q*__half2float(scales[i/64]))));
 }
}
template<bool Q5,int N,int K>
void expansion_check(const Weight& canonical,const Weight& resident,cudaStream_t stream){
 DeviceBuffer ref(std::size_t(N)*K*2),out(ref.bytes+256);
 canonical_expand<Q5,N,K><<<4096,256,0,stream>>>((const std::uint8_t*)canonical.qdata,(const std::uint8_t*)canonical.qhigh,(const __half*)canonical.scales,(__half*)ref.p);
 for(auto range:std::vector<std::pair<int,int>>{{0,N},{1,17},{N-19,19}}){
  CUDA_CHECK(cudaMemsetAsync(out.p,0x55,out.bytes,stream));
  ops::detail::resident_prefill::expand_weights<Q5,N,K><<<4096,256,0,stream>>>(
   (const std::uint16_t*)resident.qdata,(const std::uint32_t*)resident.qhigh,(const std::uint16_t*)resident.scales,(__half2*)out.p,
   std::int64_t(range.second)*K/2,range.first);
  CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
  auto expected=bytes(ref),actual=bytes(out);auto count=std::size_t(range.second)*K*2,offset=std::size_t(range.first)*K*2;
  if(!std::equal(actual.begin(),actual.begin()+count,expected.begin()+offset))throw std::runtime_error("expanded FP16 operand mismatch");
  if(!std::all_of(actual.begin()+count,actual.end(),[](auto c){return c==0x55;}))throw std::runtime_error("expanded operand guard overwritten");
 }
 printf("EXACT_EXPANSION Q%d shape=%dx%d whole_operand_and_non_aligned_row_slices_match\n",Q5?5:4,N,K);
}
void run(int t,unsigned seed,cudaStream_t stream){

 auto qk=bench::make_row_split_weight(QType::Q4G64_F16S,4096,5120,5120);
 auto outweight=bench::make_row_split_weight(QType::Q5G64_F16S,5120,6144,6144);
 auto vz=bench::make_row_split_weight(QType::Q5G64_F16S,12288,5120,5120);
 bench::t16_fill_weight(qk.weight,4096,5120,false,seed);
 bench::t16_fill_weight(outweight.weight,5120,6144,true,seed^81273U);
 bench::t16_fill_weight(vz.weight,12288,5120,true,seed^99437U);CUDA_CHECK(cudaDeviceSynchronize());
 auto qk_original=bytes(qk.storage),out_original=bytes(outweight.storage);
 WorkspaceArena conversion(32ULL<<20);
 ops::detail::sm75_radix_resident_transform(qk.weight,QuantLayout::RowSplitSm75Radix,stream);
 ops::detail::sm75_radix_resident_transform(outweight.weight,QuantLayout::RowSplitSm75Radix,stream);
 {auto scope=conversion.scope();auto w=ops::detail::sm75_radix_canonical_copy(qk.weight,conversion,stream);CUDA_CHECK(cudaStreamSynchronize(stream));expansion_check<false,4096,5120>(w,qk.weight,stream);}
 {auto scope=conversion.scope();auto w=ops::detail::sm75_radix_canonical_copy(outweight.weight,conversion,stream);CUDA_CHECK(cudaStreamSynchronize(stream));expansion_check<true,5120,6144>(w,outweight.weight,stream);}
 const auto peak=std::max(Sm75PrefillDense::projection_workspace_bytes(12288,5120,t),Sm75PrefillDense::projection_workspace_bytes(5120,6144,t));
 WorkspaceArena w0(peak),w1(peak);Sm75PrefillDense old(w0,stream),now(w1,stream);
 DeviceBuffer x(std::size_t(5120)*t*2),a(std::size_t(10240)*t*2),b(a.bytes),z(std::size_t(6144)*t*2),zz(z.bytes);
 bench::t16_fill_input(x.p,x.bytes/2,seed^471U);CUDA_CHECK(cudaDeviceSynchronize());
 Tensor xt(x.p,DType::BF16,{5120,t}),at(a.p,DType::BF16,{10240,t}),bt(b.p,DType::BF16,{10240,t}),zt(z.p,DType::BF16,{6144,t}),zzt(zz.p,DType::BF16,{6144,t});
 auto g0=[&](cudaStream_t s){auto scope=conversion.scope();const Weight canonical=ops::detail::sm75_radix_canonical_copy(qk.weight,conversion,s);if(!old.gdn_input(xt,canonical,vz.weight,at,zt,s))throw std::runtime_error("original GDN unsupported");};
 auto g1=[&](cudaStream_t s){if(!now.gdn_input(xt,qk.weight,vz.weight,bt,zzt,s))throw std::runtime_error("direct GDN unsupported");};
 g0(stream);g1(stream);CUDA_CHECK(cudaStreamSynchronize(stream));equal(a,b,"GDN output differs");equal(z,zz,"GDN z differs");
 bench::t16_fill_input(x.p,x.bytes/2,seed^571U);CUDA_CHECK(cudaDeviceSynchronize());
 replay_pair(g0,g1,[&](int repeat){if(repeat==2)CUDA_CHECK(cudaMemset(x.p,0,x.bytes));else bench::t16_fill_input(x.p,x.bytes/2,seed+repeat+571U);},[&]{equal(a,b,"GDN graph replay differs");equal(z,zz,"GDN gate replay differs");},stream);
 DeviceBuffer ox(std::size_t(6144)*t*2),initial(std::size_t(5120)*t*2),ra(initial.bytes),rb(initial.bytes);
 bench::t16_fill_input(ox.p,ox.bytes/2,seed^481U);bench::t16_fill_input(initial.p,initial.bytes/2,seed^521U);CUDA_CHECK(cudaDeviceSynchronize());
 Tensor it(ox.p,DType::BF16,{6144,t}),rat(ra.p,DType::BF16,{5120,t}),rbt(rb.p,DType::BF16,{5120,t});
 auto r0=[&](cudaStream_t s){CUDA_CHECK(cudaMemcpyAsync(ra.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,s));auto scope=conversion.scope();const auto w=ops::detail::sm75_radix_canonical_copy(outweight.weight,conversion,s);if(!old.residual(it,w,rat,s))throw std::runtime_error("old output unsupported");};
 auto r1=[&](cudaStream_t s){CUDA_CHECK(cudaMemcpyAsync(rb.p,initial.p,initial.bytes,cudaMemcpyDeviceToDevice,s));if(!now.residual(it,outweight.weight,rbt,s))throw std::runtime_error("new output unsupported");};
 r0(stream);r1(stream);CUDA_CHECK(cudaStreamSynchronize(stream));equal(ra,rb,"residual output differs");bench::t16_fill_input(ox.p,ox.bytes/2,seed^581U);CUDA_CHECK(cudaDeviceSynchronize());
 replay_pair(r0,r1,[&](int repeat){if(repeat==2)CUDA_CHECK(cudaMemset(ox.p,0,ox.bytes));else bench::t16_fill_input(ox.p,ox.bytes/2,seed+repeat+581U);},[&]{equal(ra,rb,"residual graph replay differs");},stream);
 // Malformed resident layouts must not slip through a generic valid-weight check.
 { Weight invalid=outweight.weight;invalid.group_size=32;bool rejected=false;
   try{(void)now.residual(it,invalid,rbt,stream);}catch(const std::invalid_argument&){rejected=true;}
   if(!rejected)throw std::runtime_error("invalid resident group admitted"); }
 { Weight invalid=qk.weight;invalid.scale_dtype=DType::BF16;bool rejected=false;
   try{(void)now.gdn_input(xt,invalid,vz.weight,bt,zzt,stream);}catch(const std::invalid_argument&){rejected=true;}
   if(!rejected)throw std::runtime_error("invalid resident scale admitted"); }
 ops::detail::sm75_radix_resident_transform(qk.weight,QuantLayout::RowSplit,stream);ops::detail::sm75_radix_resident_transform(outweight.weight,QuantLayout::RowSplit,stream);
 if(bytes(qk.storage)!=qk_original||bytes(outweight.storage)!=out_original)throw std::runtime_error("source weights altered");
 printf("PASS tokens=%d exact_operands_and_full_outputs_source_unchanged workspace_native=%zu direct=%zu canonical_copy=%zu\n",t,w0.peak_used(),w1.peak_used(),conversion.peak_used());
}
int main(){try{
 int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0)return 77;
 cudaDeviceProp props{};CUDA_CHECK(cudaGetDeviceProperties(&props,0));if(props.major!=7||props.minor!=5)return 77;
 cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 try{run(64,20260920U,stream);run(257,20260971U,stream);}
 catch(...){(void)cudaStreamDestroy(stream);throw;}
 CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"resident prefill test: %s\n",e.what());return 1;}}
