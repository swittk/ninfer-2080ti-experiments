#include "core/device.h"
#include "core/arena.h"
#include "../apps/dual/joint_kv/joint_residual.cuh"
#include "../apps/dual/joint_kv/integer_coordinates.cuh"
#include "../apps/dual/joint_kv/query_input_prepare.cuh"

#include <algorithm>
#include <vector>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
using namespace ninfer;
constexpr int Rows=96,H=256,K=64,Guard=64;
namespace vr=ops::value_residual;namespace jr=ops::joint_residual;namespace ij=ops::integer_joint;
unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
float unit(unsigned i){return float(int(mix(i)%2049)-1024)/1024.f;}
std::uint16_t bf(float v){return __bfloat16_as_ushort(__float2bfloat16_rn(v));}
std::uint16_t hf(float v){return __half_as_ushort(__float2half_rn(v));}
float fbf(std::uint16_t v){unsigned bits=unsigned(v)<<16;float x;std::memcpy(&x,&bits,4);return x;}
float fhf(std::uint16_t v){return __half2float(__ushort_as_half(v));}
template<class T>std::vector<T> read(DeviceBuffer& b){std::vector<T> x(b.bytes/sizeof(T));b.copy_to_host(x.data(),b.bytes);return x;}
template<class T>void upload(DeviceBuffer& b,const std::vector<T>& x){b.copy_from_host(x.data(),x.size()*sizeof(T));}
int main(){try{
 cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 DeviceBuffer input(Rows*H*2),basis(4*H*K*2),mean(4*H*2),original_codes(Rows*H),original_scale(Rows*4),
  p0(Rows*K*2+Guard),s0(Rows*4+Guard),q0(Rows*K+Guard),qs0(Rows*4+Guard),p1(p0.bytes),s1(s0.bytes),q1(q0.bytes),qs1(qs0.bytes),union_mask((4*8193+Guard)*4),metadata((104+Guard)*4);
 int union_words=4*8193;
 auto original=[&](cudaStream_t st){
  vr::quantize_queries<<<12,256,0,st>>>((const __nv_bfloat16*)input.p,(std::int8_t*)original_codes.p,(float*)original_scale.p);
  jr::project_query<<<96,256,0,st>>>((const std::int8_t*)original_codes.p,(const float*)original_scale.p,(const __half*)basis.p,(const __half*)mean.p,(__half*)p0.p,(float*)s0.p);
  ij::quantize_projected_queries<<<12,256,0,st>>>((const __half*)p0.p,(std::int8_t*)q0.p,(float*)qs0.p);
 };
 auto candidate=[&](cudaStream_t st){jr::prepare_integer_query_roles<<<96,288,0,st>>>((const __nv_bfloat16*)input.p,(const __half*)basis.p,(const __half*)mean.p,(__half*)p1.p,(float*)s1.p,(std::int8_t*)q1.p,(float*)qs1.p,(unsigned*)union_mask.p,union_words,(int*)metadata.p);};
 cudaGraph_t graphs[2]{};cudaGraphExec_t exec[2]{};
 for(int arm=0;arm<2;++arm){CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));if(arm)candidate(stream);else original(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&graphs[arm]));CUDA_CHECK(cudaGraphInstantiate(&exec[arm],graphs[arm],nullptr,nullptr,0));}
 std::size_t compared=0,scalar=0;
 for(int pattern=0;pattern<14;++pattern){
  std::vector<std::uint16_t> x(Rows*H),b(4*H*K),m(4*H);
  for(int i=0;i<Rows*H;++i)x[i]=bf(unit(i+pattern*7331)*2.f);
  for(int i=0;i<4*H*K;++i)b[i]=hf(unit(i+pattern*1777)*.03f);
  for(int i=0;i<4*H;++i)m[i]=hf(unit(i+pattern*8137)*.1f);
  if(pattern==1)std::fill(x.begin(),x.end(),0);
  if(pattern==2)std::fill(b.begin(),b.end(),0);
  if(pattern==3)for(int i=0;i<Rows*H;i+=131)x[i]=bf(fbf(x[i])*32768.f);
  if(pattern==4)for(auto& v:x)v=bf(fbf(v)*1.e-15f);
  if(pattern==5)for(int i=0;i<Rows*H;i+=137)x[i]=0x7fc1;
  if(pattern==6)for(int i=0;i<Rows*H;i+=181)x[i]=0x7f80;
  if(pattern==7)for(int i=0;i<4*H*K;i+=1337)b[i]=0x7e01;
  if(pattern==8)for(int i=0;i<4*H*K;i+=8191)b[i]=0x7c00;
  if(pattern==9)for(int i=0;i<4*H;i+=113)m[i]=0x7e01;
  if(pattern==10)for(int i=0;i<Rows*H;++i)x[i]=(i%2)?0:0x8000;
  if(pattern==11)for(int i=0;i<4*H*K;++i)b[i]=(i%2)?0:0x8000;
  upload(input,x);upload(basis,b);upload(mean,m);
  for(auto* v:{&p0,&s0,&q0,&qs0,&p1,&s1,&q1,&qs1,&union_mask,&metadata})CUDA_CHECK(cudaMemset(v->p,0x55,v->bytes));CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGraphLaunch(exec[0],stream));CUDA_CHECK(cudaGraphLaunch(exec[1],stream));CUDA_CHECK(cudaStreamSynchronize(stream));
  for(auto pair:{std::pair<DeviceBuffer*,DeviceBuffer*>{&p0,&p1},{&s0,&s1},{&q0,&q1},{&qs0,&qs1}}){
   auto a=read<unsigned char>(*pair.first),v=read<unsigned char>(*pair.second);
   if(a!=v){auto pos=std::mismatch(a.begin(),a.end(),v.begin()).first-a.begin();printf("MISMATCH pattern=%d bytes=%zu at=%zu old=%u new=%u\n",pattern,a.size(),std::size_t(pos),unsigned(a[pos]),unsigned(v[pos]));throw std::runtime_error("original query chain differs");}
   for(std::size_t i=a.size()-Guard;i<a.size();++i)if(a[i]!=0x55)throw std::runtime_error("output guard");compared+=a.size()-Guard;
  }
  if(read<std::uint16_t>(input)!=x||read<std::uint16_t>(basis)!=b||read<std::uint16_t>(mean)!=m)throw std::runtime_error("immutable input mutated");
  for(auto pair:{std::pair<DeviceBuffer*,int>{&union_mask,union_words},{&metadata,104}}){
   const auto values=read<unsigned>(*pair.first);for(int i=0;i<pair.second;++i)if(values[i]!=0)throw std::runtime_error("producer state initialization mismatch");
   for(std::size_t i=pair.second;i<values.size();++i)if(values[i]!=0x55555555U)throw std::runtime_error("initialization guard");
  }
  // Independent scalar projection on finite ordinary/zero/small/source-change cases.
  if(pattern<3||pattern==4||pattern>=10){
   auto codes=read<signed char>(original_codes);auto scales=read<float>(original_scale);auto proj=read<std::uint16_t>(p1);auto shifts=read<float>(s1);
   for(int r=0;r<Rows;++r){float maximum=0;for(int i=0;i<H;++i)maximum=std::max(maximum,std::fabs(fbf(x[r*H+i])));float scale=maximum>0?maximum/127.f:0,inv=scale>0?1.f/scale:0;
    if(scale!=scales[r])throw std::runtime_error("CPU scale");for(int i=0;i<H;++i)if(int(codes[r*H+i])!=std::clamp(int(std::nearbyint(fbf(x[r*H+i])*inv)),-127,127))throw std::runtime_error("CPU query quantizer");
    int head=(r%24)/6;
    float ms[4]{};
    for(int part=0;part<4;++part)for(int k=0;k<64;++k)
      ms[part]=std::fma(float(codes[r*H+part*64+k]),fhf(m[head*H+part*64+k]),ms[part]);
    const float expected_shift=((ms[0]+ms[1])+(ms[2]+ms[3]))*scale*.0625f;
    unsigned shift_bits,observed_shift_bits;std::memcpy(&shift_bits,&expected_shift,4);std::memcpy(&observed_shift_bits,&shifts[r],4);
    if(shift_bits!=observed_shift_bits)throw std::runtime_error("CPU ordered mean projection");
    for(int d=0;d<K;++d){float sums[4]{};for(int j=0;j<4;++j)for(int k=0;k<64;++k)sums[j]=std::fma(float(codes[r*H+j*64+k]),fhf(b[(head*H+j*64+k)*K+d]),sums[j]);float total=(sums[0]+sums[1])+(sums[2]+sums[3]);if(hf(total*scale)!=proj[r*K+d])throw std::runtime_error("CPU projection");++scalar;}
   }
   const auto final_codes=read<signed char>(q1);const auto final_scales=read<float>(qs1);
   for(int r=0;r<Rows;++r){float maximum=0;for(int d=0;d<K;++d)maximum=std::max(maximum,std::fabs(fhf(proj[r*K+d])));
    const float scale=maximum/127.f,inv=scale>0?1.f/scale:0;
    if(scale!=final_scales[r])throw std::runtime_error("CPU projected-query scale");
    for(int d=0;d<K;++d)if(int(final_codes[r*K+d])!=std::clamp(int(std::nearbyint(fhf(proj[r*K+d])*inv)),-127,127))throw std::runtime_error("CPU projected-query codes");
   }
  }
  // Keep query zero fixed while replacing every other query. The projection and
  // integer scales must not depend on other query/tokens in the verification batch.
  auto before_p=read<std::uint16_t>(p1);auto before_q=read<signed char>(q1);auto before_s=read<unsigned>(s1),before_qs=read<unsigned>(qs1);
  for(int i=H;i<Rows*H;++i)x[i]=bf(unit(i+pattern*1889)*7.f);upload(input,x);CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGraphLaunch(exec[1],stream));CUDA_CHECK(cudaStreamSynchronize(stream));auto after_p=read<std::uint16_t>(p1);auto after_q=read<signed char>(q1);
  if(!std::equal(before_p.begin(),before_p.begin()+K,after_p.begin())||!std::equal(before_q.begin(),before_q.begin()+K,after_q.begin())||before_s[0]!=read<unsigned>(s1)[0]||before_qs[0]!=read<unsigned>(qs1)[0])throw std::runtime_error("cross-query dependence");
  printf("QUERY_CHAIN_EXACT pattern=%d projection=6144 codes=6144 scales_and_shifts=192 guards_readonly_causal_graph=1\n",pattern);
 }
 for(int words:{0,1,4,124,128,132,16384,32768,32772}){
  union_words=words;CUDA_CHECK(cudaMemset(union_mask.p,0x55,union_mask.bytes));CUDA_CHECK(cudaMemset(metadata.p,0x55,metadata.bytes));
  candidate(stream);CUDA_CHECK(cudaStreamSynchronize(stream));auto u=read<unsigned>(union_mask);auto m=read<unsigned>(metadata);
  for(std::size_t i=0;i<u.size();++i)if(u[i]!=(i<std::size_t(words)?0U:0x55555555U))throw std::runtime_error("initialization extent");
  for(std::size_t i=0;i<m.size();++i)if(m[i]!=(i<104?0U:0x55555555U))throw std::runtime_error("metadata extent");
 }
 printf("QUERY_CHAIN_PASS compared_bytes=%zu independent_projection_values=%zu\n",compared,scalar);
 for(int arm=0;arm<2;++arm){CUDA_CHECK(cudaGraphExecDestroy(exec[arm]));CUDA_CHECK(cudaGraphDestroy(graphs[arm]));}
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){fprintf(stderr,"query input fusion: %s\n",e.what());return 1;}}
