#include "../apps/dual/joint_kv/query_owned_finish.cuh"
#include "core/arena.h"
#include "core/device.h"
#include <vector>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
using namespace ninfer;
namespace vr=ops::value_residual;
namespace jr=ops::joint_residual;
namespace rt=apps::dual::joint_runtime;
constexpr int Q=96,Guard=32;
template<class T>std::vector<T> read(DeviceBuffer& b){std::vector<T> x(b.bytes/sizeof(T));b.copy_to_host(x.data(),b.bytes);return x;}
template<class T>void upload(DeviceBuffer& b,const std::vector<T>& x){if(x.size()*sizeof(T)>b.bytes)throw std::runtime_error("upload extent");b.copy_from_host(x.data(),x.size()*sizeof(T));}
unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
float unit(unsigned x){return float(int(mix(x)%2049)-1024)/1024.f;}
std::uint16_t hf(float x){return __half_as_ushort(__float2half_rn(x));}
std::uint16_t bf(float x){return __bfloat16_as_ushort(__float2bfloat16_rn(x));}
struct Operation {
 int parts,splits;cudaStream_t st;
 DeviceBuffer partial,state,ep,em,el,basis,transposed,mean,c,g,ev,eg,o0,m0,l0,f0,o1,m1,l1,f1;
 cudaGraph_t graphs[2]{};cudaGraphExec_t exec[2]{};
 Operation(int p,int s,cudaStream_t stream):parts(p),splits(s),st(stream),
 partial(Q*p*32*4),state(Q*p*16),ep(Q*256*s*2),em(Q*s*4),el(Q*s*4),basis(4*256*32*2),transposed(4*256*32*2),mean(4*256*2),
 c(Q*32*4),g(Q*16),ev(Q*256*4),eg(Q*16),o0((Q*256+Guard)*4),m0((Q+Guard)*4),l0(m0.bytes),f0((4+Guard)*4),o1(o0.bytes),m1(m0.bytes),l1(l0.bytes),f1(f0.bytes){
  for(int arm=0;arm<2;++arm){CUDA_CHECK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));launch(arm);CUDA_CHECK(cudaStreamEndCapture(st,&graphs[arm]));CUDA_CHECK(cudaGraphInstantiate(&exec[arm],graphs[arm],nullptr,nullptr,0));}
 }
 void launch(int arm){
  if(!arm){
   vr::combine_coordinates_parallel<<<Q,256,0,st>>>((const float*)partial.p,(const float4*)state.p,parts,(float*)c.p,(float4*)g.p);
   jr::normalize_exact_subset<<<Q,256,0,st>>>((const __nv_bfloat16*)ep.p,(const float*)em.p,(const float*)el.p,splits,(const int*)f0.p,(float*)ev.p,(float4*)eg.p);
   rt::finish_local_state<<<Q,256,0,st>>>((const float*)c.p,(const float4*)g.p,(const float*)ev.p,(const float4*)eg.p,(const __half*)basis.p,(const __half*)mean.p,(int*)f0.p,(float*)o0.p,(float*)m0.p,(float*)l0.p);
  }else rt::query_owned_finish<<<Q,256,0,st>>>((const float*)partial.p,(const float4*)state.p,parts,(const __nv_bfloat16*)ep.p,(const float*)em.p,(const float*)el.p,splits,(const __half*)transposed.p,(const __half*)mean.p,(int*)f1.p,(float*)o1.p,(float*)m1.p,(float*)l1.p);
  CUDA_CHECK(cudaGetLastError());
 }
 std::size_t check(unsigned seed,int pattern){
  std::vector<float> pv(partial.bytes/4),maxima(em.bytes/4),denoms(el.bytes/4);
  std::vector<float4> states(Q*parts);
  std::vector<std::uint16_t> exact(ep.bytes/2),w(basis.bytes/2),meanv(mean.bytes/2);
  std::vector<int> fail(4,0);
  const bool simple=pattern==1;
  for(std::size_t i=0;i<pv.size();++i)pv[i]=simple?.25f:unit(unsigned(i)^seed)*.03f;
  for(int q=0;q<Q;++q)for(int p=0;p<parts;++p){auto& v=states[q*parts+p];v=make_float4(simple?0:unit(q*parts+p+seed),simple?1:.25f+float(mix(q*parts+p+seed)%100)/100.f,.5f,0);if(pattern==2 || (pattern==3&&p%3))v=make_float4(-INFINITY,0,0,0);}
  for(int i=0;i<Q*splits;++i){maxima[i]=simple?0:unit(i+seed+73);denoms[i]=simple?1:.25f+float(mix(i+seed+913)%127)/127.f;if(pattern==4){maxima[i]=-INFINITY;denoms[i]=0;}}
  for(std::size_t i=0;i<exact.size();++i)exact[i]=bf(simple?.5f:unit(unsigned(i)^seed)*.1f);
  for(std::size_t i=0;i<w.size();++i)w[i]=hf(simple?.03125f:unit(unsigned(i)+seed)*.03125f);
  for(std::size_t i=0;i<meanv.size();++i)meanv[i]=hf(simple?.125f:unit(unsigned(i)^seed)*.3f);
  if(pattern==5){for(auto& st:states)st=make_float4(-INFINITY,0,0,0);std::fill(maxima.begin(),maxima.end(),-INFINITY);std::fill(denoms.begin(),denoms.end(),0);}
  if(pattern==6)fail={1,1,1,1};
  if(pattern==7)fail={0,1,0,1};
  if(pattern==8)std::fill(denoms.begin(),denoms.end(),INFINITY);
  if(pattern==9)std::fill(w.begin(),w.end(),hf(INFINITY));
  if(pattern==10)std::fill(exact.begin(),exact.end(),std::uint16_t(0x7fc1));
  if(pattern==11){std::fill(pv.begin(),pv.end(),0);std::fill(exact.begin(),exact.end(),0);std::fill(w.begin(),w.end(),0);std::fill(meanv.begin(),meanv.end(),0);}
  upload(partial,pv);upload(state,states);upload(ep,exact);upload(em,maxima);upload(el,denoms);upload(basis,w);upload(mean,meanv);
  CUDA_CHECK(cudaDeviceSynchronize());
  rt::transpose_value_basis_for_queries<<<128,256,0,st>>>((const std::uint16_t*)basis.p,(std::uint16_t*)transposed.p);
  CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(st));
  const auto permuted=read<std::uint16_t>(transposed);
  for(int h=0;h<4;++h)for(int d=0;d<256;++d)for(int k=0;k<32;++k)
   if(permuted[(h*32+k)*256+d]!=w[(h*256+d)*32+k])throw std::runtime_error("prepared basis permutation changed source bits");
  for(auto* b:{&o0,&m0,&l0,&f0,&o1,&m1,&l1,&f1})CUDA_CHECK(cudaMemset(b->p,0x55,b->bytes));upload(f0,fail);upload(f1,fail);CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGraphLaunch(exec[0],st));CUDA_CHECK(cudaGraphLaunch(exec[1],st));CUDA_CHECK(cudaStreamSynchronize(st));
  const auto af=read<int>(f0),bfail=read<int>(f1);if(af!=bfail)throw std::runtime_error("head fallback flags differ");
  auto out=read<unsigned>(o0),other=read<unsigned>(o1);auto am=read<unsigned>(m0),bm=read<unsigned>(m1),al=read<unsigned>(l0),bl=read<unsigned>(l1);
  std::size_t count=0;
  for(int q=0;q<Q;++q){int h=q/24,qa=((q%24)/6)*24+h*6+(q%24)%6;
   if(fail[h]){for(int d=0;d<256;++d)if(out[qa*256+d]!=0x55555555U||other[qa*256+d]!=0x55555555U)throw std::runtime_error("pre-failed head was written");}
   if(af[h])continue;
   if(am[qa]!=bm[qa]||al[qa]!=bl[qa]){printf("STATE_DIFF q=%d %x/%x %x/%x\n",q,am[qa],bm[qa],al[qa],bl[qa]);throw std::runtime_error("log partition outputs differ");}
   for(int d=0;d<256;++d){int at=qa*256+d;if(out[at]!=other[at]){printf("DIFF parts=%d splits=%d pattern=%d q=%d d=%d %x/%x\n",parts,splits,pattern,q,d,out[at],other[at]);throw std::runtime_error("final attention values differ");}++count;}
  }
  for(int j=0;j<Guard;++j)if(out[Q*256+j]!=0x55555555U||other[Q*256+j]!=0x55555555U||am[Q+j]!=0x55555555U||bm[Q+j]!=0x55555555U||al[Q+j]!=0x55555555U||bl[Q+j]!=0x55555555U||af[4+j]!=0x55555555)throw std::runtime_error("output guard written");
  if(simple){float coordinate=.25f,vrvalue=.125f;for(int k=0;k<32;++k)vrvalue=std::fma(coordinate,.03125f,vrvalue);float exact_v=.5f;float a=float(parts),b=float(splits),total=a+b;float expected=std::fma(a/total,vrvalue,(b/total)*exact_v);
   auto data=read<float>(o1);for(int i=0;i<Q*256;++i)if(std::fabs(data[i]-expected)>1e-7f)throw std::runtime_error("independent simple formula");}
  // All inputs are immutable; distinguish scratch removal from input mutation.
  if(read<unsigned>(partial)!=[&]{std::vector<unsigned> v(pv.size());std::memcpy(v.data(),pv.data(),partial.bytes);return v;}())throw std::runtime_error("partial input mutated");
  printf("QUERY_FINISH_EXACT parts=%d splits=%d pattern=%d checked_values=%zu flags_guards_source_refresh=1\n",parts,splits,pattern,count);return count;
 }
 ~Operation(){cudaStreamSynchronize(st);for(auto e:exec)if(e)cudaGraphExecDestroy(e);for(auto g:graphs)if(g)cudaGraphDestroy(g);}
};
int main(){try{cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));std::size_t count=0;
 for(auto shape:{std::pair<int,int>{1,1},{7,16},{8,64},{9,128},{255,64},{256,64},{257,64},{512,128}}){Operation op(shape.first,shape.second,stream);for(int pattern:{0,1,2,3,4,5,6,7,8,9,10,11})count+=op.check(20261002U+pattern,pattern);}
 CUDA_CHECK(cudaStreamDestroy(stream));printf("QUERY_FINISH_TEST_PASS compared_values=%zu\n",count);return 0;
}catch(const std::exception& e){fprintf(stderr,"query finish: %s\n",e.what());return 1;}}
