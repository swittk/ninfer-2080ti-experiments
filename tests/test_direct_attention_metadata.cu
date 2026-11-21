#include "core/device.h"
#include "core/arena.h"
#include "../apps/dual/joint_kv/integer_coordinates.cuh"



#include <vector>
#include <array>
#include <stdexcept>
#include <cstdio>
using namespace ninfer;
namespace native=ops::integer_joint;
namespace candidate=ops::integer_joint;
constexpr int Old=131072,Count=Old+4,Blocks=(Count+31)/32,Storage=Old/32,Parts=(Count+511)/512;
__device__ unsigned hash(unsigned v){v^=v>>16;v*=0x7feb352dU;v^=v>>15;v*=0x846ca68bU;return v^(v>>16);}
__global__ void initialize(unsigned* keys,unsigned* values,__half* ks,__half* vs,signed char* q,float* qs,float* shift,float* cut,unsigned seed){
 int first=blockIdx.x*blockDim.x+threadIdx.x,step=blockDim.x*gridDim.x;
 for(int i=first;i<4*Storage*4*4*32;i+=step)keys[i]=hash(unsigned(i)^seed);
 for(int i=first;i<4*Storage*4*2*32;i+=step)values[i]=hash(unsigned(i)^seed^896453U);
 for(int i=first;i<4*Storage*32;i+=step){ks[i]=__float2half_rn(.015625f);vs[i]=__float2half_rn(.012f);}
 for(int i=first;i<96*64;i+=step)q[i]=static_cast<signed char>(int(hash(unsigned(i)^seed)%255)-127);
 if(first<96){qs[first]=.01f;shift[first]=float(first%7)*.02f;cut[first]=.7f+float(first%5)*.1f;}
}
template<class T>std::vector<T> read(DeviceBuffer& b){std::vector<T>v(b.bytes/sizeof(T));b.copy_to_host(v.data(),b.bytes);return v;}

__global__ void mutate_query(signed char* q,unsigned seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<96*64)q[i]=static_cast<signed char>(hash(seed^unsigned(i)));}
__global__ void native_rows8_pairs(unsigned* errors){
 const unsigned item=blockIdx.x;const int a=int(item&255)-128,b=int(item>>8)-128;
 const unsigned qa=static_cast<unsigned char>(a)*0x01010101U,kb=static_cast<unsigned char>(b)*0x01010101U;
 int c0=0,c1=0;native::mma_rows8(c0,c1,qa,qa,kb,kb);
 if(c0!=32*a*b||c1!=32*a*b)atomicAdd(errors,1U);
}
void independent_mma_check(){
 DeviceBuffer errors(4);CUDA_CHECK(cudaMemset(errors.p,0,4));native_rows8_pairs<<<65536,32>>>((unsigned*)errors.p);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
 unsigned value=0;errors.copy_to_host(&value,4);if(value)throw std::runtime_error("native M8 signed byte pair identity failed");
 printf("INDEPENDENT_SIGNED_PAIR_DOTS exact_cases=65536\n");
}

__global__ void reference_counts(const unsigned* masks,int blocks,int* counts,int* failed){
 int q=blockIdx.x,t=threadIdx.x,warp=t>>5,lane=t&31,count=0;for(int b=t;b<blocks;b+=256)count+=__popc(masks[q*blocks+b]);
 for(int off=16;off;off>>=1)count+=__shfl_down_sync(0xffffffffu,count,off);__shared__ int sums[8];if(lane==0)sums[warp]=count;__syncthreads();
 if(t==0){int total=0;for(int w=0;w<8;++w)total+=sums[w];counts[q]=total;if(total>4096)atomicOr(failed+q/24,1);}
}
__global__ void reference_union(const unsigned* masks,int blocks,unsigned* output){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<4*blocks){int head=i/blocks,b=i%blocks;unsigned mask=0;for(int q=0;q<24;++q)mask|=masks[(head*24+q)*blocks+b];output[i]=mask;}}
int main(int argc,char**argv){try{
 int device=0;CUDA_CHECK(cudaGetDevice(&device));cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,device));if(prop.major!=7||prop.minor!=5)return 77;
 independent_mma_check();
 unsigned seed=argc>1?std::stoul(argv[1]):20261001;
 cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 DeviceBuffer keys(std::size_t(4)*Storage*4*4*32*4),values(std::size_t(4)*Storage*4*2*32*4),ks(4*Storage*32*2),vs(ks.bytes),
 q(96*64),qs(96*4),shift(96*4),threshold(96*4),positions(4*4),
 mask0(96*Blocks*4+128),mask1(96*Parts*4+128),union0(4*Blocks*4+128),union1(union0.bytes),counts0(100*4+128),counts1(counts0.bytes),out0(96*Parts*32*4+128),out1(out0.bytes),state0(96*Parts*16+128),state1(state0.bytes),flush(256ULL<<20);
 int range_begin=0,range_end=Count,active_parts=Parts,first_part=0,storage_begin=0,storage_blocks=Storage;
 auto fill=[&](unsigned seed){initialize<<<2048,256,0,stream>>>((unsigned*)keys.p,(unsigned*)values.p,(__half*)ks.p,(__half*)vs.p,(signed char*)q.p,(float*)qs.p,(float*)shift.p,(float*)threshold.p,seed);CUDA_CHECK(cudaGetLastError());};
 auto launch=[&](bool revised){
  auto& c=revised?counts1:counts0;CUDA_CHECK(cudaMemsetAsync(c.p,0,100*4,stream));
  if(revised){
   CUDA_CHECK(cudaMemsetAsync(union1.p,0,4*Blocks*4,stream));
   candidate::allkey<false,true><<<4*active_parts,96,0,stream>>>((signed char*)q.p,(float*)qs.p,(float*)shift.p,(int*)positions.p,(unsigned*)keys.p,nullptr,nullptr,(unsigned*)values.p,(float*)threshold.p,Count,Old,Blocks,active_parts,(unsigned*)mask1.p,(float*)out1.p,(float4*)state1.p,Count,nullptr,nullptr,nullptr,range_begin,range_end,first_part,storage_blocks,storage_begin,(__half*)ks.p,(__half*)vs.p,(unsigned*)union1.p);
   candidate::reduce_selected_counts<<<96,256,0,stream>>>((unsigned*)mask1.p,active_parts,(int*)counts1.p,(int*)counts1.p+96);
  }else{
   CUDA_CHECK(cudaMemsetAsync(mask0.p,0,96*Blocks*4,stream));
   native::allkey<false><<<4*active_parts,96,0,stream>>>((signed char*)q.p,(float*)qs.p,(float*)shift.p,(int*)positions.p,(unsigned*)keys.p,nullptr,nullptr,(unsigned*)values.p,(float*)threshold.p,Count,Old,Blocks,active_parts,(unsigned*)mask0.p,(float*)out0.p,(float4*)state0.p,Count,nullptr,nullptr,nullptr,range_begin,range_end,first_part,storage_blocks,storage_begin,(__half*)ks.p,(__half*)vs.p);
   reference_counts<<<96,256,0,stream>>>((unsigned*)mask0.p,Blocks,(int*)counts0.p,(int*)counts0.p+96);
   reference_union<<<(4*Blocks+255)/256,256,0,stream>>>((unsigned*)mask0.p,Blocks,(unsigned*)union0.p);
  }
  CUDA_CHECK(cudaGetLastError());
 };
 auto cpu_metadata=[&]{
  auto m=read<unsigned>(mask0),a=read<unsigned>(union1),b=read<unsigned>(counts1);
  unsigned failures[4]={};for(int r=0;r<96;++r){unsigned count=0;for(int i=0;i<Blocks;++i)count+=__builtin_popcount(m[r*Blocks+i]);if(b[r]!=count)throw std::runtime_error("CPU selected count disagrees");if(count>4096)failures[r/24]=1;}
  for(int h=0;h<4;++h){if(b[96+h]!=failures[h])throw std::runtime_error("CPU failure flag disagrees");for(int i=0;i<Blocks;++i){unsigned expected=0;for(int row=0;row<24;++row)expected|=m[(h*24+row)*Blocks+i];if(a[h*Blocks+i]!=expected)throw std::runtime_error("CPU union disagrees");}}
 };
 for(int scenario=0;scenario<13;++scenario){
  fill(seed+scenario*9127U);CUDA_CHECK(cudaStreamSynchronize(stream));std::array<int,4> pos={Old,Old+1,Old+2,Old+3};
  range_begin=0;range_end=Count;active_parts=Parts;first_part=0;storage_begin=0;storage_blocks=Storage;
  if(scenario==1){std::vector<float> cut(96,-INFINITY);threshold.copy_from_host(cut.data(),threshold.bytes);}
  if(scenario==2){std::vector<float> cut(96,INFINITY);threshold.copy_from_host(cut.data(),threshold.bytes);}
  if(scenario==3){CUDA_CHECK(cudaMemsetAsync(q.p,0,q.bytes,stream));CUDA_CHECK(cudaMemsetAsync(qs.p,0,qs.bytes,stream));}
  if(scenario==4){pos={512,513,514,515};range_end=1024;}
  if(scenario==5){range_begin=65536;range_end=98304;}
  if(scenario==6){range_begin=65536;range_end=Old;first_part=128;active_parts=128;storage_begin=2048;storage_blocks=2048;}
  if(scenario==7){range_begin=1536;range_end=1544;first_part=3;active_parts=1;storage_begin=48;storage_blocks=32;}
  if(scenario==8){pos={-1,-1,-1,-1};}
  if(scenario==9){pos={31,15,-1,30};range_end=32;active_parts=1;}
  if(scenario==10){std::vector<float> a(96,NAN);threshold.copy_from_host(a.data(),threshold.bytes);}
  if(scenario==11){std::vector<float> a(96,INFINITY);qs.copy_from_host(a.data(),qs.bytes);}
  if(scenario==12){std::vector<float> a(96,-INFINITY);shift.copy_from_host(a.data(),shift.bytes);}
  positions.copy_from_host(pos.data(),16);CUDA_CHECK(cudaDeviceSynchronize());
  for(auto* b:{&mask0,&mask1,&out0,&out1,&state0,&state1,&counts0,&counts1,&union0,&union1})CUDA_CHECK(cudaMemsetAsync(b->p,0x55,b->bytes,stream));
  launch(false);launch(true);CUDA_CHECK(cudaStreamSynchronize(stream));
  for(auto pair:{std::pair<DeviceBuffer*,DeviceBuffer*>{&union0,&union1},{&counts0,&counts1},{&out0,&out1},{&state0,&state1}}){auto a=read<unsigned>(*pair.first),b=read<unsigned>(*pair.second);
   if(a!=b){std::size_t different=0;for(std::size_t i=0;i<a.size();++i)if(a[i]!=b[i]){if(different<4)printf("DIFF offset=%zu old=%x new=%x\n",i,a[i],b[i]);++different;}printf("DIFFERENCES %zu\n",different);throw std::runtime_error("integer score-loop reorder changed output");}
   for(std::size_t i=a.size()-32;i<a.size();++i)if(a[i]!=0x55555555U)throw std::runtime_error("output guard overwritten");
  }
  cpu_metadata();
  auto scratch=read<unsigned>(mask1);for(std::size_t i=scratch.size()-32;i<scratch.size();++i)if(scratch[i]!=0x55555555u)throw std::runtime_error("partial counts overflow");
  cudaGraph_t graphs[2]{};cudaGraphExec_t replay[2]{};
  for(int mode=0;mode<2;++mode){CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(mode);CUDA_CHECK(cudaStreamEndCapture(stream,&graphs[mode]));CUDA_CHECK(cudaGraphInstantiate(&replay[mode],graphs[mode],nullptr,nullptr,0));}
  // Update only the input codes at the same addresses. Captured ranges and
  // per-scenario thresholds remain intact, including infinities and empty spans.
  mutate_query<<<24,256,0,stream>>>((signed char*)q.p,seed^unsigned(scenario*9311));CUDA_CHECK(cudaGetLastError());
  for(int repeat=0;repeat<2;++repeat){
   CUDA_CHECK(cudaGraphLaunch(replay[0],stream));CUDA_CHECK(cudaGraphLaunch(replay[1],stream));CUDA_CHECK(cudaStreamSynchronize(stream));
   if(read<unsigned>(union0)!=read<unsigned>(union1)||read<unsigned>(counts0)!=read<unsigned>(counts1)||read<unsigned>(out0)!=read<unsigned>(out1)||read<unsigned>(state0)!=read<unsigned>(state1))throw std::runtime_error("captured changed-source attention differs");
  }
  for(int mode=0;mode<2;++mode){CUDA_CHECK(cudaGraphExecDestroy(replay[mode]));CUDA_CHECK(cudaGraphDestroy(graphs[mode]));}
  printf("DIRECT_METADATA_EXACT scenario=%d bytes=%zu masks_pv_softmax_guards_changed_source_graphs=1\n",scenario,union0.bytes+counts0.bytes+out0.bytes+state0.bytes);
 }

 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception&e){fprintf(stderr,"valid query integer attention test: %s\n",e.what());return 1;}}
