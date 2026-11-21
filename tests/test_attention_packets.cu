#include "core/device.h"
#include "core/arena.h"
#include "../apps/dual/attention_packet.cuh"
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
using namespace ninfer;
namespace ap=ninfer::apps::dual::attention_packet;
static_assert(ap::input_layout(4,false).bytes==49168);
static_assert(ap::input_layout(4,true).bytes==65552);
static_assert(ap::state_layout(4).bytes==99072);
__global__ void write_state(unsigned* packet,int tokens){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 const int acc=tokens*24*256,stat=tokens*24;
 if(i<acc)packet[i]=0x3f000000U+unsigned(i);
 else if(i<acc+stat)packet[i]=0x4a000000U+unsigned(i-acc);
 else if(i<acc+stat*2)packet[i]=0x5a000000U+unsigned(i-acc-stat);
}
int main(){try{
 int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0)return 77;
 for(int t:{-1,0,7,1000000}){bool invalid=false;try{(void)ap::input_layout(t,true);}catch(const std::invalid_argument&){invalid=true;}if(!invalid)return 1;}
 cudaStream_t stream;CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 const auto max=ap::input_layout(6,true);const auto maxstate=ap::state_layout(6);
 DeviceBuffer q(max.query_bytes),k(max.kv_bytes),v(max.kv_bytes),pos(6*4),dst(max.bytes+64),state(maxstate.bytes+64);
 std::vector<unsigned char> query(q.bytes),keys(k.bytes),values(v.bytes);
 std::size_t checked=0;
 for(int tokens:{4,5,1,6,3,2,4})for(bool kv:{false,true}){
  const auto layout=ap::input_layout(tokens,kv);const auto sl=ap::state_layout(tokens);
  if(layout.bytes>max.bytes||sl.bytes>maxstate.bytes||layout.key_offset%16||layout.value_offset%16)throw std::runtime_error("packet allocation/alignment mismatch");
  cudaGraph_t graph{};cudaGraphExec_t execution{};
  CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  ap::pack_input<<<(layout.bytes/16+255)/256,256,0,stream>>>((const uint4*)q.p,(const int*)pos.p,kv?(const uint4*)k.p:nullptr,kv?(const uint4*)v.p:nullptr,(uint4*)dst.p,tokens,kv);
  write_state<<<(sl.bytes/4+255)/256,256,0,stream>>>((unsigned*)state.p,tokens);
  CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamEndCapture(stream,&graph));CUDA_CHECK(cudaGraphInstantiate(&execution,graph,nullptr,nullptr,0));
  for(int round=0;round<3;++round){
   for(std::size_t i=0;i<query.size();++i)query[i]=static_cast<unsigned char>(i*71+round*53+tokens);
   for(std::size_t i=0;i<keys.size();++i){keys[i]=static_cast<unsigned char>(i*33+round*67);values[i]=static_cast<unsigned char>(i*127-round*43);}
   int positions[6];for(int i=0;i<6;++i)positions[i]=round==1?-1:262130+i+round;
   q.copy_from_host(query.data(),q.bytes);k.copy_from_host(keys.data(),k.bytes);v.copy_from_host(values.data(),v.bytes);pos.copy_from_host(positions,sizeof(positions));CUDA_CHECK(cudaDeviceSynchronize());
   CUDA_CHECK(cudaMemsetAsync(dst.p,0x55,dst.bytes,stream));CUDA_CHECK(cudaMemsetAsync(state.p,0x55,state.bytes,stream));
   CUDA_CHECK(cudaGraphLaunch(execution,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
   std::vector<unsigned char> expected(dst.bytes,0x55),actual(dst.bytes);
   std::memcpy(expected.data(),query.data(),layout.query_bytes);
   std::fill(expected.begin()+layout.position_offset,expected.begin()+layout.key_offset,0);
   std::memcpy(expected.data()+layout.position_offset,positions,layout.position_bytes);
   if(kv){std::memcpy(expected.data()+layout.key_offset,keys.data(),layout.kv_bytes);std::memcpy(expected.data()+layout.value_offset,values.data(),layout.kv_bytes);}
   dst.copy_to_host(actual.data(),dst.bytes);if(expected!=actual)throw std::runtime_error("query/KV/position bit patterns or packet guards changed");
   std::vector<unsigned> sx(state.bytes/4);state.copy_to_host(sx.data(),state.bytes);
   const auto ac=std::size_t(tokens)*24*256,st=std::size_t(tokens)*24;
   for(std::size_t i=0;i<sx.size();++i){const unsigned exp=i<ac?0x3f000000U+i:i<ac+st?0x4a000000U+(i-ac):i<ac+2*st?0x5a000000U+(i-ac-st):0x55555555U;if(sx[i]!=exp)throw std::runtime_error("numerator/max/normalizer overlap or state tail overwritten");}
   checked+=dst.bytes+state.bytes;
  }
  CUDA_CHECK(cudaGraphExecDestroy(execution));CUDA_CHECK(cudaGraphDestroy(graph));
 }
 printf("PASS packet widths1..6, KV/no-KV, source reuse, position padding, state layout and graph guards: %zu bytes checked\n",checked);
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){fprintf(stderr,"attention packet: %s\n",e.what());return 1;}}
