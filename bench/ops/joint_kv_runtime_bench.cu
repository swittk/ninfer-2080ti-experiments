#include "core/device.h"
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ops/launcher/gqa_attention.h"
#include "../../apps/dual/headscale_gqa.cuh"
#include "../../apps/dual/joint_kv/runtime.cuh"
#include "../../.chatgpt/production_serve_20260918/joint_kv_runtime/steady_integration_2220/owned_coordinate_storage/original_runtime.cuh"
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <cstdio>
#include <cstring>
#include <chrono>
using namespace ninfer;
namespace jr = ninfer::apps::dual::joint_runtime;
namespace fs = std::filesystem;
using Clock=std::chrono::steady_clock;
std::vector<char> read_bytes(const fs::path& path){
 std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open "+path.string());
 const auto size=fs::file_size(path);std::vector<char> data(size);
 if(!in.read(data.data(),size))throw std::runtime_error("short capture "+path.string());return data;
}
struct Rank {
 int device,pages,first,end,splits=85;cudaStream_t stream=nullptr;
 std::unique_ptr<jr::Allocation> k,v,ks,vs,table,q,p,acc,m,l,part,pm,pl;
 PagedKVLayerView cache;Tensor qt,pt,at,mt,lt,pat,pmt,plt;
 Rank(int d,const fs::path& folder,const char* label):device(d){
  jr::cuda_check(cudaSetDevice(d),"rank device");jr::cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"rank stream");
  auto load=[&](std::unique_ptr<jr::Allocation>& dst,const std::string& name){auto host=read_bytes(folder/(name+".bin"));dst=std::make_unique<jr::Allocation>(d,host.size());jr::cuda_check(cudaMemcpyAsync(dst->p,host.data(),host.size(),cudaMemcpyHostToDevice,stream),"capture upload");jr::cuda_check(cudaStreamSynchronize(stream),"capture bytes ready");};
  std::string pre(label);load(k,pre+"_k");load(v,pre+"_v");load(ks,pre+"_ks");load(vs,pre+"_vs");load(table,pre+"_table");load(q,"query");load(p,"positions");
  pages=k->bytes/(256*64*4);if(k->bytes!=std::size_t(pages)*256*64*4 || table->bytes!=4096*4)throw std::runtime_error("capture envelope");
  auto tb=read_bytes(folder/(pre+"_table.bin"));std::vector<int> map(tb.size()/4);std::memcpy(map.data(),tb.data(),tb.size());first=-1;end=-1;
  for(int i=0;i<int(map.size());++i)if(map[i]>=0){if(first<0)first=i*64;end=(i+1)*64;}
  cache={.k_pages=Tensor(k->p,DType::I8,{256,64,4,pages}),.v_pages=Tensor(v->p,DType::I8,{256,64,4,pages}),
   .k_scale_pages=Tensor(ks->p,DType::FP16,{4,64,4,pages}),.v_scale_pages=Tensor(vs->p,DType::FP16,{4,64,4,pages}),
   .block_table=Tensor(table->p,DType::I32,{4096}),.head_dim=256,.num_kv_heads=4,.dtype=DType::I8,.quant_group=64};
  qt=Tensor(q->p,DType::BF16,{256,24,4});pt=Tensor(p->p,DType::I32,{4});
  acc=std::make_unique<jr::Allocation>(d,96*256*4);m=std::make_unique<jr::Allocation>(d,96*4);l=std::make_unique<jr::Allocation>(d,96*4);
  part=std::make_unique<jr::Allocation>(d,96*256*splits*2);pm=std::make_unique<jr::Allocation>(d,96*splits*4);pl=std::make_unique<jr::Allocation>(d,96*splits*4);
  at=Tensor(acc->p,DType::FP32,{256,24,4});mt=Tensor(m->p,DType::FP32,{24,4});lt=Tensor(l->p,DType::FP32,{24,4});
  pat=Tensor(part->p,DType::BF16,{256,24,4,splits});pmt=Tensor(pm->p,DType::FP32,{24,4,splits});plt=Tensor(pl->p,DType::FP32,{24,4,splits});
 }
 ~Rank(){(void)cudaSetDevice(device);if(stream){(void)cudaStreamSynchronize(stream);(void)cudaStreamDestroy(stream);}}
 void sync(){jr::cuda_check(cudaSetDevice(device),"rank wait device");jr::cuda_check(cudaStreamSynchronize(stream),"rank wait");}
 void native(){
  jr::cuda_check(cudaSetDevice(device),"native device");const int unit=((262144/32+splits-1)/splits)*32;
  if(first%unit || (end!=262144 && end%unit))throw std::runtime_error("native split partition mismatch");
  const int begin=first/unit,count=(end==262144?splits:end/unit)-begin;
  ops::headscale_qk::launch_headscale_range(qt,pt,.0625f,cache,begin,count,splits,262144,pat,pmt,plt,stream,true);
  ops::detail::gqa_attention_small_t_reduce_range_state_launch(24,4,splits,begin,count,pat,pmt,plt,at,mt,lt,stream);
 }
};
float bf(std::uint16_t x){unsigned u=unsigned(x)<<16;float f;std::memcpy(&f,&u,4);return f;}
void check_case(const fs::path& root,int layer,int first_device){
 const auto folder=root/("layer"+std::to_string(layer));
 Rank a(first_device,folder,"owner"),b(1-first_device,folder,"helper");
 auto pb=read_bytes(folder/"positions.bin");std::array<int,4> pos;std::memcpy(pos.data(),pb.data(),pb.size());
 if(pos[0]<32768||pos[3]-pos[0]!=3)throw std::runtime_error("not full D3 capture");
 jr::cuda_check(cudaSetDevice(a.device),"merge device");
 jr::Allocation remote_acc(a.device,96*256*4),remote_m(a.device,96*4),remote_l(a.device,96*4),result(a.device,96*256*2);
 Tensor ra(remote_acc.p,DType::FP32,{256,24,4}),rm(remote_m.p,DType::FP32,{24,4}),rl(remote_l.p,DType::FP32,{24,4}),out(result.p,DType::BF16,{256,24,4});
 std::vector<char> ha(remote_acc.bytes),hm(remote_m.bytes),hl(remote_l.bytes);
 auto merge=[&]{
  b.sync();jr::cuda_check(cudaMemcpy(ha.data(),b.acc->p,ha.size(),cudaMemcpyDeviceToHost),"partial download");
  jr::cuda_check(cudaMemcpy(hm.data(),b.m->p,hm.size(),cudaMemcpyDeviceToHost),"max download");jr::cuda_check(cudaMemcpy(hl.data(),b.l->p,hl.size(),cudaMemcpyDeviceToHost),"sum download");
  jr::cuda_check(cudaSetDevice(a.device),"merge owner");
  jr::cuda_check(cudaMemcpyAsync(remote_acc.p,ha.data(),ha.size(),cudaMemcpyHostToDevice,a.stream),"partial upload");
  jr::cuda_check(cudaMemcpyAsync(remote_m.p,hm.data(),hm.size(),cudaMemcpyHostToDevice,a.stream),"max upload");jr::cuda_check(cudaMemcpyAsync(remote_l.p,hl.data(),hl.size(),cudaMemcpyHostToDevice,a.stream),"sum upload");
  if(a.first<b.first)ops::detail::gqa_attention_small_t_merge_two_states_launch(24,4,a.at,a.mt,a.lt,ra,rm,rl,out,a.stream);
  else ops::detail::gqa_attention_small_t_merge_two_states_launch(24,4,ra,rm,rl,a.at,a.mt,a.lt,out,a.stream);
  a.sync();
 };
 auto download=[&]{std::vector<std::uint16_t> v(result.bytes/2);jr::cuda_check(cudaSetDevice(a.device),"result device");jr::cuda_check(cudaMemcpy(v.data(),result.p,result.bytes,cudaMemcpyDeviceToHost),"result read");return v;};
 auto native=[&]{a.native();b.native();merge();};
 native();const auto reference=download();auto capture=read_bytes(folder/"native_output.bin");std::size_t rawdiff=0;
 for(std::size_t i=0;i<capture.size();++i)rawdiff+=capture[i]!=reinterpret_cast<const char*>(reference.data())[i];
 std::printf("NATIVE_CAPTURE layer=%d first=%d differing_bytes=%zu\n",layer,first_device,rawdiff);std::fflush(stdout);
 jr::Collection collection(a.device,b.device,262144,pos[0],1,a.cache,b.cache,a.stream,b.stream);
 // Exercise the explicit immutable-prefix phase and its idempotent reuse.
 // This must not execute attention, append KV or alter any source bytes.
 collection.prepare_prefix(0,a.cache,b.cache);
 collection.prepare_prefix(0,a.cache,b.cache);
 bool source_rejected=false;
 auto changed_source=a.cache;changed_source.v_pages.data=a.cache.k_pages.data;
 try{collection.prepare_prefix(0,changed_source,b.cache);}catch(const std::runtime_error&){source_rejected=true;}
 if(!source_rejected)throw std::runtime_error("different prefix source silently reused");
 std::printf("PREFIX_LIFECYCLE layer=%d first=%d idempotent=1 different_source_rejected=1\n",layer,first_device);
 bool accepted=false;
 auto candidate=[&](bool eager=false){accepted=collection.execute(0,a.cache,b.cache,a.qt,a.pt,b.qt,b.pt,pos[3]+1,a.at,a.mt,a.lt,b.at,b.mt,b.lt,eager);if(!accepted){a.native();b.native();}merge();};
 auto start=Clock::now();candidate();const double prepare_wall=std::chrono::duration<double>(Clock::now()-start).count();
 const auto actual=download();
 {
  ninfer::apps::dual::joint_full_storage_reference::Collection original(a.device,b.device,262144,pos[0],1,a.cache,b.cache,a.stream,b.stream);
  original.prepare_prefix(0,a.cache,b.cache);
  const bool used=original.execute(0,a.cache,b.cache,a.qt,a.pt,b.qt,b.pt,pos[3]+1,a.at,a.mt,a.lt,b.at,b.mt,b.lt,false);
  if(!used)throw std::runtime_error("original storage reference unexpectedly fell back");
  a.sync();b.sync();merge();a.sync();
  if(download()!=actual)throw std::runtime_error("owned-coordinate result differs bitwise from full allocation");
  std::printf("STORAGE_PARITY layer=%d first=%d exact_original_output=1\n",layer,first_device);
 }
 double maxrel=0,mincos=1;std::size_t bad=0;
 for(int row=0;row<96;++row){double err=0,norm=0,cross=0,other=0;for(int d=0;d<256;++d){double x=bf(reference[row*256+d]),y=bf(actual[row*256+d]);err+=(x-y)*(x-y);norm+=x*x;cross+=x*y;other+=y*y;bad+=!std::isfinite(y);}
 maxrel=std::max(maxrel,std::sqrt(err/std::max(norm,1e-30)));mincos=std::min(mincos,cross/std::sqrt(std::max(norm*other,1e-30)));}
 std::printf("LIVE_PCA layer=%d first=%d used=%d queries=96 max_rel=%.9g min_cos=%.9g nonfinite=%zu initial_seconds=%.6f\n",layer,first_device,accepted,maxrel,mincos,bad,prepare_wall);std::fflush(stdout);
 if(!accepted || bad || maxrel>.15 || mincos<.995)throw std::runtime_error("live two-owner numerical admission failed");
 candidate();if(download()!=actual)throw std::runtime_error("live representation repeated output changed");
 const auto original_query=read_bytes(folder/"query.bin");const auto original_pos=pos;
 for(int pass=0;pass<4;++pass){
  auto changed=original_query;
  if(pass==1||pass==2){
   // Permute existing query-head rows at stable addresses, not just repeat a
   // graph with unchanged inputs. Causal positions also move in pass2.
   constexpr int row_bytes=256*2;
   for(int row=0;row<96;++row)std::memcpy(changed.data()+row*row_bytes,original_query.data()+((row/24)*24+(row%24+1)%24)*row_bytes,row_bytes);
  }
  pos=original_pos;if(pass==2)for(auto& x:pos)x-=7;
  for(Rank* rank:{&a,&b}){
   jr::cuda_check(cudaSetDevice(rank->device),"test mutable source");
   jr::cuda_check(cudaMemcpyAsync(rank->q->p,changed.data(),changed.size(),cudaMemcpyHostToDevice,rank->stream),"updated query bytes");
   jr::cuda_check(cudaMemcpyAsync(rank->p->p,pos.data(),16,cudaMemcpyHostToDevice,rank->stream),"updated causal positions");rank->sync();
  }
  candidate(true);const auto eager_result=download();const bool eager_used=accepted;
  candidate(false);if(accepted!=eager_used || download()!=eager_result)throw std::runtime_error("graph/eager source-update or position parity failed");
 }
 if(download()!=actual)throw std::runtime_error("restored-query graph output changed");
 std::printf("GRAPH_DYNAMIC layer=%d first=%d cases=4 exact_eager_output_and_fallback_decision=1\n",layer,first_device);std::fflush(stdout);

 // A request beyond the retained-tail contract must refuse before reading any
 // stale/future coordinates. The caller's original all-key fallback is exact.
 const bool tail_accept=collection.execute(0,a.cache,b.cache,a.qt,a.pt,b.qt,b.pt,(pos[0]/32)*32+jr::MaxTail+1,a.at,a.mt,a.lt,b.at,b.mt,b.lt);
 if(tail_accept)throw std::runtime_error("tail overflow did not fall back");native();if(download()!=reference)throw std::runtime_error("tail fallback differed");
 std::vector<double> nt,ct;
 for(int i=0;i<6;++i){auto t=Clock::now();native();nt.push_back(std::chrono::duration<double,std::milli>(Clock::now()-t).count());t=Clock::now();candidate();ct.push_back(std::chrono::duration<double,std::milli>(Clock::now()-t).count());}
 std::sort(nt.begin(),nt.end());std::sort(ct.begin(),ct.end());
 if(download()!=actual)throw std::runtime_error("after timing output changed");
 std::printf("PARTITION_COST layer=%d first=%d native_ms=%.6f joint_ms=%.6f ratio=%.6f fallback_exact=1 repeat_exact=1\n",layer,first_device,(nt[2]+nt[3])/2,(ct[2]+ct[3])/2,(nt[2]+nt[3])/(ct[2]+ct[3]));std::fflush(stdout);
}
int main(int argc,char**argv){try{if(argc<3)throw std::invalid_argument("capture root and physical first required");int first=std::stoi(argv[2]);
 for(int layer:{3,31,63})check_case(argv[1],layer,first);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"RUNTIME_TEST_ERROR %s\n",e.what());return 1;}}
