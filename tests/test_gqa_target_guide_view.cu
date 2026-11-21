#include "core/device.h"
#include "ops/launcher/gqa_attention.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <set>
#include <vector>
using namespace ninfer;
struct Buffer{void* p=nullptr;std::size_t bytes;explicit Buffer(std::size_t n):bytes(n){CUDA_CHECK(cudaMalloc(&p,n));}~Buffer(){if(p)(void)cudaFree(p);}Buffer(const Buffer&)=delete;};
int main(){try{
 DeviceContext device(0);constexpr int Pages=256,Rows=2,View=132,Head=4,Dim=256;
 const std::size_t elements=std::size_t(Pages)*64*Head*Dim;
 Buffer guide(Pages*24*4*4),guide_position(4*4),limit_buffer(4),key(elements),scale(elements/64*2),tables(Pages*Rows*4),owner(4),positions(6*4),query(6*24*Dim*2),scores(Pages*24*4),norm(24*4),localpos(6*4),localtable(View*4),localrow(4);
 std::vector<std::int8_t> data(elements,0);std::vector<__half> scales(elements/64,__float2half_rn(.03125f));
 std::vector<int> source(Pages*Rows);for(int p=0;p<Pages;++p){source[p]=p;source[Pages+p]=(p*73+29)%Pages;}
 for(int p=0;p<Pages;++p)for(int h=0;h<4;++h)for(int t=0;t<64;++t)data[((std::size_t(source[Pages+p])*4+h)*64+t)*256]=static_cast<std::int8_t>(p-127);
 source[Pages+17]=-1;
 CUDA_CHECK(cudaMemcpy(key.p,data.data(),data.size(),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(scale.p,scales.data(),scale.bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(tables.p,source.data(),tables.bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaDeviceSynchronize());
 int row=1;CUDA_CHECK(cudaMemcpy(owner.p,&row,4,cudaMemcpyHostToDevice));
 PagedKVBatchLayerView cache{.k_pages=Tensor(key.p,DType::I8,{Dim,64,4,Pages}),.v_pages=Tensor(key.p,DType::I8,{Dim,64,4,Pages}),.k_scale_pages=Tensor(scale.p,DType::FP16,{4,64,4,Pages}),.v_scale_pages=Tensor(scale.p,DType::FP16,{4,64,4,Pages}),.block_tables=Tensor(tables.p,DType::I32,{Pages,Rows}),.head_dim=256,.num_kv_heads=4,.dtype=DType::I8,.quant_group=64};
 Tensor own(owner.p,DType::I32,{1}),score(scores.p,DType::FP32,{Pages,24}),nr(norm.p,DType::FP32,{24}),tb(localtable.p,DType::I32,{View,1}),zr(localrow.p,DType::I32,{1});
 int cases=0;
 for(int first:{0,127,4097,8191,8257,12031,16200})for(int width:{1,4,6})for(int sign:{-1,0,1})for(int future:{0,1}){
  std::vector<int> pos(width);std::iota(pos.begin(),pos.end(),first);
  std::vector<__nv_bfloat16> q(width*24*256,__float2bfloat16_rn(0));for(int h=0;h<24;++h)q[h*256]=__float2bfloat16_rn(float(sign));
  CUDA_CHECK(cudaMemcpy(positions.p,pos.data(),pos.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(query.p,q.data(),q.size()*2,cudaMemcpyHostToDevice));
  std::vector<float> masses(Pages*24*4);for(int h=0;h<24;++h)for(int p=0;p<Pages;++p)masses[h*Pages+p]=float(sign*(p-127))/512.f;
  int anchor=future?first+1:0;
  CUDA_CHECK(cudaMemcpy(guide.p,masses.data(),guide.bytes,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(guide_position.p,&anchor,sizeof(int),cudaMemcpyHostToDevice));
  Tensor gg(guide.p,DType::FP32,{Pages,24}),gp(guide_position.p,DType::I32,{1});
  CUDA_CHECK(cudaStreamSynchronize(nullptr)); // Complete pageable fixture uploads before nonblocking execution.
  Tensor qp(query.p,DType::BF16,{256,24,width}),pp(positions.p,DType::I32,{width}),lp(localpos.p,DType::I32,{width});
  ops::detail::gqa_target_guided_proposal_view_launch(gg,gp,pp,cache,own,8192,score,nr,lp,tb,zr,device.stream);device.synchronize();
  std::vector<int> got(width),view(View);CUDA_CHECK(cudaMemcpy(got.data(),localpos.p,width*4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(view.data(),localtable.p,view.size()*4,cudaMemcpyDeviceToHost));
  const int tail=std::max(2,std::max(0,first+1-(future?8192:4096))/64);
  std::vector<int> eligible;for(int p=2;p<tail;++p)if(source[Pages+p]>=0)eligible.push_back(p);
  int selected=future?0:std::min(64,int(eligible.size()));
  for(int j=0;j<width;++j){int l=got[j];if(l<0||l>=View*64||view[l/64]!=source[Pages+pos[j]/64]||l%64!=pos[j]%64)throw std::runtime_error("append slot changed");}
  if(view[0]!=source[Pages]||view[1]!=source[Pages+1])throw std::runtime_error("sink changed");
  std::vector<int> chosen;for(int i=0;i<selected;++i){auto at=std::find(source.begin()+Pages,source.end(),view[i+2]);if(at==source.end())throw std::runtime_error("foreign page");chosen.push_back(int(at-source.begin())-Pages);}
  if(!std::is_sorted(chosen.begin(),chosen.end())||std::set<int>(chosen.begin(),chosen.end()).size()!=chosen.size())throw std::runtime_error("repeated/reordered retrieved pages");
  for(int p:chosen)if(p<2||p>=tail||source[Pages+p]<0)throw std::runtime_error("future or missing page was selected");
  if(sign!=0){if(sign>0)std::reverse(eligible.begin(),eligible.end());eligible.resize(selected);std::sort(eligible.begin(),eligible.end());if(chosen!=eligible){fprintf(stderr,"rank first=%d width=%d sign=%d tail=%d expected_count=%zu got_count=%zu\n",first,width,sign,tail,eligible.size(),chosen.size());for(int p:chosen)fprintf(stderr," %d",p);fprintf(stderr,"\nexpected:");for(int p:eligible)fprintf(stderr," %d",p);fprintf(stderr,"\n");throw std::runtime_error("retrieval rank disagrees with independent monotone score");}}
  for(int i=2+selected;i<View;++i){int original=tail+i-2-selected;if(view[i]!=(original<Pages?source[Pages+original]:-1))throw std::runtime_error("tail alias mismatch");}
  ops::detail::gqa_target_guided_proposal_view_launch(gg,gp,pp,cache,own,8192,score,nr,lp,tb,zr,device.stream);device.synchronize();std::vector<int> repeated(View);CUDA_CHECK(cudaMemcpy(repeated.data(),localtable.p,View*4,cudaMemcpyDeviceToHost));if(repeated!=view)throw std::runtime_error("retrieval nondeterministic");
  ++cases;
 }

 // Queries after the whole verifier batch may still use only its accepted rows.
 // Adversarially changing every rejected row must never affect page selection.
 for(int width:{1,4})for(int accepted:{1,2,3,4}){
  std::vector<int> pos(width);std::iota(pos.begin(),pos.end(),14000);
  const int gp_values[4]={13000,13001,13002,13003};
  std::vector<float> masses(Pages*24*4);
  for(int c=0;c<4;++c)for(int h=0;h<24;++h)for(int p=0;p<Pages;++p)
   masses[(c*24+h)*Pages+p]=float((c%2==0?1:-1)*(p-127))/512.f;
  CUDA_CHECK(cudaMemcpy(positions.p,pos.data(),width*4,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(guide_position.p,gp_values,sizeof(gp_values),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(limit_buffer.p,&accepted,sizeof(int),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(guide.p,masses.data(),guide.bytes,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaStreamSynchronize(nullptr));
  Tensor gg(guide.p,DType::FP32,{Pages,24,4}),gp(guide_position.p,DType::I32,{4}),limit(limit_buffer.p,DType::I32,{1});
  Tensor pp(positions.p,DType::I32,{width}),lp(localpos.p,DType::I32,{width});
  ops::detail::gqa_target_guided_proposal_view_launch(gg,gp,pp,cache,own,8192,score,nr,lp,tb,zr,device.stream,&limit);device.synchronize();
  std::vector<int> before(View),after(View);CUDA_CHECK(cudaMemcpy(before.data(),localtable.p,View*4,cudaMemcpyDeviceToHost));
  const int tail=std::max(2,(14000+1-4096)/64);std::vector<int> selected;
  for(int page=2;page<tail;++page)if(source[Pages+page]>=0)selected.push_back(page);
  if(accepted%2==1)std::reverse(selected.begin(),selected.end());selected.resize(64);std::sort(selected.begin(),selected.end());
  for(int i=0;i<64;++i)if(before[2+i]!=source[Pages+selected[i]])throw std::runtime_error("wrong accepted query selected");
  for(int c=accepted;c<4;++c)for(int i=0;i<Pages*24;++i)masses[c*Pages*24+i]=float((i*173)%1000)-500.f;
  CUDA_CHECK(cudaMemcpy(guide.p,masses.data(),guide.bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaStreamSynchronize(nullptr));
  ops::detail::gqa_target_guided_proposal_view_launch(gg,gp,pp,cache,own,8192,score,nr,lp,tb,zr,device.stream,&limit);device.synchronize();
  CUDA_CHECK(cudaMemcpy(after.data(),localtable.p,View*4,cudaMemcpyDeviceToHost));
  if(before!=after)throw std::runtime_error("rejected target queries leaked into proposal context");
  ++cases;
 }
 std::vector<std::int8_t> after(data.size());std::vector<int> tab(source.size());CUDA_CHECK(cudaMemcpy(after.data(),key.p,key.bytes,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(tab.data(),tables.p,tables.bytes,cudaMemcpyDeviceToHost));if(after!=data||tab!=source)throw std::runtime_error("canonical source changed");
 printf("PASS cases=%d target_guide_ranks accepted_row_only rejected_rows_poisoned future_guide_fallback sink_tail append_identity missing_pages physical_permutation deterministic_ties canonical_sources_unchanged\n",cases);
 return 0;
}catch(const std::exception& e){fprintf(stderr,"retrieval_test: %s\n",e.what());return 1;}}
