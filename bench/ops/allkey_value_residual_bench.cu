#include <cuda_profiler_api.h>
#include "core/device.h"
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/launcher/gqa_attention.h"
#include "ninfer_bench_common.h"
#include "../../.chatgpt/production_serve_20260918/allkey_value_residual_correction/cuda/value_residual.cuh"
#include "../../.chatgpt/production_serve_20260918/allkey_value_residual_correction/cuda/fallback.cuh"
#include "../../.chatgpt/production_serve_20260918/allkey_value_residual_correction/cuda/sample_order_statistic.cuh"
#include "../../.chatgpt/production_serve_20260918/allkey_value_residual_correction/cuda/pipelined_coordinates.cuh"
#include "../../.chatgpt/production_serve_20260918/allkey_value_residual_correction/cuda/shared_union.cuh"
#include "../../.chatgpt/production_serve_20260918/allkey_value_residual_correction/cuda/native_coordinate_kernel.cuh"
#include "../../.chatgpt/production_serve_20260918/allkey_value_residual_correction/cuda/split_qk_coordinate.cuh"
#include "../../.chatgpt/production_serve_20260918/allkey_value_residual_correction/cuda/sample_packet.cuh"
#include "../../.chatgpt/production_serve_20260918/allkey_value_residual_correction/cuda/indexed_subset_attention.cuh"
#include "../../.chatgpt/production_serve_20260918/allkey_value_residual_correction/cuda/subset_attention.cuh"
#include "../../.chatgpt/production_serve_20260918/allkey_value_residual_correction/cuda/parallel_coordinate_reduce.cuh"
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
using namespace ninfer;
namespace vr=ninfer::ops::value_residual;
namespace fb=ninfer::ops::value_residual_fallback;
namespace {
void read_file(const std::filesystem::path& path,DeviceBuffer& buffer){
 if(std::filesystem::file_size(path)!=buffer.bytes)throw std::runtime_error("unexpected file size: "+path.string());
 std::vector<char> contents(buffer.bytes);std::ifstream file(path,std::ios::binary);
 if(!file.read(contents.data(),contents.size()))throw std::runtime_error("read failed: "+path.string());
 buffer.copy_from_host(contents.data(),contents.size());
}
template<class T>std::vector<T> download(DeviceBuffer& b){std::vector<T> v(b.bytes/sizeof(T));b.copy_to_host(v.data(),b.bytes);return v;}
template<class T>void write_file(const std::filesystem::path& p,const std::vector<T>& v){std::ofstream f(p,std::ios::binary|std::ios::trunc);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));if(!f)throw std::runtime_error("output write failed");}
float bf16(std::uint16_t x){union{unsigned u;float f;}a{unsigned(x)<<16};return a.f;}
__global__ void force_threshold(float* t,int mode){int i=threadIdx.x;if(i<96)t[i]=mode>0?CUDART_INF_F:-CUDART_INF_F;}
struct Error {double relative=0,cosine=1;int bad=0;};
Error compare_rows(const std::vector<std::uint16_t>& output,const std::vector<float>& expected){
 Error result;
 for(int row=0;row<96;++row){double err=0,norm=0,ynorm=0,dot=0;
  for(int d=0;d<256;++d){double a=expected[row*256+d],b=bf16(output[row*256+d]);err+=(a-b)*(a-b);norm+=a*a;ynorm+=b*b;dot+=a*b;result.bad+=!std::isfinite(b);}
  result.relative=std::max(result.relative,std::sqrt(err/std::max(norm,1e-30)));result.cosine=std::min(result.cosine,dot/std::sqrt(std::max(norm*ynorm,1e-30)));
 }
 return result;
}
}
int main(int argc,char**argv){try{
 if(argc<4)throw std::invalid_argument("usage: fixture_dir keys old_end [output_prefix]");
 const std::filesystem::path directory=argv[1];const int keys=std::stoi(argv[2]),old_end=std::stoi(argv[3]);
 if(keys<256||keys>262144||old_end<=0||old_end%32||old_end>=keys||keys-old_end>64)throw std::invalid_argument("unsupported bounded fixture");
 const int pages=(keys+63)/64,padded=pages*64,blocks=(keys+31)/32,parts=ops::detail::gqa_attention_split_capacity(24,4,DType::I8,ops::GqaExecutionEnvelope{static_cast<unsigned>(padded),static_cast<unsigned>(padded)}),samples=std::min(16384,old_end),order=std::clamp((3072*samples+old_end-1)/old_end,1,samples);
 const std::size_t elems=static_cast<std::size_t>(pages)*4*64*256;
 cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 DeviceBuffer key(elems),value(elems),ks(elems/64*2),vs(elems/64*2),query(96*256*2),positions(16),table(pages*4),
 linear(std::size_t(4)*padded*32*2),packed(std::size_t(4)*blocks*512*4),basis(4*256*32*2),mean(4*256*2),
 expected_buffer(96*256*4),reference_buffer(96*256*4),reference((96*256+128)*2),output(reference.bytes),flush(256ULL<<20);
 read_file(directory/"k.bin",key);read_file(directory/"v.bin",value);read_file(directory/"ks.bin",ks);read_file(directory/"vs.bin",vs);
 read_file(directory/"query.bin",query);read_file(directory/"positions.bin",positions);read_file(directory/"table.bin",table);
 read_file(directory/"coefficients.bin",linear);read_file(directory/"basis.bin",basis);read_file(directory/"mean.bin",mean);
 read_file(directory/"expected_union.bin",expected_buffer);read_file(directory/"reference.bin",reference_buffer);
 CUDA_CHECK(cudaDeviceSynchronize());
 vr::pack_coordinates<<<(packed.bytes/4+255)/256,256,0,stream>>>((const __half*)linear.p,(unsigned*)packed.p,padded,blocks);
 CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
 DeviceBuffer qcodes(96*256),qscales(96*4),sample_scores(std::size_t(96)*samples*4),sorted(sample_scores.bytes),offsets(97*4),threshold(96*4),
 masks(std::size_t(96)*blocks*4),selected((96*4096+128)*4),counts(96*4),fallback(4*4),partial(std::size_t(96)*parts*32*4),
 states(std::size_t(96)*parts*16),coordinate(96*32*4),global(96*16),corrections(std::size_t(96)*129*289*4);
 std::vector<int> off(97);for(int i=0;i<97;++i)off[i]=i*samples;offsets.copy_from_host(off.data(),offsets.bytes);CUDA_CHECK(cudaDeviceSynchronize());
 std::size_t sort_bytes=0;
 CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeysDescending(nullptr,sort_bytes,(const float*)sample_scores.p,(float*)sorted.p,96*samples,96,(const int*)offsets.p,(const int*)offsets.p+1,0,32,stream));
 DeviceBuffer sort_work(sort_bytes);
 DeviceBuffer sample_packet(std::size_t(4)*((samples+31)/32)*8*4*32*8),sample_scales(std::size_t(4)*samples*2);
 vr::prepare_sample_packet<<<(sample_packet.bytes/8+255)/256,256,0,stream>>>((const std::int8_t*)key.p,(const __half*)ks.p,(const int*)table.p,old_end,samples,(uint2*)sample_packet.p,(__half*)sample_scales.p);
 CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));

 constexpr int union_splits=64;
 DeviceBuffer union_masks(std::size_t(4)*blocks*4),union_ids((std::size_t(4)*vr::UnionCapacity+128)*4),union_counts(16),union_pos(64),union_table(vr::UnionPages*4),
 union_k(std::size_t(vr::UnionRows)*4*256),union_v(union_k.bytes),union_ks(union_k.bytes/64*2),union_vs(union_ks.bytes),
 subset_partial(std::size_t(96)*256*union_splits*2),subset_m(96*union_splits*4),subset_l(subset_m.bytes),weighted_selected(96*256*4),selected_mass(96*4);
 std::vector<int> ut(vr::UnionPages);for(int i=0;i<vr::UnionPages;++i)ut[i]=i;union_table.copy_from_host(ut.data(),union_table.bytes);CUDA_CHECK(cudaDeviceSynchronize());

 const ops::GqaExecutionEnvelope envelope{static_cast<unsigned>(padded),static_cast<unsigned>(padded)};
 const int splits=ops::detail::gqa_attention_split_capacity(24,4,DType::I8,envelope);
 DeviceBuffer native_partial(std::size_t(96)*256*splits*2),native_m(96*splits*4),native_l(native_m.bytes);
 auto original=[&](cudaStream_t st){
  ops::headscale_qk::gqa_attention_headscale_qk_kernel<ops::Gqa27Geometry,4,8,2,16,false,false,false,ops::GqaCachedInput,true>
  <<<dim3(4,splits),256,0,st>>>((const __nv_bfloat16*)query.p,{},(const int*)positions.p,(std::int8_t*)key.p,(std::int8_t*)value.p,(__half*)ks.p,(__half*)vs.p,(const int*)table.p,nullptr,nullptr,pages,4,0,padded,.0625f,0,splits,splits,1,padded,(__nv_bfloat16*)native_partial.p,(float*)native_m.p,(float*)native_l.p);
  ops::gqa_attention_small_t_reduce_output_kernel<ops::Gqa27Geometry,64,true,false,false,false><<<dim3(24,4,4),256,0,st>>>((const __nv_bfloat16*)native_partial.p,(const float*)native_m.p,(const float*)native_l.p,(const int*)positions.p,nullptr,4,4,0,1,splits,(__nv_bfloat16*)reference.p);
  CUDA_CHECK(cudaGetLastError());
 };
 auto sampling=[&](cudaStream_t st){
  vr::quantize_queries<<<12,256,0,st>>>((const __nv_bfloat16*)query.p,(std::int8_t*)qcodes.p,(float*)qscales.p);
  vr::sampled_packet_qk<<<(4*((samples+31)/32)*2+7)/8,256,0,st>>>((const std::int8_t*)qcodes.p,(const float*)qscales.p,(const std::int8_t*)sample_packet.p,(const __half*)sample_scales.p,(const int*)table.p,old_end,samples,(float*)sample_scores.p);
  vr::sample_order_statistic<<<96,256,0,st>>>((const float*)sample_scores.p,samples,order,(float*)threshold.p);
 };
 auto dense_coordinates=[&](cudaStream_t st){
  CUDA_CHECK(cudaMemsetAsync(masks.p,0,masks.bytes,st));
  ops::split_qk_coordinate::coordinate_kernel<ops::Gqa27Geometry,4,8,2,16,false,false,false,ops::GqaCachedInput,true>
  <<<dim3(4,parts),256,0,st>>>((const __nv_bfloat16*)query.p,{},(const int*)positions.p,(std::int8_t*)key.p,(std::int8_t*)linear.p,(__half*)ks.p,nullptr,(const int*)table.p,nullptr,nullptr,pages,4,0,padded,.0625f,0,parts,parts,1,padded,(float*)partial.p,nullptr,nullptr,(float4*)states.p,(const float*)threshold.p,(unsigned*)masks.p,blocks,old_end);
 };
 auto select_and_normalize=[&](cudaStream_t st){
  CUDA_CHECK(cudaMemsetAsync(fallback.p,0,fallback.bytes,st));
  vr::compact_masks<<<96,256,0,st>>>((const unsigned*)masks.p,blocks,old_end,(int*)selected.p,(int*)counts.p,(int*)fallback.p);
  vr::combine_coordinates_parallel<<<96,256,0,st>>>((const float*)partial.p,(const float4*)states.p,parts,(float*)coordinate.p,(float4*)global.p);
 };
 auto gathered_correction=[&](cudaStream_t st){
  vr::union_masks<<<(4*blocks+255)/256,256,0,st>>>((const unsigned*)masks.p,blocks,(unsigned*)union_masks.p);
  vr::compact_union<<<4,256,0,st>>>((const unsigned*)union_masks.p,blocks,old_end,(int*)union_ids.p,(int*)union_counts.p,(int*)fallback.p);
  vr::union_positions<<<1,32,0,st>>>((const int*)positions.p,(const int*)union_counts.p,old_end,(int*)union_pos.p);
  vr::gather_union<<<dim3(128,4),256,0,st>>>((const std::int8_t*)key.p,(const std::int8_t*)value.p,(const __half*)ks.p,(const __half*)vs.p,(const int*)table.p,(const int*)union_ids.p,(const int*)union_counts.p,(const int*)fallback.p,keys,old_end,(std::int8_t*)union_k.p,(std::int8_t*)union_v.p,(__half*)union_ks.p,(__half*)union_vs.p);
  ops::value_residual_subset::gqa_attention_headscale_qk_kernel<ops::Gqa27Geometry,4,8,2,16,false,false,false,ops::GqaCachedInput,true>
  <<<dim3(4,union_splits),256,0,st>>>((const __nv_bfloat16*)query.p,{},(const int*)union_pos.p,(std::int8_t*)union_k.p,(std::int8_t*)union_v.p,(__half*)union_ks.p,(__half*)union_vs.p,(const int*)union_table.p,nullptr,nullptr,vr::UnionPages,4,0,vr::UnionRows,.0625f,0,union_splits,union_splits,1,0,(__nv_bfloat16*)subset_partial.p,(float*)subset_m.p,(float*)subset_l.p,(const int*)fallback.p);
  vr::combine_subset<<<96,256,0,st>>>((const __nv_bfloat16*)subset_partial.p,(const float*)subset_m.p,(const float*)subset_l.p,(const int*)union_pos.p,union_splits,(const int*)fallback.p,(const float4*)global.p,(float*)weighted_selected.p,(float*)selected_mass.p);
  vr::finish_shared_residual<<<96,256,0,st>>>((const float*)coordinate.p,(const float*)weighted_selected.p,(const float*)selected_mass.p,(const __half*)basis.p,(const __half*)mean.p,(const int*)fallback.p,(__nv_bfloat16*)output.p);
 };
 auto sparse_correction=[&](cudaStream_t st){
  vr::union_masks<<<(4*blocks+255)/256,256,0,st>>>((const unsigned*)masks.p,blocks,(unsigned*)union_masks.p);
  vr::compact_union<<<4,256,0,st>>>((const unsigned*)union_masks.p,blocks,old_end,(int*)union_ids.p,(int*)union_counts.p,(int*)fallback.p);
  vr::union_positions<<<1,32,0,st>>>((const int*)positions.p,(const int*)union_counts.p,old_end,(int*)union_pos.p);
  ops::value_residual_indexed_subset::gqa_attention_headscale_qk_kernel<ops::Gqa27Geometry,4,8,2,16,false,false,false,ops::GqaCachedInput,true>
  <<<dim3(4,union_splits),256,0,st>>>((const __nv_bfloat16*)query.p,{},(const int*)union_pos.p,(std::int8_t*)key.p,(std::int8_t*)value.p,(__half*)ks.p,(__half*)vs.p,(const int*)union_table.p,nullptr,nullptr,vr::UnionPages,4,0,vr::UnionRows,.0625f,0,union_splits,union_splits,1,0,(__nv_bfloat16*)subset_partial.p,(float*)subset_m.p,(float*)subset_l.p,(const int*)fallback.p,(const int*)union_ids.p,(const int*)union_counts.p,old_end,(const int*)table.p);
  vr::combine_subset<<<96,256,0,st>>>((const __nv_bfloat16*)subset_partial.p,(const float*)subset_m.p,(const float*)subset_l.p,(const int*)union_pos.p,union_splits,(const int*)fallback.p,(const float4*)global.p,(float*)weighted_selected.p,(float*)selected_mass.p);
  vr::finish_shared_residual<<<96,256,0,st>>>((const float*)coordinate.p,(const float*)weighted_selected.p,(const float*)selected_mass.p,(const __half*)basis.p,(const __half*)mean.p,(const int*)fallback.p,(__nv_bfloat16*)output.p);
 };
 auto exact_fallback=[&](cudaStream_t st){
  fb::gqa_attention_headscale_qk_kernel<ops::Gqa27Geometry,4,8,2,16,false,false,false,ops::GqaCachedInput,true>
  <<<dim3(4,splits),256,0,st>>>((const __nv_bfloat16*)query.p,{},(const int*)positions.p,(std::int8_t*)key.p,(std::int8_t*)value.p,(__half*)ks.p,(__half*)vs.p,(const int*)table.p,nullptr,nullptr,pages,4,0,padded,.0625f,0,splits,splits,1,padded,(__nv_bfloat16*)native_partial.p,(float*)native_m.p,(float*)native_l.p,(const int*)fallback.p);
  fb::reduce_fallback<ops::Gqa27Geometry,64,true,false,false,false><<<dim3(24,4,4),256,0,st>>>((const __nv_bfloat16*)native_partial.p,(const float*)native_m.p,(const float*)native_l.p,(const int*)positions.p,nullptr,4,4,0,1,splits,(__nv_bfloat16*)output.p,(const int*)fallback.p);
 };
 auto candidate=[&](cudaStream_t st){sampling(st);dense_coordinates(st);select_and_normalize(st);sparse_correction(st);exact_fallback(st);CUDA_CHECK(cudaGetLastError());};
 CUDA_CHECK(cudaMemsetAsync(reference.p,0x55,reference.bytes,stream));CUDA_CHECK(cudaMemsetAsync(output.p,0x55,output.bytes,stream));CUDA_CHECK(cudaMemsetAsync(selected.p,0x55,selected.bytes,stream));
 original(stream);candidate(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
 auto expected=download<float>(expected_buffer),true_reference=download<float>(reference_buffer);auto actual=download<std::uint16_t>(output),native=download<std::uint16_t>(reference);
 auto flagged=download<int>(fallback),numbers=download<int>(counts),indices=download<int>(selected);
 auto thresholds=download<float>(threshold);int falls=0;
 auto prior_sample_scores=download<float>(sample_scores);
 vr::sample_qk<<<(4*((samples+31)/32)*2+7)/8,256,0,stream>>>((const std::int8_t*)qcodes.p,(const float*)qscales.p,(const std::int8_t*)key.p,(const __half*)ks.p,(const int*)table.p,old_end,samples,(float*)sample_scores.p);
 CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
 if(download<float>(sample_scores)!=prior_sample_scores)throw std::runtime_error("packed sample differs from canonical source scores");
 printf("SAMPLE_PACKET exact_current_query_scores prepared_source_bytes=%zu\n",sample_packet.bytes+sample_scales.bytes);

 CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeysDescending(sort_work.p,sort_bytes,(const float*)sample_scores.p,(float*)sorted.p,96*samples,96,(const int*)offsets.p,(const int*)offsets.p+1,0,32,stream));
 vr::choose_threshold<<<1,128,0,stream>>>((const float*)sorted.p,samples,order,(float*)threshold.p);CUDA_CHECK(cudaStreamSynchronize(stream));
 if(download<float>(threshold)!=thresholds)throw std::runtime_error("radix-selected threshold differs from complete CUB sort");
 printf("INDEPENDENT_SELECTION exact_CUB_thresholds\n");
 for(int h=0;h<4;++h)if(flagged[h]){++falls;for(int tok=0;tok<4;++tok)for(int qh=h*6;qh<h*6+6;++qh)for(int d=0;d<256;++d)expected[(tok*24+qh)*256+d]=true_reference[(tok*24+qh)*256+d];}
 for(int q=0;q<96;++q){if(flagged[q/24])continue;int previous=-1;for(int i=0;i<numbers[q];++i){int keyid=indices[q*4096+i];if(keyid<=previous||keyid>=old_end)throw std::runtime_error("selected-index order/bounds failed");previous=keyid;}}
 for(std::size_t i=96*256;i<actual.size();++i)if(actual[i]!=0x5555U)throw std::runtime_error("output guard overwritten");
 for(std::size_t i=96*4096;i<indices.size();++i)if(indices[i]!=0x55555555)throw std::runtime_error("selected-index guard overwritten");
 auto before_subset=download<std::uint16_t>(subset_partial);auto before_m=download<float>(subset_m);auto before_l=download<float>(subset_l);
 gathered_correction(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
 if(download<std::uint16_t>(output)!=actual||download<std::uint16_t>(subset_partial)!=before_subset||download<float>(subset_m)!=before_m||download<float>(subset_l)!=before_l)throw std::runtime_error("indexed/gathered subset parity failed");
 printf("INDEXED_SUBSET bit_exact_to_explicit_gathered_KV_and_native_tensor_attention\n");
 auto union_sizes=download<int>(union_counts);auto union_indices=download<int>(union_ids);
 for(int h=0;h<4;++h){if(flagged[h])continue;int prev=-1;for(int i=0;i<union_sizes[h];++i){int id=union_indices[h*vr::UnionCapacity+i];if(id<=prev||id>=old_end)throw std::runtime_error("union index order/bounds failed");prev=id;}}
 printf("UNION_COUNTS %d %d %d %d\n",union_sizes[0],union_sizes[1],union_sizes[2],union_sizes[3]);
 auto algorithm=compare_rows(actual,expected),quality=compare_rows(actual,true_reference),native_error=compare_rows(native,true_reference);
 printf("ACTUAL_INPUT rows=96 keys=%d old_end=%d selected_min=%d selected_max=%d fallback_heads=%d\n",keys,old_end,*std::min_element(numbers.begin(),numbers.end()),*std::max_element(numbers.begin(),numbers.end()),falls);
 printf("REPRESENTED_REFERENCE max_relative_l2=%.9g min_cosine=%.9g nonfinite=%d\n",algorithm.relative,algorithm.cosine,algorithm.bad);
 printf("NATIVE_DISTANCE max_relative_l2=%.9g min_cosine=%.9g reference_native_error=%.9g\n",quality.relative,quality.cosine,native_error.relative);
 if(argc>4){std::filesystem::path prefix=argv[4];write_file(prefix.string()+".union_counts.bin",union_sizes);write_file(prefix.string()+".selected_values.bin",download<float>(weighted_selected));write_file(prefix.string()+".selected_mass.bin",download<float>(selected_mass));write_file(prefix.string()+".coordinates.bin",download<float>(coordinate));write_file(prefix.string()+".output.bin",actual);write_file(prefix.string()+".threshold.bin",thresholds);write_file(prefix.string()+".counts.bin",numbers);write_file(prefix.string()+".selected.bin",indices);write_file(prefix.string()+".qcodes.bin",download<std::int8_t>(qcodes));write_file(prefix.string()+".qscales.bin",download<float>(qscales));}
 if(algorithm.relative>.025||algorithm.cosine<.999||algorithm.bad)throw std::runtime_error("independent represented algorithm mismatch");
 if(quality.relative>.15||quality.cosine<.995)throw std::runtime_error("actual-query numerical gate failed");
 for(int mode:{-1,1}){
  sampling(stream);force_threshold<<<1,128,0,stream>>>((float*)threshold.p,mode);CUDA_CHECK(cudaMemsetAsync(selected.p,0x55,selected.bytes,stream));
  dense_coordinates(stream);select_and_normalize(stream);sparse_correction(stream);exact_fallback(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
  auto f=download<std::uint16_t>(output);auto c=download<int>(selected);if(f!=native)throw std::runtime_error("forced fallback differs from original");
  for(auto value:c)if(value!=0x55555555)throw std::runtime_error("out-of-range selection wrote indices");
  printf("FALLBACK mode=%d native_output_bit_exact bounded_index_storage\n",mode);
 }
 candidate(stream);CUDA_CHECK(cudaStreamSynchronize(stream));if(download<std::uint16_t>(output)!=actual)throw std::runtime_error("normal operation changed after fallback");
 cudaFuncAttributes attr{};CUDA_CHECK(cudaFuncGetAttributes(&attr,ops::split_qk_coordinate::coordinate_kernel<ops::Gqa27Geometry,4,8,2,16,false,false,false,ops::GqaCachedInput,true>));printf("RESOURCE main_registers=%d shared_bytes=%zu local_bytes=%zu\n",attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes);
 printf("MEMORY prepared_coefficients=%zu sort_workspace=%zu decode_workspace_excluding_native_fallback=%zu prefix_basis_construction_not_timed\n",packed.bytes,sort_work.bytes,qcodes.bytes+qscales.bytes+sample_scores.bytes+sorted.bytes+offsets.bytes+threshold.bytes+masks.bytes+selected.bytes+counts.bytes+fallback.bytes+partial.bytes+states.bytes+coordinate.bytes+global.bytes+corrections.bytes+sort_work.bytes);
 cudaGraph_t gb{},gc{};cudaGraphExec_t eb{},ec{};
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));original(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gb));CUDA_CHECK(cudaGraphInstantiate(&eb,gb,nullptr,nullptr,0));
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));candidate(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gc));CUDA_CHECK(cudaGraphInstantiate(&ec,gc,nullptr,nullptr,0));
 auto rb=[&](cudaStream_t st){CUDA_CHECK(cudaGraphLaunch(eb,st));};auto rc=[&](cudaStream_t st){CUDA_CHECK(cudaGraphLaunch(ec,st));};
 if(std::getenv("NINFER_VALUE_RESIDUAL_PROFILE")!=nullptr){
  CUDA_CHECK(cudaProfilerStart());CUDA_CHECK(cudaGraphLaunch(eb,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGraphLaunch(ec,stream));CUDA_CHECK(cudaStreamSynchronize(stream));CUDA_CHECK(cudaProfilerStop());
 }
 for(int pass=0;pass<2;++pass){double b,c;if(pass==0){b=bench::measure_cold_launch(rb,flush,stream,3,12).median_us;c=bench::measure_cold_launch(rc,flush,stream,3,12).median_us;}else{c=bench::measure_cold_launch(rc,flush,stream,3,12).median_us;b=bench::measure_cold_launch(rb,flush,stream,3,12).median_us;}
  printf("COMPLETE_COST pass=%d native_us=%.3f candidate_us=%.3f speedup=%.6f\n",pass,b,c,b/c);fflush(stdout);
 }
 if(download<std::uint16_t>(output)!=actual)throw std::runtime_error("graph-repeat output changed");
 printf("GRAPH_REPEAT exact_to_first_candidate_output\n");
 CUDA_CHECK(cudaGraphExecDestroy(eb));CUDA_CHECK(cudaGraphExecDestroy(ec));CUDA_CHECK(cudaGraphDestroy(gb));CUDA_CHECK(cudaGraphDestroy(gc));
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& error){fprintf(stderr,"value residual: %s\n",error.what());return 1;}}
