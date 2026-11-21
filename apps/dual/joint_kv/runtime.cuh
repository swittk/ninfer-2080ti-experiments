#pragma once
#include "runtime_kernels.cuh"
#include "integer_coordinates.cuh"
#include "query_owned_finish.cuh"
#include "query_input_prepare.cuh"
#include <cstdlib>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <array>
#include <memory>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <cmath>
#include <cstring>
namespace ninfer::apps::dual::joint_runtime {
inline void cuda_check(cudaError_t r,const char* s){if(r!=cudaSuccess)throw std::runtime_error(std::string("jointKV ")+s+": "+cudaGetErrorString(r));}
inline void blas_check(cublasStatus_t r,const char* s){if(r!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string("jointKV BLAS ")+s+" code="+std::to_string(int(r)));}
inline void solver_check(cusolverStatus_t r,const char* s){if(r!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error(std::string("jointKV solver ")+s+" code="+std::to_string(int(r)));}
struct Allocation {
 int device;std::size_t bytes;void* p=nullptr;
 Allocation(int d,std::size_t n):device(d),bytes(n){cuda_check(cudaSetDevice(d),"allocate device");cuda_check(cudaMalloc(&p,std::max<std::size_t>(n,16)),"allocate");}
 ~Allocation(){if(p){(void)cudaSetDevice(device);(void)cudaFree(p);}}
 // Only preparation may retire staging allocations, after its CUDA work has
 // completed. Inference allocations remain stable throughout graph lifetime.
 void release_completed() {
  if(!p)return;
  cuda_check(cudaSetDevice(device),"release completed staging device");
  cuda_check(cudaFree(p),"release completed staging allocation");p=nullptr;bytes=0;
 }
 Allocation(const Allocation&)=delete;Allocation&operator=(const Allocation&)=delete;
};
struct PreparedLayer {
 int device,blocks,samples,block_begin;bool ready=false;const void* original_k=nullptr;const void* original_v=nullptr;
 Allocation k,v,k_basis,v_basis,k_mean,v_mean,sample;
 std::unique_ptr<Allocation> integer_k,integer_k_scale,integer_v,integer_v_scale,integer_sample,integer_sample_scale;
 void prepare_integer(cudaStream_t stream){
  cuda_check(cudaSetDevice(device),"integer coordinate device");
  integer_k=std::make_unique<Allocation>(device,std::size_t(4)*blocks*32*64);
  integer_k_scale=std::make_unique<Allocation>(device,std::size_t(4)*blocks*32*2);
  integer_v=std::make_unique<Allocation>(device,std::size_t(4)*blocks*32*32);
  integer_v_scale=std::make_unique<Allocation>(device,std::size_t(4)*blocks*32*2);
  integer_sample=std::make_unique<Allocation>(device,std::size_t(4)*((samples+31)/32)*32*64);
  integer_sample_scale=std::make_unique<Allocation>(device,std::size_t(4)*samples*2);
  integer_joint::quantize_keys<<<(4*blocks*32+7)/8,256,0,stream>>>(static_cast<const unsigned*>(k.p),static_cast<std::int8_t*>(integer_k->p),static_cast<__half*>(integer_k_scale->p),blocks,blocks*32);
  integer_joint::quantize_values<<<(4*blocks*32+7)/8,256,0,stream>>>(static_cast<const unsigned*>(v.p),static_cast<std::int8_t*>(integer_v->p),static_cast<__half*>(integer_v_scale->p),blocks);
  integer_joint::quantize_keys<<<(4*samples+7)/8,256,0,stream>>>(static_cast<const unsigned*>(sample.p),static_cast<std::int8_t*>(integer_sample->p),static_cast<__half*>(integer_sample_scale->p),(samples+31)/32,samples);
  cuda_check(cudaGetLastError(),"integer projected-coordinate preparation");
 }

 void release_float_coordinates() {
  if(ready || execution || !integer_k || !integer_v || !integer_sample)
   throw std::logic_error("coordinate staging release requires completed integer preparation before activation");
  // Quantized coordinates, their bases/means, and original canonical KV are
  // sufficient for integer inference, exact corrections and native fallback.
  k.release_completed();v.release_completed();sample.release_completed();
 }

 bool finish_basis_transposed=false;
 cudaGraph_t graph=nullptr;cudaGraphExec_t execution=nullptr;
 std::array<const void*,10> graph_sources{};
 ~PreparedLayer(){
  (void)cudaSetDevice(device);
  // Collection drains both streams before any prepared source or graph is freed.
  if(execution)(void)cudaGraphExecDestroy(execution);
  if(graph)(void)cudaGraphDestroy(graph);
 }

 PreparedLayer(int d,int owned_begin,int prefix_rows,int sample_count):device(d),blocks(std::max(1,prefix_rows/32)),samples(sample_count),block_begin(owned_begin/32),
  k(d,std::size_t(4)*blocks*1024*4),v(d,std::size_t(4)*blocks*512*4),k_basis(d,4*256*64*2),v_basis(d,4*256*32*2),k_mean(d,4*256*2),v_mean(d,4*256*2),sample(d,std::size_t(4)*((samples+31)/32)*1024*4){
   cuda_check(cudaMemset(k.p,0,k.bytes),"zero unowned key coordinates");cuda_check(cudaMemset(v.p,0,v.bytes),"zero unowned value coordinates");cuda_check(cudaDeviceSynchronize(),"coordinate storage initialized");
 }
 std::size_t bytes()const{
  std::size_t total=k.bytes+v.bytes+k_basis.bytes+v_basis.bytes+k_mean.bytes+v_mean.bytes+sample.bytes;
  for(const auto* b:{integer_k.get(),integer_k_scale.get(),integer_v.get(),integer_v_scale.get(),integer_sample.get(),integer_sample_scale.get()})if(b)total+=b->bytes;
  return total;
 }
};
struct DeviceWorkspace {
 int device,capacity,blocks,parts,samples,range_begin,range_end,prefix_rows;
 cudaStream_t stream;cublasHandle_t blas=nullptr;
 Allocation query_codes,query_scale,projected_query,query_shift,sample_scores,threshold,masks,union_masks,union_indices,metadata,subset_positions,identity_table,
  partial,states,coords,global,subset_partial,subset_m,subset_l,selected_values,exact_state,
  matrix,projected,moment_parts,moments,center_mean,covariance,basis_float;
 std::array<int,104> host_meta{};void* pinned_meta=nullptr;
 std::array<double,256> host_moments{};std::vector<float> host_covariance;
 DeviceWorkspace(int d,int cap,int sample_count,int prefix_end,int begin,int end,cudaStream_t st):
  device(d),capacity(cap),blocks((cap+31)/32),parts((cap+511)/512),samples(sample_count),range_begin(begin),range_end(end),prefix_rows(std::max(0,std::min(end,prefix_end)-begin)),stream(st),
  query_codes(d,96*256),query_scale(d,96*4),projected_query(d,96*64*2),query_shift(d,96*4),sample_scores(d,std::size_t(96)*samples*4),threshold(d,96*4),
  masks(d,std::size_t(96)*blocks*4),union_masks(d,std::size_t(4)*blocks*4),union_indices(d,std::size_t(4)*UnionLimit*4),metadata(d,104*4),subset_positions(d,16*4),identity_table(d,SubsetPages*4),
  partial(d,std::size_t(96)*parts*32*4),states(d,std::size_t(96)*parts*16),coords(d,96*32*4),global(d,96*16),
  subset_partial(d,std::size_t(96)*256*LocalSubsetSplits*2),subset_m(d,96*LocalSubsetSplits*4),subset_l(d,96*LocalSubsetSplits*4),selected_values(d,96*256*4),exact_state(d,96*16),
  matrix(d,std::size_t(std::max(1,prefix_rows))*256*4),projected(d,std::size_t(std::max(1,prefix_rows))*64*4),moment_parts(d,std::size_t((prefix_rows+255)/256+1)*256*8),moments(d,256*8),center_mean(d,256*2),covariance(d,256*256*4),basis_float(d,256*64*4),host_covariance(256*256){
   cuda_check(cudaSetDevice(d),"workspace device");cuda_check(cudaMallocHost(&pinned_meta,metadata.bytes),"pinned count metadata");
   std::vector<int> table(SubsetPages);for(int i=0;i<SubsetPages;++i)table[i]=i;
   cuda_check(cudaMemcpy(identity_table.p,table.data(),identity_table.bytes,cudaMemcpyHostToDevice),"subset table initialization");cuda_check(cudaDeviceSynchronize(),"subset table ready");
   blas_check(cublasCreate(&blas),"create");blas_check(cublasSetStream(blas,stream),"stream");blas_check(cublasSetMathMode(blas,CUBLAS_DEFAULT_MATH),"math mode");blas_check(cublasSetAtomicsMode(blas,CUBLAS_ATOMICS_NOT_ALLOWED),"deterministic products");
 }
 ~DeviceWorkspace(){(void)cudaSetDevice(device);if(blas)(void)cublasDestroy(blas);if(pinned_meta)(void)cudaFreeHost(pinned_meta);}
 int* counts(){return static_cast<int*>(metadata.p);}int* union_counts(){return static_cast<int*>(metadata.p)+96;}int* failed(){return static_cast<int*>(metadata.p)+100;}
 void synchronize(){cuda_check(cudaSetDevice(device),"sync device");cuda_check(cudaStreamSynchronize(stream),"stream ready");}
 void decode(const PagedKVLayerView& cache,int head,bool values){
  cuda_check(cudaSetDevice(device),"decode device");
  if(prefix_rows){decode_plane<<<(std::size_t(prefix_rows)*256+255)/256,256,0,stream>>>(
    static_cast<const std::int8_t*>(values?cache.v_pages.data:cache.k_pages.data),static_cast<const __half*>(values?cache.v_scale_pages.data:cache.k_scale_pages.data),static_cast<const int*>(cache.block_table.data),head,range_begin,prefix_rows,values,static_cast<float*>(matrix.p));
   moment_blocks<<<(prefix_rows+255)/256,256,0,stream>>>(static_cast<const float*>(matrix.p),prefix_rows,static_cast<double*>(moment_parts.p));
   finish_moments<<<1,256,0,stream>>>(static_cast<const double*>(moment_parts.p),(prefix_rows+255)/256,static_cast<double*>(moments.p));
  }else cuda_check(cudaMemsetAsync(moments.p,0,moments.bytes,stream),"empty prefix moments");
  cuda_check(cudaGetLastError(),"prefix moment kernels");cuda_check(cudaMemcpyAsync(host_moments.data(),moments.p,moments.bytes,cudaMemcpyDeviceToHost,stream),"moment download");
 }
 void cov(const std::array<__half,256>& mean,int total_prefix){
  cuda_check(cudaSetDevice(device),"covariance device");cuda_check(cudaMemcpyAsync(center_mean.p,mean.data(),center_mean.bytes,cudaMemcpyHostToDevice,stream),"global mean upload");
  if(prefix_rows){
   center_matrix<<<(std::size_t(prefix_rows)*256+255)/256,256,0,stream>>>(static_cast<float*>(matrix.p),static_cast<const __half*>(center_mean.p),prefix_rows);
   const float alpha=1.0f/total_prefix,beta=0;
   blas_check(cublasSgemm(blas,CUBLAS_OP_N,CUBLAS_OP_T,256,256,prefix_rows,&alpha,static_cast<const float*>(matrix.p),256,static_cast<const float*>(matrix.p),256,&beta,static_cast<float*>(covariance.p),256),"live prefix covariance");
  }else cuda_check(cudaMemsetAsync(covariance.p,0,covariance.bytes,stream),"empty covariance");
  cuda_check(cudaMemcpyAsync(host_covariance.data(),covariance.p,covariance.bytes,cudaMemcpyDeviceToHost,stream),"covariance download");
 }
 void project(PreparedLayer& layer,int head,bool values,const std::vector<__half>& basis,const std::array<__half,256>& mean){
  cuda_check(cudaSetDevice(device),"projection device");const int rank=values?32:64;
  auto* destination=static_cast<__half*>(values?layer.v_basis.p:layer.k_basis.p)+head*256*rank;
  cuda_check(cudaMemcpyAsync(destination,basis.data(),basis.size()*2,cudaMemcpyHostToDevice,stream),"shared basis upload");
  cuda_check(cudaMemcpyAsync(static_cast<__half*>(values?layer.v_mean.p:layer.k_mean.p)+head*256,mean.data(),512,cudaMemcpyHostToDevice,stream),"shared mean upload");
  if(prefix_rows){
   basis_to_float<<<(256*rank+255)/256,256,0,stream>>>(destination,256*rank,static_cast<float*>(basis_float.p));
   const float alpha=1,beta=0;
   // Row-major [256,rank] equals column-major [rank,256]. The live
   // centered matrix is column-major [256,prefix_rows].
   blas_check(cublasSgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,rank,prefix_rows,256,&alpha,static_cast<const float*>(basis_float.p),rank,static_cast<const float*>(matrix.p),256,&beta,static_cast<float*>(projected.p),rank),"source-derived coordinates");
   if(values)pack_projected<32,false><<<(std::size_t(prefix_rows/32)*512+255)/256,256,0,stream>>>(static_cast<const float*>(projected.p),range_begin,prefix_rows,head,layer.blocks,static_cast<unsigned*>(layer.v.p),layer.block_begin);
   else pack_projected<64,true><<<(std::size_t(prefix_rows/32)*1024+255)/256,256,0,stream>>>(static_cast<const float*>(projected.p),range_begin,prefix_rows,head,layer.blocks,static_cast<unsigned*>(layer.k.p),layer.block_begin);
   cuda_check(cudaGetLastError(),"coordinate packing");
  }
 }
 void prepare_finish_basis(PreparedLayer& layer) {
  if(!layer.integer_k)return;
  if(layer.finish_basis_transposed)throw std::logic_error("jointKV basis already prepared for query completion");
  if(basis_float.bytes<layer.v_basis.bytes)throw std::logic_error("jointKV value-basis staging is too small");
  cuda_check(cudaSetDevice(device),"value-basis consumer layout device");
  cuda_check(cudaMemcpyAsync(basis_float.p,layer.v_basis.p,layer.v_basis.bytes,cudaMemcpyDeviceToDevice,stream),"value-basis staging copy");
  transpose_value_basis_for_queries<<<(4*256*32+255)/256,256,0,stream>>>(static_cast<const std::uint16_t*>(basis_float.p),static_cast<std::uint16_t*>(layer.v_basis.p));
  cuda_check(cudaGetLastError(),"value-basis consumer layout");
  layer.finish_basis_transposed=true;
 }
 void enqueue(PreparedLayer& layer,const PagedKVLayerView& cache,const Tensor& q,const Tensor& pos,int prefix_end,float* out,float* out_m,float* out_l){
  cuda_check(cudaSetDevice(device),"execute device");
  // Query positions are read on-device at replay. The bounded work envelope
  // is fixed for graph lifetime; invisible partitions emit neutral states.
  const int first_part=range_begin/512,active_end=std::min(range_end,prefix_end+MaxTail),active_parts=std::max(1,(std::max(range_begin,active_end)+511)/512-first_part);
  const int tail_begin=std::max(prefix_end,range_begin),tail_end=range_end;
  const int rank_order=std::clamp((3072*samples+prefix_end-1)/prefix_end,1,samples);
  const bool direct_metadata=static_cast<bool>(layer.integer_k);
  const bool fused=direct_metadata;
  if(fused!=layer.finish_basis_transposed)throw std::logic_error("jointKV completion/source-layout mismatch");
  if(!direct_metadata) {
   cuda_check(cudaMemsetAsync(metadata.p,0,metadata.bytes,stream),"clear selected metadata");
   cuda_check(cudaMemsetAsync(masks.p,0,masks.bytes,stream),"clear unowned query masks");
  }
  if(direct_metadata) {
   joint_residual::prepare_integer_query_roles<<<96,288,0,stream>>>(
     static_cast<const __nv_bfloat16*>(q.data),static_cast<const __half*>(layer.k_basis.p),
     static_cast<const __half*>(layer.k_mean.p),static_cast<__half*>(projected_query.p),
     static_cast<float*>(query_shift.p),static_cast<std::int8_t*>(query_codes.p),static_cast<float*>(query_scale.p),
     static_cast<unsigned*>(union_masks.p),static_cast<int>(union_masks.bytes/sizeof(unsigned)),static_cast<int*>(metadata.p));
  } else {
  value_residual::quantize_queries<<<12,256,0,stream>>>(static_cast<const __nv_bfloat16*>(q.data),static_cast<std::int8_t*>(query_codes.p),static_cast<float*>(query_scale.p));
  joint_residual::project_query<<<96,256,0,stream>>>(static_cast<const std::int8_t*>(query_codes.p),static_cast<const float*>(query_scale.p),static_cast<const __half*>(layer.k_basis.p),static_cast<const __half*>(layer.k_mean.p),static_cast<__half*>(projected_query.p),static_cast<float*>(query_shift.p));
  }
  if(layer.integer_k){
   integer_joint::sample_scores<<<(4*((samples+31)/32)*2+7)/8,256,0,stream>>>(static_cast<const std::int8_t*>(query_codes.p),static_cast<const float*>(query_scale.p),static_cast<const float*>(query_shift.p),static_cast<const unsigned*>(layer.integer_sample->p),static_cast<const __half*>(layer.integer_sample_scale->p),samples,static_cast<float*>(sample_scores.p));
  }else joint_residual::sample_scores<<<(4*((samples+31)/32)*2+7)/8,256,0,stream>>>(static_cast<const __half*>(projected_query.p),static_cast<const float*>(query_shift.p),static_cast<const unsigned*>(layer.sample.p),samples,static_cast<float*>(sample_scores.p));
  value_residual::sample_order_statistic<<<96,256,0,stream>>>(static_cast<const float*>(sample_scores.p),samples,rank_order,static_cast<float*>(threshold.p));
  if(direct_metadata) integer_joint::allkey<false,true><<<4*active_parts,96,0,stream>>>(static_cast<const std::int8_t*>(query_codes.p),static_cast<const float*>(query_scale.p),static_cast<const float*>(query_shift.p),static_cast<const int*>(pos.data),static_cast<const unsigned*>(layer.integer_k->p),nullptr,nullptr,static_cast<const unsigned*>(layer.integer_v->p),static_cast<const float*>(threshold.p),capacity,prefix_end,blocks,active_parts,static_cast<unsigned*>(masks.p),static_cast<float*>(partial.p),static_cast<float4*>(states.p),capacity,nullptr,nullptr,nullptr,range_begin,active_end,first_part,layer.blocks,layer.block_begin,static_cast<const __half*>(layer.integer_k_scale->p),static_cast<const __half*>(layer.integer_v_scale->p),static_cast<unsigned*>(union_masks.p));

  else joint_partition_scan::allkey<false><<<(4*active_parts*2+3)/4,128,0,stream>>>(static_cast<const __half*>(projected_query.p),static_cast<const float*>(query_shift.p),static_cast<const int*>(pos.data),static_cast<const unsigned*>(layer.k.p),nullptr,nullptr,static_cast<const unsigned*>(layer.v.p),static_cast<const float*>(threshold.p),capacity,prefix_end,blocks,active_parts,static_cast<unsigned*>(masks.p),static_cast<float*>(partial.p),static_cast<float4*>(states.p),capacity,nullptr,nullptr,nullptr,range_begin,active_end,first_part,layer.blocks,layer.block_begin);
  if(direct_metadata)integer_joint::reduce_selected_counts<<<96,256,0,stream>>>(static_cast<const unsigned*>(masks.p),active_parts,counts(),failed());
  else {
  count_local_masks<<<96,256,0,stream>>>(static_cast<const unsigned*>(masks.p),blocks,counts(),failed());
  union_masks_local<<<(4*blocks+255)/256,256,0,stream>>>(static_cast<const unsigned*>(masks.p),blocks,static_cast<unsigned*>(union_masks.p));
  }
  // Integer execution wrote per-partition counts into masks; float execution
  // wrote query masks. Both have now been consumed. Reuse their fixed allocation
  // for ordered compaction without adding storage or changing graph addresses.
  // Direct metadata overwrites each active count on the next replay and only
  // needs to clear the much smaller unowned head-union region.
  const int compact_tiles=ordered_union::tiles_for(blocks);
  auto* word_prefix=static_cast<int*>(masks.p);
  auto* tile_prefix=word_prefix+ordered_union::kHeads*blocks;
  ordered_union::tile_prefix<<<dim3(compact_tiles,4),256,0,stream>>>(
      static_cast<const unsigned*>(union_masks.p),blocks,word_prefix,tile_prefix,failed());
  ordered_union::head_prefix<<<4,32,0,stream>>>(blocks,tile_prefix,union_counts(),failed());
  ordered_union::scatter<<<dim3(compact_tiles,4),256,0,stream>>>(
      static_cast<const unsigned*>(union_masks.p),blocks,word_prefix,tile_prefix,
      static_cast<int*>(union_indices.p),failed());
  if(!fused) value_residual::combine_coordinates_parallel<<<96,256,0,stream>>>(static_cast<const float*>(partial.p),static_cast<const float4*>(states.p),active_parts,static_cast<float*>(coords.p),static_cast<float4*>(global.p));
  joint_runtime::subset_positions<<<1,32,0,stream>>>(static_cast<const int*>(pos.data),union_counts(),tail_begin,tail_end,static_cast<int*>(subset_positions.p));
  value_residual_indexed_subset::gqa_attention_headscale_qk_kernel<Gqa27Geometry,4,8,2,16,false,false,false,GqaCachedInput,true>
  <<<dim3(4,LocalSubsetSplits),256,0,stream>>>(static_cast<const __nv_bfloat16*>(q.data),{},static_cast<const int*>(subset_positions.p),static_cast<std::int8_t*>(cache.k_pages.data),static_cast<std::int8_t*>(cache.v_pages.data),static_cast<__half*>(cache.k_scale_pages.data),static_cast<__half*>(cache.v_scale_pages.data),static_cast<const int*>(identity_table.p),nullptr,nullptr,SubsetPages,4,0,SubsetRows,.0625f,0,LocalSubsetSplits,LocalSubsetSplits,1,0,static_cast<__nv_bfloat16*>(subset_partial.p),static_cast<float*>(subset_m.p),static_cast<float*>(subset_l.p),failed(),static_cast<const int*>(union_indices.p),union_counts(),tail_begin,static_cast<const int*>(cache.block_table.data));
  if(fused) {
   query_owned_finish<<<96,256,0,stream>>>(static_cast<const float*>(partial.p),static_cast<const float4*>(states.p),active_parts,
     static_cast<const __nv_bfloat16*>(subset_partial.p),static_cast<const float*>(subset_m.p),static_cast<const float*>(subset_l.p),LocalSubsetSplits,
     static_cast<const __half*>(layer.v_basis.p),static_cast<const __half*>(layer.v_mean.p),failed(),out,out_m,out_l);
  } else {
  joint_residual::normalize_exact_subset<<<96,256,0,stream>>>(static_cast<const __nv_bfloat16*>(subset_partial.p),static_cast<const float*>(subset_m.p),static_cast<const float*>(subset_l.p),LocalSubsetSplits,failed(),static_cast<float*>(selected_values.p),static_cast<float4*>(exact_state.p));
  finish_local_state<<<96,256,0,stream>>>(static_cast<const float*>(coords.p),static_cast<const float4*>(global.p),static_cast<const float*>(selected_values.p),static_cast<const float4*>(exact_state.p),static_cast<const __half*>(layer.v_basis.p),static_cast<const __half*>(layer.v_mean.p),failed(),out,out_m,out_l);
  }
  cuda_check(cudaGetLastError(),"joint attention pipeline");cuda_check(cudaMemcpyAsync(pinned_meta,metadata.p,metadata.bytes,cudaMemcpyDeviceToHost,stream),"global-count gate transfer");
 }
 void execute(PreparedLayer& layer,const PagedKVLayerView& cache,const Tensor& q,const Tensor& pos,int visible,int prefix_end,float* out,float* out_m,float* out_l,bool eager=false){
  cuda_check(cudaSetDevice(device),"graph execution device");
  if(eager){enqueue(layer,cache,q,pos,prefix_end,out,out_m,out_l);return;}
  const std::array<const void*,10> sources{q.data,pos.data,cache.k_pages.data,cache.v_pages.data,
    cache.k_scale_pages.data,cache.v_scale_pages.data,cache.block_table.data,out,out_m,out_l};
  if(!layer.execution){
   cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"joint graph begin");
   try {enqueue(layer,cache,q,pos,prefix_end,out,out_m,out_l);}
   catch(...){cudaGraph_t abandoned=nullptr;(void)cudaStreamEndCapture(stream,&abandoned);if(abandoned)(void)cudaGraphDestroy(abandoned);throw;}
   cuda_check(cudaStreamEndCapture(stream,&layer.graph),"joint graph end");
   cuda_check(cudaGraphInstantiate(&layer.execution,layer.graph,nullptr,nullptr,0),"joint graph instantiate");
   layer.graph_sources=sources;
  }else if(layer.graph_sources!=sources){
   throw std::runtime_error("joint attention graph source/destination addresses changed");
  }
  cuda_check(cudaGraphLaunch(layer.execution,stream),"joint graph replay");
 }
};
struct LayerPair {std::unique_ptr<PreparedLayer> a,b;bool ready=false;};
class Collection {
 int devices_[2];int capacity_,prefix_end_,samples_,layers_,begin_[2],end_[2];
 const bool integer_coordinates_ = std::getenv("NINFER_INTEGER_JOINT_COORDINATES") != nullptr;
 std::unique_ptr<DeviceWorkspace> workspace_[2];std::vector<LayerPair> layer_;
 cusolverDnHandle_t solver_=nullptr;std::unique_ptr<Allocation> eigen_,eigenvalues_,eigenwork_,info_;int work_size_=0;
 std::vector<int> table_[2];
 std::uint64_t calls_=0,fallbacks_=0,prepared_bytes_=0;double preparation_seconds_=0;
 std::array<std::uint64_t,16> layer_calls_{},layer_fallbacks_{};
 using Clock=std::chrono::steady_clock;
 static std::pair<int,int> ownership(const PagedKVLayerView& cache,int capacity,int device,cudaStream_t stream,std::vector<int>& table){
  cuda_check(cudaSetDevice(device),"ownership device");table.resize((capacity+63)/64);
  cuda_check(cudaMemcpyAsync(table.data(),cache.block_table.data,table.size()*4,cudaMemcpyDeviceToHost,stream),"ownership table");cuda_check(cudaStreamSynchronize(stream),"ownership table ready");
  int first=-1,last=-1;for(int i=0;i<int(table.size());++i)if(table[i]>=0){if(first<0)first=i;last=i;if(table[i]>=cache.k_pages.ne[3])throw std::runtime_error("jointKV physical page out of range");}
  if(first<0)throw std::runtime_error("jointKV empty physical owner");
  for(int i=first;i<=last;++i)if(table[i]<0)throw std::runtime_error("jointKV noncontiguous owner requires unsupported gather layout");
  return {first*64,std::min(capacity,(last+1)*64)};
 }
 void prepare_layer(int index,const PagedKVLayerView& ca,const PagedKVLayerView& cb){
  auto start=Clock::now();auto& pair=layer_[index];
  pair.a=std::make_unique<PreparedLayer>(devices_[0],begin_[0],workspace_[0]->prefix_rows,samples_);pair.b=std::make_unique<PreparedLayer>(devices_[1],begin_[1],workspace_[1]->prefix_rows,samples_);
  pair.a->original_k=ca.k_pages.data;pair.a->original_v=ca.v_pages.data;pair.b->original_k=cb.k_pages.data;pair.b->original_v=cb.v_pages.data;
  PreparedLayer* out[2]={pair.a.get(),pair.b.get()};const PagedKVLayerView* cache[2]={&ca,&cb};
  for(int head=0;head<4;++head)for(bool values:{false,true}){
   for(int r=0;r<2;++r)workspace_[r]->decode(*cache[r],head,values);
   for(auto& w:workspace_)w->synchronize();
   std::array<__half,256> mean;
   for(int i=0;i<256;++i)mean[i]=__float2half_rn(float((workspace_[0]->host_moments[i]+workspace_[1]->host_moments[i])/prefix_end_));
   for(auto& w:workspace_)w->cov(mean,prefix_end_);
   for(auto& w:workspace_)w->synchronize();
   std::vector<double> covariance(256*256);
   for(int col=0;col<256;++col)for(int row=0;row<256;++row){
    const auto sum=[&](int i){return double(workspace_[0]->host_covariance[i])+double(workspace_[1]->host_covariance[i]);};
    covariance[col*256+row]=.5*(sum(col*256+row)+sum(row*256+col));
   }
   cuda_check(cudaSetDevice(devices_[0]),"eigensolver device");auto st=workspace_[0]->stream;
   cuda_check(cudaMemcpyAsync(eigen_->p,covariance.data(),eigen_->bytes,cudaMemcpyHostToDevice,st),"global covariance upload");
   solver_check(cusolverDnDsyevd(solver_,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_UPPER,256,static_cast<double*>(eigen_->p),256,static_cast<double*>(eigenvalues_->p),static_cast<double*>(eigenwork_->p),work_size_,static_cast<int*>(info_->p)),"global prefix eigensolve");
   int status=0;cuda_check(cudaMemcpyAsync(&status,info_->p,4,cudaMemcpyDeviceToHost,st),"eigensolver convergence");workspace_[0]->synchronize();
   if(status!=0)throw std::runtime_error("jointKV covariance eigensolve failed: "+std::to_string(status));
   const int rank=values?32:64;Allocation half_basis(devices_[0],256*rank*2);
   select_basis<<<rank,256,0,st>>>(static_cast<const double*>(eigen_->p),rank,static_cast<__half*>(half_basis.p));cuda_check(cudaGetLastError(),"fixed eigenbasis selection");
   std::vector<__half> basis(256*rank);cuda_check(cudaMemcpyAsync(basis.data(),half_basis.p,basis.size()*2,cudaMemcpyDeviceToHost,st),"shared basis download");workspace_[0]->synchronize();
   for(int r=0;r<2;++r)workspace_[r]->project(*out[r],head,values,basis,mean);
   for(auto& w:workspace_)w->synchronize();
  }
  std::vector<unsigned> sample[2];
  for(int r=0;r<2;++r){cuda_check(cudaSetDevice(devices_[r]),"sample source device");auto& w=*workspace_[r];auto& p=*out[r];
   sample_from_partition<<<(p.sample.bytes/4+255)/256,256,0,w.stream>>>(static_cast<const unsigned*>(p.k.p),static_cast<const int*>(cache[r]->block_table.data),p.blocks,prefix_end_,samples_,static_cast<unsigned*>(p.sample.p),p.block_begin);
   cuda_check(cudaGetLastError(),"sample source packing");sample[r].resize(p.sample.bytes/4);cuda_check(cudaMemcpyAsync(sample[r].data(),p.sample.p,p.sample.bytes,cudaMemcpyDeviceToHost,w.stream),"sample-partition download");}
  for(auto& w:workspace_)w->synchronize();
  for(std::size_t i=0;i<sample[0].size();++i){if(sample[0][i]&&sample[1][i])throw std::runtime_error("jointKV sample has overlapping owner contributions");sample[0][i]|=sample[1][i];}
  for(int r=0;r<2;++r){cuda_check(cudaSetDevice(devices_[r]),"shared sample device");cuda_check(cudaMemcpyAsync(out[r]->sample.p,sample[0].data(),out[r]->sample.bytes,cudaMemcpyHostToDevice,workspace_[r]->stream),"global sample upload");}
  if(integer_coordinates_)for(int r=0;r<2;++r)out[r]->prepare_integer(workspace_[r]->stream);
  for(auto& w:workspace_)w->synchronize();
  if(integer_coordinates_){
   for(int r=0;r<2;++r)workspace_[r]->prepare_finish_basis(*out[r]);
   for(auto& w:workspace_)w->synchronize();
   for(auto* prepared:out)prepared->release_float_coordinates();
  }
  pair.a->ready=pair.b->ready=pair.ready=true;prepared_bytes_+=pair.a->bytes()+pair.b->bytes();
  const double seconds=std::chrono::duration<double>(Clock::now()-start).count();preparation_seconds_+=seconds;
  std::fprintf(stderr,"jointKV prepared layer_slot=%d prefix=%d seconds=%.6f additional_bytes_both_devices=%zu\n",index,prefix_end_,seconds,pair.a->bytes()+pair.b->bytes());
 }
public:
 Collection(int device0,int device1,int capacity,int populated,int layers,const PagedKVLayerView& a,const PagedKVLayerView& b,cudaStream_t stream0,cudaStream_t stream1):
 devices_{device0,device1},capacity_(capacity),prefix_end_((populated/32)*32),samples_(std::min(16384,prefix_end_)),layers_(layers),layer_(layers){
  if(capacity<32768||capacity>262144||prefix_end_<32768||device0==device1||layers<=0||layers>16)throw std::invalid_argument("jointKV unsupported full-context envelope");
  auto first=ownership(a,capacity,device0,stream0,table_[0]),second=ownership(b,capacity,device1,stream1,table_[1]);
  begin_[0]=first.first;end_[0]=first.second;begin_[1]=second.first;end_[1]=second.second;
  for(int i=0;i<int(table_[0].size());++i)if((table_[0][i]>=0)+(table_[1][i]>=0)!=1)throw std::runtime_error("jointKV source ownership gap/overlap");
  if((begin_[0]%32)||(end_[0]%32)||(begin_[1]%32)||(end_[1]%32))throw std::runtime_error("jointKV ownership not aligned to immutable coordinate blocks");
  for(int rank=0;rank<2;++rank){
   const int d=devices_[rank];
   const int owned_prefix=std::max(0,std::min(end_[rank],prefix_end_)-begin_[rank]);
   const auto packed_blocks=std::max(1,owned_prefix/32);
   const auto integer_bytes=integer_coordinates_ ? std::size_t(4)*packed_blocks*(32*64+32*2+32*32+32*2)+std::size_t(4)*((samples_+31)/32)*32*64+std::size_t(4)*samples_*2 : 0;
   const auto prepare_bytes=std::size_t(4)*packed_blocks*(1024+512)*4+std::size_t(4)*((samples_+31)/32)*1024*4+4*256*(64+32+2)*2+integer_bytes;
   cuda_check(cudaSetDevice(d),"reservation device");std::size_t free=0,total=0;cuda_check(cudaMemGetInfo(&free,&total),"memory reservation");
   // Includes the largest temporary F32 matrix/projected coordinates, count
   // workspaces and solver scratch. Do not silently shrink context to fit.
   const auto bases_bytes=std::size_t(4)*256*(64+32+2)*2;
   const auto float_coordinate_bytes=prepare_bytes-integer_bytes-bases_bytes;
   // Each completed layer keeps only integer coordinates. One current layer
   // temporarily owns both representations during conversion; count that peak.
   const auto prepared_peak=integer_coordinates_
       ? (integer_bytes+bases_bytes)*layers+float_coordinate_bytes
       : prepare_bytes*layers;
   const std::size_t required=prepared_peak+std::size_t(capacity)*320*4+(128ULL<<20);
   if(required>free)throw std::runtime_error("jointKV additional reservation needs "+std::to_string(required)+" bytes, available "+std::to_string(free));
  }
  workspace_[0]=std::make_unique<DeviceWorkspace>(device0,capacity,samples_,prefix_end_,begin_[0],end_[0],stream0);
  workspace_[1]=std::make_unique<DeviceWorkspace>(device1,capacity,samples_,prefix_end_,begin_[1],end_[1],stream1);
  cuda_check(cudaSetDevice(device0),"solver initialize device");solver_check(cusolverDnCreate(&solver_),"create");solver_check(cusolverDnSetStream(solver_,stream0),"stream");
  eigen_=std::make_unique<Allocation>(device0,256*256*8);eigenvalues_=std::make_unique<Allocation>(device0,256*8);info_=std::make_unique<Allocation>(device0,4);
  solver_check(cusolverDnDsyevd_bufferSize(solver_,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_UPPER,256,static_cast<double*>(eigen_->p),256,static_cast<double*>(eigenvalues_->p),&work_size_),"workspace size");eigenwork_=std::make_unique<Allocation>(device0,std::size_t(work_size_)*8);
 }
 ~Collection(){
  for(auto& w:workspace_)if(w){(void)cudaSetDevice(w->device);(void)cudaStreamSynchronize(w->stream);}
  if(solver_){(void)cudaSetDevice(devices_[0]);(void)cusolverDnDestroy(solver_);}
  for(int i=0;i<layers_;++i)if(layer_calls_[i])std::fprintf(stderr,"jointKV layer=%d calls=%llu fallback=%llu\n",i,(unsigned long long)layer_calls_[i],(unsigned long long)layer_fallbacks_[i]);
  std::fprintf(stderr,"jointKV terminal calls=%llu fallback_layers=%llu prepared_bytes_both_devices=%llu preparation_seconds=%.6f\n",(unsigned long long)calls_,(unsigned long long)fallbacks_,(unsigned long long)prepared_bytes_,preparation_seconds_);
 }
 void prepare_prefix(int index,const PagedKVLayerView& a,const PagedKVLayerView& b) {
  if(index<0||index>=layers_||a.dtype!=DType::I8||b.dtype!=DType::I8)
   throw std::invalid_argument("jointKV invalid prefix preparation layer");
  auto& layer=layer_[index];
  if(!layer.ready){for(auto& w:workspace_)w->synchronize();prepare_layer(index,a,b);}
  if(layer.a->original_k!=a.k_pages.data||layer.a->original_v!=a.v_pages.data||
     layer.b->original_k!=b.k_pages.data||layer.b->original_v!=b.v_pages.data)
   throw std::runtime_error("jointKV prefix preparation source changed");
 }
 bool execute(int index,const PagedKVLayerView& a,const PagedKVLayerView& b,const Tensor& qa,const Tensor& pa,const Tensor& qb,const Tensor& pb,int visible,
              const Tensor& oa,const Tensor& om,const Tensor& ol,const Tensor& ba,const Tensor& bm,const Tensor& bl,bool eager=false){
  if(index<0||index>=layers_||qa.ne[2]!=4||qb.ne[2]!=4||visible<prefix_end_||visible-prefix_end_>MaxTail)return false;
  if(a.dtype!=DType::I8||b.dtype!=DType::I8)throw std::runtime_error("jointKV requires canonical I8 owners");
  auto& layer=layer_[index];
  if(!layer.ready){for(auto& w:workspace_)w->synchronize();prepare_layer(index,a,b);}
  if(layer.a->original_k!=a.k_pages.data||layer.a->original_v!=a.v_pages.data||layer.b->original_k!=b.k_pages.data||layer.b->original_v!=b.v_pages.data)throw std::runtime_error("jointKV prepared source identity changed");
  workspace_[0]->execute(*layer.a,a,qa,pa,visible,prefix_end_,static_cast<float*>(oa.data),static_cast<float*>(om.data),static_cast<float*>(ol.data),eager);
  workspace_[1]->execute(*layer.b,b,qb,pb,visible,prefix_end_,static_cast<float*>(ba.data),static_cast<float*>(bm.data),static_cast<float*>(bl.data),eager);
  for(auto& w:workspace_)w->synchronize();++calls_;
  for(int r=0;r<2;++r)std::memcpy(workspace_[r]->host_meta.data(),workspace_[r]->pinned_meta,104*4);
  bool pass=true;
  for(int q=0;q<96;++q){int total=workspace_[0]->host_meta[q]+workspace_[1]->host_meta[q];if(total<2048||total>4096)pass=false;}
  for(int h=0;h<4;++h)if(workspace_[0]->host_meta[100+h]||workspace_[1]->host_meta[100+h]||workspace_[0]->host_meta[96+h]+workspace_[1]->host_meta[96+h]>UnionLimit)pass=false;
  ++layer_calls_[index];
  if(!pass){++fallbacks_;++layer_fallbacks_[index];}
  return pass;
 }
};
}
