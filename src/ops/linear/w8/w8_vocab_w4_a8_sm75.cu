#include "ops/linear/w8/w8_target_candidate_union.cuh"
#include "ops/linear/w8/w8_vocab_w4_a8_sm75.h"
#include "ops/linear/w8/w8_vocab_b2_partial.h"
#include "core/device.h"
#include "ops/kernel/sampling_device.cuh"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include <cub/block/block_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <map>
#include <memory>
#include <mutex>
#include <string>
namespace ninfer::ops::detail { namespace {
using b1=nvcuda::wmma::experimental::precision::b1;
constexpr int kN=248320,kK=5120,kTokens=4,kGroups=40,kExactGroups=160;
constexpr int kRows=16,kWarps=8,kThreads=256,kCandidateCount=128,kMandatoryRows=16,kMandatoryMaxTokens=5;
constexpr int kProjectCtas=kN/(kRows*kWarps),kCtasPerPartial=8;
constexpr int kPartialBlocks=(kProjectCtas+kCtasPerPartial-1)/kCtasPerPartial,kGroupPartials=16,kGroupBlocks=(kPartialBlocks+kGroupPartials-1)/kGroupPartials;
using PartialGeometry=W8VocabB2PartialGeometry;
static_assert(kN==PartialGeometry::kVocabularyRows);
static_assert(kRows*kWarps==PartialGeometry::kRowsPerCta);
static_assert(kCandidateCount==PartialGeometry::kCandidateCount);
static_assert(kCtasPerPartial==PartialGeometry::kCtasPerPartial);
static_assert(kProjectCtas==PartialGeometry::kProjectCtas);
static_assert(kPartialBlocks==PartialGeometry::kPartialBlocks);
using PartialSort=cub::BlockRadixSort<unsigned long long,256,8>;
using FinalSort=cub::BlockRadixSort<unsigned long long,256,8>;
using ExactGeometry=W8LinearGeometry<kTokens*kCandidateCount,kK>;
using ExactGeometryT1=W8LinearGeometry<kCandidateCount,kK>;
using MandatoryGeometry=W8LinearGeometry<kMandatoryRows,kK>;
using ExactSchedule=W8SmallTMmaSchedule<8,8,2,W8SmallTMmaScaleAccess::Direct>;
constexpr std::size_t kPackedBytes=(std::size_t)kN*kK/8,kScaleBytes=(std::size_t)kN*kGroups*sizeof(__half);
struct State{const void*source=nullptr;uint2*packed=nullptr;std::uint32_t*scales=nullptr;uint2*bfrag=nullptr;__half*as=nullptr;unsigned long long*partial=nullptr,*group=nullptr;std::int32_t*ids=nullptr;std::int8_t*codes=nullptr;std::uint16_t*exact_scales=nullptr;__nv_bfloat16*exact=nullptr;std::int32_t*mandatory_ids=nullptr;__nv_bfloat16*mandatory_exact=nullptr;
    int device = -1;
    ~State() {
        if (device < 0) { return; }
        int previous = device;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(device);
        if (packed) { (void)cudaFree(packed); }
        if (scales) { (void)cudaFree(scales); }
        if (bfrag) { (void)cudaFree(bfrag); }
        if (as) { (void)cudaFree(as); }
        if (partial) { (void)cudaFree(partial); }
        if (group) { (void)cudaFree(group); }
        if (ids) { (void)cudaFree(ids); }
        if (codes) { (void)cudaFree(codes); }
        if (exact_scales) { (void)cudaFree(exact_scales); }
        if (exact) { (void)cudaFree(exact); }
        if (mandatory_ids) { (void)cudaFree(mandatory_ids); }
        if (mandatory_exact) { (void)cudaFree(mandatory_exact); }
        if (previous != device) { (void)cudaSetDevice(previous); }
    }
};

// One allocation set per source weight. Engines have independent source weights
// and execution streams; a process-global singleton corrupts both ownership and
// memory accounting when independently resident GPU replicas alternate.
struct Cache {
    std::mutex mutex;
    std::map<const void*, std::unique_ptr<State>> entries;
};
Cache& cache() { static Cache value; return value; }
State* lookup(const Weight& weight) noexcept {
    auto& values = cache();
    std::lock_guard lock(values.mutex);
    const auto entry = values.entries.find(weight.qdata);
    return entry == values.entries.end() ? nullptr : entry->second.get();
}
State& required_state(const Weight& weight) {
    State* prepared = lookup(weight);
    if (!prepared) { throw std::logic_error("GPU head weight was not prepared"); }
    return *prepared;
}
template<class T>
void allocate(T** pointer, std::size_t bytes) {
    const cudaError_t status = cudaMalloc(pointer, bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("GPU head cache allocation failed: ") +
                                 cudaGetErrorString(status));
    }
}
struct IndexedRows{const std::int32_t*ids;static constexpr int kOutputRowsPerCta=16;__device__ __forceinline__ int weight_row(int row0,int local)const{return ids[row0+local];}};
__device__ __forceinline__ void bmma(int&a,int&b,unsigned x,unsigned y){asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc {%0,%1}, {%2}, {%3}, {%0,%1};\n":"+r"(a),"+r"(b):"r"(x),"r"(y));}
__global__ void transform_b1(const std::int8_t*code,const std::uint16_t*src_sc,uint2*out,std::uint32_t*out_sc){
 __shared__ __align__(16) std::uint32_t bh[2][32],rh[2][32];
 __shared__ std::uint16_t rs[2][8];
 int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,gid=lane>>2,lid=lane&3;
 int rb=(int)blockIdx.x,row=rb*16+warp*8+gid;
 for(int g=0;g<kGroups;++g){
  const auto*p=code+(std::int64_t)row*kK+g*128+lid*32;
  float ss=__half2float(__ushort_as_half(src_sc[(std::int64_t)row*kExactGroups+g*4+lid]));
  float sum=0.f;
#pragma unroll
  for(int i=0;i<32;++i)sum+=fabsf((float)p[i]*ss);
  sum+=__shfl_xor_sync(0xffffffffu,sum,1,4);sum+=__shfl_xor_sync(0xffffffffu,sum,2,4);
  float alpha=sum*(1.f/128.f);std::uint32_t hi=0;
#pragma unroll
  for(int i=0;i<32;++i)hi|=(std::uint32_t)(((float)p[i]*ss)<0.f)<<i;
  bh[warp][lane]=hi;
  if(lid==0)rs[warp][gid]=__half_as_ushort(__float2half_rn(alpha));
  __syncwarp();
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,8,8,128,b1,nvcuda::wmma::row_major> ah;
  nvcuda::wmma::load_matrix_sync(ah,bh[warp],128);rh[warp][lane]=ah.x[0];
  __syncthreads();
  if(warp==0){
   out[((std::size_t)rb*kGroups+g)*32+lane]=make_uint2(rh[0][lane],rh[1][lane]);
   if(lane<8)out_sc[((std::size_t)rb*kGroups+g)*8+lane]=(std::uint32_t)rs[0][lane]|((std::uint32_t)rs[1][lane]<<16);
  }
  __syncthreads();
 }
}
__global__ void prepare_b2(const __nv_bfloat16*x,uint2*out,__half*sc){
 __shared__ __align__(16) std::uint32_t bh[8][32],bl[8][32];int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,g=(int)blockIdx.x*8+warp;if(g>=kGroups)return;bh[warp][lane]=0;bl[warp][lane]=0;
 if(lane<16){int t=lane>>2,w=lane&3;const auto*p=x+(std::int64_t)t*kK+g*128+w*32;float sum=0.f;
#pragma unroll
  for(int i=0;i<32;++i)sum+=fabsf(__bfloat162float(p[i]));sum+=__shfl_xor_sync(0x0000ffffu,sum,1,4);sum+=__shfl_xor_sync(0x0000ffffu,sum,2,4);float alpha=sum*(1.f/(128.f*1.5f));
#pragma unroll
  for(int it=0;it<2;++it){float num=0.f,den=0.f;
#pragma unroll
   for(int i=0;i<32;++i){float av=fabsf(__bfloat162float(p[i]));float m=av>2.f*alpha?3.f:1.f;num+=av*m;den+=m*m;}num+=__shfl_xor_sync(0x0000ffffu,num,1,4);num+=__shfl_xor_sync(0x0000ffffu,num,2,4);den+=__shfl_xor_sync(0x0000ffffu,den,1,4);den+=__shfl_xor_sync(0x0000ffffu,den,2,4);alpha=den>0.f?num/den:0.f;}
  std::uint32_t hi=0,lo=0;
#pragma unroll
  for(int i=0;i<32;++i){float v=__bfloat162float(p[i]),av=fabsf(v);bool neg=v<0.f,strong=av>2.f*alpha;hi|=(std::uint32_t)neg<<i;lo|=(std::uint32_t)(strong?neg:!neg)<<i;}bh[warp][t*4+w]=hi;bl[warp][t*4+w]=lo;if(w==0)sc[t*kGroups+g]=__float2half_rn(alpha);}
 __syncwarp();nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,8,8,128,b1,nvcuda::wmma::col_major>fh,fl;nvcuda::wmma::load_matrix_sync(fh,bh[warp],128);nvcuda::wmma::load_matrix_sync(fl,bl[warp],128);out[g*32+lane]=make_uint2(fh.x[0],fl.x[0]);
}
__global__ void project_b1_local_top2(const uint2*pw,const std::uint32_t*ws,const uint2*bf,const __half*as,unsigned long long*partial){
 __shared__ unsigned long long local[kWarps*kTokens*16];
 int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,cta=(int)blockIdx.x;
 int rb=cta*kWarps+warp,rowbase=rb*16,gid=lane>>2,lid=lane&3;
 float f0=0,f1=0,f2=0,f3=0;uint2 p=pw[((std::int64_t)rb*kGroups)*32+lane];
#pragma unroll 1
 for(int g=0;g<kGroups;++g){
  uint2 pn{};if(g+1<kGroups)pn=pw[((std::int64_t)rb*kGroups+g+1)*32+lane];
  uint2 bv=bf[g*32+lane];int q0=0,q1=0,q2=0,q3=0,a=0,b=0;
  bmma(a,b,p.x,bv.x);q0+=2*(128-2*a);q1+=2*(128-2*b);
  a=b=0;bmma(a,b,p.x,bv.y);q0+=128-2*a;q1+=128-2*b;
  a=b=0;bmma(a,b,p.y,bv.x);q2+=2*(128-2*a);q3+=2*(128-2*b);
  a=b=0;bmma(a,b,p.y,bv.y);q2+=128-2*a;q3+=128-2*b;
  int ta=2*lid,tb=ta+1;std::uint32_t sw=0;
  if(lid==0)sw=ws[((std::int64_t)rb*kGroups+g)*8+gid];
  sw=__shfl_sync(0xffffffffu,sw,gid*4);
  float w0=__half2float(__ushort_as_half((std::uint16_t)sw));
  float w1=__half2float(__ushort_as_half((std::uint16_t)(sw>>16)));
  float s0=ta<4?__half2float(as[ta*kGroups+g]):0.f;
  float s1=tb<4?__half2float(as[tb*kGroups+g]):0.f;
  f0=__fmaf_rn((float)q0,w0*s0,f0);f1=__fmaf_rn((float)q1,w0*s1,f1);
  f2=__fmaf_rn((float)q2,w1*s0,f2);f3=__fmaf_rn((float)q3,w1*s1,f3);p=pn;
 }
 int ta=2*lid,tb=ta+1,r0=rowbase+gid,r1=r0+8;
 if(ta<4){int o=(warp*kTokens+ta)*16+gid;local[o]=sampling_sort_key(__bfloat162float(__float2bfloat16_rn(f0)),r0);local[o+8]=sampling_sort_key(__bfloat162float(__float2bfloat16_rn(f2)),r1);}
 if(tb<4){int o=(warp*kTokens+tb)*16+gid;local[o]=sampling_sort_key(__bfloat162float(__float2bfloat16_rn(f1)),r0);local[o+8]=sampling_sort_key(__bfloat162float(__float2bfloat16_rn(f3)),r1);}
 __syncthreads();
 if(lane<kTokens){int o=(warp*kTokens+lane)*16;unsigned long long best0=0,best1=0;
#pragma unroll
  for(int i=0;i<16;++i){unsigned long long v=local[o+i];if(v>best0){best1=best0;best0=v;}else if(v>best1)best1=v;}
  int part=cta/kCtasPerPartial,slot=cta%kCtasPerPartial;
  int dst=((lane*kPartialBlocks+part)*kCandidateCount)+slot*(2*kWarps)+2*warp;
  partial[dst]=best0;partial[dst+1]=best1;
 }
}
__global__ void prepare_b2_t1(const __nv_bfloat16*x,uint2*out,__half*sc){
 __shared__ __align__(16) std::uint32_t bh[8][32],bl[8][32];int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,g=(int)blockIdx.x*8+warp;if(g>=kGroups)return;bh[warp][lane]=0;bl[warp][lane]=0;
 if(lane<4){int w=lane;const auto*q=x+g*128+w*32;float sum=0.f;
#pragma unroll
  for(int i=0;i<32;++i)sum+=fabsf(__bfloat162float(q[i]));sum+=__shfl_xor_sync(0x0fu,sum,1,4);sum+=__shfl_xor_sync(0x0fu,sum,2,4);float alpha=sum*(1.f/(128.f*1.5f));
#pragma unroll
  for(int it=0;it<2;++it){float num=0.f,den=0.f;
#pragma unroll
   for(int i=0;i<32;++i){float av=fabsf(__bfloat162float(q[i]));float m=av>2.f*alpha?3.f:1.f;num+=av*m;den+=m*m;}num+=__shfl_xor_sync(0x0fu,num,1,4);num+=__shfl_xor_sync(0x0fu,num,2,4);den+=__shfl_xor_sync(0x0fu,den,1,4);den+=__shfl_xor_sync(0x0fu,den,2,4);alpha=den>0.f?num/den:0.f;}
  std::uint32_t hi=0,lo=0;
#pragma unroll
  for(int i=0;i<32;++i){float v=__bfloat162float(q[i]),av=fabsf(v);bool neg=v<0.f,strong=av>2.f*alpha;hi|=(std::uint32_t)neg<<i;lo|=(std::uint32_t)(strong?neg:!neg)<<i;}bh[warp][w]=hi;bl[warp][w]=lo;if(w==0)sc[g]=__float2half_rn(alpha);}
 __syncwarp();nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,8,8,128,b1,nvcuda::wmma::col_major>fh,fl;nvcuda::wmma::load_matrix_sync(fh,bh[warp],128);nvcuda::wmma::load_matrix_sync(fl,bl[warp],128);out[g*32+lane]=make_uint2(fh.x[0],fl.x[0]);
}
__global__ void project_b1_local_top2_t1(const uint2*pw,const std::uint32_t*ws,const uint2*bf,const __half*as,unsigned long long*partial){
 __shared__ unsigned long long local[kWarps*16];int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,cta=(int)blockIdx.x,rb=cta*kWarps+warp,rowbase=rb*16,gid=lane>>2,lid=lane&3;float top=0.f,bottom=0.f;uint2 packed=pw[((std::int64_t)rb*kGroups)*32+lane];
#pragma unroll 1
 for(int g=0;g<kGroups;++g){uint2 next{};if(g+1<kGroups)next=pw[((std::int64_t)rb*kGroups+g+1)*32+lane];uint2 act=bf[g*32+lane];int tq=0,bq=0,a=0,ign=0;bmma(a,ign,packed.x,act.x);tq+=2*(128-2*a);a=ign=0;bmma(a,ign,packed.x,act.y);tq+=128-2*a;a=ign=0;bmma(a,ign,packed.y,act.x);bq+=2*(128-2*a);a=ign=0;bmma(a,ign,packed.y,act.y);bq+=128-2*a;std::uint32_t sw=0;if(lid==0)sw=ws[((std::int64_t)rb*kGroups+g)*8+gid];sw=__shfl_sync(0xffffffffu,sw,gid*4);float w0=__half2float(__ushort_as_half((std::uint16_t)sw)),w1=__half2float(__ushort_as_half((std::uint16_t)(sw>>16))),sa=__half2float(as[g]);top=__fmaf_rn((float)tq,w0*sa,top);bottom=__fmaf_rn((float)bq,w1*sa,bottom);packed=next;}
 if(lid==0){int o=warp*16+gid;local[o]=sampling_sort_key(__bfloat162float(__float2bfloat16_rn(top)),rowbase+gid);local[o+8]=sampling_sort_key(__bfloat162float(__float2bfloat16_rn(bottom)),rowbase+gid+8);}__syncthreads();if(lane==0){int o=warp*16;unsigned long long b0=0,b1v=0;
#pragma unroll
  for(int i=0;i<16;++i){auto v=local[o+i];if(v>b0){b1v=b0;b0=v;}else if(v>b1v)b1v=v;}int part=cta/kCtasPerPartial,slot=cta%kCtasPerPartial,dst=part*kCandidateCount+slot*(2*kWarps)+2*warp;partial[dst]=b0;partial[dst+1]=b1v;}
}
// The final partial contains fewer than kCtasPerPartial producer CTAs. Mask its unwritten tail
// while reading instead of relying on a separate memset node before every CUDA-Graph replay.
__global__ void topk_group(const unsigned long long*in,unsigned long long*out){int col=(int)blockIdx.y,g=(int)blockIdx.x,tid=(int)threadIdx.x,first=g*kGroupPartials,count=min(kGroupPartials,kPartialBlocks-first);unsigned long long keys[8];
#pragma unroll
 for(int i=0;i<8;++i){int q=tid*8+i,p=q/kCandidateCount,r=q-p*kCandidateCount,absolute=first+p;bool valid=p<count&&PartialGeometry::rank_written(absolute,r);keys[i]=valid?in[(col*kPartialBlocks+absolute)*kCandidateCount+r]:0ull;}__shared__ typename PartialSort::TempStorage st;PartialSort(st).SortDescending(keys);
#pragma unroll
 for(int i=0;i<8;++i){int rank=tid*8+i;if(rank<kCandidateCount)out[(col*kGroupBlocks+g)*kCandidateCount+rank]=keys[i];}
}
__global__ void topk_final(const unsigned long long*in,std::int32_t*ids){int col=(int)blockIdx.x,tid=(int)threadIdx.x;unsigned long long keys[8];
#pragma unroll
 for(int i=0;i<8;++i){int q=tid*8+i;keys[i]=q<kGroupBlocks*kCandidateCount?in[col*kGroupBlocks*kCandidateCount+q]:0ull;}__shared__ typename FinalSort::TempStorage st;FinalSort(st).SortDescending(keys);
#pragma unroll
 for(int i=0;i<8;++i){int rank=tid*8+i;if(rank<kCandidateCount)ids[col*kCandidateCount+rank]=sampling_key_index(keys[i]);}}
__global__ void gather(const std::int32_t*ids,const std::int8_t*code,const std::uint16_t*sc,std::int8_t*out,std::uint16_t*out_sc){int cr=(int)blockIdx.x,sr=ids[cr];for(int d=(int)threadIdx.x;d<kK;d+=blockDim.x)out[(std::int64_t)cr*kK+d]=code[(std::int64_t)sr*kK+d];for(int g=(int)threadIdx.x;g<kExactGroups;g+=blockDim.x)out_sc[(std::int64_t)cr*kExactGroups+g]=sc[(std::int64_t)sr*kExactGroups+g];}
__global__ void prepare_mandatory_ids(const std::int32_t*drafts,std::int32_t*ids,int active_cols){
 int row=(int)threadIdx.x;if(row<kMandatoryRows)ids[row]=drafts[min(row,active_cols-2)];
}
__global__ void finish_mandatory_scores(const __nv_bfloat16*exact,__nv_bfloat16*metadata,int physical_rows,int active_cols){
 int col=(int)threadIdx.x;if(col<active_cols-1)metadata[(std::int64_t)col*physical_rows+512]=exact[(std::int64_t)col*kMandatoryRows+col];
}
__global__ void finish(const __nv_bfloat16*exact,const std::int32_t*ids,std::int32_t*tokens,__nv_bfloat16*metadata,int physical_rows){
 int col=(int)blockIdx.x,tid=(int)threadIdx.x;__shared__ unsigned long long keys[kCandidateCount];
 if(tid<kCandidateCount){
  int token=ids[col*kCandidateCount+tid];
  __nv_bfloat16 score=exact[(std::int64_t)col*(kTokens*kCandidateCount)+col*kCandidateCount+tid];
  keys[tid]=sampling_sort_key(__bfloat162float(score),token);
  auto* meta_ids=reinterpret_cast<std::int32_t*>(metadata+(std::int64_t)col*physical_rows);
  meta_ids[tid]=token;
  metadata[(std::int64_t)col*physical_rows+256+tid]=score;
 }
 __syncthreads();
 if(tid==0){unsigned long long best=keys[0];for(int i=1;i<kCandidateCount;++i)if(keys[i]>best)best=keys[i];tokens[col]=sampling_key_index(best);}
}
__global__ void finish_t1(const __nv_bfloat16*exact,const std::int32_t*ids,std::int32_t*token,__nv_bfloat16*metadata){
 int tid=(int)threadIdx.x;__shared__ unsigned long long keys[kCandidateCount];
 if(tid<kCandidateCount){int candidate=ids[tid];__nv_bfloat16 score=exact[tid];keys[tid]=sampling_sort_key(__bfloat162float(score),candidate);auto*meta_ids=reinterpret_cast<std::int32_t*>(metadata);meta_ids[tid]=candidate;metadata[256+tid]=score;}
 __syncthreads();if(tid==0){unsigned long long best=keys[0];for(int i=1;i<kCandidateCount;++i)if(keys[i]>best)best=keys[i];token[0]=sampling_key_index(best);}
}
}
std::size_t w8_vocab_w4_a8_reservation_bytes() noexcept{return kPackedBytes+kScaleBytes+(std::size_t)kGroups*32*sizeof(uint2)+(std::size_t)kTokens*kGroups*2+(std::size_t)kTokens*kPartialBlocks*kCandidateCount*8+(std::size_t)kTokens*kGroupBlocks*kCandidateCount*8+(std::size_t)kTokens*kCandidateCount*4+(std::size_t)kTokens*kCandidateCount*kK+(std::size_t)kTokens*kCandidateCount*kExactGroups*2+(std::size_t)kTokens*kCandidateCount*kTokens*2+(std::size_t)kMandatoryRows*sizeof(std::int32_t)+(std::size_t)kMandatoryRows*kMandatoryMaxTokens*sizeof(__nv_bfloat16);}
void w8_vocab_w4_a8_prepare(const Weight&w,cudaStream_t stream){if(w.qtype!=QType::W8G32_F16S||w.n!=kN||w.k!=kK||!w.qdata||!w.scales)throw std::invalid_argument("SM75 B2 vocabulary preparation requires exact W8 weight");auto&values=cache();std::lock_guard lock(values.mutex);if(values.entries.contains(w.qdata))return;auto prepared=std::make_unique<State>();auto&s=*prepared;CUDA_CHECK(cudaGetDevice(&s.device));allocate(&s.packed,kPackedBytes);allocate(&s.scales,kScaleBytes);allocate(&s.bfrag,(std::size_t)kGroups*32*sizeof(uint2));allocate(&s.as,(std::size_t)kTokens*kGroups*2);allocate(&s.partial,(std::size_t)kTokens*kPartialBlocks*kCandidateCount*8);allocate(&s.group,(std::size_t)kTokens*kGroupBlocks*kCandidateCount*8);allocate(&s.ids,(std::size_t)kTokens*kCandidateCount*4);allocate(&s.codes,(std::size_t)kTokens*kCandidateCount*kK);allocate(&s.exact_scales,(std::size_t)kTokens*kCandidateCount*kExactGroups*2);allocate(&s.exact,(std::size_t)kTokens*kCandidateCount*kTokens*2);allocate(&s.mandatory_ids,(std::size_t)kMandatoryRows*sizeof(std::int32_t));allocate(&s.mandatory_exact,(std::size_t)kMandatoryRows*kMandatoryMaxTokens*sizeof(__nv_bfloat16));transform_b1<<<kN/kRows,64,0,stream>>>((const std::int8_t*)w.qdata,(const std::uint16_t*)w.scales,s.packed,s.scales);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));s.source=w.qdata;values.entries.emplace(w.qdata,std::move(prepared));}
void w8_vocab_w4_a8_release(const Weight& weight) noexcept {
    auto& values = cache();
    std::lock_guard lock(values.mutex);
    values.entries.erase(weight.qdata);
}
bool w8_vocab_w4_a8_ready(const Weight&w) noexcept {
    const State* prepared = lookup(w);
    return prepared != nullptr && prepared->packed != nullptr;
}
void w8_vocab_w4_a8_launch(const Tensor&x,const Weight&w,Tensor&out,Tensor&tokens,cudaStream_t stream){auto&s=required_state(w);if(!w8_vocab_w4_a8_ready(w)||x.ne[1]!=4||out.ne[0]!=kN||tokens.ne[0]!=4)throw std::invalid_argument("SM75 B2 hybrid vocabulary requires prepared T4 geometry");prepare_b2<<<(kGroups+7)/8,kThreads,0,stream>>>((const __nv_bfloat16*)x.data,s.bfrag,s.as);project_b1_local_top2<<<kProjectCtas,kThreads,0,stream>>>(s.packed,s.scales,s.bfrag,s.as,s.partial);topk_group<<<dim3(kGroupBlocks,kTokens,1),256,0,stream>>>(s.partial,s.group);topk_final<<<kTokens,256,0,stream>>>(s.group,s.ids);const W8ContiguousOutput eo{s.exact,kTokens*kCandidateCount};w8_small_t_mma_kernel<ExactGeometry,kTokens,ExactSchedule,W8ContiguousOutput,W8SmallTMmaStoreEpilogue,IndexedRows><<<(kTokens*kCandidateCount)/ExactSchedule::kRowsPerCta,ExactSchedule::kThreads,0,stream>>>((const __nv_bfloat16*)x.data,(const std::uint8_t*)w.qdata,(const std::uint8_t*)w.scales,eo,W8SmallTMmaStoreEpilogue{},IndexedRows{s.ids});finish<<<kTokens,kCandidateCount,0,stream>>>(s.exact,s.ids,(std::int32_t*)tokens.data,static_cast<__nv_bfloat16*>(out.data),out.ne[0]);CUDA_CHECK(cudaGetLastError());}
void w8_vocab_w4_a8_launch_t1(const Tensor&x,const Weight&w,Tensor&out,Tensor&token,cudaStream_t stream){auto&s=required_state(w);if(!w8_vocab_w4_a8_ready(w)||x.dtype!=DType::BF16||x.ne[0]!=kK||x.ne[1]!=1||out.dtype!=DType::BF16||out.ne[0]!=kN||out.ne[1]!=1||token.dtype!=DType::I32||token.ne[0]!=1)throw std::invalid_argument("SM75 B2 hybrid vocabulary requires prepared T1 geometry");prepare_b2_t1<<<(kGroups+7)/8,kThreads,0,stream>>>((const __nv_bfloat16*)x.data,s.bfrag,s.as);project_b1_local_top2_t1<<<kProjectCtas,kThreads,0,stream>>>(s.packed,s.scales,s.bfrag,s.as,s.partial);topk_group<<<dim3(kGroupBlocks,1,1),256,0,stream>>>(s.partial,s.group);topk_final<<<1,256,0,stream>>>(s.group,s.ids);const W8ContiguousOutput eo{s.exact,kCandidateCount};w8_small_t_mma_kernel<ExactGeometryT1,1,ExactSchedule,W8ContiguousOutput,W8SmallTMmaStoreEpilogue,IndexedRows><<<kCandidateCount/ExactSchedule::kRowsPerCta,ExactSchedule::kThreads,0,stream>>>((const __nv_bfloat16*)x.data,(const std::uint8_t*)w.qdata,(const std::uint8_t*)w.scales,eo,W8SmallTMmaStoreEpilogue{},IndexedRows{s.ids});finish_t1<<<1,kCandidateCount,0,stream>>>(s.exact,s.ids,(std::int32_t*)token.data,static_cast<__nv_bfloat16*>(out.data));CUDA_CHECK(cudaGetLastError());}
void w8_vocab_w4_a8_include_drafts(const Tensor& x,const Weight& w,const Tensor& drafts,
                                   const Tensor& extents,Tensor& metadata,Tensor& tokens,cudaStream_t stream) {
    auto& s=required_state(w);
    if(x.dtype!=DType::BF16 || x.ne[0]!=kK || x.ne[1]!=4 || !x.is_contiguous() ||
       drafts.dtype!=DType::I32 || drafts.ne[0]!=3 || !drafts.is_contiguous() ||
       extents.dtype!=DType::I32 || extents.ne[0]!=1 ||
       metadata.dtype!=DType::BF16 || metadata.ne[0]!=kN || metadata.ne[1]!=4 ||
       !metadata.is_contiguous() || tokens.dtype!=DType::I32 || tokens.ne[0]!=4)
        throw std::invalid_argument("target draft union requires native contiguous C1 D3 head tensors");
    target_union::prepare<<<1,32,0,stream>>>(static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(extents.data),s.mandatory_ids,kN);
    const W8ContiguousOutput output{s.mandatory_exact,kMandatoryRows};
    w8_small_t_mma_kernel<MandatoryGeometry,4,ExactSchedule,W8ContiguousOutput,
        W8SmallTMmaStoreEpilogue,IndexedRows><<<1,ExactSchedule::kThreads,0,stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales),output,W8SmallTMmaStoreEpilogue{},IndexedRows{s.mandatory_ids});
    target_union::merge<<<1,32,0,stream>>>(s.mandatory_exact,static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(extents.data),static_cast<__nv_bfloat16*>(metadata.data),
        static_cast<std::int32_t*>(tokens.data),kN);
    CUDA_CHECK(cudaGetLastError());
}

void w8_vocab_w4_a8_score_mandatory_drafts(const Tensor&x,const Weight&w,const Tensor&drafts,Tensor&metadata,cudaStream_t stream){
 auto&s=required_state(w);
 const int active_cols=static_cast<int>(x.ne[1]);
 if(!w8_vocab_w4_a8_ready(w)||x.dtype!=DType::BF16||x.ne[0]!=kK||(active_cols!=4&&active_cols!=5)||drafts.dtype!=DType::I32||drafts.ne[0]!=active_cols-1||metadata.dtype!=DType::BF16||metadata.ne[0]!=kN||metadata.ne[1]!=active_cols)throw std::invalid_argument("SM75 mandatory draft rerank requires T4/T5 hidden, D3/D4 drafts, and vocabulary metadata");
 prepare_mandatory_ids<<<1,kMandatoryRows,0,stream>>>((const std::int32_t*)drafts.data,s.mandatory_ids,active_cols);
 const W8ContiguousOutput eo{s.mandatory_exact,kMandatoryRows};
 if(active_cols==4){
  w8_small_t_mma_kernel<MandatoryGeometry,4,ExactSchedule,W8ContiguousOutput,W8SmallTMmaStoreEpilogue,IndexedRows><<<1,ExactSchedule::kThreads,0,stream>>>((const __nv_bfloat16*)x.data,(const std::uint8_t*)w.qdata,(const std::uint8_t*)w.scales,eo,W8SmallTMmaStoreEpilogue{},IndexedRows{s.mandatory_ids});
 }else{
  w8_small_t_mma_kernel<MandatoryGeometry,5,ExactSchedule,W8ContiguousOutput,W8SmallTMmaStoreEpilogue,IndexedRows><<<1,ExactSchedule::kThreads,0,stream>>>((const __nv_bfloat16*)x.data,(const std::uint8_t*)w.qdata,(const std::uint8_t*)w.scales,eo,W8SmallTMmaStoreEpilogue{},IndexedRows{s.mandatory_ids});
 }
 finish_mandatory_scores<<<1,32,0,stream>>>(s.mandatory_exact,static_cast<__nv_bfloat16*>(metadata.data),metadata.ne[0],active_cols);
 CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
