#include "core/device.h"
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/launcher/gqa_attention.h"
#include "ninfer_bench_common.h"
#include "../../apps/dual/headscale_gqa.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
#include <stdexcept>
using namespace ninfer;
__device__ unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;}
__global__ void initialize(std::int8_t* k,std::int8_t* v,__half* ks,__half* vs,std::uint64_t elements,unsigned seed){
 std::uint64_t i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<elements){k[i]=static_cast<std::int8_t>(int(mix(unsigned(i)^seed)%255)-127);v[i]=static_cast<std::int8_t>(int(mix(unsigned(i)^seed^1777u)%255)-127);}
 if(i<elements/64){ks[i]=__float2half_rn(.001f+float(mix(unsigned(i)^seed^9987u)%4096)*.000005f);vs[i]=__float2half_rn(.001f+float(mix(unsigned(i)^seed^2345u)%4096)*.000005f);}
}
__global__ void headscale_key(const std::int8_t* codes,const __half* scales,
    std::int8_t* output,__half* outscales,std::uint64_t count){
    const int lane=threadIdx.x&31;
    const auto tokenhead=(static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    if(tokenhead>=count/256)return;
    const auto source_scale=scales+tokenhead*4;
    const float value=lane<4?__half2float(source_scale[lane]):0.f;
    float maximum=value;
    for(int off=16;off;off>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,off));
    const float inverse=maximum>0.f?1.f/maximum:0.f;
    if(lane<4)outscales[tokenhead*4+lane]=__float2half_rn(maximum);
#pragma unroll
    for(int j=0;j<8;++j){
        const auto index=tokenhead*256+j*32+lane;
        const float original_scale=__half2float(source_scale[j/2]);
        output[index]=static_cast<std::int8_t>(max(-127,min(127,
            __float2int_rn(static_cast<float>(codes[index])*original_scale*inverse))));
    }
}
int main(int argc,char**argv){try{
 const int window=argc>1?std::stoi(argv[1]):262144;const unsigned seed=argc>2?std::stoul(argv[2]):1234567u;
 if(window<2048||window>262144)throw std::invalid_argument("window");
 constexpr int dim=256,heads=24,kvheads=4,tokens=4;
 const int pages=(window+kPagedKVPageSize-1)/kPagedKVPageSize;
 const std::size_t elements=static_cast<std::size_t>(pages)*kPagedKVPageSize*kvheads*dim;
 cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 DeviceBuffer k(elements),v(elements),ks(elements/64*2),vs(elements/64*2),converted(elements),converted_scales(elements/64*2),query(dim*heads*tokens*2),positions(tokens*4),tables(pages*4),ref(dim*heads*tokens*2),out(ref.bytes),flush(256ULL<<20);
 initialize<<<(elements+255)/256,256,0,stream>>>((std::int8_t*)k.p,(std::int8_t*)v.p,(__half*)ks.p,(__half*)vs.p,elements,seed);
 headscale_key<<<(elements/256+7)/8,256,0,stream>>>((const std::int8_t*)k.p,(const __half*)ks.p,(std::int8_t*)converted.p,(__half*)converted_scales.p,elements);CUDA_CHECK(cudaGetLastError());
 std::vector<int>table(pages);for(int i=0;i<pages;++i)table[i]=pages-1-i;
 tables.copy_from_host(table.data(),tables.bytes);
 std::vector<int>pos{window-4,window-3,window-2,window-1};positions.copy_from_host(pos.data(),positions.bytes);
 std::vector<std::uint16_t>qh(dim*heads*tokens);
 for(std::size_t i=0;i<qh.size();++i){float x=(float((unsigned(i)*1664525u+seed)%1024)-511.f)*.003f;qh[i]=bench::f32_to_bf16(x);}
 query.copy_from_host(qh.data(),query.bytes);CUDA_CHECK(cudaStreamSynchronize(stream));
 const ops::GqaExecutionEnvelope envelope{static_cast<unsigned>(window),static_cast<unsigned>(window)};
 const int splits=ops::detail::gqa_attention_split_capacity(heads,tokens,DType::I8,envelope);
 DeviceBuffer part(dim*heads*tokens*splits*2ULL),m(heads*tokens*splits*4ULL),l(m.bytes);
 Tensor qt(query.p,DType::BF16,{dim,heads,tokens}),pt(positions.p,DType::I32,{tokens});
 Tensor pa(part.p,DType::BF16,{dim,heads,tokens,splits}),pm(m.p,DType::FP32,{heads,tokens,splits}),pl(l.p,DType::FP32,{heads,tokens,splits});
 PagedKVLayerView cache{.k_pages=Tensor(k.p,DType::I8,{dim,kPagedKVPageSize,kvheads,pages}),.v_pages=Tensor(v.p,DType::I8,{dim,kPagedKVPageSize,kvheads,pages}),.k_scale_pages=Tensor(ks.p,DType::FP16,{4,kPagedKVPageSize,kvheads,pages}),.v_scale_pages=Tensor(vs.p,DType::FP16,{4,kPagedKVPageSize,kvheads,pages}),.block_table=Tensor(tables.p,DType::I32,{pages}),.head_dim=dim,.num_kv_heads=kvheads,.dtype=DType::I8,.quant_group=64};
 auto invoke=[&](cudaStream_t s,bool faster,const void* values,const void* scales,void* dest){
  if(faster) {
   ops::headscale_qk::gqa_attention_headscale_qk_kernel<ops::Gqa27Geometry,4,8,2,16,false,false,false,ops::GqaCachedInput,true><<<dim3(kvheads,splits),256,0,s>>>((const __nv_bfloat16*)query.p,ops::GqaCachedInput{},(const int*)positions.p,(std::int8_t*)converted.p,(std::int8_t*)values,(__half*)converted_scales.p,(__half*)scales,(const int*)tables.p,nullptr,nullptr,pages,4,0,window,.0625f,0,splits,splits,1,0,(__nv_bfloat16*)part.p,(float*)m.p,(float*)l.p);
  } else {
   ops::headscale_qk::gqa_attention_headscale_qk_kernel<ops::Gqa27Geometry,4,8,2,16,false,false,false,ops::GqaCachedInput><<<dim3(kvheads,splits),256,0,s>>>((const __nv_bfloat16*)query.p,ops::GqaCachedInput{},(const int*)positions.p,(std::int8_t*)converted.p,(std::int8_t*)values,(__half*)converted_scales.p,(__half*)scales,(const int*)tables.p,nullptr,nullptr,pages,4,0,window,.0625f,0,splits,splits,1,0,(__nv_bfloat16*)part.p,(float*)m.p,(float*)l.p);
  }
  ops::gqa_attention_small_t_reduce_output_kernel<ops::Gqa27Geometry,64,true,false,false,false><<<dim3(heads,4,tokens),256,0,s>>>((const __nv_bfloat16*)part.p,(const float*)m.p,(const float*)l.p,(const int*)positions.p,nullptr,4,4,0,1,splits,(__nv_bfloat16*)dest);CUDA_CHECK(cudaGetLastError());
 };
 auto reference=[&](cudaStream_t s){invoke(s,false,v.p,vs.p,ref.p);};
 auto candidate=[&](cudaStream_t s){invoke(s,true,v.p,vs.p,out.p);};
 reference(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
 std::vector<std::uint8_t>p0(part.bytes),m0(m.bytes),l0(l.bytes);part.copy_to_host(p0.data(),p0.size());m.copy_to_host(m0.data(),m0.size());l.copy_to_host(l0.data(),l0.size());
 candidate(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
 std::vector<std::uint8_t>p1(part.bytes),m1(m.bytes),l1(l.bytes);part.copy_to_host(p1.data(),p1.size());m.copy_to_host(m1.data(),m1.size());l.copy_to_host(l1.data(),l1.size());
 std::vector<std::uint16_t>a(ref.bytes/2),b(out.bytes/2);ref.copy_to_host(a.data(),ref.bytes);out.copy_to_host(b.data(),out.bytes);
 auto diff=[](const auto&a,const auto&b){std::size_t n=0;for(std::size_t i=0;i<a.size();++i)n+=a[i]!=b[i];return n;};
 auto bfvalue=[](std::uint16_t v){union {unsigned u;float f;}x{static_cast<unsigned>(v)<<16};return x.f;};
 double error=0,norm=0,dot=0,ynorm=0;size_t nonfinite=0;
 for(size_t i=0;i<a.size();++i){double x=bfvalue(a[i]),y=bfvalue(b[i]);error+=(x-y)*(x-y);norm+=x*x;ynorm+=y*y;dot+=x*y;nonfinite+=!std::isfinite(y);}
 printf("NUMERICAL window=%d seed=%u output=%zu partial=%zu max=%zu sum=%zu rel_l2=%.9g cosine=%.9g nonfinite=%zu representation=headscaleQK_originalVcodes_packedFP16_noBF16round\n",window,seed,diff(a,b),diff(p0,p1),diff(m0,m1),diff(l0,l1),std::sqrt(error/std::max(norm,1e-30)),dot/std::sqrt(norm*ynorm),nonfinite);
 if(nonfinite)return 2;
 cudaGraph_t gr{},gc{};cudaGraphExec_t er{},ec{};
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));reference(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gr));CUDA_CHECK(cudaGraphInstantiate(&er,gr,nullptr,nullptr,0));
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));candidate(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gc));CUDA_CHECK(cudaGraphInstantiate(&ec,gc,nullptr,nullptr,0));
 for(int pass=0;pass<2;++pass){auto a=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(er,s));},flush,stream,3,12);auto b=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(ec,s));},flush,stream,3,12);printf("PERF window=%d seed=%u pass=%d baseline_us=%.3f prepared_us=%.3f ratio=%.6f\n",window,seed,pass,a.median_us,b.median_us,a.median_us/b.median_us);}
 CUDA_CHECK(cudaGraphExecDestroy(er));CUDA_CHECK(cudaGraphExecDestroy(ec));CUDA_CHECK(cudaGraphDestroy(gr));CUDA_CHECK(cudaGraphDestroy(gc));
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception&e){fprintf(stderr,"prepared_values: %s\n",e.what());return 1;}
}
