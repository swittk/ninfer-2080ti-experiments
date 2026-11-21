#include "core/device.h"
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/launcher/gqa_attention.h"
#include "ninfer_bench_common.h"
#include "../../apps/dual/headscale_gqa.cuh"
#include "../../.chatgpt/production_serve_20260917/centered_value_integer_pv/centered_pv.cuh"
#include "../../.chatgpt/production_serve_20260917/centered_value_integer_pv/query_reuse/centered_query_reuse.cuh"
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
// Each lane encodes one entire channel; adjacent lanes read adjacent values.
__global__ void prepare_centered(const std::int8_t* src,const __half* scales,
    unsigned char* dst,__half2* metadata,std::size_t count){
    const std::size_t unit=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(unit>=count/32)return;
    const int d=unit%256;const auto block=unit/256,pagehead=block/2;const int halfblock=block%2;
    float values[32],sum=0;
#pragma unroll
    for(int key=0;key<32;++key){const auto token=pagehead*64+halfblock*32+key;
        values[key]=__half2float(__float2half_rn(float(src[token*256+d])*__half2float(scales[token*4+d/64])));
        sum+=values[key];
    }
    const __half mean=__float2half_rn(sum*(1.f/32));const float mu=__half2float(mean);float maximum=0;
#pragma unroll
    for(int key=0;key<32;++key)maximum=fmaxf(maximum,fabsf(values[key]-mu));
    const __half hs=__float2half_rn(maximum/7.f);const float sc=__half2float(hs),inv=sc>0?1.f/sc:0;
    metadata[unit]=__halves2half2(hs,mean);
#pragma unroll
    for(int pair=0;pair<16;++pair){
        const int a=max(-7,min(7,__float2int_rn((values[pair*2]-mu)*inv)));
        const int b=max(-7,min(7,__float2int_rn((values[pair*2+1]-mu)*inv)));
        dst[unit*16+(pair^((d&1)*8))]=static_cast<unsigned char>((a&15)|((b&15)<<4));
    }
}
__global__ void s4_permute(const unsigned char* src,unsigned char* dst,std::size_t bytes){
 const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=bytes)return;
 const int dim=(i/16)%256;dst[i]=src[i^((dim&1)*8)];
}
float bf_value(std::uint16_t bits){union{unsigned u;float f;}x{unsigned(bits)<<16};return x.f;}
int nibble(const std::vector<unsigned char>& v,std::size_t block,int d,int key){
    unsigned b=v[(block*256+d)*16+((key/2)^((d&1)*8))];unsigned n=(b>>(4*(key&1)))&15;
    return int(n^8)-8;
}
// Scalar CPU oracle knows neither the native MMA mapping nor the GPU PV reducer.
void scalar_check(int window,int pages,const std::vector<int>& table,const std::vector<int>& positions,
    const std::vector<std::uint16_t>& qbits,DeviceBuffer& k,DeviceBuffer& ks,
    DeviceBuffer& codes,DeviceBuffer& meta,DeviceBuffer& output){
    std::vector<std::int8_t> key(k.bytes);std::vector<std::uint16_t> scales(ks.bytes/2),params(meta.bytes/2),got(output.bytes/2);
    std::vector<unsigned char> cv(codes.bytes);k.copy_to_host(key.data(),key.size());ks.copy_to_host(scales.data(),ks.bytes);
    codes.copy_to_host(cv.data(),cv.size());meta.copy_to_host(params.data(),meta.bytes);output.copy_to_host(got.data(),output.bytes);
    double maxrel=0,mincos=1;
    for(int token=0;token<4;++token)for(int head=0;head<24;++head){
        const int row=token*24+head,kh=head/6;std::vector<float> q(256),logits(window),y(256,0);float am=0;
        for(int d=0;d<256;++d){q[d]=bf_value(qbits[row*256+d]);am=std::max(am,std::fabs(q[d]));}
        const float qs=am/127;std::vector<int> qi(256);for(int d=0;d<256;++d)qi[d]=qs>0?int(std::nearbyint(q[d]/qs)):0;
        float m=-INFINITY;
        for(int pos=0;pos<window;++pos){
            if(pos>positions[token]){logits[pos]=-INFINITY;continue;}
            const auto ph=table[pos/64]*4+kh;const auto ix=(static_cast<std::size_t>(ph)*64+(pos%64))*256;int dot=0;
            for(int d=0;d<256;++d)dot+=qi[d]*int(key[ix+d]);
            const float sc=__half2float(__ushort_as_half(scales[ix/64]));logits[pos]=float(dot)*qs*sc/16;m=std::max(m,logits[pos]);
        }
        double denominator=0;std::vector<float> p(window);for(int pos=0;pos<window;++pos){p[pos]=std::exp(logits[pos]-m);denominator+=p[pos];}
        for(int start=0;start<window;start+=32){const int end=std::min(window,start+32);float maximum=0,sum=0;
            for(int pos=start;pos<end;++pos){maximum=std::max(maximum,p[pos]);sum+=p[pos];}
            const float ps=maximum/127,inv=ps>0?1.f/ps:0;int pi[32]{};for(int pos=start;pos<end;++pos)pi[pos-start]=int(std::nearbyint(p[pos]*inv));
            const auto block=(static_cast<std::size_t>(table[start/64])*4+kh)*2+(start%64)/32;
            for(int d=0;d<256;++d){int sumint=0;for(int j=0;j<32;++j)sumint+=pi[j]*nibble(cv,block,d,j);
                const auto mi=(block*256+d)*2;const float sc=__half2float(__ushort_as_half(params[mi])),mu=__half2float(__ushort_as_half(params[mi+1]));
                y[d]+=float(sumint)*ps*sc+sum*mu;
            }
        }
        double err=0,norm=0,other=0,cross=0;for(int d=0;d<256;++d){double expected=y[d]/denominator,actual=bf_value(got[row*256+d]);err+=(expected-actual)*(expected-actual);norm+=expected*expected;other+=actual*actual;cross+=expected*actual;}
        maxrel=std::max(maxrel,std::sqrt(err/std::max(norm,1e-30)));mincos=std::min(mincos,cross/std::sqrt(norm*other));
    }
    printf("SCALAR_REFERENCE query_rows=96 max_relative_l2=%.9g min_cosine=%.9g includes_mean_and_integer_residual\n",maxrel,mincos);
    if(maxrel>.025 || mincos<.999)throw std::runtime_error("independent scalar PV output mismatch");
}
int main(int argc,char**argv){try{
 const int window=argc>1?std::stoi(argv[1]):262144;const unsigned seed=argc>2?std::stoul(argv[2]):1234567u;
 if(window<128||window>262144)throw std::invalid_argument("window");
 constexpr int dim=256,heads=24,kvheads=4,tokens=4;
 const int pages=(window+kPagedKVPageSize-1)/kPagedKVPageSize;
 const std::size_t elements=static_cast<std::size_t>(pages)*kPagedKVPageSize*kvheads*dim;
 cudaStream_t stream{};CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));{
 DeviceBuffer query_i8(96*256),query_scales(96*4),s4v(elements/2),packedv(elements/2),packedvs(elements/32*4),k(elements),v(elements),ks(elements/64*2),vs(elements/64*2),converted(elements),converted_scales(elements/64*2),query(dim*heads*tokens*2),positions(tokens*4),tables(pages*4),ref(dim*heads*tokens*2),out(ref.bytes),flush(256ULL<<20);
 initialize<<<(elements+255)/256,256,0,stream>>>((std::int8_t*)k.p,(std::int8_t*)v.p,(__half*)ks.p,(__half*)vs.p,elements,seed);
 headscale_key<<<(elements/256+7)/8,256,0,stream>>>((const std::int8_t*)k.p,(const __half*)ks.p,(std::int8_t*)converted.p,(__half*)converted_scales.p,elements);CUDA_CHECK(cudaGetLastError());
 std::vector<int>table(pages);for(int i=0;i<pages;++i)table[i]=pages-1-i;
 tables.copy_from_host(table.data(),tables.bytes);
 std::vector<int>pos{window-4,window-3,window-2,window-1};positions.copy_from_host(pos.data(),positions.bytes);
 std::vector<std::uint16_t>qh(dim*heads*tokens);
 for(std::size_t i=0;i<qh.size();++i){float x=(float((unsigned(i)*1664525u+seed)%1024)-511.f)*.003f;qh[i]=bench::f32_to_bf16(x);}
 query.copy_from_host(qh.data(),query.bytes);CUDA_CHECK(cudaStreamSynchronize(stream));
 CUDA_CHECK(cudaDeviceSynchronize());
 cudaEvent_t prep_a{},prep_b{};CUDA_CHECK(cudaEventCreate(&prep_a));CUDA_CHECK(cudaEventCreate(&prep_b));
 CUDA_CHECK(cudaEventRecord(prep_a,stream));
 prepare_centered<<<(elements/32+255)/256,256,0,stream>>>((const std::int8_t*)v.p,(const __half*)vs.p,(unsigned char*)packedv.p,(__half2*)packedvs.p,elements);
 CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaEventRecord(prep_b,stream));CUDA_CHECK(cudaEventSynchronize(prep_b));float prep_ms=0;CUDA_CHECK(cudaEventElapsedTime(&prep_ms,prep_a,prep_b));
 printf("PREPARATION once_ms=%.6f additional_paired_cache_bytes=%zu not_free_startup_or_model_memory\n",prep_ms,packedv.bytes+packedvs.bytes);
 CUDA_CHECK(cudaEventDestroy(prep_a));CUDA_CHECK(cudaEventDestroy(prep_b));
 s4_permute<<<(packedv.bytes+255)/256,256,0,stream>>>((const unsigned char*)packedv.p,(unsigned char*)s4v.p,packedv.bytes);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));

 const ops::GqaExecutionEnvelope envelope{static_cast<unsigned>(window),static_cast<unsigned>(window)};
 const int splits=ops::detail::gqa_attention_split_capacity(heads,tokens,DType::I8,envelope);
 DeviceBuffer part(dim*heads*tokens*splits*2ULL),m(heads*tokens*splits*4ULL),l(m.bytes);
 Tensor qt(query.p,DType::BF16,{dim,heads,tokens}),pt(positions.p,DType::I32,{tokens});
 Tensor pa(part.p,DType::BF16,{dim,heads,tokens,splits}),pm(m.p,DType::FP32,{heads,tokens,splits}),pl(l.p,DType::FP32,{heads,tokens,splits});
 PagedKVLayerView cache{.k_pages=Tensor(k.p,DType::I8,{dim,kPagedKVPageSize,kvheads,pages}),.v_pages=Tensor(v.p,DType::I8,{dim,kPagedKVPageSize,kvheads,pages}),.k_scale_pages=Tensor(ks.p,DType::FP16,{4,kPagedKVPageSize,kvheads,pages}),.v_scale_pages=Tensor(vs.p,DType::FP16,{4,kPagedKVPageSize,kvheads,pages}),.block_table=Tensor(tables.p,DType::I32,{pages}),.head_dim=dim,.num_kv_heads=kvheads,.dtype=DType::I8,.quant_group=64};
 bool s4_mode=false;
 auto invoke=[&](cudaStream_t s,bool faster,const void* values,const void* scales,void* dest){
  if(faster) {
   if(s4_mode){
    ops::centered_query_reuse::prepare_queries<<<12,256,0,s>>>((const __nv_bfloat16*)query.p,(std::int8_t*)query_i8.p,(float*)query_scales.p);
   ops::centered_query_reuse::gqa_attention_integer_pv_kernel<ops::Gqa27Geometry,4,8,3,32,false,false,false,ops::GqaCachedInput><<<dim3(kvheads,splits),256,0,s>>>((const std::int8_t*)query_i8.p,(const float*)query_scales.p,ops::GqaCachedInput{},(const int*)positions.p,(std::int8_t*)converted.p,(std::int8_t*)s4v.p,(__half*)converted_scales.p,(__half*)packedvs.p,(const int*)tables.p,nullptr,nullptr,pages,4,0,window,.0625f,0,splits,splits,1,0,(__nv_bfloat16*)part.p,(float*)m.p,(float*)l.p);
   }else{
   ops::centered_pv::gqa_attention_integer_pv_kernel<ops::Gqa27Geometry,4,8,2,32,false,false,false,ops::GqaCachedInput><<<dim3(kvheads,splits),256,0,s>>>((const __nv_bfloat16*)query.p,ops::GqaCachedInput{},(const int*)positions.p,(std::int8_t*)converted.p,(std::int8_t*)packedv.p,(__half*)converted_scales.p,(__half*)packedvs.p,(const int*)tables.p,nullptr,nullptr,pages,4,0,window,.0625f,0,splits,splits,1,0,(__nv_bfloat16*)part.p,(float*)m.p,(float*)l.p);
   }
  } else {
   ops::headscale_qk::gqa_attention_headscale_qk_kernel<ops::Gqa27Geometry,4,8,2,16,false,false,false,ops::GqaCachedInput,true><<<dim3(kvheads,splits),256,0,s>>>((const __nv_bfloat16*)query.p,ops::GqaCachedInput{},(const int*)positions.p,(std::int8_t*)converted.p,(std::int8_t*)values,(__half*)converted_scales.p,(__half*)scales,(const int*)tables.p,nullptr,nullptr,pages,4,0,window,.0625f,0,splits,splits,1,0,(__nv_bfloat16*)part.p,(float*)m.p,(float*)l.p);
  }
  ops::gqa_attention_small_t_reduce_output_kernel<ops::Gqa27Geometry,64,true,false,false,false><<<dim3(heads,4,tokens),256,0,s>>>((const __nv_bfloat16*)part.p,(const float*)m.p,(const float*)l.p,(const int*)positions.p,nullptr,4,4,0,1,splits,(__nv_bfloat16*)dest);CUDA_CHECK(cudaGetLastError());
 };
 auto reference=[&](cudaStream_t s){invoke(s,false,v.p,vs.p,ref.p);};
 auto candidate=[&](cudaStream_t s){invoke(s,true,v.p,vs.p,out.p);};
 reference(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
 std::vector<std::uint8_t>p0(part.bytes),m0(m.bytes),l0(l.bytes);part.copy_to_host(p0.data(),p0.size());m.copy_to_host(m0.data(),m0.size());l.copy_to_host(l0.data(),l0.size());
 candidate(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
 std::vector<char> refout(out.bytes),refpart(part.bytes),refmax(m.bytes),refsum(l.bytes);
 out.copy_to_host(refout.data(),out.bytes);part.copy_to_host(refpart.data(),part.bytes);m.copy_to_host(refmax.data(),m.bytes);l.copy_to_host(refsum.data(),l.bytes);
 s4_mode=true;candidate(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
 std::vector<char> gotout(out.bytes),gotpart(part.bytes),gotmax(m.bytes),gotsum(l.bytes);
 out.copy_to_host(gotout.data(),out.bytes);part.copy_to_host(gotpart.data(),part.bytes);m.copy_to_host(gotmax.data(),m.bytes);l.copy_to_host(gotsum.data(),l.bytes);
 if(gotout!=refout||gotpart!=refpart||gotmax!=refmax||gotsum!=refsum)throw std::runtime_error("nativeS4 changed the centeredS8 arithmetic");
 printf("QUERY_REUSE output_partial_max_sum_byte_exact_to_centeredS8\n");
 if(window<=2049)scalar_check(window,pages,table,pos,qh,converted,converted_scales,packedv,packedvs,out);
 std::vector<std::uint8_t>p1(part.bytes),m1(m.bytes),l1(l.bytes);part.copy_to_host(p1.data(),p1.size());m.copy_to_host(m1.data(),m1.size());l.copy_to_host(l1.data(),l1.size());
 std::vector<std::uint16_t>a(ref.bytes/2),b(out.bytes/2);ref.copy_to_host(a.data(),ref.bytes);out.copy_to_host(b.data(),out.bytes);
 auto diff=[](const auto&a,const auto&b){std::size_t n=0;for(std::size_t i=0;i<a.size();++i)n+=a[i]!=b[i];return n;};
 auto bfvalue=[](std::uint16_t v){union {unsigned u;float f;}x{static_cast<unsigned>(v)<<16};return x.f;};
 double error=0,norm=0,dot=0,ynorm=0;size_t nonfinite=0;
 for(size_t i=0;i<a.size();++i){double x=bfvalue(a[i]),y=bfvalue(b[i]);error+=(x-y)*(x-y);norm+=x*x;ynorm+=y*y;dot+=x*y;nonfinite+=!std::isfinite(y);}
 printf("NUMERICAL window=%d seed=%u output=%zu partial=%zu max=%zu sum=%zu rel_l2=%.9g cosine=%.9g nonfinite=%zu representation=original_headscaleQK_centered_channelV4_integer_PV\n",window,seed,diff(a,b),diff(p0,p1),diff(m0,m1),diff(l0,l1),std::sqrt(error/std::max(norm,1e-30)),dot/std::sqrt(norm*ynorm),nonfinite);
 if(nonfinite)return 2;
 // Random synthetic output distance is reported, not a real-model quality gate.

 cudaGraph_t gr{},gc{};cudaGraphExec_t er{},ec{};
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));reference(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gr));CUDA_CHECK(cudaGraphInstantiate(&er,gr,nullptr,nullptr,0));
 CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));candidate(stream);CUDA_CHECK(cudaStreamEndCapture(stream,&gc));CUDA_CHECK(cudaGraphInstantiate(&ec,gc,nullptr,nullptr,0));
 for(int pass=0;pass<2;++pass){auto a=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(er,s));},flush,stream,3,12);auto b=bench::measure_cold_launch([&](cudaStream_t s){CUDA_CHECK(cudaGraphLaunch(ec,s));},flush,stream,3,12);printf("PERF window=%d seed=%u pass=%d baseline_us=%.3f prepared_us=%.3f ratio=%.6f\n",window,seed,pass,a.median_us,b.median_us,a.median_us/b.median_us);}
 std::vector<std::uint16_t> replay(out.bytes/2);out.copy_to_host(replay.data(),out.bytes);
 if(replay!=b)throw std::runtime_error("candidate graph output changed");
 printf("REPLAY all_output_bits_equal static_encoded_cache_cost_not_tail_runtime_integration\n");
 CUDA_CHECK(cudaGraphExecDestroy(er));CUDA_CHECK(cudaGraphExecDestroy(ec));CUDA_CHECK(cudaGraphDestroy(gr));CUDA_CHECK(cudaGraphDestroy(gc));
 }CUDA_CHECK(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception&e){fprintf(stderr,"prepared_values: %s\n",e.what());return 1;}
}
