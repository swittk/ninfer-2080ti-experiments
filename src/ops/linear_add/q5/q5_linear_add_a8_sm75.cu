#include "q5_linear_add_a8_sm75.h"
#include "core/device.h"
#include "ops/common/mma.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
namespace ninfer::ops::detail {
namespace {
constexpr int kN=5120,kK=17408,kGroupK=64,kGroups=kK/kGroupK;
constexpr int kRowsPerBlock=16,kLanes=32,kWarps=2,kThreads=kWarps*32;
constexpr int kRowBlocks=kN/kRowsPerBlock;
constexpr int kLowRowBytes=kGroups*32,kHighRowBytes=kGroups*8;
constexpr int kLowBlockBytes=kRowsPerBlock*kLowRowBytes;
constexpr int kHighBlockBytes=kRowsPerBlock*kHighRowBytes;
constexpr int kScratchBlockBytes=kLowBlockBytes+kHighBlockBytes;
constexpr int kTransformCtas=64;
constexpr std::size_t kScratchBytes=(std::size_t)kTransformCtas*kScratchBlockBytes;

__device__ __forceinline__ std::uint32_t expand_q5(std::uint16_t low,std::uint32_t hi4){
    const std::uint32_t x=(std::uint32_t)low;
    const std::uint32_t lo=__byte_perm(x&0x0f0fu,(x>>4)&0x0f0fu,0x5140);
    const std::uint32_t m=((hi4&1u)<<4)|((hi4&2u)<<11)|((hi4&4u)<<18)|((hi4&8u)<<25);
    return lo|(m*15u);
}

__device__ __forceinline__ std::uint32_t canonical_hi4(const std::uint8_t* high,int row,int group,int k0){
    const auto* p=high+(row*kGroups+group)*8+(k0>>3);
    return ((*p)>>(k0&7))&15u;
}

template<int Tokens>
__global__ void quantize_a8(const __nv_bfloat16* x,std::int8_t* q,__half* scales){
    int warp=(int)threadIdx.x>>5,lane=(int)threadIdx.x&31,unit=(int)blockIdx.x*8+warp;
    if(unit>=Tokens*kGroups)return;
    int t=unit/kGroups,g=unit-t*kGroups,d0=g*kGroupK+lane,d1=d0+32;
    float x0=__bfloat162float(x[(std::int64_t)t*kK+d0]);
    float x1=__bfloat162float(x[(std::int64_t)t*kK+d1]);
    float m=fmaxf(fabsf(x0),fabsf(x1));
#pragma unroll
    for(int off=16;off;off>>=1)m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,off));
    float s=m?m/127.f:0.f,inv=s?1.f/s:0.f;
    int a=__float2int_rn(x0*inv),b=__float2int_rn(x1*inv);
    a=max(-127,min(127,a));b=max(-127,min(127,b));
    q[(std::int64_t)t*kK+d0]=(std::int8_t)a;
    q[(std::int64_t)t*kK+d1]=(std::int8_t)b;
    if(lane==0)scales[t*kGroups+g]=__float2half_rn(s);
}

template<int Tokens>
__global__ void q5_a8_add_kernel(const std::uint8_t* low,const std::uint8_t* high,
                                 const std::uint8_t* scale_bytes,const std::int8_t* aq,
                                 const __half* as,__nv_bfloat16* out){
    int lane=(int)threadIdx.x&31,warp=(int)threadIdx.x>>5;
    int rb=(int)blockIdx.x*kWarps+warp,rowbase=rb*16,gid=lane>>2,lid=lane&3;
    int r0=rowbase+gid,r1=r0+8;
    float f0=0,f1=0,f2=0,f3=0;
    uint4 p=*reinterpret_cast<const uint4*>(low+(((std::int64_t)rb*kGroups)*32+lane)*16);
    std::uint32_t h=*reinterpret_cast<const std::uint32_t*>(high+(((std::int64_t)rb*kGroups)*32+lane)*4);
#pragma unroll 1
    for(int g=0;g<kGroups;++g){
        uint4 pn{};std::uint32_t hn=0;
        if(g+1<kGroups){
            pn=*reinterpret_cast<const uint4*>(low+(((std::int64_t)rb*kGroups+g+1)*32+lane)*16);
            hn=*reinterpret_cast<const std::uint32_t*>(high+(((std::int64_t)rb*kGroups+g+1)*32+lane)*4);
        }
        int c0=0,c1=0,c2=0,c3=0;
        int tok=lane>>2,q=lane&3;unsigned b0=0,b1=0;
        if(tok<Tokens){const auto* ab=reinterpret_cast<const std::uint32_t*>(aq+(std::int64_t)tok*kK+g*64);b0=ab[q];b1=ab[4+q];}
        mma_s8(c0,c1,c2,c3,
               expand_q5((std::uint16_t)p.x,h),expand_q5((std::uint16_t)(p.x>>16),h>>4),
               expand_q5((std::uint16_t)p.y,h>>8),expand_q5((std::uint16_t)(p.y>>16),h>>12),b0,b1);
        if(tok<Tokens){const auto* ab=reinterpret_cast<const std::uint32_t*>(aq+(std::int64_t)tok*kK+g*64+32);b0=ab[q];b1=ab[4+q];}else b0=b1=0;
        mma_s8(c0,c1,c2,c3,
               expand_q5((std::uint16_t)p.z,h>>16),expand_q5((std::uint16_t)(p.z>>16),h>>20),
               expand_q5((std::uint16_t)p.w,h>>24),expand_q5((std::uint16_t)(p.w>>16),h>>28),b0,b1);
        int ca=2*lid,cb=ca+1;
        float w0=0,w1=0;
        if(lid==0){
            const auto* ss=reinterpret_cast<const std::uint16_t*>(scale_bytes);
            const std::int64_t si=((std::int64_t)rb*kGroups+g)*kRowsPerBlock;
            w0=__half2float(__ushort_as_half(ss[si+gid]));
            w1=__half2float(__ushort_as_half(ss[si+8+gid]));
        }
        w0=__shfl_sync(0xffffffffu,w0,gid*4);w1=__shfl_sync(0xffffffffu,w1,gid*4);
        float s0=0,s1=0;
        if(gid==0){s0=ca<Tokens?__half2float(as[ca*kGroups+g]):0;s1=cb<Tokens?__half2float(as[cb*kGroups+g]):0;}
        s0=__shfl_sync(0xffffffffu,s0,lid);s1=__shfl_sync(0xffffffffu,s1,lid);
        f0=__fmaf_rn((float)c0,w0*s0,f0);f1=__fmaf_rn((float)c1,w0*s1,f1);
        f2=__fmaf_rn((float)c2,w1*s0,f2);f3=__fmaf_rn((float)c3,w1*s1,f3);
        p=pn;h=hn;
    }
    int ca=2*lid,cb=ca+1;
    if(ca<Tokens){
        std::int64_t i0=(std::int64_t)ca*kN+r0,i1=(std::int64_t)ca*kN+r1;
        out[i0]=__float2bfloat16_rn(__bfloat162float(out[i0])+f0);
        out[i1]=__float2bfloat16_rn(__bfloat162float(out[i1])+f2);
    }
    if(cb<Tokens){
        std::int64_t i0=(std::int64_t)cb*kN+r0,i1=(std::int64_t)cb*kN+r1;
        out[i0]=__float2bfloat16_rn(__bfloat162float(out[i0])+f1);
        out[i1]=__float2bfloat16_rn(__bfloat162float(out[i1])+f3);
    }
}


__launch_bounds__(128, 4) __global__ void q5_a8_add_split4_interleave_t4(
    const std::uint8_t* low,const std::uint8_t* high,const std::uint8_t* scale_bytes,
    const std::int8_t* aq,const __half* as,__nv_bfloat16* out){
    constexpr int Parts=4;
    __shared__ float partial[Parts][32][4];
    int lane=(int)threadIdx.x&31,part=(int)threadIdx.x>>5,rb=(int)blockIdx.x;
    int rowbase=rb*16,gid=lane>>2,lid=lane&3;
    float f0=0,f1=0,f2=0,f3=0;
    const int begin=part,end=kGroups;
    uint4 p=*reinterpret_cast<const uint4*>(low+(((std::int64_t)rb*kGroups+begin)*32+lane)*16);
    std::uint32_t h=*reinterpret_cast<const std::uint32_t*>(high+(((std::int64_t)rb*kGroups+begin)*32+lane)*4);
    for(int g=begin;g<end;g+=Parts){
        uint4 pn{};std::uint32_t hn=0;
        if(g+Parts<end){
            pn=*reinterpret_cast<const uint4*>(low+(((std::int64_t)rb*kGroups+g+Parts)*32+lane)*16);
            hn=*reinterpret_cast<const std::uint32_t*>(high+(((std::int64_t)rb*kGroups+g+Parts)*32+lane)*4);
        }
        int c0=0,c1=0,c2=0,c3=0;int tok=lane>>2,q=lane&3;unsigned b0=0,b1=0;
        if(tok<4){const auto* ab=reinterpret_cast<const std::uint32_t*>(aq+(std::int64_t)tok*kK+g*64);b0=ab[q];b1=ab[4+q];}
        mma_s8(c0,c1,c2,c3,
               expand_q5((std::uint16_t)p.x,h),expand_q5((std::uint16_t)(p.x>>16),h>>4),
               expand_q5((std::uint16_t)p.y,h>>8),expand_q5((std::uint16_t)(p.y>>16),h>>12),b0,b1);
        if(tok<4){const auto* ab=reinterpret_cast<const std::uint32_t*>(aq+(std::int64_t)tok*kK+g*64+32);b0=ab[q];b1=ab[4+q];}else b0=b1=0;
        mma_s8(c0,c1,c2,c3,
               expand_q5((std::uint16_t)p.z,h>>16),expand_q5((std::uint16_t)(p.z>>16),h>>20),
               expand_q5((std::uint16_t)p.w,h>>24),expand_q5((std::uint16_t)(p.w>>16),h>>28),b0,b1);
        int ca=2*lid,cb=ca+1,r0=rowbase+gid,r1=r0+8;
        float w0=0,w1=0;
        if(lid==0){
            const auto* ss=reinterpret_cast<const std::uint16_t*>(scale_bytes);
            const std::int64_t si=((std::int64_t)rb*kGroups+g)*kRowsPerBlock;
            w0=__half2float(__ushort_as_half(ss[si+gid]));
            w1=__half2float(__ushort_as_half(ss[si+8+gid]));
        }
        w0=__shfl_sync(0xffffffffu,w0,gid*4);w1=__shfl_sync(0xffffffffu,w1,gid*4);
        float s0=0,s1=0;
        if(gid==0){s0=ca<4?__half2float(as[ca*kGroups+g]):0;s1=cb<4?__half2float(as[cb*kGroups+g]):0;}
        s0=__shfl_sync(0xffffffffu,s0,lid);s1=__shfl_sync(0xffffffffu,s1,lid);
        f0=__fmaf_rn((float)c0,w0*s0,f0);f1=__fmaf_rn((float)c1,w0*s1,f1);
        f2=__fmaf_rn((float)c2,w1*s0,f2);f3=__fmaf_rn((float)c3,w1*s1,f3);
        p=pn;h=hn;
    }
    partial[part][lane][0]=f0;partial[part][lane][1]=f1;
    partial[part][lane][2]=f2;partial[part][lane][3]=f3;
    __syncthreads();
    if(part==0){
        float a0=partial[0][lane][0],a1=partial[0][lane][1],a2=partial[0][lane][2],a3=partial[0][lane][3];
#pragma unroll
        for(int p=1;p<Parts;++p){a0+=partial[p][lane][0];a1+=partial[p][lane][1];a2+=partial[p][lane][2];a3+=partial[p][lane][3];}
        int ca=2*lid,cb=ca+1,r0=rowbase+gid,r1=r0+8;
        if(ca<4){
            auto i0=(std::int64_t)ca*kN+r0,i1=(std::int64_t)ca*kN+r1;
            out[i0]=__float2bfloat16_rn(__bfloat162float(out[i0])+a0);
            out[i1]=__float2bfloat16_rn(__bfloat162float(out[i1])+a2);
        }
        if(cb<4){
            auto i0=(std::int64_t)cb*kN+r0,i1=(std::int64_t)cb*kN+r1;
            out[i0]=__float2bfloat16_rn(__bfloat162float(out[i0])+a1);
            out[i1]=__float2bfloat16_rn(__bfloat162float(out[i1])+a3);
        }
    }
}

template<bool ToFragment>
__global__ void transform_kernel(std::uint8_t* low,std::uint8_t* high,std::uint8_t* scales,std::uint8_t* scratch){
    std::uint8_t* tmp=scratch+(std::size_t)blockIdx.x*kScratchBlockBytes;
    std::uint8_t* low_tmp=tmp;
    std::uint8_t* high_tmp=tmp+kLowBlockBytes;
    for(int rb=(int)blockIdx.x;rb<kRowBlocks;rb+=gridDim.x){
        std::uint8_t* low_base=low+(std::size_t)rb*kLowBlockBytes;
        std::uint8_t* high_base=high+(std::size_t)rb*kHighBlockBytes;
        for(int v=(int)threadIdx.x;v<kLowBlockBytes/16;v+=blockDim.x)
            reinterpret_cast<uint4*>(low_tmp)[v]=reinterpret_cast<const uint4*>(low_base)[v];
        for(int v=(int)threadIdx.x;v<kHighBlockBytes/16;v+=blockDim.x)
            reinterpret_cast<uint4*>(high_tmp)[v]=reinterpret_cast<const uint4*>(high_base)[v];
        __syncthreads();
        if constexpr(ToFragment){
            for(int item=(int)threadIdx.x;item<kGroups*32;item+=blockDim.x){
                int g=item>>5,lane=item&31,r=lane>>2,q=lane&3,r0=r,r1=r+8;
                auto* dst=low_base+(std::size_t)item*16;
                auto* d16=reinterpret_cast<std::uint16_t*>(dst);
                std::uint32_t hp=0;
#pragma unroll
                for(int hh=0;hh<2;++hh){
                    int bo=hh*16+q*2,k0=hh*32+q*4;
                    const auto* l0=low_tmp+(r0*kGroups+g)*32;
                    const auto* l1=low_tmp+(r1*kGroups+g)*32;
                    d16[hh*4+0]=*reinterpret_cast<const std::uint16_t*>(l0+bo);
                    d16[hh*4+1]=*reinterpret_cast<const std::uint16_t*>(l1+bo);
                    d16[hh*4+2]=*reinterpret_cast<const std::uint16_t*>(l0+bo+8);
                    d16[hh*4+3]=*reinterpret_cast<const std::uint16_t*>(l1+bo+8);
                    hp|=canonical_hi4(high_tmp,r0,g,k0)<<((hh*4+0)*4);
                    hp|=canonical_hi4(high_tmp,r1,g,k0)<<((hh*4+1)*4);
                    hp|=canonical_hi4(high_tmp,r0,g,k0+16)<<((hh*4+2)*4);
                    hp|=canonical_hi4(high_tmp,r1,g,k0+16)<<((hh*4+3)*4);
                }
                *reinterpret_cast<std::uint32_t*>(high_base+(std::size_t)item*4)=hp;
            }
        }else{
            const int low_pairs=kRowsPerBlock*kGroups*16;
            for(int item=(int)threadIdx.x;item<low_pairs;item+=blockDim.x){
                int row=item/(kGroups*16),rem=item-row*kGroups*16,g=rem/16,cp=rem-g*16;
                int q=cp&3,quarter=cp>>2,lane=4*(row&7)+q;
                const auto* src=reinterpret_cast<const std::uint16_t*>(low_tmp+(std::size_t)(g*32+lane)*16);
                *reinterpret_cast<std::uint16_t*>(low_base+(row*kGroups+g)*32+cp*2)=src[quarter*2+(row>=8?1:0)];
            }
            const int high_bytes=kRowsPerBlock*kGroups*8;
            for(int item=(int)threadIdx.x;item<high_bytes;item+=blockDim.x){
                int row=item/(kGroups*8),rem=item-row*kGroups*8,g=rem/8,byte=rem-g*8;
                int seg=byte>>1,pair=byte&1,q0=2*pair,q1=q0+1;
                int l0=4*(row&7)+q0,l1=l0+1,shift=seg*8+(row>=8?4:0);
                std::uint32_t h0=*reinterpret_cast<const std::uint32_t*>(high_tmp+(std::size_t)(g*32+l0)*4);
                std::uint32_t h1=*reinterpret_cast<const std::uint32_t*>(high_tmp+(std::size_t)(g*32+l1)*4);
                high_base[(row*kGroups+g)*8+byte]=(std::uint8_t)(((h0>>shift)&15u)|(((h1>>shift)&15u)<<4));
            }
        }
        __syncthreads();
        auto* scale_base=reinterpret_cast<std::uint16_t*>(scales)+(std::size_t)rb*kRowsPerBlock*kGroups;
        auto* scale_tmp=reinterpret_cast<std::uint16_t*>(low_tmp);
        constexpr int kScaleTileElems=kRowsPerBlock*kGroups;
        for(int i=(int)threadIdx.x;i<kScaleTileElems;i+=blockDim.x)scale_tmp[i]=scale_base[i];
        __syncthreads();
        for(int i=(int)threadIdx.x;i<kScaleTileElems;i+=blockDim.x){
            if constexpr(ToFragment){
                int row=i/kGroups,g=i-row*kGroups;
                scale_base[g*kRowsPerBlock+row]=scale_tmp[i];
            }else{
                int row=i/kGroups,g=i-row*kGroups;
                scale_base[i]=scale_tmp[g*kRowsPerBlock+row];
            }
        }
        __syncthreads();
    }
}
}

std::size_t q5_linear_add_sm75_a8_workspace_bytes(std::int32_t tokens){
    if(tokens<1||tokens>8)return 0;
    return (std::size_t)kK*tokens+(std::size_t)kGroups*tokens*sizeof(__half);
}
std::size_t q5_linear_add_sm75_transform_scratch_bytes(){return kScratchBytes;}

template<int Tokens>
void launch_tokens(const Tensor& x,const Weight& w,Tensor& residual,WorkspaceArena& ws,cudaStream_t stream){
    auto scope=ws.scope();
    Tensor aq=ws.alloc(DType::I8,{kK,Tokens});
    Tensor as=ws.alloc(DType::FP16,{kGroups,Tokens});
    quantize_a8<Tokens><<<(Tokens*kGroups+7)/8,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x.data),static_cast<std::int8_t*>(aq.data),static_cast<__half*>(as.data));
    CUDA_CHECK(cudaGetLastError());
    if constexpr(Tokens==4){
        q5_a8_add_split4_interleave_t4<<<kN/16,128,0,stream>>>(static_cast<const std::uint8_t*>(w.qdata),static_cast<const std::uint8_t*>(w.qhigh),static_cast<const std::uint8_t*>(w.scales),static_cast<const std::int8_t*>(aq.data),static_cast<const __half*>(as.data),static_cast<__nv_bfloat16*>(residual.data));
    }else{
        q5_a8_add_kernel<Tokens><<<kN/(16*kWarps),kThreads,0,stream>>>(static_cast<const std::uint8_t*>(w.qdata),static_cast<const std::uint8_t*>(w.qhigh),static_cast<const std::uint8_t*>(w.scales),static_cast<const std::int8_t*>(aq.data),static_cast<const __half*>(as.data),static_cast<__nv_bfloat16*>(residual.data));
    }
    CUDA_CHECK(cudaGetLastError());
}
void q5_linear_add_sm75_a8_prequant_launch(const Tensor& aq,const Tensor& as,const Weight& w,Tensor& residual,cudaStream_t stream){
    if(w.layout!=QuantLayout::RowSplitSm75Fragment||w.n!=kN||w.k!=kK||aq.dtype!=DType::I8||as.dtype!=DType::FP16||aq.ne[1]!=4||as.ne[1]!=4)throw std::invalid_argument("SM75 Q5 prequant linear_add requires T4 A8 input");
    q5_a8_add_split4_interleave_t4<<<kN/16,128,0,stream>>>(static_cast<const std::uint8_t*>(w.qdata),static_cast<const std::uint8_t*>(w.qhigh),static_cast<const std::uint8_t*>(w.scales),static_cast<const std::int8_t*>(aq.data),static_cast<const __half*>(as.data),static_cast<__nv_bfloat16*>(residual.data));
    CUDA_CHECK(cudaGetLastError());
}
void q5_linear_add_sm75_a8_launch(const Tensor& x,const Weight& w,Tensor& residual,WorkspaceArena& ws,cudaStream_t stream){
    if(w.layout!=QuantLayout::RowSplitSm75Fragment||w.n!=kN||w.k!=kK)throw std::invalid_argument("SM75 Q5 A8 linear_add requires fragment 5120x17408 weight");
    switch(x.ne[1]){case 1:launch_tokens<1>(x,w,residual,ws,stream);return;case 2:launch_tokens<2>(x,w,residual,ws,stream);return;case 3:launch_tokens<3>(x,w,residual,ws,stream);return;case 4:launch_tokens<4>(x,w,residual,ws,stream);return;case 5:launch_tokens<5>(x,w,residual,ws,stream);return;case 6:launch_tokens<6>(x,w,residual,ws,stream);return;case 7:launch_tokens<7>(x,w,residual,ws,stream);return;case 8:launch_tokens<8>(x,w,residual,ws,stream);return;default:throw std::invalid_argument("SM75 Q5 A8 linear_add requires T1..T8");}
}
void q5_linear_add_sm75_transform_layout(Weight& w,QuantLayout target,void* scratch,std::size_t scratch_bytes,cudaStream_t stream){
    if(w.qtype!=QType::Q5G64_F16S||w.n!=kN||w.k!=kK||w.qdata==nullptr||w.qhigh==nullptr||scratch==nullptr||scratch_bytes<kScratchBytes||(target!=QuantLayout::RowSplit&&target!=QuantLayout::RowSplitSm75Fragment))throw std::invalid_argument("SM75 Q5 fragment transform invalid arguments");
    if(w.layout==target)return;
    if(w.layout!=QuantLayout::RowSplit&&w.layout!=QuantLayout::RowSplitSm75Fragment)throw std::invalid_argument("SM75 Q5 fragment transform unsupported source layout");
    auto* low=const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(w.qdata));auto* high=const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(w.qhigh));auto* scales=const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(w.scales));auto* tmp=static_cast<std::uint8_t*>(scratch);
    if(scales==nullptr)throw std::invalid_argument("SM75 Q5 fragment transform requires scales");
    if(target==QuantLayout::RowSplitSm75Fragment)transform_kernel<true><<<kTransformCtas,256,0,stream>>>(low,high,scales,tmp);else transform_kernel<false><<<kTransformCtas,256,0,stream>>>(low,high,scales,tmp);
    CUDA_CHECK(cudaGetLastError());w.layout=target;
}
} // namespace ninfer::ops::detail
