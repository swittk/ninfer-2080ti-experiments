#include "targets/qwen3_6_27b/impl/sm75_prefill_dense.h"
#include "core/device.h"
#include "ops/linear/sm75_radix_projection.h"
#include "ops/linear/sm75_radix_prefill.cuh"
#include "core/layout.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <tuple>
namespace ninfer::targets::qwen3_6_27b::detail {
namespace {
constexpr std::size_t kLibraryWorkspace=4ULL<<20;
void checked(cublasStatus_t result) {
    if(result!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error(
        "SM75 prefill cuBLASLt failure="+std::to_string(result));
}
// Gate/up is expanded in four row sections. The FP32 output remains full-sized
// and column-major, so SwiGLU and its native rounding boundary are unchanged.
// Large-chunk alternative: bound intermediate live storage without changing
// input limits or accepting a smaller KV reservation. Fixed for this process,
// shared by planning and execution; existing profile stays unchanged by default.
bool streamed_prefill() noexcept {
    static const bool value=[] {
        const char* text=std::getenv("NINFER_SERVE_PREFILL_STREAMED");
        return text && text[0]=='1' && text[1]=='\0';
    }();
    return value;
}
constexpr int kStreamedRows=4352;
int expanded_rows(int n,int k,int t) noexcept {
    if(streamed_prefill() && t>=1024 && n==5120 && k==17408) return 2560;
    return n==34816?8704:n;
}
template<class Arena> void streamed_gate_layout(Arena& arena,int t) {
    (void)arena.alloc(DType::FP16,{5120,kStreamedRows});
    (void)arena.alloc(DType::FP16,{5120,t});
    (void)arena.alloc(DType::FP32,{kStreamedRows,t});
    (void)arena.alloc(DType::FP32,{kStreamedRows,t});
    (void)arena.alloc_bytes(kLibraryWorkspace);
}
template<class Arena> void streamed_value_layout(Arena& arena,int t) {
    (void)arena.alloc(DType::FP16,{5120,6144});
    (void)arena.alloc(DType::FP16,{5120,t});
    (void)arena.alloc(DType::FP32,{6144,t});
    (void)arena.alloc_bytes(kLibraryWorkspace);
}
template<class Arena> void projection_layout(Arena& arena,int n,int k,int t) {
    (void)arena.alloc(DType::FP16,{k,expanded_rows(n,k,t)});
    (void)arena.alloc(DType::FP16,{k,t});
    (void)arena.alloc(DType::FP32,{n,t});
    (void)arena.alloc_bytes(kLibraryWorkspace);
}
template<bool Q5,bool Fragment,int N,int K>
__global__ void unpack_weights(const std::uint8_t* low,const std::uint8_t* high,
                               const std::uint16_t* scales,__half2* out,
                               std::int64_t pairs, int row_begin) {
    constexpr int n=N, k=K;
    constexpr int groups=k/64;
    for(std::int64_t i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        i<pairs;i+=static_cast<std::int64_t>(blockDim.x)*gridDim.x) {
        const auto source_pair = i + static_cast<std::int64_t>(row_begin) * groups * 32;
        const auto group_index=source_pair/32;
        const int pair=source_pair%32;
        std::uint8_t bits,h=0;std::uint16_t scale;
        if constexpr(!Fragment) {
            bits=low[source_pair];scale=scales[group_index];
            if constexpr(Q5) h=high[group_index*8+(pair>>2)];
        } else {
            const int row=group_index/groups, group=group_index%groups;
            if constexpr(Q5) {
                const int rb=row/16,r=row%16,pair16=pair/2;
                const int lane=4*(r&7)+(pair16&3);
                const auto source=((static_cast<std::int64_t>(rb)*groups+group)*32+lane)*8+
                                  (pair16>>2)*2+(r>=8?1:0);
                bits=static_cast<std::uint8_t>(reinterpret_cast<const std::uint16_t*>(low)[source]>>((pair&1)*8));
                const int byte=pair>>2,hlane=4*(r&7)+2*(byte&1);
                const auto hs=(static_cast<std::int64_t>(rb)*groups+group)*32+hlane;
                const int shift=(byte>>1)*8+(r>=8?4:0);
                const auto* hi=reinterpret_cast<const std::uint32_t*>(high);
                h=static_cast<std::uint8_t>(((hi[hs]>>shift)&15u)|(((hi[hs+1]>>shift)&15u)<<4));
                scale=scales[(static_cast<std::int64_t>(rb)*groups+group)*16+r];
            } else {
                const int rows=n/2,local=row%rows,rb=local/8,r=local%8;
                const int lane=r*4+(pair%8)/2;
                const auto source=((static_cast<std::int64_t>(rb)*groups+group)*32+lane)*16+
                    ((pair/16)+(row>=rows?2:0))*4+((pair%16)>=8?2:0)+(pair&1);
                bits=low[source];
                scale=scales[((static_cast<std::int64_t>(rb)*groups+group)*8+r)*2+(row>=rows?1:0)];
            }
        }
        int a=bits&15,b=bits>>4;
        if constexpr(Q5) {int shift=(pair&3)*2;a|=((h>>shift)&1)<<4;b|=((h>>(shift+1))&1)<<4;a=(a^16)-16;b=(b^16)-16;}
        else {a=(a^8)-8;b=(b^8)-8;}
        const float s=__half2float(__ushort_as_half(scale));
        out[i]=__floats2half2_rn(__bfloat162float(__float2bfloat16_rn(a*s)),
                               __bfloat162float(__float2bfloat16_rn(b*s)));
    }
}
__global__ void to_half(const __nv_bfloat16* input,__half* output,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) output[i]=__float2half_rn(__bfloat162float(input[i]));
}
template<bool Materialized> __global__ void swiglu(const float* projection,__nv_bfloat16* output,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) {const int t=i/17408,r=i%17408;
        float g=projection[static_cast<std::int64_t>(t)*34816+r];
        float u=projection[static_cast<std::int64_t>(t)*34816+r+17408];
        if constexpr (Materialized) {
            g=__bfloat162float(__float2bfloat16_rn(g));
            u=__bfloat162float(__float2bfloat16_rn(u));
        }
        output[i]=__float2bfloat16_rn((g/(1.f+expf(-g)))*u);
    }
}
template<bool Materialized>
__global__ void streamed_swiglu(const float* gate,const float* up,__nv_bfloat16* output,int tokens,int begin) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<kStreamedRows*tokens) {
        float g=gate[i],u=up[i];
        if constexpr(Materialized) {
            g=__bfloat162float(__float2bfloat16_rn(g));
            u=__bfloat162float(__float2bfloat16_rn(u));
        }
        output[static_cast<std::int64_t>(i/kStreamedRows)*17408+begin+i%kStreamedRows]=
            __float2bfloat16_rn((g/(1.f+expf(-g)))*u);
    }
}
__global__ void gdn_scatter(const float* source,__nv_bfloat16* qkv,__nv_bfloat16* z,int tokens,bool value_z) {
    const int rows=value_z?12288:4096;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*tokens) {
        const int t=i/rows,r=i%rows;
        const auto rounded=__float2bfloat16_rn(source[i]);
        if(!value_z) qkv[static_cast<std::int64_t>(t)*10240+r]=rounded;
        else if(r<6144) qkv[static_cast<std::int64_t>(t)*10240+4096+r]=rounded;
        else z[static_cast<std::int64_t>(t)*6144+r-6144]=rounded;
    }
}
template<bool OutputGate>
__global__ void streamed_value_scatter(const float* source,__nv_bfloat16* qkv,__nv_bfloat16* z,int tokens) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<6144*tokens) {
        const auto rounded=__float2bfloat16_rn(source[i]);
        if constexpr(OutputGate) z[i]=rounded;
        else qkv[static_cast<std::int64_t>(i/6144)*10240+4096+i%6144]=rounded;
    }
}
__global__ void attention_scatter(const float* source, __nv_bfloat16* large,
                                  __nv_bfloat16* small, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 7168 * tokens) {
        const int t = i / 7168, r = i % 7168;
        const auto rounded = __float2bfloat16_rn(source[i]);
        if (r < 6144) large[static_cast<std::int64_t>(t) * 6144 + r] = rounded;
        else small[static_cast<std::int64_t>(t) * 1024 + r - 6144] = rounded;
    }
}
__global__ void add(const float* projection,__nv_bfloat16* output,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) {
        // Existing Q5 CtaCollectiveResidual rounds its projection before the add.
        const float projected=__bfloat162float(__float2bfloat16_rn(projection[i]));
        output[i]=__float2bfloat16_rn(__bfloat162float(output[i])+projected);
    }
}
struct Plan {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t a{},b{},c{};
    cublasLtMatmulAlgo_t algorithm{};
    Plan(cublasLtHandle_t handle,int n,int k,int t,int output_stride) {
        cublasLtMatmulPreference_t preference{};
        try {
            checked(cublasLtMatmulDescCreate(&operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
            cublasOperation_t transpose=CUBLAS_OP_T;
            checked(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
            checked(cublasLtMatrixLayoutCreate(&a,CUDA_R_16F,k,n,k));
            checked(cublasLtMatrixLayoutCreate(&b,CUDA_R_16F,k,t,k));
            checked(cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,n,t,output_stride));
            checked(cublasLtMatmulPreferenceCreate(&preference));
            const std::size_t bytes=kLibraryWorkspace;
            checked(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
            cublasLtMatmulHeuristicResult_t best{};int found=0;
            checked(cublasLtMatmulAlgoGetHeuristic(handle,operation,a,b,c,c,preference,1,&best,&found));
            if(found!=1||best.state!=CUBLAS_STATUS_SUCCESS||best.workspaceSize>bytes) throw std::runtime_error("SM75 prefill has no bounded dense GEMM algorithm");
            algorithm=best.algo;
            cublasLtMatmulPreferenceDestroy(preference);
        } catch(...) {if(preference)cublasLtMatmulPreferenceDestroy(preference);release();throw;}
    }
    ~Plan(){release();}
    void release() noexcept {
        if(c)cublasLtMatrixLayoutDestroy(c);if(b)cublasLtMatrixLayoutDestroy(b);
        if(a)cublasLtMatrixLayoutDestroy(a);if(operation)cublasLtMatmulDescDestroy(operation);
        a=b=c=nullptr;operation=nullptr;
    }
};
}
struct Sm75PrefillDense::Impl {
    WorkspaceArena& workspace;
    cudaStream_t stream;
    cublasLtHandle_t handle{};
    std::map<std::tuple<int,int,int,int>,std::unique_ptr<Plan>> plans;
    Impl(WorkspaceArena& w,cudaStream_t s):workspace(w),stream(s){checked(cublasLtCreate(&handle));}
    ~Impl(){plans.clear();if(handle)cublasLtDestroy(handle);}
    void streamed_gate(const Tensor& x,const Weight& weight,Tensor& activation) {
        const int t=x.ne[1],k=5120;
        auto scope=workspace.scope();
        Tensor expanded=workspace.alloc(DType::FP16,{k,kStreamedRows});
        Tensor input=workspace.alloc(DType::FP16,{k,t});
        Tensor gate=workspace.alloc(DType::FP32,{kStreamedRows,t});
        Tensor up=workspace.alloc(DType::FP32,{kStreamedRows,t});
        const auto scratch=workspace.alloc_bytes(kLibraryWorkspace);
        to_half<<<(k*t+255)/256,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x.data),static_cast<__half*>(input.data),k*t);
        CUDA_CHECK(cudaGetLastError());
        const auto key=std::make_tuple(kStreamedRows,k,t,kStreamedRows);
        auto it=plans.find(key);
        if(it==plans.end()) it=plans.emplace(key,std::make_unique<Plan>(handle,kStreamedRows,k,t,kStreamedRows)).first;
        const Plan& p=*it->second;const float alpha=1.f,beta=0.f;
        const auto project_section=[&](int row_begin,const Tensor& destination) {
            const auto* low=static_cast<const std::uint8_t*>(weight.qdata);
            const auto* scales=static_cast<const std::uint16_t*>(weight.scales);
            auto* out=static_cast<__half2*>(expanded.data);
            constexpr auto pairs=static_cast<std::int64_t>(kStreamedRows)*5120/2;
            if(weight.layout==QuantLayout::RowSplitSm75Fragment)
                unpack_weights<false,true,34816,5120><<<4096,256,0,stream>>>(low,nullptr,scales,out,pairs,row_begin);
            else unpack_weights<false,false,34816,5120><<<4096,256,0,stream>>>(low,nullptr,scales,out,pairs,row_begin);
            CUDA_CHECK(cudaGetLastError());
            checked(cublasLtMatmul(handle,p.operation,&alpha,expanded.data,p.a,input.data,p.b,
                                  &beta,destination.data,p.c,destination.data,p.c,&p.algorithm,
                                  scratch.data,scratch.bytes,stream));
        };
        const auto plan=ops::detail::q4_linear_swiglu_resolve_plan({34816,17408,5120,5120,t});
        for(int begin=0;begin<17408;begin+=kStreamedRows) {
            project_section(begin,gate);
            project_section(begin+17408,up);
            if(plan.schedule==ops::detail::Q4LinearSwiGluScheduleId::Materialized)
                streamed_swiglu<true><<<(kStreamedRows*t+255)/256,256,0,stream>>>(static_cast<const float*>(gate.data),static_cast<const float*>(up.data),static_cast<__nv_bfloat16*>(activation.data),t,begin);
            else streamed_swiglu<false><<<(kStreamedRows*t+255)/256,256,0,stream>>>(static_cast<const float*>(gate.data),static_cast<const float*>(up.data),static_cast<__nv_bfloat16*>(activation.data),t,begin);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    void streamed_value_z(const Tensor& x,const Weight& weight,Tensor& qkv,Tensor& z) {
        const int t=x.ne[1];
        constexpr int k=5120,rows=6144;
        auto scope=workspace.scope();
        Tensor expanded=workspace.alloc(DType::FP16,{k,rows});
        Tensor input=workspace.alloc(DType::FP16,{k,t});
        Tensor output=workspace.alloc(DType::FP32,{rows,t});
        const auto scratch=workspace.alloc_bytes(kLibraryWorkspace);
        to_half<<<(k*t+255)/256,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x.data),static_cast<__half*>(input.data),k*t);
        CUDA_CHECK(cudaGetLastError());
        const auto key=std::make_tuple(rows,k,t,rows);
        auto it=plans.find(key);
        if(it==plans.end())it=plans.emplace(key,std::make_unique<Plan>(handle,rows,k,t,rows)).first;
        const Plan& p=*it->second;const float alpha=1.f,beta=0.f;
        const auto* low=static_cast<const std::uint8_t*>(weight.qdata);
        const auto* high=static_cast<const std::uint8_t*>(weight.qhigh);
        const auto* scales=static_cast<const std::uint16_t*>(weight.scales);
        auto* out=static_cast<__half2*>(expanded.data);
        constexpr auto pairs=static_cast<std::int64_t>(rows)*k/2;
        for(int row_begin : {0,rows}) {
            if(weight.layout==QuantLayout::RowSplitSm75Fragment)
                unpack_weights<true,true,12288,5120><<<4096,256,0,stream>>>(low,high,scales,out,pairs,row_begin);
            else unpack_weights<true,false,12288,5120><<<4096,256,0,stream>>>(low,high,scales,out,pairs,row_begin);
            CUDA_CHECK(cudaGetLastError());
            checked(cublasLtMatmul(handle,p.operation,&alpha,expanded.data,p.a,input.data,p.b,
                                  &beta,output.data,p.c,output.data,p.c,&p.algorithm,
                                  scratch.data,scratch.bytes,stream));
            if(row_begin==0)
                streamed_value_scatter<false><<<(rows*t+255)/256,256,0,stream>>>(static_cast<const float*>(output.data),static_cast<__nv_bfloat16*>(qkv.data),static_cast<__nv_bfloat16*>(z.data),t);
            else streamed_value_scatter<true><<<(rows*t+255)/256,256,0,stream>>>(static_cast<const float*>(output.data),static_cast<__nv_bfloat16*>(qkv.data),static_cast<__nv_bfloat16*>(z.data),t);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    Tensor project(const Tensor& x,const Weight& weight) {
        const int n=weight.n,k=weight.k,t=x.ne[1];
        Tensor expanded=workspace.alloc(DType::FP16,{k,expanded_rows(n,k,t)});
        Tensor input=workspace.alloc(DType::FP16,{k,t});
        Tensor output=workspace.alloc(DType::FP32,{n,t});
        const auto scratch=workspace.alloc_bytes(kLibraryWorkspace);
        const bool q5=weight.qtype==QType::Q5G64_F16S;
        const bool fragment=weight.layout==QuantLayout::RowSplitSm75Fragment;
        const bool resident=weight.layout==QuantLayout::RowSplitSm75Radix;
        to_half<<<(k*t+255)/256,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x.data),static_cast<__half*>(input.data),k*t);
        CUDA_CHECK(cudaGetLastError());
        for (int row_begin = 0; row_begin < n; row_begin += expanded_rows(n,k,t)) {
        const int rows = std::min(expanded_rows(n,k,t), n - row_begin);
        const auto pairs=static_cast<std::int64_t>(rows)*k/2;
        const auto* low=static_cast<const std::uint8_t*>(weight.qdata);
        const auto* high=static_cast<const std::uint8_t*>(weight.qhigh);
        const auto* scales=static_cast<const std::uint16_t*>(weight.scales);
        auto* out=static_cast<__half2*>(expanded.data);
        const auto unpack=[&]<bool Q5,int N,int K>() {
            if(resident) {
                if constexpr ((!Q5 && N==4096 && K==5120) || (Q5 && N==5120 && K==6144)) {
                    ops::detail::resident_prefill::expand_weights<Q5,N,K><<<4096,256,0,stream>>>(
                        reinterpret_cast<const std::uint16_t*>(low),
                        reinterpret_cast<const std::uint32_t*>(high),scales,out,pairs,row_begin);
                } else throw std::invalid_argument("resident prefill weight shape unsupported");
            } else if(fragment) unpack_weights<Q5,true,N,K><<<4096,256,0,stream>>>(low,high,scales,out,pairs,row_begin);
            else unpack_weights<Q5,false,N,K><<<4096,256,0,stream>>>(low,high,scales,out,pairs,row_begin);
        };
        if(q5 && n==5120 && k==17408) unpack.template operator()<true,5120,17408>();
        else if(q5 && n==5120 && k==6144) {
            if(fragment) throw std::invalid_argument("output projection has no fragment format");
            unpack.template operator()<true,5120,6144>();
        }
        else if(q5 && n==12288 && k==5120) unpack.template operator()<true,12288,5120>();
        else if(!q5 && n==34816 && k==5120) unpack.template operator()<false,34816,5120>();
        else if(!q5 && n==4096 && k==5120) {
            if(fragment) throw std::invalid_argument("QK projection has no fragment format");
            unpack.template operator()<false,4096,5120>();
        }
        else if (n == 7168 && k == 5120) {
            if (fragment) throw std::invalid_argument("attention input has no fragment format");
            if (q5) unpack.template operator()<true,7168,5120>();
            else unpack.template operator()<false,7168,5120>();
        }
        else throw std::invalid_argument("unsupported prefill dense projection shape");
        CUDA_CHECK(cudaGetLastError());
        const auto key=std::make_tuple(rows,k,t,n);
        auto it=plans.find(key);
        if(it==plans.end()) it=plans.emplace(key,std::make_unique<Plan>(handle,rows,k,t,n)).first;
        const Plan& p=*it->second;const float alpha=1.f,beta=0.f;
        auto* destination = static_cast<float*>(output.data) + row_begin;
        checked(cublasLtMatmul(handle,p.operation,&alpha,expanded.data,p.a,input.data,p.b,
                              &beta,destination,p.c,destination,p.c,&p.algorithm,
                              scratch.data,scratch.bytes,stream));
        }
        return output;
    }
};
bool Sm75PrefillDense::enabled() noexcept {
    static const bool value=[] {const char* s=std::getenv("NINFER_SERVE_PREFILL_BLAS");return s&&s[0]=='1'&&s[1]=='\0';}();
    return value;
}
std::size_t Sm75PrefillDense::workspace_bytes(std::int32_t t) {
    if(t<32||t>4096) throw std::invalid_argument("SM75 dense prefill T must be 32..4096");
    WorkspaceLayoutBuilder layout;(void)layout.alloc(DType::BF16,{17408,t});
    {auto scope=layout.scope();if(streamed_prefill() && t>=1024)streamed_gate_layout(layout,t);else projection_layout(layout,34816,5120,t);}
    {auto scope=layout.scope();projection_layout(layout,5120,17408,t);}
    std::size_t peak=layout.peak_bytes(1);
    if(streamed_prefill() && t>=1024) {
        // This is an interval reservation, not only the exact endpoint shape.
        // A ragged tail below the streamed threshold uses the previous full
        // output plan and can exceed the new T1024 streamed plan.
        WorkspaceLayoutBuilder tail;
        constexpr int fallback_tokens=1023;
        (void)tail.alloc(DType::BF16,{17408,fallback_tokens});
        {auto scope=tail.scope();projection_layout(tail,34816,5120,fallback_tokens);}
        {auto scope=tail.scope();projection_layout(tail,5120,17408,fallback_tokens);}
        peak=std::max(peak,tail.peak_bytes(1));
    }
    return peak;
}
std::size_t Sm75PrefillDense::projection_workspace_bytes(int n,int k,int t) {
    if(t<kMinimumTokens||t>4096 || !((n==5120&&k==6144)||(n==12288&&k==5120)||(n==4096&&k==5120)||(n==7168&&k==5120)))
        throw std::invalid_argument("unsupported prefill projection workspace shape");
    WorkspaceLayoutBuilder layout;
    if(streamed_prefill() && t>=1024 && n==12288 && k==5120) {
        streamed_value_layout(layout,t);
        // A smaller, nonstreamed suffix retains the old full projection plan.
        WorkspaceLayoutBuilder tail;projection_layout(tail,n,k,1023);
        return std::max(layout.peak_bytes(1),tail.peak_bytes(1));
    }
    projection_layout(layout,n,k,t);return layout.peak_bytes(1);
}
namespace {
bool valid_weight(const Weight& w) noexcept {
    return w.qdata&&w.scales&&w.k>0&&w.n>0&&w.k%64==0&&w.padded_shape[1]==w.k&&
           w.scale_dtype==DType::FP16&&(w.layout==QuantLayout::RowSplit||w.layout==QuantLayout::RowSplitSm75Fragment);
}
bool valid_resident_projection(const Weight& w) noexcept {
    return w.layout==QuantLayout::RowSplitSm75Radix && w.qdata && w.scales &&
           w.scale_dtype==DType::FP16 && w.group_size==64 && w.padded_shape[1]==w.k &&
           ((w.qtype==QType::Q4G64_F16S && w.n==4096 && w.k==5120) ||
            (w.qtype==QType::Q5G64_F16S && w.n==5120 && w.k==6144 && w.qhigh));
}
}
bool Sm75PrefillDense::residual(const Tensor& x,const Weight& source,Tensor& output,cudaStream_t stream) {
    const int t=x.ne[1];
    if(t<kMinimumTokens||t>4096||source.qtype!=QType::Q5G64_F16S||source.n!=5120||source.k!=6144) return false;
    if(stream!=impl_->stream)throw std::invalid_argument("resident prefill stream mismatch");
    auto scope=impl_->workspace.scope();
    const Weight& weight=source;
    const bool resident=valid_resident_projection(weight);
    if(stream!=impl_->stream||(!resident && (!valid_weight(weight)||weight.layout!=QuantLayout::RowSplit))||!weight.qhigh||
       x.dtype!=DType::BF16||output.dtype!=DType::BF16||!x.is_contiguous()||!output.is_contiguous()||
       x.ne[0]!=6144||output.ne[0]!=5120||output.ne[1]!=t)
        throw std::invalid_argument("prefill residual projection shape or stream mismatch");
    Tensor y=impl_->project(x,weight);
    add<<<(5120*t+255)/256,256,0,stream>>>(static_cast<const float*>(y.data),static_cast<__nv_bfloat16*>(output.data),5120*t);
    CUDA_CHECK(cudaGetLastError());return true;
}
bool Sm75PrefillDense::attention_input(const Tensor& x, const Weight& qk, const Weight& gv,
                                       Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                       cudaStream_t stream) {
    const int t = x.ne[1];
    if (t < 64 || t > 4096 || qk.qtype != QType::Q4G64_F16S ||
        gv.qtype != QType::Q5G64_F16S || qk.n != 7168 || qk.k != 5120 ||
        gv.n != 7168 || gv.k != 5120) return false;
    const auto valid = [t](const Tensor& y, int rows) {
        return y.dtype == DType::BF16 && y.is_contiguous() && y.ne[0] == rows && y.ne[1] == t;
    };
    if (stream != impl_->stream || !valid_weight(qk) || !valid_weight(gv) || !gv.qhigh ||
        qk.layout != QuantLayout::RowSplit || gv.layout != QuantLayout::RowSplit ||
        !valid(x,5120) || !valid(q,6144) || !valid(gate,6144) || !valid(k,1024) || !valid(v,1024))
        throw std::invalid_argument("prefill attention input shape or stream mismatch");
    // Preserve separate request-local projections; MRoPE, normalization, KV and
    // attention itself remain in the existing family runtime.
    { auto scope = impl_->workspace.scope(); Tensor y = impl_->project(x,qk);
      attention_scatter<<<(7168*t+255)/256,256,0,stream>>>(static_cast<const float*>(y.data),
          static_cast<__nv_bfloat16*>(q.data),static_cast<__nv_bfloat16*>(k.data),t);
      CUDA_CHECK(cudaGetLastError()); }
    { auto scope = impl_->workspace.scope(); Tensor y = impl_->project(x,gv);
      attention_scatter<<<(7168*t+255)/256,256,0,stream>>>(static_cast<const float*>(y.data),
          static_cast<__nv_bfloat16*>(gate.data),static_cast<__nv_bfloat16*>(v.data),t);
      CUDA_CHECK(cudaGetLastError()); }
    return true;
}
bool Sm75PrefillDense::gdn_input(const Tensor& x,const Weight& qk,const Weight& vz,
                                 Tensor& qkv,Tensor& z,cudaStream_t stream) {
    const int t=x.ne[1];
    if(t<kMinimumTokens||t>4096||qk.qtype!=QType::Q4G64_F16S||vz.qtype!=QType::Q5G64_F16S||
       qk.n!=4096||qk.k!=5120||vz.n!=12288||vz.k!=5120) return false;
    const bool resident=valid_resident_projection(qk);
    if(stream!=impl_->stream||(!resident && (!valid_weight(qk)||qk.layout!=QuantLayout::RowSplit))||!valid_weight(vz)||!vz.qhigh||
       x.dtype!=DType::BF16||qkv.dtype!=DType::BF16||z.dtype!=DType::BF16||
       !x.is_contiguous()||!qkv.is_contiguous()||!z.is_contiguous()||x.ne[0]!=5120||qkv.ne[0]!=10240||z.ne[0]!=6144||qkv.ne[1]!=t||z.ne[1]!=t)
        throw std::invalid_argument("prefill GDN projection shape or stream mismatch");
    {auto scope=impl_->workspace.scope();Tensor y=impl_->project(x,qk);
     gdn_scatter<<<(4096*t+255)/256,256,0,stream>>>(static_cast<const float*>(y.data),static_cast<__nv_bfloat16*>(qkv.data),static_cast<__nv_bfloat16*>(z.data),t,false);CUDA_CHECK(cudaGetLastError());}
    if(streamed_prefill() && t>=1024) impl_->streamed_value_z(x,vz,qkv,z);
    else {auto scope=impl_->workspace.scope();Tensor y=impl_->project(x,vz);
     gdn_scatter<<<(12288*t+255)/256,256,0,stream>>>(static_cast<const float*>(y.data),static_cast<__nv_bfloat16*>(qkv.data),static_cast<__nv_bfloat16*>(z.data),t,true);CUDA_CHECK(cudaGetLastError());}
    return true;
}
Sm75PrefillDense::Sm75PrefillDense(WorkspaceArena& workspace,cudaStream_t stream):impl_(std::make_unique<Impl>(workspace,stream)){}
Sm75PrefillDense::~Sm75PrefillDense()=default;
bool Sm75PrefillDense::execute(std::int32_t,const Tensor& x,const Weight& gate,
                              const Weight& down,Tensor& residual,cudaStream_t stream) {
    const int t=x.ne[1];
    if(t<kMinimumTokens||t>4096||gate.qtype!=QType::Q4G64_F16S||down.qtype!=QType::Q5G64_F16S) return false;
    const auto supported=[](const Weight& w){return w.layout==QuantLayout::RowSplit||w.layout==QuantLayout::RowSplitSm75Fragment;};
    if(stream!=impl_->stream||x.dtype!=DType::BF16||residual.dtype!=DType::BF16||
       x.ne[0]!=5120||residual.ne[0]!=5120||residual.ne[1]!=t||!x.is_contiguous()||!residual.is_contiguous()||
       gate.n!=34816||gate.k!=5120||down.n!=5120||down.k!=17408||!supported(gate)||!supported(down)||
       !valid_weight(gate)||!valid_weight(down)||!down.qhigh)
        throw std::invalid_argument("SM75 prefill MLP shape, source, or stream mismatch");
    auto scope=impl_->workspace.scope();
    Tensor activation=impl_->workspace.alloc(DType::BF16,{17408,t});
    if(streamed_prefill() && t>=1024) impl_->streamed_gate(x,gate,activation);
    else {auto weight_scope=impl_->workspace.scope();Tensor y=impl_->project(x,gate);
     // Preserve the native operator's actual materialized/fused rounding boundary,
     // including ragged tails, rather than extending its T128 casting to all widths.
     const auto plan=ops::detail::q4_linear_swiglu_resolve_plan({34816,17408,5120,5120,t});
     if(plan.schedule==ops::detail::Q4LinearSwiGluScheduleId::Materialized)
         swiglu<true><<<(17408*t+255)/256,256,0,stream>>>(static_cast<const float*>(y.data),static_cast<__nv_bfloat16*>(activation.data),17408*t);
     else
         swiglu<false><<<(17408*t+255)/256,256,0,stream>>>(static_cast<const float*>(y.data),static_cast<__nv_bfloat16*>(activation.data),17408*t);
     CUDA_CHECK(cudaGetLastError());}
    {auto weight_scope=impl_->workspace.scope();Tensor y=impl_->project(activation,down);
     add<<<(5120*t+255)/256,256,0,stream>>>(static_cast<const float*>(y.data),static_cast<__nv_bfloat16*>(residual.data),5120*t);CUDA_CHECK(cudaGetLastError());}
    return true;
}
}
