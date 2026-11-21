#include <stdexcept>
// ninfer::ops - gqa_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include "../../../.chatgpt/production_serve_20260918/proposal_page_retrieval/retrieval.cuh"
#include "../../../.chatgpt/production_serve_20260918/target_guided_proposal/guide_view.cuh"

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                               float scale, const CacheView& cache,
                                               Metadata metadata, Tensor& out,
                                               cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    // The dynamic-smem opt-in is device-local. A process may execute this template on more than
    // one GPU, so a function-static one-time result is incorrect: it configures only the first
    // current device and leaves later devices at the default 48 KiB ceiling.
    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    if (cache.dtype == DType::I8) {
        CUDA_CHECK(cudaFuncSetAttribute(gqa_attention_prefill_i8_kernel<Geometry, Metadata>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kGqaPrefillI8SmemBytes));
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillI8Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        gqa_attention_prefill_i8_kernel<Geometry, Metadata>
            <<<attention_grid, kGqaPrefillI8Threads, kGqaPrefillI8SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache_k.data),
                static_cast<const std::int8_t*>(cache_v.data),
                static_cast<const __half*>(cache_k_scale.data),
                static_cast<const __half*>(cache_v_scale.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    } else {
        CUDA_CHECK(cudaFuncSetAttribute(gqa_attention_prefill_bf16_kernel<Geometry, Metadata>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kGqaPrefillSmemBytes));
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        gqa_attention_prefill_bf16_kernel<Geometry, Metadata>
            <<<attention_grid, kGqaPrefillThreads, kGqaPrefillSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(cache_k.data),
                static_cast<const __nv_bfloat16*>(cache_v.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_kv_append_launch_for(const Tensor& k, const Tensor& v, const Tensor& positions,
                              CacheView cache, Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    Tensor& cache_k   = cache.k_pages;
    Tensor& cache_v   = cache.v_pages;
    if (cache.dtype == DType::I8) {
        Tensor& cache_k_scale    = cache.k_scale_pages;
        Tensor& cache_v_scale    = cache.v_scale_pages;
        constexpr int kFillBlock = 256;
        if (tokens >= 128 && Geometry::KVHeads == 2) {
            constexpr int kPageBlock     = 256;
            constexpr int kTokensPerTile = 8;
            const int max_tiles          = div_up(tokens + kTokensPerTile - 1, kTokensPerTile);
            const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                                 static_cast<unsigned>(Geometry::KVHeads),
                                 static_cast<unsigned>(kGqaKvQuantGroups));
            gqa_attention_prefill_fill_i8_page_kernel<Geometry, Metadata>
                <<<fill_grid, kPageBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::int8_t*>(cache_k.data),
                    static_cast<std::int8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        } else {
            constexpr int kFillWarps = kFillBlock / 32;
            const std::int64_t fill_units =
                static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kGqaKvQuantGroups;
            const int fill_grid =
                static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kFillWarps)));
            gqa_attention_prefill_fill_i8_kernel<Geometry, Metadata>
                <<<fill_grid, kFillBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::int8_t*>(cache_k.data),
                    static_cast<std::int8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        }
        CUDA_CHECK(cudaGetLastError());
    } else {
        constexpr int kBlock           = Geometry::KVHeads == 4 ? 128 : 96;
        constexpr int kFillVecElems    = 8;
        const std::int64_t kv_elements = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                                         (kGqaPrefillHeadDim / kFillVecElems);
        const int fill_grid =
            static_cast<int>(div_up(kv_elements, static_cast<std::int64_t>(kBlock)));
        gqa_attention_prefill_fill_bf16_kernel<Geometry, Metadata>
            <<<fill_grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(k.data),
                                               static_cast<const __nv_bfloat16*>(v.data),
                                               static_cast<const std::int32_t*>(positions.data),
                                               metadata, static_cast<__nv_bfloat16*>(cache_k.data),
                                               static_cast<__nv_bfloat16*>(cache_v.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, Tensor& out,
                                           cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_prompt_attention_launch_for<Gqa27Geometry>(q, positions, scale, cache,
                                                                 metadata, out, stream);
        return;
    }
    gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(q, positions, scale, cache, metadata,
                                                             out, stream);
}

namespace {
__global__ void build_recent_view(const std::int32_t* positions,const std::int32_t* source,
    const std::int32_t* source_row,std::int32_t source_stride,std::int32_t source_rows,
    std::int32_t width,std::int32_t recent,std::int32_t sink,
    std::int32_t* local_positions,std::int32_t* tables,std::int32_t pages,std::int32_t* row) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    constexpr int Page=64;
    const int selected=*source_row;
    const int first=positions[0];
    const int tail_page=max(sink/Page,max(0,first+1-recent)/Page);
    if(i<width)local_positions[i]=positions[i]<sink ? positions[i] : positions[i]-tail_page*Page+sink;
    if(i<pages) {
        const int original=i<sink/Page ? i : tail_page+i-sink/Page;
        tables[i]=selected>=0&&selected<source_rows&&original>=0&&original<source_stride
            ? source[static_cast<long long>(selected)*source_stride+original] : -1;
    }
    if(i==0)*row=0;
}
}
void gqa_recent_view_launch(const Tensor& positions,const Tensor& source_tables,
    const Tensor& source_row,std::uint32_t recent_tokens,std::uint32_t sink_tokens,
    Tensor& local_positions,Tensor& local_tables,Tensor& local_row,cudaStream_t stream) {
    if(recent_tokens<128 || recent_tokens%64 || sink_tokens%64 ||
       positions.dtype!=DType::I32 || positions.ne[0]<1 || positions.ne[0]>6 ||
       source_tables.dtype!=DType::I32 || source_row.dtype!=DType::I32 ||
       local_positions.dtype!=DType::I32 || local_positions.ne[0]!=positions.ne[0] ||
       local_tables.dtype!=DType::I32 || local_tables.ne[0]!=static_cast<int>((recent_tokens+sink_tokens)/64+2) ||
       local_tables.ne[1]!=1 || local_row.dtype!=DType::I32) {
        throw std::invalid_argument("invalid request-local MTP attention window view");
    }
    const int count=local_tables.ne[0];
    build_recent_view<<<(count+127)/128,128,0,stream>>>(
        static_cast<const std::int32_t*>(positions.data),static_cast<const std::int32_t*>(source_tables.data),
        static_cast<const std::int32_t*>(source_row.data),source_tables.ne[0],source_tables.ne[1],positions.ne[0],
        recent_tokens,sink_tokens,static_cast<std::int32_t*>(local_positions.data),
        static_cast<std::int32_t*>(local_tables.data),count,static_cast<std::int32_t*>(local_row.data));
    CUDA_CHECK(cudaGetLastError());
}

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (k.ne[1] == Gqa27Geometry::KVHeads) {
        gqa_kv_append_launch_for<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
        return;
    }
    gqa_kv_append_launch_for<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
}

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const GqaPrefillBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == Gqa27Geometry::QHeads) {
            gqa_kv_append_launch_for<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
            gqa_attention_prompt_attention_launch_for<Gqa27Geometry>(q, positions, scale, cache,
                                                                     metadata, out, stream);
            return;
        }
        gqa_kv_append_launch_for<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
        gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(q, positions, scale, cache,
                                                                 metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
