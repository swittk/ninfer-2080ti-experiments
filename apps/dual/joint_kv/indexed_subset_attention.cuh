#pragma once
#include <stdexcept>

// ninfer::ops - split-KV GQA small-T attention, int8 KV-cache partial kernel.
// Historical design: docs/archive/optimization-era/2026-07-08-gqa-decode-int8-kernel-redesign.md.
//
//   * QK runs on native m16n8k32.s8 tensor cores. Q is quantized on-chip to int8
//     per (row, 64-group); K stays int8 in the cache and is read straight into
//     smem (no dequant). The int32 MMA output is rescaled per 64-group by
//     qs[row,g]*ks[key,g]. This halves the QK MMA count vs bf16 and removes the
//     entire K dequant.
//   * PV stays bf16 (V is quantized per key, so its scale cannot be factored out
//     of a key-contracted int8 accumulation): V int8 is staged, dequanted once to
//     a bf16 tile, then the existing bf16 PV MMA runs. V is still read from DRAM
//     as int8, so the bandwidth win is kept.
//   * All keys (history AND the current/diagonal tokens) are read from the
//     quantized cache; the fused append writes the new tokens first and a
//     __syncthreads orders the in-block readback. No from_new special-casing.
//
// Standalone from the bf16 kernel; shared scaffolding (layout constants, ldmatrix
// helpers, the s8/bf16 MMA helpers, the reducer) lives in gqa_attention_decode.cuh.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cstdint>

namespace ninfer::ops::value_residual_indexed_subset {

// Bc16 probability-tile XOR swizzle. Bc32 uses two row bits; Bc16 uses one
// so each address stays inside its 16-column row while preserving ldmatrix layout.
__device__ __forceinline__ __half gqa_bf16_rounded_f16(float value) {
    const __nv_bfloat16 b = float_to_bf16_rn(value);
    return __float2half_rn(__bfloat162float(b));
}

__device__ __forceinline__ int4 gqa_kv_dequant_i8x8_to_f16_from(
    const std::int8_t* codes8, float scale) {
    const int4 bf = gqa_kv_dequant_i8x8_from(codes8, scale);
    return make_int4(static_cast<int>(bf16x2_to_f16x2(static_cast<unsigned>(bf.x))),
                     static_cast<int>(bf16x2_to_f16x2(static_cast<unsigned>(bf.y))),
                     static_cast<int>(bf16x2_to_f16x2(static_cast<unsigned>(bf.z))),
                     static_cast<int>(bf16x2_to_f16x2(static_cast<unsigned>(bf.w))));
}

__device__ __forceinline__ int4 gqa_value_packed_fp16(
    const std::int8_t* codes8,float scale) {
    const uint2 raw=*reinterpret_cast<const uint2*>(codes8);
    const unsigned x=raw.x^0x80808080u,y=raw.y^0x80808080u;
    union Pair {unsigned u;__half2 h;};
    Pair a{__byte_perm(x,0x64646464u,0x4140)},b{__byte_perm(x,0x64646464u,0x4342)};
    Pair c{__byte_perm(y,0x64646464u,0x4140)},d{__byte_perm(y,0x64646464u,0x4342)};
    const __half2 bias=__float2half2_rn(1152.f),factor=__float2half2_rn(scale);
    a.h=__hmul2(__hsub2(a.h,bias),factor);b.h=__hmul2(__hsub2(b.h,bias),factor);
    c.h=__hmul2(__hsub2(c.h,bias),factor);d.h=__hmul2(__hsub2(d.h,bias),factor);
    return make_int4(a.u,b.u,c.u,d.u);
}

template <int Columns>
__device__ __forceinline__ int gqa_small_t_probability_swz(int row, int col) {
    static_assert(Columns == 16 || Columns == 32 || Columns == 64);
    if constexpr (Columns == 16) {
        return (((col >> 3) ^ (row & 1)) << 3) | (col & 7);
    } else {
        return gqa_small_t_tc_swz32(row, col);
    }
}

// Store one int8 code into a d-contiguous-as-b16 swizzled tile so the same
// gqa_small_t_tc_swz / ldmatrix path that serves bf16 tiles serves the int8 tile.
// A b16 lane holds two packed int8 (d even = low byte, d odd = high byte); this
// matches the byte layout a 16 B cp.async of d-contiguous cache bytes produces
// (see the design doc / kernel comments), so Q (byte stores) and K (cp.async)
// agree.
__device__ __forceinline__ void gqa_small_t_i8_store_swz(std::int8_t* tile, int row, int d,
                                                         int d_b16_stride, std::int8_t code) {
    const int c   = d >> 1;
    const int lo  = d & 1;
    const int off = (row * d_b16_stride + gqa_small_t_tc_swz(row, c)) * 2 + lo;
    tile[off]     = code;
}

// Decode-specialized producer/consumer kernel for T=1..6. One producer warp per
// m16 row tile computes QK + online softmax, while all CTA warps partition the
// tile's 256-wide PV output. This keeps each thread's PV accumulator at 16, 32,
// or 64 floats instead of 128 and uses otherwise-idle warps for useful output
// work.
//
// Q has a dedicated shared tile so producers can reload one 64-dimension group
// at a time. K/V codes and scales are staged asynchronously; non-producer warps
// dequantize V while producers execute QK. After both consume the code tile, the
// next K/V tile is prefetched into the same arena while the current PV runs.
template <typename Geometry, int TokenTile, int WarpsPerCta, int MinBlocksPerSm, int KeyBlock,
          bool DynamicArena, bool MultiBatch, bool Masked, typename CacheInput, bool ValueFp16 = false>
__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm) __global__
    void gqa_attention_headscale_qk_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
        std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, float scale,
        std::int32_t split_offset, std::int32_t total_split_count,
        std::int32_t launch_base_split_count, std::int32_t split_subdivide,
        std::int32_t partition_capacity,
        __nv_bfloat16* partial_acc, float* partial_m, float* partial_l, const int* active_heads = nullptr, const int* union_indices = nullptr, const int* union_counts = nullptr, int old_end = 0, const int* source_table = nullptr) {
    if(active_heads != nullptr && active_heads[blockIdx.x] != 0) return;
    pos += static_cast<std::size_t>(blockIdx.x) * TokenTile;
    constexpr int Wc                   = WarpsPerCta;
    constexpr int RowCount             = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCount + 15) / 16;
    constexpr int Br                   = RowTiles * 16;
    constexpr int Bc                   = KeyBlock;
    constexpr int D                    = kGqaHeadDim;
    constexpr int DB16                 = D / 2;
    constexpr int Threads              = Wc * 32;
    constexpr int Groups               = kGqaKvQuantGroups;
    constexpr int GroupKc              = kGqaKvQuantGroup / 32;
    constexpr int QKKs                 = D / 32;
    constexpr int QKNt                 = Bc / 8;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp          = D / (ConsumerWarpsPerTile * 8);
    constexpr int PVKs                 = Bc / 16;
    // The GQA Op's 262144-key maximum envelope spans at most 49 pages in one 27B split.
    constexpr int PageIds         = 64;
    constexpr int ProducerThreads = RowTiles * 32;
    constexpr int VLoaderThreads  = Threads - ProducerThreads;
    constexpr float Log2E         = 1.4426950408889634074f;
    constexpr unsigned FullMask   = 0xffffffffu;

    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(Bc == 16 || Bc == 32 || Bc == 64);
    static_assert(RowTiles >= 1 && RowTiles <= 3);
    static_assert(Wc % RowTiles == 0);
    static_assert(PVNtPerWarp == 2 || PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);
    static_assert(QKKs == Groups * GroupKc);

    // Keep Q in a compact dedicated tile so the producer can reload one
    // 64-dimension group at a time instead of carrying all eight fragments in
    // registers across the whole kernel. The main arena holds K i8, V i8, and
    // V bf16 during the key loop.
    __shared__ __align__(16) std::int8_t q_s[Br * D];
    __shared__ __align__(16) std::int8_t static_r_s[DynamicArena ? 16 : 4 * Bc * D];
    extern __shared__ __align__(16) std::int8_t dynamic_r_s[];
    std::int8_t* r_s      = DynamicArena ? dynamic_r_s : static_r_s;
    std::int8_t* q_i8     = q_s;
    float* q_scale_tmp    = reinterpret_cast<float*>(r_s);
    std::int8_t* k_i8     = r_s;
    __nv_bfloat16* q_b16  = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16  = reinterpret_cast<__nv_bfloat16*>(k_i8);
    std::int8_t* v_i8     = r_s + Bc * D;
    __half* v_f16 = reinterpret_cast<__half*>(r_s + 2 * Bc * D);
    __shared__ __align__(16) __half p_s[Br * Bc];
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) __half k_scale_s[Bc * Groups];
    __shared__ __align__(16) __half v_scale_s[Bc * Groups];
    __shared__ std::int32_t physical_pages_s[PageIds];

    const int kv_head       = static_cast<int>(blockIdx.x);
    const int local_split   = static_cast<int>(blockIdx.y);
    const int subdivision   = split_subdivide > 0 ? split_subdivide : 1;
    const int split         = split_offset * subdivision + local_split;
    const int batch         = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count   = total_split_count * subdivision;
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads *
                       TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = 0.0f;
            }
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, TokenTile)] =
                    float_to_bf16_rn(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) { return; }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    // Physical ownership is fixed at startup. Only the causal endpoint moves.
    const int partition_window=partition_capacity>0 ? partition_capacity : window;
    const int base_active_split_count =
        gqa_small_t_active_splits<Geometry, true>(partition_window, total_split_count, TokenTile);

    constexpr int SplitTile = (Bc == 16 && !CacheInput::writes_cache) ? 32 : Bc;
    const int logical_tiles = div_up(partition_window, SplitTile);
    const bool base_tile_split = logical_tiles >= base_active_split_count;
    const int base_units_per_split = base_tile_split
        ? div_up(logical_tiles, base_active_split_count)
        : div_up(partition_window, base_active_split_count);
    const int base_unit_keys = base_units_per_split * (base_tile_split ? SplitTile : 1);
    const int base_local_split = local_split / subdivision;
    const int sub_split = local_split - base_local_split * subdivision;
    if (base_local_split >= launch_base_split_count) {
        write_neutral();
        return;
    }
    const int base_split = split_offset + base_local_split;
    const int base_begin = base_split * base_unit_keys;
    const int requested_base_end = base_begin + base_unit_keys;
    const int base_end = requested_base_end < window ? requested_base_end : window;
    if (base_split >= base_active_split_count || base_begin >= base_end) {
        write_neutral();
        return;
    }

    const int range_tiles = div_up(base_end - base_begin, SplitTile);
    const int active_local_splits = range_tiles < subdivision ? range_tiles : subdivision;
    if (sub_split >= active_local_splits) {
        write_neutral();
        return;
    }
    const int local_units = div_up(range_tiles, active_local_splits);
    const int split_start = base_begin + sub_split * local_units * SplitTile;
    const int split_limit = split_start + local_units * SplitTile;
    const int split_end = split_limit < base_end ? split_limit : base_end;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {
        // The owning split quantizes each current row before its cache tile is consumed.
        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            int physical_page       = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kGqaKvQuantGroup + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = gqa_kv_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = gqa_kv_new_index<Geometry>(kv_head, d1, token);
            const float kv0         = __bfloat162float(input.k[src0]);
            const float kv1         = __bfloat162float(input.k[src1]);
            const float vv0         = __bfloat162float(input.v[src0]);
            const float vv1         = __bfloat162float(input.v[src1]);
            float kamax             = fmaxf(fabsf(kv0), fabsf(kv1));
            float vamax             = fmaxf(fabsf(vv0), fabsf(vv1));
            kamax                   = warp_max(kamax, FullMask);
            vamax                   = warp_max(vamax, FullMask);
            const __half ksh        = __float2half_rn(kamax > 0.0f ? kamax / 127.0f : 0.0f);
            const __half vsh        = __float2half_rn(vamax > 0.0f ? vamax / 127.0f : 0.0f);
            const float ks          = __half2float(ksh);
            const float vs          = __half2float(vsh);
            const float k_inv       = ks > 0.0f ? 1.0f / ks : 0.0f;
            const float v_inv       = vs > 0.0f ? 1.0f / vs : 0.0f;
            physical_page           = __shfl_sync(FullMask, physical_page, 0);
            cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
                gqa_kv_quant_code(kv0, k_inv);
            cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
                gqa_kv_quant_code(kv1, k_inv);
            cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
                gqa_kv_quant_code(vv0, v_inv);
            cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
                gqa_kv_quant_code(vv1, v_inv);
            if (lane == 0) {
                const std::int64_t so =
                    gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, grp, page_offset);
                cache_k_scale[so] = ksh;
                cache_v_scale[so] = vsh;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < Br * D; i += Threads) { q_i8[i] = 0; }
    for (int i = tid; i < RowCount * Groups; i += Threads) { q_scale_tmp[i] = 0.0f; }
    __syncthreads();

    for(int row=warp;row<RowCount;row+=Wc) {
        int q_head=0,token=0;
        gqa_small_t_tc_row_to_qt<Geometry>(row,TokenTile,kv_head,q_head,token);
        float values[D/32],amax=0.f;
#pragma unroll
        for(int j=0;j<D/32;++j){
            values[j]=__bfloat162float(q[gqa_q_index<Geometry>(q_head,lane+j*32,token)]);
            amax=fmaxf(amax,fabsf(values[j]));
        }
        amax=warp_max(amax,FullMask);
        const float qs=amax>0.f?amax/127.f:0.f,inv=qs>0.f?1.f/qs:0.f;
#pragma unroll
        for(int j=0;j<D/32;++j)
            gqa_small_t_i8_store_swz(q_i8,row,lane+j*32,DB16,gqa_kv_quant_code(values[j],inv));
        if(lane==0){for(int g=0;g<Groups;++g)q_scale_tmp[row*Groups+g]=qs;}
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    float q_scale_r0[Groups];
    float q_scale_r1[Groups];
    if (warp < RowTiles) {
        const int producer_row0 = warp * 16 + gid;
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
            float qs0     = (lid == 0 && producer_row0 < RowCount)
                                ? q_scale_tmp[producer_row0 * Groups + g]
                                : 0.0f;
            float qs1     = (lid == 0 && producer_row0 + 8 < RowCount)
                                ? q_scale_tmp[(producer_row0 + 8) * Groups + g]
                                : 0.0f;
            q_scale_r0[g] = __shfl_sync(FullMask, qs0, gid * 4);
            q_scale_r1[g] = __shfl_sync(FullMask, qs1, gid * 4);
        }
    }
    __syncthreads();

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    auto issue_kv_tile = [&](int tile_k0, int physical_page) {
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                const int source = key < union_counts[kv_head]
                    ? union_indices[kv_head * 65536 + key] : old_end + key - union_counts[kv_head];
                const std::int64_t off = gqa_kv_quant_scale_index<Geometry>(
                    source_table[source / 64], kv_head, 0, source & kPagedKVPageMask);
                ninfer::ops::cp_async<8>(&k_scale_s[key_l * Groups], &cache_k_scale[off]);
                ninfer::ops::cp_async<8>(&v_scale_s[key_l * Groups], &cache_v_scale[off]);
            } else {
                store_vec(&k_scale_s[key_l * Groups], make_int2(0, 0));
                store_vec(&v_scale_s[key_l * Groups], make_int2(0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                const int source = key < union_counts[kv_head]
                    ? union_indices[kv_head * 65536 + key] : old_end + key - union_counts[kv_head];
                const std::int64_t off = gqa_kv_quant_code_index<Geometry>(
                    source_table[source / 64], kv_head, d, source & kPagedKVPageMask);
                std::int8_t* dst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                ninfer::ops::cp_async<16>(dst, &cache_k_i8[off]);
                ninfer::ops::cp_async<16>(&v_i8[key_l * D + d], &cache_v_i8[off]);
            } else {
                std::int8_t* dst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                store_vec(dst, make_int4(0, 0, 0, 0));
                store_vec(&v_i8[key_l * D + d], make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    int physical_page = physical_pages_s[0];
    issue_kv_tile(first_tile, physical_page);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;

        // One warp per row tile produces P and alpha while the remaining warps
        // stream/dequant V.
        if (warp < RowTiles) {
            const int producer_row_base = warp * 16;
            __half* p_sw                = &p_s[producer_row_base * Bc];
            int sums[QKNt][4];
#pragma unroll
            for(int nt=0;nt<QKNt;++nt)for(int i=0;i<4;++i)sums[nt][i]=0;
#pragma unroll
            for(int g=0;g<Groups;++g){
                unsigned af[GroupKc][4];
#pragma unroll
                for(int kk=0;kk<GroupKc;++kk){
                    const int k=g*GroupKc+kk,acol=k*16+a_coloff;
                    ldmatrix_x4(af[kk][0],af[kk][1],af[kk][2],af[kk][3],
                        smem_addr(&q_b16[(producer_row_base+a_rowoff)*DB16+
                            gqa_small_t_tc_swz(producer_row_base+a_rowoff,acol)]));
                }
#pragma unroll
                for(int nt=0;nt<QKNt;++nt){
#pragma unroll
                    for(int kk=0;kk<GroupKc;++kk){
                        const int k=g*GroupKc+kk,brow=nt*8+b_rin,bcol=k*16+b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(bf[0],bf[1],smem_addr(&k_b16[brow*DB16+gqa_small_t_tc_swz(brow,bcol)]));
                        mma_s8(sums[nt][0],sums[nt][1],sums[nt][2],sums[nt][3],
                               af[kk][0],af[kk][1],af[kk][2],af[kk][3],bf[0],bf[1]);
                    }
                }
            }
            float score[QKNt][4];
#pragma unroll
            for(int nt=0;nt<QKNt;++nt){
                const int keya=nt*8+2*lid,keyb=keya+1;
                float ka=0.f,kb2=0.f;
                if(gid==0){ka=__half2float(k_scale_s[keya*Groups]);kb2=__half2float(k_scale_s[keyb*Groups]);}
                ka=__shfl_sync(FullMask,ka,lid);kb2=__shfl_sync(FullMask,kb2,lid);
                score[nt][0]=q_scale_r0[0]*ka*static_cast<float>(sums[nt][0]);
                score[nt][1]=q_scale_r0[0]*kb2*static_cast<float>(sums[nt][1]);
                score[nt][2]=q_scale_r1[0]*ka*static_cast<float>(sums[nt][2]);
                score[nt][3]=q_scale_r1[0]*kb2*static_cast<float>(sums[nt][3]);
            }

            const int row0 = producer_row_base + gid;
            const int row1 = row0 + 8;
            int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head0, token0);
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head1, token1);
            const int qabs0 = (row0 < RowCount) ? pos[token0] : -1;
            const int qabs1 = (row1 < RowCount) ? pos[token1] : -1;
            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = k0 + col0;
                const int key1 = k0 + col1;
                score[nt][0] =
                    (row0 < RowCount && key0 >= split_start && key0 < split_end && key0 <= qabs0)
                        ? score[nt][0] * scale
                        : -CUDART_INF_F;
                score[nt][1] =
                    (row0 < RowCount && key1 >= split_start && key1 < split_end && key1 <= qabs0)
                        ? score[nt][1] * scale
                        : -CUDART_INF_F;
                score[nt][2] =
                    (row1 < RowCount && key0 >= split_start && key0 < split_end && key0 <= qabs1)
                        ? score[nt][2] * scale
                        : -CUDART_INF_F;
                score[nt][3] =
                    (row1 < RowCount && key1 >= split_start && key1 < split_end && key1 <= qabs1)
                        ? score[nt][3] * scale
                        : -CUDART_INF_F;
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            const float nm0    = fmaxf(m0, bm0);
            const float nm1    = fmaxf(m1, bm1);
            const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
            const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

            float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                      ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                      : 0.0f;
                const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                      ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                      : 0.0f;
                const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                      ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                      : 0.0f;
                const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                      ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_sw[gid * Bc + gqa_small_t_probability_swz<Bc>(gid, col0)]           = gqa_bf16_rounded_f16(p00);
                p_sw[gid * Bc + gqa_small_t_probability_swz<Bc>(gid, col1)]           = gqa_bf16_rounded_f16(p01);
                p_sw[(gid + 8) * Bc + gqa_small_t_probability_swz<Bc>(gid + 8, col0)] = gqa_bf16_rounded_f16(p10);
                p_sw[(gid + 8) * Bc + gqa_small_t_probability_swz<Bc>(gid + 8, col1)] = gqa_bf16_rounded_f16(p11);
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);

            l0 = l0 * alpha0 + bl0;
            l1 = l1 * alpha1 + bl1;
            m0 = nm0;
            m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else {
            const int loader_tid = tid - ProducerThreads;
#pragma unroll 1
            for (int chunk = loader_tid; chunk < Bc * (D / 8); chunk += VLoaderThreads) {
                const int key_l    = chunk / (D / 8);
                const int dc       = chunk - key_l * (D / 8);
                const int d        = dc * 8;
                const int key      = k0 + key_l;
                __half* dst = &v_f16[key_l * D + gqa_small_t_tc_swz(key_l, d)];
                if (key >= split_start && key < split_end) {
                    const int grp = d >> 6;
                    float vs      = 0.0f;
                    if ((lane & 7) == 0) { vs = __half2float(v_scale_s[key_l * Groups + grp]); }
                    vs = __shfl_sync(FullMask, vs, grp * 8);
                    if constexpr (ValueFp16) {
                        // The stored KV representation is unchanged. This opt-in
                        // path rounds directly to FP16 instead of BF16 then FP16.
                        store_vec(dst,gqa_value_packed_fp16(&v_i8[key_l*D+d],vs));
                    } else {
                    store_vec(dst, gqa_kv_dequant_i8x8_to_f16_from(&v_i8[key_l * D + d], vs));
                    }
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) {
            const int next_k0 = k0 + Bc;
            if ((next_k0 & kPagedKVPageMask) == 0) {
                physical_page = physical_pages_s[(next_k0 >> kPagedKVPageShift) - first_page];
            }
            issue_kv_tile(next_k0, physical_page);
        }

        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        __half* p_consumer          = &p_s[consumer_row_base * Bc];
        const float alpha0          = alpha_s[consumer_row_base + gid];
        const float alpha1          = alpha_s[consumer_row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(
                pf[0], pf[1], pf[2], pf[3],
                smem_addr(&p_consumer[a_rowoff * Bc +
                                      gqa_small_t_probability_swz<Bc>(a_rowoff, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = consumer_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                        pf[0], pf[1], pf[2], pf[3], vf[0], vf[1]);
#else
                // This diagnostic is SM75-only. Preserve compilability for
                // other architectures by treating staged FP16 as FP16 MMA.
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                        pf[0], pf[1], pf[2], pf[3], vf[0], vf[1]);
#endif
            }
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
    }

    if (warp < RowTiles && lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l0;
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l1;
        }
    }

#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        const int d0                = (consumer_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        const int row0              = consumer_row_base + gid;
        const int row1              = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<unsigned*>(&partial_acc[dst]) = pack_bf16x2(acc[n][0], acc[n][1]);
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<unsigned*>(&partial_acc[dst]) = pack_bf16x2(acc[n][2], acc[n][3]);
        }
    }
}

} // namespace ninfer::ops::value_residual_indexed_subset

namespace ninfer::ops::value_residual_indexed_subset {
__global__ void requantize_paged_keys(std::int8_t* codes,__half* scales,const std::int32_t* table,
    const std::int32_t* positions,int count,int prefix,int capacity,int physical_pages) {
    const int lane=threadIdx.x&31,u=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(u>=count*Gqa27Geometry::KVHeads)return;
    const int token=positions?positions[u/Gqa27Geometry::KVHeads]:u/Gqa27Geometry::KVHeads;
    const int head=u%Gqa27Geometry::KVHeads;
    if(token<0 || token>=capacity || (!positions&&token>=prefix))return;
    const int page=table[token/kPagedKVPageSize];
    if(page<0 || page>=physical_pages)return;
    const auto si=gqa_kv_quant_scale_index<Gqa27Geometry>(page,head,0,token&kPagedKVPageMask);
    const auto vi=gqa_kv_quant_code_index<Gqa27Geometry>(page,head,0,token&kPagedKVPageMask);
    float scale=lane<4?__half2float(scales[si+lane]):0.f;
    for(int offset=16;offset;offset>>=1)scale=fmaxf(scale,__shfl_xor_sync(0xffffffffu,scale,offset));
    const float inverse=scale>0.f?1.f/scale:0.f;
    float old[4];
#pragma unroll
    for(int g=0;g<4;++g)old[g]=__half2float(scales[si+g]);
    __syncwarp();
#pragma unroll
    for(int j=0;j<8;++j){
        const int channel=lane+j*32;
        codes[vi+channel]=static_cast<std::int8_t>(max(-127,min(127,
            __float2int_rn(static_cast<float>(codes[vi+channel])*old[j/2]*inverse))));
    }
    __syncwarp();
    if(lane<4)scales[si+lane]=__float2half_rn(scale);
}
inline void prepare_headscale_keys(const PagedKVLayerView& cache,const Tensor* positions,
                                  std::uint32_t prefix,cudaStream_t stream) {
    const int count=positions?positions->ne[0]:static_cast<int>(prefix);
    if(count==0)return;
    if(cache.dtype!=DType::I8 || cache.head_dim!=256 || cache.num_kv_heads!=4 || cache.quant_group!=64)
        throw std::invalid_argument("headscale requires native27 INT8G64 cache view");
    requantize_paged_keys<<<(count*4+7)/8,256,0,stream>>>(
        static_cast<std::int8_t*>(cache.k_pages.data),static_cast<__half*>(cache.k_scale_pages.data),
        static_cast<const std::int32_t*>(cache.block_table.data),
        positions?static_cast<const std::int32_t*>(positions->data):nullptr,count,prefix,
        cache.block_table.ne[0]*kPagedKVPageSize,cache.k_pages.ne[3]);
    CUDA_CHECK(cudaGetLastError());
}
inline void launch_headscale_range(const Tensor& q,const Tensor& positions,float scale,
    const PagedKVLayerView& cache,std::int32_t range_begin,std::int32_t range_count,
    std::int32_t total_splits,std::uint32_t capacity,Tensor& partial_acc,Tensor& partial_m,
    Tensor& partial_l,cudaStream_t stream,bool value_fp16 = false) {
    if(q.ne[2]<1||q.ne[2]>4||cache.dtype!=DType::I8||q.ne[1]!=24||capacity!=cache.block_table.ne[0]*64)
        throw std::invalid_argument("headscale complete range requires supported D3 INT8 cache");
    const auto launch=[&]<int T,bool PackedValues>() {
        gqa_attention_headscale_qk_kernel<Gqa27Geometry,T,8,2,16,false,false,false,GqaCachedInput,PackedValues>
        <<<dim3(4,range_count),256,0,stream>>>(static_cast<const __nv_bfloat16*>(q.data),{},
            static_cast<const int*>(positions.data),static_cast<std::int8_t*>(cache.k_pages.data),
            static_cast<std::int8_t*>(cache.v_pages.data),static_cast<__half*>(cache.k_scale_pages.data),
            static_cast<__half*>(cache.v_scale_pages.data),static_cast<const int*>(cache.block_table.data),
            nullptr,nullptr,cache.block_table.ne[0],T,0,capacity,scale,range_begin,total_splits,range_count,1,
            capacity,static_cast<__nv_bfloat16*>(partial_acc.data),static_cast<float*>(partial_m.data),
            static_cast<float*>(partial_l.data));
    };
    const auto dispatch=[&]<bool PackedValues>() {
        switch(q.ne[2]) {
            case 1:launch.template operator()<1,PackedValues>();break;
            case 2:launch.template operator()<2,PackedValues>();break;
            case 3:launch.template operator()<3,PackedValues>();break;
            case 4:launch.template operator()<4,PackedValues>();break;
        }
    };
    if(value_fp16)dispatch.template operator()<true>();
    else dispatch.template operator()<false>();
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::value_residual_indexed_subset

namespace ninfer::ops::value_residual_indexed_subset {
template <typename Geometry, int DChunk, bool Int8, bool MultiBatch, bool Masked, bool Offset>
__launch_bounds__(256) __global__ void reduce_fallback(
    const __nv_bfloat16* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    std::int32_t split_count, __nv_bfloat16* out, const int* active_heads) {
    if(active_heads[blockIdx.x / Geometry::GroupSize] == 0) return;
    static_assert(DChunk > 0 && DChunk <= kGqaHeadDim);

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    if constexpr (Offset) { positions += column_begin; }
    if constexpr (MultiBatch) { positions += batch * full_width; }
    const int last_pos = positions[tokens - 1];
    int output_column  = token;
    if constexpr (Offset) { output_column += column_begin; }
    if constexpr (MultiBatch) { output_column += batch * full_width; }

    if constexpr (MultiBatch) {
        const std::int64_t partial_acc_row = static_cast<std::int64_t>(batch) * kGqaHeadDim *
                                             Geometry::QHeads * tokens * split_count;
        const std::int64_t partial_stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_acc += partial_acc_row;
        partial_m += partial_stat_row;
        partial_l += partial_stat_row;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, Int8>(window, split_count, tokens);

    __shared__ float reduce[256];

    float local_m = -CUDART_INF_F;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        local_m = fmaxf(local_m,
                        partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)]);
    }
    reduce[tid] = local_m;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]); }
        __syncthreads();
    }
    const float head_m = reduce[0];
    __syncthreads();

    if (head_m == -CUDART_INF_F) {
        const int d = d_start + tid;
        if (tid < DChunk && d < kGqaHeadDim) {
            out[gqa_q_index<Geometry>(q_head, d, output_column)] = float_to_bf16_rn(0.0f);
        }
        return;
    }

    float local_l = 0.0f;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        const float tile_l =
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
        if (tile_l > 0.0f) {
            local_l +=
                tile_l *
                expf(partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] -
                     head_m);
        }
    }
    reduce[tid] = local_l;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] += reduce[tid + stride]; }
        __syncthreads();
    }
    const float head_l = reduce[0];

    const int d = d_start + tid;
    if (tid >= DChunk || d >= kGqaHeadDim) { return; }

    float numerator = 0.0f;
    if (head_l > 0.0f) {
        for (int split = 0; split < active_split_count; ++split) {
            const float tile_l =
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
            if (tile_l <= 0.0f) { continue; }
            const float weight = expf(
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] - head_m);
            numerator +=
                __bfloat162float(
                    partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)]) *
                weight;
        }
    }
    bool valid = true;
    if constexpr (Masked) {
        int absolute_column = token;
        if constexpr (Offset) { absolute_column += column_begin; }
        valid = absolute_column < valid_columns[batch];
    }
    const float value = (valid && head_l > 0.0f) ? numerator / head_l : 0.0f;
    out[gqa_q_index<Geometry>(q_head, d, output_column)] = float_to_bf16_rn(value);
}
}
