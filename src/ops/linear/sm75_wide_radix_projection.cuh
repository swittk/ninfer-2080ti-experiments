#pragma once
#include "ops/linear/sm75_radix_projection.cuh"
namespace ninfer::ops::detail::wide_radix {
template<int Tokens>
__global__ void quantize(const __nv_bfloat16* x,signed char* q,float* scales,int k){
 int group=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x&31;
 int groups=k/64;if(group>=Tokens*groups)return;
 int t=group/groups,g=group%groups;
 float a=__bfloat162float(x[t*k+g*64+lane]),b=__bfloat162float(x[t*k+g*64+lane+32]);
 float mx=fmaxf(fabsf(a),fabsf(b));
#pragma unroll
 for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
 // Power-of-two radix preserves BF16 mantissa bits rather than rounding
 // every value to an arbitrary scale. High and low limbs share exact exponents.
 int exponent=0;float fraction=frexpf(mx,&exponent);
 int shift=exponent-7+(fraction>127.f/128.f?1:0);
 float s=mx?ldexpf(1.f,shift):0.f,inv=mx?ldexpf(1.f,-shift):0.f;
 int a0=max(-127,min(127,__float2int_rn(a*inv))),b0=max(-127,min(127,__float2int_rn(b*inv)));
 int a1=max(-127,min(127,__float2int_rn(__fmaf_rn(-float(a0),s,a)*inv*128.f)));
 int b1=max(-127,min(127,__float2int_rn(__fmaf_rn(-float(b0),s,b)*inv*128.f)));
 q[t*k+g*64+lane]=a0;q[t*k+g*64+lane+32]=b0;
 q[(t+Tokens)*k+g*64+lane]=a1;q[(t+Tokens)*k+g*64+lane+32]=b1;
 if(lane==0)scales[t*groups+g]=s;
}

// Each warp streams one weight fragment once and feeds all independent T4 groups.
// The eight IMMA columns within each group retain the original high/low limbs.
template<int GroupsOfFour, int Parts, bool Q5>
__global__ void matmul(const uint4* weights, const unsigned* high,
                       const __half* weight_scales, const signed char* q,
                       const float* activation_scales, float* partial,
                       int n, int k) {
    static_assert(GroupsOfFour == 2 || GroupsOfFour == 4);
    constexpr int Tokens = GroupsOfFour * 4;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row_block = blockIdx.x * 4 + warp;
    if (row_block >= n / 16) return;
    const int row = lane >> 2;
    const int lane_in_group = lane & 3;
    const int groups = k / 64;
    const int part = blockIdx.y;
    const int first = groups * part / Parts;
    const int last = groups * (part + 1) / Parts;
    float sums[GroupsOfFour][4] = {};
    for (int group = first; group < last; ++group) {
        const auto at = (static_cast<long long>(row_block) * groups + group) * 32 + lane;
        const uint4 p = weights[at];
        unsigned h = 0;
        if constexpr (Q5) h = high[at];
        const unsigned a0 = radix_projection::expand_typed<Q5>(p.x & 65535U, h);
        const unsigned a1 = radix_projection::expand_typed<Q5>(p.y & 65535U, h >> 8);
        const unsigned a2 = radix_projection::expand_typed<Q5>(p.x >> 16, h >> 4);
        const unsigned a3 = radix_projection::expand_typed<Q5>(p.y >> 16, h >> 12);
        const unsigned b0 = radix_projection::expand_typed<Q5>(p.z & 65535U, h >> 16);
        const unsigned b1 = radix_projection::expand_typed<Q5>(p.w & 65535U, h >> 24);
        const unsigned b2 = radix_projection::expand_typed<Q5>(p.z >> 16, h >> 20);
        const unsigned b3 = radix_projection::expand_typed<Q5>(p.w >> 16, h >> 28);
        const float sw0 = __half2float(weight_scales[(row_block * groups + group) * 16 + row]);
        const float sw1 = __half2float(weight_scales[(row_block * groups + group) * 16 + row + 8]);
        const int ta = lane_in_group * 2;
        const int tb = ta + 1;
#pragma unroll
        for (int block = 0; block < GroupsOfFour; ++block) {
            const int input_column = block * 4 + (row & 3) + (row >= 4 ? Tokens : 0);
            const auto* x = reinterpret_cast<const unsigned*>(q + input_column * k + group * 64);
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            ops::mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, x[lane_in_group], x[4 + lane_in_group]);
            ops::mma_s8(c0, c1, c2, c3, b0, b1, b2, b3, x[8 + lane_in_group], x[12 + lane_in_group]);
            const float sa = activation_scales[(block * 4 + (ta & 3)) * groups + group] *
                             (ta >= 4 ? 1.f / 128 : 1.f);
            const float sb = activation_scales[(block * 4 + (tb & 3)) * groups + group] *
                             (tb >= 4 ? 1.f / 128 : 1.f);
            sums[block][0] = __fmaf_rn(float(c0), sw0 * sa, sums[block][0]);
            sums[block][1] = __fmaf_rn(float(c1), sw0 * sb, sums[block][1]);
            sums[block][2] = __fmaf_rn(float(c2), sw1 * sa, sums[block][2]);
            sums[block][3] = __fmaf_rn(float(c3), sw1 * sb, sums[block][3]);
        }
    }
    const int ta = lane_in_group * 2;
    const int tb = ta + 1;
    const int r = row_block * 16 + row;
#pragma unroll
    for (int block = 0; block < GroupsOfFour; ++block) {
        const auto offset = (static_cast<long long>(part) * GroupsOfFour + block) * 8 * n;
        partial[offset + ta * n + r] = sums[block][0];
        partial[offset + tb * n + r] = sums[block][1];
        partial[offset + ta * n + r + 8] = sums[block][2];
        partial[offset + tb * n + r + 8] = sums[block][3];
    }
}

template<int GroupsOfFour, int Parts, bool Residual>
__global__ void finish(const float* partial, const __nv_bfloat16* initial,
                       __nv_bfloat16* output, int n, int leading_dimension) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= GroupsOfFour * 4 * n) return;
    const int block = i / (4 * n);
    const int local = i % (4 * n);
    float sum = 0.f;
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
        const auto offset = (static_cast<long long>(part) * GroupsOfFour + block) * 8 * n;
        sum += partial[offset + local] + partial[offset + local + 4 * n];
    }
    const int destination = (i / n) * leading_dimension + i % n;
    if constexpr (Residual) sum += __bfloat162float(initial[destination]);
    output[destination] = __float2bfloat16_rn(sum);
}
} // namespace ninfer::ops::detail::wide_radix
