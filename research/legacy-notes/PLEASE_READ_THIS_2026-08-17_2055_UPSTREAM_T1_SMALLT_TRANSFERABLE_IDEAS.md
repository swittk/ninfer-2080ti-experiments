# PLEASE READ — upstream Aug-16 T=1/small-T methodology is transferable; FP8/NVFP4 kernels are not

Adviser research note, 2026-08-17 20:55 +07. **Do not interrupt the current dual-MTP B2 acceptance/correctness work for this.** This is a targeted later research lead for the SM75 single-GPU and sharded small-T kernels.

## What changed upstream

Current Neroued/ninfer master added Qwen3.8 NVFP4/FP8 routes after the RTX3090 base we forked from. Relevant upstream commits include:

- `c8ca80b51f31` — `feat(ops): add fp8 t1 projection routes`
- `9f7277266976` — `perf(ops): qualify fp8 projection routes` (adds dedicated small-T families)
- `935c44a5b47e` — `perf(ops): qualify fp8 gdn snapshot and record`
- `5d2c1f5590b8` — `feat(qwen3.8): integrate nvfp4 artifact runtime`

Here **T=1 means one activation/token column**, not a new Tensor-Core generation or scheduler mode.

## Do NOT directly port the FP8/NVFP4 kernels

Those exact kernels/artifact routes target RTX 5090 / `sm_120a`, row-scaled FP8 E4M3 and NVFP4. Turing SM75 does not have the Blackwell FP8/NVFP4 Tensor Core execution model, so direct cherry-picking would be wrong. Our deployment weights remain the SM75-appropriate groupwise Q4/Q5/W8 routes.

## The important transferable result

For upstream's first representative FP8 T=1 geometry `[N,K]=[14336,5120]`, the measured winning route on RTX5090 was **not native FP8 MMA**. It was a direct CUDA-core memory-streaming GEMV:

- tuned schedule: 8 warps/CTA, 2 rows/warp, 8 values/lane, 4 FP32 accumulator chains, default code-cache policy, phase-unroll 2, launch bound 2 blocks/SM;
- public Linear A16 median ~`50.464 us`;
- one-read effective bandwidth ~`1455.8 GB/s`, ~`86.94%` of their sustained-read probe;
- best native FP8-MMA candidate was ~`69.632 us`, materially slower.

Do not copy those schedule constants to TU102. The transferable conclusion is that **single-column decode remains a memory/scheduling problem even on newer hardware; Tensor Core use is not automatically the winning T=1 route.** This reinforces our SIMT/bandwidth-first TU102 work.

## More important: upstream now has a dedicated small-T family

The later `9f7277...` work does not simply run T=1 repeatedly. It introduces a compile-time small-T contraction where one loaded/decoded weight pair is reused across a token tile before advancing K. The tuning space includes:

- token tile width;
- warps/CTA;
- rows/warp;
- values/lane;
- independent accumulator-chain count;
- activation access strategy (direct/token-packed vs shared-phase);
- global code-cache policy;
- phase unroll;
- minimum blocks/SM / occupancy;
- block ordering (rows-contiguous vs token-contiguous);
- optional paired-row output;
- fused output/epilogue policies so specialized projections do not materialize an unnecessary parent output.

This is directly relevant to our **T4 target verification, B2/B3 packed requests, MTP proposal projections, and other exact small-width SM75 shapes**, even though the FP8 decoder itself is not.

## Upstream also chooses activation route by exact geometry + T

Their FP8 `AllowA8` selector is not globally `A8 whenever possible`. It uses measured per-geometry token thresholds (different for attention input, GDN input, MLP gate/up, residual projections), and A16 can win for some small-T ranges. **Do not copy their thresholds**; use this as methodology. Audit SM75 Q4/Q5/W8 A16-vs-A8 decisions independently for the actual Qwen3.8 geometries and T=1/T2/T3/T4/T5/T6/B2... widths.

## Our current SM75 code appears less systematic in this specific dimension

Current TU102 sources already have several useful exact small-T variants (`q5_rowsplit...` split2/split4, row-count variants, T4-specialized Q4/A8 paths, etc.), so this is not a claim that we missed small-T specialization entirely. But the current schedules are more hand-selected/hard-coded and do not expose the same systematic compile-time sweep over rows/warp, values/lane, accumulator chains, cache policy, phase unroll, token tile, block order, and activation staging across all dominant exact geometries.

After the current dual milestone and after requalifying the ~72 lane, consider building a **bounded TU102 schedule-sweep harness** for the dominant T=1 and T=2..6 projection shapes, then promote only measured winners through the existing public Op contracts. Do not giant-refactor kernels merely to imitate upstream.

## Qwen3.8 status

Generic Qwen3.8-27B support is **already in our ancestry**: the `v0.6.0-rtx3090` base already had Qwen3.8 as a first-class groupwise target, and current dual qualification is literally running `models/qwen3_8_27b.ninfer`. Do not re-port generic Qwen3.8 support.

What is newer upstream is mainly the **Qwen3.8 NVFP4/row-FP8 artifact profile and its Blackwell-specific execution routes**. Those are not directly useful on SM75.

Generic post-base runtime fixes (prefix-reuse semantics, multimodal admission, serving correctness, etc.) may still be worth a selective upstream audit later; merge by behavior/test, not blind cherry-pick, because the RTX3090/SM75 fork already diverged materially.

## Priority

1. Current issue first: restore dual MTP proposal/acceptance behavior; the 128-token B2 runs are semantically correct but acceptance is only ~33.3% versus the single-GPU control ~55.2%, masking pipeline potential.
2. Finish/qualify dual MTP and bounded multi-user performance.
3. Return to single-GPU >=80 track as already instructed.
4. Then use the upstream T1/small-T methodology as one concrete TU102 optimization avenue.
