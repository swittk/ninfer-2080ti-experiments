# ADVISER — 2026-08-21 13:52 +07 — after current prefill geometry qualification

Read `AGENTS.md`. This note does NOT reopen closed short-T Q4/Q5 micro-tuning, Marlin decode, Humming, FlashQLA ReplaySSM decode, or any model training. Finish the currently-running split/chunk qualification first.

## Current good result

The new dual MTP prefill pipeline is real and should be qualified/promoted if the existing full-decode/snapshot gates remain exact:
- old MTP 4K: ~182.6 t/s
- pipelined MTP 4K: ~314.5-315.2 t/s
- pipelined MTP 65K repeats: 284.40 / 284.43 t/s, exact same first-token digest/output as target-only baseline
- full 4K 128-token decode A/B: identical digest, exact IDs, 50 rounds, 74/150 accepted, 33/23/18 by-position; decode 56.57 vs 56.40 t/s (neutral)
- snapshot capture/reload preserves exact output/trajectory
- split36/chunk256 at32K: 344.12 t/s vs split32/chunk128 304.47, same digest
- split36/chunk512 4K with enlarged stage0 workspace: 364.23 t/s, same digest; long-context qualification still required

## External confirmation: MTP prefill overhead should be small

The maintained dual-2080Ti vLLM Qwen3.6 sweep reports FP8 prefill:
- 4K noMTP 1677.0 t/s vs MTP3 1614.1 t/s
- 65K noMTP 1303.9 t/s vs MTP3 1253.9 t/s
So a well-integrated MTP prefill path only loses ~4%, consistent with our new pipeline eliminating the old serial MTP tax. Source: `weicj/vLLM-2080Ti-Definitive/docs/qwen36-kv-throughput-sweep.md`.

## New P0-A2 after geometry: HIGH-M DENSE PREFILL ORACLE

Nsight `nsys4k_target/profile_summary.json` partitions prefill GPU time:
- dense post-mixer: ~66.67%
- GDN recurrence: ~22.66%
- full attention: ~10.64%

The SM75 FlashQLA fork itself reports ~2.1x GDN-stage speedup but only ~7.17% whole-request prefill gain. Therefore FlashQLA alone cannot explain or close the remaining 3-4x external prefill gap.

Most important consequence: prior Marlin/Humming closures were at tiny verifier T (T4/T5) and DO NOT classify high-M prefill. Current NInfer profile shows production high-M kernels such as `q4_rowsplit_gemm_mma_kernel` and `q5_rowsplit_gemm_mma_kernel` dominate. A high-M external oracle is materially new.

### Do this only after current chunk/split qualification
1. Extract the exact production prefill GEMM shapes for full/GDN layers at M=128,256,512 (K/N, Q4/Q5 codec, epilogue/fusion).
2. Benchmark current NInfer kernels on the loaded faster card at those M values, with real weight format and clocks. Report achieved effective weight GB/s and/or TFLOP/s, not only kernel us.
3. In the already-cloned `vllm-2080ti-definitive-sidecar`, benchmark the maintained patched SM75 **high-M** AWQ-Marlin / FP8-Marlin dense route on matching or nearest production shapes. Do not extrapolate from T4 decode timings. If exact NInfer groupwise format cannot be fed, use this strictly as a dataflow ceiling/oracle.
4. Also compare an ordinary FP16/BF16 cuBLASLt/CUTLASS tensor-core GEMM with dequantized weights as a compute ceiling. This is an oracle, not permission to expand production weights beyond capacity.
5. Attribute whether NInfer is limited by weight decode/dequant, shared-memory staging, MMA occupancy, or output/epilogue. Use Nsight Compute only on 1-2 dominant shapes after the timing discriminator, not a giant sweep.

### Hard promotion gate
A dense-prefill mechanism should project >=1.5x complete 65K prefill from the best qualified split/chunk baseline or remove >=30% TTFT. Since dense post-mixer is ~2/3 of GPU time, a real 2x dense-stage win would be material (~1.5x whole prefill). Anything <~20% dense-stage win is not worth a new production kernel family.

### Do NOT conflate external totals
The public vLLM ~1.2-1.6k t/s rows use different checkpoint formats and NVLink TP=2, so they are not an apples-to-apples product target for our no-P2P NInfer route. Use them only to prove that SM75 high-M prefill can be much faster and to identify source dataflow.

## Long-context / dual research status

P0-B LeanAttention/PersistentKV is correctly CLOSED without code: NInfer already has KV-head tile reuse, sequence splitting and exact `(m,l,acc)` merge; even deleting all query/state-return/merge overhead saves only ~4.8% of the 262K T5-like floor.

P0-C PipeInfer is correctly CLOSED for current factory MTP: independent-proposer target-only oracle is ~143.5 t/s, but factory-MTP cancellation economics give only ~79 t/s realistic / ~92 t/s even with free proposal+fold. Reopen only with an owner-allowed independent proposer whose measured placement does not contend, or if target microbatches stop restreaming the full model.

## Execution order
1. Finish split/chunk 32K and 65K qualification; prove chunk512 workspace/correctness at long context before keeping it.
2. Promote the coarse MTP prefill pipeline only after exact full-decode + snapshot qualification and matched long-context repeats.
3. Then run the high-M dense-prefill oracle above.
4. Do not stop after a negative high-M oracle; recompute cold 65K/200K TTFT and return to the hard decode gates.
