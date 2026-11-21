# RESUME HERE — INT8 >=80: STOP SHORT-GQA POLISH, START Q4/Q5 BYTE REDUCTION

Owner directive 2026-08-17 15:03 +07. Worktree `../ninfer-2080ti-int8-tu102`, branch `perf/sm75-int8-tu102`, base `0a2b776`. Do not rediscover trajectory. Hard gates: ONE GPU + INT8-G64 KV >=80 useful decode tok/s; dual performance mode >=100+. INT8 is the deployment lane.

## Latest truth
Stable canonical INT8 baseline: **65.31 / 65.29 / 65.27 tok/s**, digest `2fb75f62ab6e0e31`, 380 MTP3 rounds, 642 accepted, 56.37%, 2.69 useful tok/round, fallback 1. Use 65.29 as baseline. Current clean/general BF16 control here is ~65.68; do not confuse it with the separately-qualified historical ~72 turbo lane.

TU102 split-cap tuning improved isolated T3/8K GQA from roughly **270.5 us -> 163-164 us (~39%)**, but real-model splitcap16 runs were **65.25 / 65.28 / 65.32**, i.e. zero meaningful E2E gain. Therefore **close further short-context GQA geometry tuning now**. Preserve it for 32K/64K/128K/262K later, where attention matters more.

NCU still showed useful TU102 ugliness (low occupancy, long-scoreboard/wait/barrier, shared-store conflicts), but canonical model evidence says it is not the lever for the 80-TPS floor.

## Clean current GQA diagnostics
Current tree has a diagnostic `NINFER_GQA_SPLIT_CAPACITY` override, the SM75 T6 legal-route change, and a new GQA bench. Preserve artifacts. Keep the bench if clean. Qualify the TU102 T6 <=8K legal 32-key route with `ninfer_gqa_attention_test`; retain that minimal legality fix if green. **Do not promote the split-cap env override** as production policy; remove/restore it after recording the conclusion. No more broad 8K split sweeps without model-level evidence.

## NEXT PRIMARY: A0 production-faithful SwiGLU block64 oracle audit
Do this before native S4, Q3 formats, or more dual work.

The T4 fast path in `src/targets/qwen3_6_27b/impl/variant.cpp` already produces:
`activation_q: I8 [intermediate,4]` and `activation_s: FP16 [intermediate/64,4]`
before `q5_linear_add_sm75_a8_prequant_launch`. One **64-channel block is already one quant group**. Use this representation; do not first invent a BF16 capture path.

### A0a salience audit, no sparse kernel
After Q4 SwiGLU produces `activation_q/scales` and before Q5 down, compute block64 salience from the actual production A8 activation (`code * scale`; max and/or L2). For retained densities 90/80/70/60/50%, record energy retained, active groups, layer distribution, and stability across rounds/prompts.

**CRITICAL T4 trap:** verifier processes 4 columns together and shares weight reads. Record BOTH per-column mask density and **union-of-4 density**. Also test one T4-shared mask selected by max/summed salience across all four columns. Hardware savings depend on the common/union mask. If 40% per-token sparsity collapses to ~5% after T4 union, discover that now and kill the idea early.

### A0b quality audit
Diagnostic-only: zero complete unselected 64-channel `activation_q` groups before Q5 down; scales may remain. Run bounded canonical/prose/code/JSON/multilingual cases and evaluate under the established NINFER numerical/first-fork contract, not SHA religion.

Stop rules: if useful T4-shared/union sparsity is <~20%, likely insufficient leverage. If **30-40%+ block64 skipping** preserves acceptable behavior, advance to A1. If only finer masks work, audit block32/16 before abandoning, but account for worse memory/layout behavior.

## A1 only if A0 wins: sparse Q5 down first
The mask is known after Q4 SwiGLU, so no predictor is needed yet. Implement an oracle/diagnostic Q5-down path that **skips entire K group64 weight blocks** for zero activation groups. Measure real E2E versus 65.29. This is the easiest true byte-saving upper bound. If 30-40% group skipping produces no useful E2E gain because control/gather overhead eats it, stop/rethink layout before building a predictor.

## A2 only after A1 proves hardware value
Then predict the mask before Q4 gate/up so gate/up weight reads can also be skipped. Bounded candidates: native Turing `s4.s4` / A4 proxy, or B1/B2/BMMA proxy reusing proposal-head screening. Proxy output is ranking/mask only; retained channels still use trusted Q4/Q5 math. Do not build S4 just because the TOPS number is sexy; first prove sparse upper bound exists.

If A0/A1 fail, pivot to stored-byte reduction audits: Q5->Q4 recalibration, Q4/Q5 core + sparse residual/outlier correction, then Q4->Q3 + sparse correction only if byte/quality accounting supports it. Audit before rewriting artifact formats.

## Profiling discipline
Latest NSYS included load-time transforms and prompt work; do not treat its category percentages as steady decode percentages. If more system attribution is needed, use a decode-only NVTX/profile window. For sparse candidates report bytes/verification-round, bytes/useful-token, E2E tok/s, MTP useful tok/round, block density + T4 union/shared density, kernel latency, DRAM bytes/bandwidth, stalls, registers/occupancy/barriers.

## Immediate order
1. Preserve GQA artifacts; qualify minimal T6 legality fix; remove diagnostic split override.
2. **Start A0 block64 SwiGLU oracle audit immediately.**
3. If A0 passes, zero-mask quality gate.
4. If quality passes, sparse Q5-down oracle kernel + E2E upper bound.
5. Only then S4/B2 proxy for pre-Q4 mask selection.
6. Long-context GQA + dual >=100 return after the single-GPU INT8 byte-reduction path is pushed properly.

An isolated kernel win with zero E2E gain is a pivot signal, not a reason to stop.
