# START HERE — CONTINUE THE LATEST INT8/TU102 WORK, DO NOT REDISCOVER IT

**2026-08-17 15:07 +07 — this is the continuation pointer.**

Use worktree `../ninfer-2080ti-int8-tu102`, branch `perf/sm75-int8-tu102`. There are currently no live jobs. Resume from the existing dirty diagnostic state; do not start a new worktree and do not go back to dual/multi-user work yet.

## Read these notes first, in this order

1. `PLEASE_READ_THIS_2026-08-17_1402_CORRECTED_HARD_GATES_INT8_SINGLE80_DUAL100PLUS.md`
   - hard gates: **single GPU + INT8 >=80 tok/s**, dual performance mode **>=100+ tok/s**.
2. `PLEASE_READ_THIS_2026-08-17_1420_RESEARCH_LEADS_INT8_80_Q4_Q5.md`
   - explains why dense Q4/Q5 polishing alone is unlikely to reach 80 and lays out sparse-SwiGLU / native-S4 / lower-bit research.
3. `PLEASE_READ_THIS_2026-08-17_1503_RESUME_HERE_INT8_80_NEXT_Q4_Q5_BYTE_REDUCTION.md`
   - contains the exact latest measurements and the required next experiment sequence.

Do not spend time rereading older continuity notes unless one of those three explicitly points you there.

## Continue from this exact latest result

Stable INT8 canonical baseline: **65.31 / 65.29 / 65.27 tok/s**; use **65.29** as the comparison baseline.

The latest TU102 GQA work improved isolated T3/8K attention by about **39%** (~270.5 us -> ~163-164 us), but the real model stayed **65.25 / 65.28 / 65.32 tok/s**. Therefore the short-context GQA tuning phase is closed for now: preserve the benchmark/results and revisit it later for long populated contexts, but **do not keep sweeping 8K attention geometry** for the 80-TPS target.

Current dirty GQA state includes the useful benchmark, a TU102 T6 legal-route fix, and a diagnostic split-cap override. Preserve artifacts; qualify/keep the minimal legality fix if green; remove the diagnostic split override from anything production-like.

## What to do next

**Next primary task: A0 block-64 SwiGLU oracle sparsity audit.**

Use the existing fast T4 verify representation in `src/targets/qwen3_6_27b/impl/variant.cpp`: `activation_q` is INT8 `[intermediate,4]` and `activation_s` is FP16 `[intermediate/64,4]` immediately before Q5 down. One 64-channel block already matches one quant group.

1. Measure block-64 salience at retained densities 90/80/70/60/50%. Record both **per-column density** and the **union/shared-mask density across the actual T4 verifier columns**. This union-density check is critical because four individually sparse masks may become nearly dense together.
2. If useful shared/union sparsity looks like **30-40%+**, zero complete low-salience block64 activation groups before Q5 down and run bounded prose/code/JSON/multilingual quality checks. Do not build a sparse kernel yet.
3. If quality survives, implement the simplest oracle sparse **Q5-down** path that skips whole zero K-group64 weight blocks and measure real E2E against 65.29. Only if that produces a meaningful model-level win should you build a pre-Q4 predictor using native Turing `s4.s4` or B1/B2/BMMA so gate/up reads can also be skipped.

If block64 shared sparsity is weak or quality fails, follow the fallback path in the 14:20/15:03 notes: block32/16 audit, then Q5->Q4 / low-bit-core + sparse-residual audits.

**Do not stop after a microbenchmark win with zero E2E movement. That result means move to the next leverage point.**
