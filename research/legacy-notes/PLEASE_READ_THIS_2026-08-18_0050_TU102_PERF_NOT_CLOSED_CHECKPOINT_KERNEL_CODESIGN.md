# PLEASE READ — TU102 PERF IS NOT CLOSED: REOPEN CHECKPOINT/KERNEL CO-DESIGN, NOT BLIND PREFETCH SWEEPS

Owner/adviser correction, 2026-08-18 00:50 +07. Do not undo the good bounded conclusions from `TU102_SMALLT_KERNEL_CONCLUSION.md`: the specific current-layout Q4/Q5/GDN lookahead/warp/barrier experiments were measured and may stay rejected. But the phrase “campaign complete” must NOT be interpreted as “target-side TU102 performance is exhausted.” It only closes those local schedule variants under the current persistent numeric recipe and data layout.

Do not interrupt a coherent batch/concurrency experiment already in flight. Re-ground on this before declaring another target-side avenue exhausted.

## Major search spaces not closed

### 1. Hardware-aware checkpoint numeric recipe

The current recipe is role-fixed across all 64 layers: MLP gate/up Q4G64, MLP down Q5G64; Q/K mostly Q4; value/output mostly Q5; vocab endpoints W8. There has been no serious per-layer sensitivity-guided bit allocation.

Use the original BF16 source and measure layer/role sensitivity. A Q5G64 -> Q4G64 conversion changes logical weight cost from 5.25 to 4.25 bpw (~19% fewer persistent/streamed bits for that matrix). The text core has ~11.32B Q5 weights, so even a subset can remove meaningful decode traffic. Spend bits back only where sensitivity warrants it (selected Q4 -> Q5/W8), under a fixed quality and/or average-bpw budget. Do not assume every layer of the same role deserves the same bit width.

### 2. Better encoder with the SAME existing runtime formats

The canonical encoder is simple per-group symmetric MAXABS + FP16 scale + RNE. The format docs explicitly allow a documented checkpoint-specific alternative encoder as long as emitted codes/scales obey the same persistent scheme. Therefore calibration/clipping/reconstruction/activation-aware code selection can potentially improve Q4/Q5 fidelity with zero hot-kernel format change.

Before inventing a new on-device format, test whether MSE/activation-weighted clipping, GPTQ/AWQ-like calibration, or block reconstruction on the BF16 source lets more Q5 roles/layers move to Q4 at equal quality. This is likely higher leverage than another prefetch constant.

### 3. Group geometry / scale-budget co-design

G64 is historical/current, not proven TU102-optimal. Examples worth OFFLINE quality/byte testing before kernel work: Q4 with finer grouping (e.g. G32 costs 4.5 bpw with FP16 scales, still below Q5G64 5.25), or other bounded group/scale arrangements. Only implement kernels if the quality-vs-byte result is compelling.

### 4. Rotation/outlier transforms and activation-aware co-design

QuaRot/SpinQuant/AWQ/GPTQ-class ideas were never actually exhausted on the original BF16 checkpoint. Equivalent/foldable transforms may make weights/activations easier to quantize and allow more A8 or lower-bit execution without sacrificing quality. Do not equate earlier requantization of the already-quantized NInfer artifact with a real BF16-source PTQ study.

### 5. New A8 fragment layout != old rejected layout campaign

The current `both-mlp-stage1-all` A8 fragment path is a materially newer execution/data-layout regime. Old pre-A8 cache/layout/schedule negatives are evidence, not universal proof against revisiting a data-layout hypothesis that specifically attacks current NCU stalls. Current Q4 T4 is ~86.6% DRAM with ~69% long-scoreboard; Q5 down ~84.7% DRAM with ~60.5% long-scoreboard. There is still measurable hardware headroom. Do not reopen blind warp/prefetch sweeps, but a genuinely different byte packing/interleave/load-stream hypothesis remains valid.

### 6. Packed B2 is not globally dead anymore

After the A8 work, the requalified packed-B2 target path shows much faster target-stage execution (~1.31/1.34 s over 100 rounds) even though current serialized orchestration is only ~75.5 aggregate tok/s. Earlier “packed B2 loses” does not prove the target kernels are unhelpful. Investigate whether exact-B microbatches can be combined with rolling/pipeline overlap (especially C4/C8) via the inherited 3090 scheduler rather than choosing either batching OR pipeline overlap.

### 7. Long-context target profile remains separate

8K fast-path success does not qualify populated 128K/262K execution. Profile attention/KV dominance at real populated frontiers before claiming the long-context optimization space is closed.

## Qualification discipline

Keep `PLEASE_READ_THIS_2026-08-17_2236_MTP_ACCEPTANCE_IS_NOT_ENGINE_SPEED_QUALIFICATION_RULE.md` authoritative. Separate engine round-cost gains from acceptance gains. Use BF16/reference quality evaluation for checkpoint-recipe experiments; do not choose layers/bits solely from greedy-output coincidence.

## Priority

Preserve `acc9b4b` and the measured A8 wins. Finish the current bounded batch/concurrency work cleanly. Then the highest-leverage target-side research should be **BF16-source hardware-aware quantization/bit allocation under a real quality budget**, while continuing proposal/target concurrency. Single-GPU >=80 remains open; shared kernel/recipe wins should feed both single and dual.

Do not declare TU102 performance exhausted merely because local prefetch variants stopped winning.
