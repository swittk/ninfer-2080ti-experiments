# OWNER/ADVISER — 2026-08-21 16:05 +07 — CONTINUE, DO NOT STOP AT THE PREVIOUS CHECKLIST

Read repo-root `AGENTS.md` first. Hard gates remain single>=80, dual-short>=100, populated65K>=100, populated262K>=100 useful decode t/s. NO agents. NO local behavioral training/finetuning/distillation. Do not reopen CLOSED lanes unless this note identifies a materially different mechanism.

The previous pass was disciplined, but stopping after the supplied checklist is not completion. When a supplied lane closes, directly profile/research the next material system mechanism yourself and continue.

## Current accepted/trusted state
- Single fixed short factory-MTP D3: ~76.6-76.7 useful t/s; six-family D3 mean 68.25.
- Accepted short dual layer-shard ~65.3; owner+TP2 helper 66.37 is rejected because owner-only is faster.
- Populated65K INT8 physical-KVP factory-D4/T5: 58.75 useful t/s, 67.563 ms/round, 3.90625 useful/round.
- 262K physical INT8 KVP + dedup T5 TP fits; no trusted full populated262K E2E decode number. Exact 262K KVP scan ~28.24 ms/round across 16 full-attention layers.
- ACCEPTED cold-prefill change: dual factory-MTP prefill double-buffers pinned boundaries / overlaps stages. Populated65K 414.073 -> median 229.569 s, 157.68 -> 284.42 tok/s (1.804x, 44.56% TTFT reduction), exact repeated digest/output, exact 128-token continuation/MTP counters and snapshot capture/reload. Keep chunk128; wider chunks were correctly rejected for production.
- LeanAttention/PersistentKV scheduler-only mapping CLOSED; PipeInfer with current factory MTP CLOSED; global lookup-tail CLOSED (copy/edit special-mode oracle remains interesting but is not the global frontier).

## P0-A3 — HIGH-M MARLIN IS NOT CLOSED BY PER-PROMPT REPACK

The latest byte-valid high-M oracle contains a stronger ceiling than the terminal prose emphasizes:
- native dense modeled slice: 713.855 ms
- exact Marlin GEMM work: 145.368 ms
- direct repack charged on critical path: 279.196 ms
- scale permutation: 24.726 ms
- current repack-on-prompt candidate: 449.290 ms dense, 1.328x whole-prefill / 24.71% TTFT reduction.

That closes only **repack on the prompt critical path**. Production vLLM Marlin repacks weights after loading and replaces the runtime weight; repack is setup/post-load processing, not paid every GEMM. If repack+scale permutation are amortized away, the same measured slice projects `(1070.7825 - 713.855 + 145.368) = 502.296 ms`, about **2.13x** versus baseline / ~53% reduction before layout-switch/fill-drain cost. This is material and must be discriminated.

### Model TWO architectures before coding

#### A. Partial resident dual-layout weights
Keep canonical NInfer decode weights plus Marlin-packed copies only for matrices with best saved-ms / extra-GiB. Measured packed payload totals across all layers (scales excluded):
- Q4 GDN QK 0.469 GiB
- Q4 attention QK 0.273 GiB
- Q4 gate/up 5.312 GiB
- Q5->Marlin-int8 GDN value/Z 1.758 GiB
- Q5->Marlin-int8 attention gate/value 0.342 GiB
- Q5->Marlin-int8 down 3.320 GiB
- Q5->Marlin-int8 output 1.172 GiB
- all 12.646 GiB.

Compute exact per-stage memory including scales/workspace at the actual split and compare against measured 262K peak allocations. Search subsets by **complete-prefill ms saved per extra GiB**. A resident layout must preserve 262K capacity.

#### B. Macroblock weight-stationary prefill (preferred if resident copy does not fit)
Do not repack every 128-token chunk. For a macroblock (model 4K/8K/16K):
1. retain canonical hidden for that macroblock;
2. for one layer, direct-repack native weights into one bounded Marlin scratch layout ONCE;
3. run every 128-token chunk in that macroblock through the layer while the packed layout stays resident;
4. reuse scratch for the next layer;
5. carry GDN recurrent state / attention KV exactly across chunks and macroblocks;
6. double-buffer macroblocks so stage1 consumes macroblock N while stage0 works on N+1.

This is NOT equivalent to increasing `--prefill-chunk`: keep the qualified M128 geometry while amortizing repack over 32/64/128 chunks. It needs macroblock hidden storage plus one-layer packed scratch, not a second 12.6-GiB model copy.

First build a measured ceiling using current stage0/stage1 time, repack, Marlin per-layer timing, boundary bytes, hidden/scratch memory and pipeline fill/drain. Implementation gate: >=30% complete cold65K TTFT reduction after all costs (roughly <=160.7 s from accepted 229.57 s). If it passes with margin, prototype 4K then qualify 32K/65K. Correctness: identical digest/IDs, >=128-token continuation with identical MTP rounds/drafted/accepted/by-position, and snapshot capture/reload. No prompt-specific paths.

## P0-E — 262K OLD-KV BYTE REDUCTION: INT4 TIER ORACLE (NO CUDA FIRST)

Scheduler-only long-KV work is closed because NInfer already sequence-splits and reuses K/V tiles. The remaining ~28.24-ms 262K KVP wall is predominantly bytes. Attack bytes directly.

External evidence makes this a legitimate inference-only oracle, not permission to train:
- KIVI is tuning-free and supports 2/4-bit KV; its current code supports GQA and motivates per-channel K / per-token V plus a recent residual cache.
- Hugging Face QuantizedCache exposes INT4/INT2 (`hqq`/`quanto`) with a residual recent-token cache.
- KVQuant reports sub-4-bit long-KV quality and custom-kernel speedups in other model families. None of those quality numbers may be assumed for Qwen3.8.

Oracle sequence:
1. Preserve current INT8 cache as canonical. Quantize only **old full-attention K/V pages** to 4-bit; keep newest 128 or 256 tokens/pages in current INT8. Start INT4, not INT2.
2. Test K per-channel / V per-token and a simpler per-group64 fallback aligned with NInfer page machinery. Quantize completed pages; no per-token repack in decode critical path.
3. Using saved 4K/65K states and a populated/constructed 262K attention oracle, compare against canonical INT8: relative L2, max error, top-logit margin where available, and exact target-token decisions.
4. If promising, run held-out general/code/JSON/Thai/Japanese/Arabic plus retrieval/passkey trajectories using an offline/dequant reference BEFORE building a fused INT4 attention kernel. Useful output matters more than low L2.
5. Then model a native SM75 packed-INT4 K/V scan including scale bytes/dequant. Raw payload halving would move the ~28.24-ms scan toward a ~14-ms-class byte floor, which is material. Promotion requires >=8 ms/round complete target saving after all overhead and portable target-output qualification.
6. If INT4 quality fails, try asymmetric/per-channel details or a larger residual window before INT2. Do not train anything. Close if no portable output-preserving regime has material economics.

INT4 old-KV can also free several GiB at 262K, potentially enabling partial resident Marlin prefill layouts. Model that interaction explicitly.

## Independent proposer status
A fresh public search found no new professionally published **independent** Qwen3.8-27B proposer that solves PipeInfer's dependency. Public exact-target options remain factory MTP/MTP-only artifacts and RadixArk DSpark, which consumes target auxiliary features. Do not spend another pass re-downloading DSpark/DFlash/MTP under new names. Reopen PipeInfer only for a genuinely independent published proposer or a target execution path that stops restreaming the full model.

## Execution order / autonomy requirement
1. Preserve the already-qualified MTP prefill pipeline. Review current dirty diff/tests and commit/promote it cleanly if repository policy allows; do not lose it.
2. Finish/classify the existing high-M artifacts, then immediately build the P0-A3 macroblock/phase-layout ceiling. This is highest priority because the existing measured no-repack ceiling is >2x.
3. In parallel by direct analysis, start P0-E INT4-old-KV numerical oracle from existing snapshots/states. No CUDA until quality+ceiling pass.
4. If P0-A3 passes, prototype it. If it fails, move directly to P0-E or another material mechanism.
5. After every lane append `progress.md`, preserve evidence, and recompute single80 / dual100 / 65K100 / 262K100.
6. **DO NOT STOP merely because every item in this note was classified.** If gates remain unmet, profile the new dominant complete-system bucket, research externally for a genuinely different mechanism, bound it, and continue. The owner wants autonomous continuation, not checklist completion.
