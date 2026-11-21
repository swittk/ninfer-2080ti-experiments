# OWNER ADVISER — 2026-08-21 11:50 +07 — next unblock after Marlin/DSpark/Humming closures

Read repo-root `AGENTS.md` first. Hard gates remain single>=80, dual-short>=100, populated65K>=100, populated262K>=100 useful decode t/s. NO agents. NO local behavioral training. Same-family Q4/Q5 micro-tuning stays CLOSED.

## Current verified frontier / closures
- Single-owner short factory MTP D3: ~76.6-76.7 useful t/s fixed workload; six-family D3 mean 68.25.
- Short layer-sharded dual ~65.3; owner+TP2 helper 66.37 CLOSED.
- 65K INT8 physical-KVP depth4/T5: 58.75 useful t/s.
- 262K physical INT8 capacity works; exact KVP scan ~28.24 ms/round across 16 full-attention layers; no trusted full262K E2E frontier.
- SSD/SpecBranch current mapping CLOSED (~96.1 t/s state-feasible impossible ceiling).
- Real patched SM75 Marlin, published DSpark, MTP-Marlin substitution, mechanical FP8 DSpark, Humming and FlashQLA T4/T5 have all been bounded and CLOSED at current economics.

These close mechanisms, not goals. Continue.

## P0-A — COLD PREFILL IS A HUGE UNRESOLVED ENGINE GAP
Our cold dual65K prefill is ~65,291 tokens /414s = ~157.7 t/s (chunk128). External SM75 same-family baselines are far higher: dual modified2080Ti Qwen3.6 AWQ/vLLM/FlashQLA+FlashInfer ~1,294 t/s at64K and ~1,842 t/s at4K; even single-card 2080Ti llama.cpp ~383 t/s at64K. NVLink/checkpoint differences do not explain the whole gap. For 200K coding this is P0.

Bounded profile before code:
1. Cold single-owner and dual 4K/32K/65K identical model/KV, no snapshots. Record chunk, clocks, full-attention, GDN, dense-linear, stage-boundary/transport wall.
2. Determine why dual long prefill is chunk128 while single paths previously qualified chunk256/512. Do not blindly sweep; identify workspace/state constraint and highest safe chunk.
3. Source-audit actual SM75 vLLM prefill: FlashQLA legacy GDN PREFILL (decode recurrence closure does not close this), FlashInfer/FA2 full-attention prefill, chunked-prefill scheduling/batched-token geometry.
4. Implement only mechanisms projecting >=1.5x cold65K prefill or >=30% TTFT removal.
Prefix snapshots/reuse remain operationally mandatory but must not hide bad cold prefill.

## P0-B — 200K/262K EXACT GQA: LEANATTENTION / PERSISTENTKV ORACLE
262K physical KVP is ~28.24 ms/round, so different long-attention scheduling can matter. Source-audit:
- LeanAttention: exact online-softmax stream-K decode scheduling; published ~1.73x average vs FlashDecoding and ~2.18x at256K in its reported setting.
- PersistentKV (2026): native paged GQA, KV-head-group tile reuse, sequence splitting, compact persistent/workqueue scheduling; ~1.399x synchronized B1 long-context wall vs FlashInfer on RTX3060 in its reported setting.

NInfer already has sequence-sharded partial `(acc,m,l)` merge, so first compare mechanisms, do not assume novelty. Measure CTA/SM occupancy, effective KV bytes/s, split/tail imbalance, and whether one K/V tile is reused across all query heads sharing a KV head. Benchmark-only alternative only if a concrete dataflow difference remains. Hard gate: >=8% projected complete262K-round saving (~7-8ms/round) with exact same-dtype attention semantics.

## P0-C — DUAL NO-P2P: PIPEINFER-STYLE ASYNC PIPELINE ORACLE
SSD/SpecBranch closure does NOT close asynchronous speculative pipeline parallelism. PipeInfer (SC24) targets single-request inference over low-bandwidth interconnects by keeping multiple speculative verification microbatches in flight across target pipeline stages and canceling invalid work early; its paper reports up to2.15x over standard speculative inference and explicitly discusses scaling to a single multi-GPU machine.

This is distinct from closed per-layer TP2, complete-next-round helper caching, ordinary serial PP, and SSD full-round fanout. TRACE/MODEL ONLY first using measured stage0/stage1 costs, boundary bytes, proposal costs and factory-MTP acceptance. Model pipeline fill/drain, several verification microbatches in flight, cancellation depth and host-staged boundary traffic. An upstream published drafter may be used only as existing weights; no training. Implement only if realistic 128-256-token one-request completion exceeds100 useful t/s with margin on current measured clocks.

The 2026 Speculative Pipeline Decoding work is also a scheduling oracle, but its learned intermediate predictor is not permission to train anything.

## P0-D — CODING-SPECIFIC HYBRID LOOKUP TAIL: TRACE ORACLE ONLY
Fresh upstream Qwen3.8 work (`syv-ai/qwen38-27b-rtx3090`) combines DFlash2 with request-context lookup to fill extra verify slots. It reports 15/16 accepted on document-reproduction/edit-like workloads and ~+9% on ordinary chat. This is materially different from the CLOSED standalone history drafter: lookup is a near-free tail filler after a stronger proposer and amortizes a longer verifier block.

Do not implement. Reopen only as trace oracle:
1. Existing six-family traces + at least one repo-edit/document-rewrite workload.
2. Test factory-MTP-D3 + lookup-tail and, if possible from existing sidecar traces, exact upstream DFlash2 + lookup-tail.
3. Verify widths 8/12/16 against real measured target costs/grouped N17-N64 route.
4. Lookup may only derive from current request/context; no curated token maps.
5. A spectacular copy/edit row is a special coding mode, not the global frontier. Promotion requires portable complete-round economics.

## Execution order
1. Confirm clean source except owner docs/progress; no agent/training process.
2. Start P0-A prefill profile immediately; the cold65K gap is already material and directly blocks practical200K coding.
3. In parallel by analysis only, P0-B long-KV mechanism comparison and P0-C PipeInfer trace ceiling. Whichever has larger complete-round ceiling gets implementation priority.
4. P0-D lookup-tail trace oracle only after target width costs are assembled.
5. Recompute single80 / dual100 / 65K100 / 262K100 after every material result. Do not stop when a lane closes.
