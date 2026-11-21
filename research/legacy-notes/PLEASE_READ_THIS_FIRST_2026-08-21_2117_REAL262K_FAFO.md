# 21:17 adviser — real 262K authority, single80 package, FAFO next

Read repo-root `AGENTS.md` first. Stock GPU settings only. No PCI/reset/clock/power/fan manipulation. No agents. No training. Gates remain 80 / 100 / 100 / 100 useful complete-path tok/s.

## New authority

The genuine populated-262K physical route is now measured, deterministic, and much worse than prior modeled placeholders:
- 261,900-token cold capture: 1201.235 s / 218.03 tok/s.
- physical INT8 sequence-sharded KV + deduplicated T5 TP median: **32.76 useful tok/s** across 32.84 / 32.76 / 32.71.
- exact trajectory: 42 rounds, 167 drafted, 83 accepted, by-pos 40,39,3,1.

Use this as the current 262K authority. Do not continue quoting the old ~50.85 modeled/fail-closed number as frontier.

## P0-M first — finish the real single80 discriminator

The combined mechanical package has a measured/bounded projection of **80.332 useful tok/s**:
- mechanical INT4-G128 factory-MTP body saving 1.664 ms with held-out D3 useful ratio 0.994178;
- exact Q5-down resident subset, 38 layers, saving 0.344 ms, 3.253 GiB packed storage;
- projected round 36.552 ms, useful/round 2.936292.

This is close-margin, so do not spend time polishing the model. Implement the minimal opt-in E2E discriminator now and let complete-path stock-card output decide. Promotion requires representative six-family behavior/trajectory quality; one favorable prompt is not enough. Mechanical quantization is allowed; no retraining or modified learned weights.

## Long-context research: FAFO is materially different and preserves final correctness

Fresh ICML-2026 research worth evaluating after P0-M:

**FAFO: Lossy KV Cache Compression for Lossless Inference Acceleration via Draftless Fumble Decoding**
OpenReview: https://openreview.net/forum?id=oSk9tP5Mgs

Key distinction from the KV4/KVarN/TurboQuant lanes already closed: FAFO does **not** promote the lossy-cache output as final. It lets a compressed/lossy KV path generate n-gram guesses, while the exact/lossless target verifies them in parallel. The paper reports 1.20–2.71x latency speedup while preserving baseline generation quality. This makes it a candidate architecture for our strict-output contract rather than another lossy-KV production format.

Do not port blindly. First build a complete-round ceiling using the **actual 262K authority**:
1. model a cheap compressed/sparse attention path only as a guess generator;
2. exact canonical INT8/full-KV target remains the acceptance oracle;
3. use our real target-stage / proposal / context-packet timings and measured Qwen trajectory;
4. account for candidate multiplicity and exact verification width;
5. advance only if a realistic width/yield model can approach/exceed 100, not merely improve 32.76 by 20%.

If the dependency structure collapses to the already-closed lookup-tail or multi-branch PipeInfer oracle, close it explicitly rather than reimplementing it. The question is whether FAFO's same-pass lossy-guess + exact-verify dataflow creates a new overlap/yield regime on this hybrid target.

## Dustin / Faster-Flash are research oracles, not promotion contracts

**Dustin: Draft-Augmented Sparse Verification for Efficient Long-Context Generation with Speculative Decoding** (ICML 2026) reports very large long-context speculative-verification gains by combining draft lookahead with target historical attention.
**Faster Than Flash / Faster Flash Decoding** (ICML 2026) fuses low-bit content-aware scanning and sparse attention and reports up to 2.37x E2E at 256K.

Both are useful for selector/dataflow ideas, but their published contract is accuracy preservation / negligible degradation, not bit-identical target generation. Do not promote an approximate target-attention path. Use them only to inform a FAFO-like or other exact-verification wrapper unless an exact acceptance proof is constructed.

## Already closed; do not recycle

- direct KV4/KIVI/KVarN/TurboQuant production cache routes;
- Quest/min-max page selection;
- perfect semantic retrieval / lookup-tail as global route;
- multi-branch successor PipeInfer ceiling (best impossible row ~84.34);
- resident Q4 Marlin target;
- same-family Q4/Q5 microtuning;
- direct sparse-attention approximation without exact verification.

## Autonomy

Do not stop when P0-M or FAFO classifies. Recompute all four gates from authoritative stock-card complete-path measurements, then continue to the next materially different architecture until all four pass.
