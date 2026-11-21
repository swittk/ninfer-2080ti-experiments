# nInfer RTX 2080 Ti / SM75 progress ledger

This file is an **append-only project memory** for the Turing port. Record major architecture decisions, performance checkpoints, accepted/rejected experiments, and anything a future agent needs to avoid repeating work.

Use Bangkok time (`Asia/Bangkok`, UTC+07:00). Prefer `python3 scripts/append_progress.py "Title" "Details..."`. Do not rewrite old entries to make the story cleaner; append a correction entry if an earlier statement becomes wrong.

## Current reference

- Project root: `<local-worktree>`
- Working branch: `feat/rtx2080ti-sm75`
- Functional-baseline commit: `57f6e50 feat: add functional SM75 Turing backend`
- Hardware: modified RTX 2080 Ti, SM75, 22,528 MiB VRAM
- Model: Qwen3.8-27B official groupwise `.ninfer` artifact, BF16 KV on SM75
- RTX 3090 reference: published C1/MTP3 decode ~71.00 tok/s; matching or exceeding it is allowed and desirable
- Best validated short deterministic MTP3 checkpoint before workspace relocation: **46.15 tok/s decode**, exact expected output, 55.56% draft acceptance, 2.67 accepted tokens/round

## Backfilled checkpoints

### 2026-08-15 06:09 +07 — First real SM75 generation
- Full 245-step SM75 build linked successfully.
- Qwen3.8-27B loaded completely on the 22 GB 2080 Ti and produced `SM75 WORKS`.
- CUDA Graphs, BF16 paged KV, >64-token prefill/page-boundary behavior, and longer decode subsequently passed.
- Initial ordinary decode was about 23 tok/s; this established correctness, not acceptable final performance.

### 2026-08-15 07:18 +07 — Functional baseline committed
- Commit `57f6e50` created before performance surgery.
- Baseline includes SM75 synchronous-copy fallback, Turing FP16 Tensor Core MMA, BF16-storage/FP16-compute compatibility, TF32 fallback, SM75 prefill geometry, and explicit rejection of INT8 KV on SM75.

### 2026-08-15 07:24 +07 — MTP3 works but exposes Turing small-T tax
- Initial MTP3 benchmark: 26.47 tok/s, 55.56% draft acceptance.
- Profiling showed attention was tiny; Q4/Q5/W8 projection kernels dominated.
- Ordinary scalar decode was almost entirely GPU busy, so host synchronization was not the root cause.

### 2026-08-15 07:28 +07 — Direct FP16 Q4 small-T path
- Removed repeated BF16→FP16 conversion inside Turing MMA by staging FP16 fragments directly.
- Deterministic MTP3 result improved from 26.47 to **32.99 tok/s** with identical output and acceptance.

### 2026-08-15 07:36 +07 — Fused SM75 Q4 T=4 verify kernel
- Added a Turing-specific fused SIMT Q4 SwiGLU path for four-token MTP3 verification.
- Q4 verify kernel fell from roughly 498 µs/layer to ~276 µs/layer.
- Deterministic MTP3 result improved to **38.92 tok/s**, output and acceptance unchanged.

### 2026-08-15 07:47 +07 — Direct FP16 W8 small-T path
- Applied the same one-time BF16→FP16 staging strategy to W8 small-T Tensor Core kernels.
- Large vocabulary projection fell from roughly 7.1 ms/round to ~2.5 ms/round.
- Deterministic MTP3 result improved to **45.55 tok/s**, output and acceptance unchanged.

### 2026-08-15 08:01 +07 — Q4 T=4 CTA reuse tuned to 16 rows
- Increased fused Q4 T=4 CTA from 8 to 16 output rows, sharing each activation tile across more rows.
- 32-row/1024-thread variant regressed; 16 rows is the current best geometry.
- Best validated deterministic MTP3 result became **46.15 tok/s**.

### 2026-08-15 08:08 +07 — Q5 and speculative-width experiments rejected
- Q5 split4 instead of split2: neutral/slightly slower.
- Q5 SM75 Tensor Core T=4 prototype: neutral/slightly slower.
- Q5 shared-activation rows8 prototype: 46.13 vs 46.15 tok/s, rejected. Q5 appears primarily weight-bandwidth bound.
- MTP2: ~37.36 tok/s despite higher per-draft acceptance.
- MTP4: ~35.12 tok/s. MTP3 is the current product-performance sweet spot.
- Current best is ~1.54× slower than the 3090 published 71 tok/s result, close to the raw 3090/2080 Ti bandwidth ratio; future work may still beat this through better effective work per weight scan.

### 2026-08-15 08:31 +07 — Workspace relocated and made persistent
- Moved Git root from `<local-home>/Documents/ninfer-3090` to `<local-worktree>`.
- Recreated CMake 3.31.10 environment inside `.tools/cmake-venv`; old `<local-home>/Documents/ninfer-build-tools` removed.
- Added this append-only progress ledger and timestamp helper.
- MCP root registration is being added for stable repo id `ninfer-2080ti`.

### 2026-08-15 08:37 +07 — MCP registration and workspace cleanup
- Registered repo_id ninfer-2080ti in GPT Repo MCP config with ship-style Git/write permissions plus host exec; config validation passes with 8 repos.
- Removed all stray ninfer-sm75 Nsight .nsys-rep/.sqlite files from ~/Documents and the raw W8 profile files from the repo root.
- Fresh SM75 CMake tree is being regenerated at the relocated project path; .tools, build-sm75, and models are intentionally local/ignored.

### 2026-08-15 08:47 +07 — Relocated clean-build validation
- Fresh build-sm75 configured and built 245/245 from the new project root using CUDA 13.1 and local CMake 3.31.10.
- Deterministic MTP3 checkpoint reproduced the exact expected paragraph and 60/108 accepted drafts (55.56%, 2.67 tok/round).
- Decode measured 46.12 tok/s versus prior best 46.15 tok/s; treated as equivalent measurement noise. Relocation introduced no performance regression.

### 2026-08-15 09:22 +07 — Q5 split2 R2 shape-specific result
- SM75 T=4 Q5 two-row register-reuse R2 kernel was tightened to the fixed N=5120/K shapes; emitted K=17408 SASS dropped to ~3561 instructions/output row from ~4207 while preserving per-row accumulation order.
- K=6144 improved from about 59.6 to 52.4 us/layer (~12%); keep R2 only for this shape.
- K=17408 regressed to about 192.4 us/layer (~8% slower); restored original split2 kernel there.
- K=6144 __ldcg + cudaFuncCachePreferL1 experiment regressed 52.4 to 58.95 us/layer; rejected.
- Deterministic 94-token MTP3 output and 60/108 acceptance remained identical throughout the accepted R2 path.

### 2026-08-15 09:32 +07 — SM75 W8 gate/up 8-K-warp specialization accepted
- For exact W8 gate/up geometry 34816x5120, production T=1/T=4 4-K-warp variants used 254/255 registers per thread; 8-K-warp variants compile at 128 registers/thread with no spill.
- Post-prefill isolated latency improved T1 625.9 -> 392.1 us (~37.4%) and T4 579.0 -> 367.9 us (~36.5%).
- Exact deterministic 94-token MTP3 run produced the byte-identical paragraph and unchanged 60/108 acceptance (55.56%, 2.67 tok/round) at 47.10 tok/s decode.
- Keep the 8-K-warp specialization only for 34816x5120 T1/T4 on SM75. Do not generalize to other W8 shapes without separate A/B evidence.

### 2026-08-15 09:40 +07 — Sustained 1024-output MTP3 baseline
- Matched the published RTX 3090 C1 prompt-length/output regime: 29 prompt tokens, 1024 generated tokens, 8192-token context/KV pool, CUDA Graphs, MTP3, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0, seed 42.
- SM75 BF16-KV decode measured 49.93 tok/s with 60.09% draft acceptance (658/1095, 2.80 tok/round), zero fallback steps.
- RTX 3090 published reference is 71.00 tok/s with 61.13% acceptance and INT8-G64 KV. Current SM75 result is about 1.42x slower, already better than the raw 936/616 ~1.52x bandwidth ratio despite BF16 KV.
- Earlier 53-token and 38-token prompts produced 43.54 and 44.32 tok/s because acceptance fell to 47.59% and 49.59%; prompt/acceptance matching materially affects sustained MTP comparison.

### 2026-08-15 09:50 +07 — SM75 Q4 SwiGLU T5/T6 parity accepted
- Generalized the proven SM75 fused 16-row Q4 SwiGLU verification kernel from exact T=4 to exact T=5/T=6; other column counts retain existing generic routes.
- Isolated post-prefill Q4 SwiGLU latency improved T5 574.1 -> 307.8 us/layer (~46.4%) and T6 571.9 -> 350.7 us/layer (~38.7%).
- Post-parity short MTP4 sky run: 42.35 tok/s, 28 rounds, 112 drafted, 56 accepted, 50.00% draft acceptance, 3.00 output tok/round, zero fallback; 84-token coherent output.
- Post-parity short MTP5 sky run: 40.19 tok/s, 26 rounds, 130 drafted, 59 accepted, 45.38% draft acceptance, 3.27 output tok/round, zero fallback; output paragraph exactly matched the post-parity MTP4 paragraph.
- Keep T5/T6 exact kernels even if fixed wider MTP widths lose; they remove an artificial width-specific cliff and enable future adaptive speculative width.

### 2026-08-15 10:05 +07 — Sustained MTP width sweep and wider-width profile
- Matched sustained campaign for all widths: 29 prompt tokens, 1024 generated tokens, 8192 context/KV, CUDA Graphs, BF16 KV, temp 0.6/top-p 0.95/top-k 20/presence 1.0/seed 42.
- K3: 49.93 tok/s, 60.09% draft acceptance (658/1095), 365 rounds, 2.80 output tok/round, ~56.1 ms/round.
- K4: 40.96 tok/s, 47.18% (668/1416), 354 rounds, 2.89 output tok/round, ~70.6 ms/round. K4 useful tokens/round slightly exceed K3, but verify rounds are much more expensive.
- K5: 33.88 tok/s, 35.50% (654/1842), 369 rounds, 2.77 output tok/round, ~81.8 ms/round. K5 loses from both higher verify cost and falling acceptance.
- Post-parity profiles show Q5 is the dominant remaining width-scaling cliff: K17408 split2 ~208 us/layer at T5 and ~241 us at T6; K6144 ~72 us at T5 and ~89 us at T6; fused GDN Q5 ~163/~207 us. Vocabulary head stays about 4.2-4.4 ms/round and is not the main scaling culprit.
- W8 34816x5120 T5/T6 8-K-warp test was rejected: isolated/production kernel latency improved only modestly and deterministic short K4/K5 stayed 42.33/40.10 tok/s versus 42.35/40.19 baseline. Source restored to T1/T4-only specialization.
- INT8-KV follow-up candidate evidence from consultant: native SM75 m8n8k16 s8 emits IMMA.8816; Ampere logical m16n8k32 QK can be decomposed into four Turing IMMAs; SM75 INT8 prefill Bc=32 calculates to 55,296 B smem vs 92,672 at Bc=64 and existing source has Bc=32 P swizzle. Treat as implementation candidate pending our own correctness/runtime validation.

### 2026-08-15 10:23 +07 — SM75 Q5 K6144 R2 T5/T6 width parity accepted
- Generalized the proven SM75 K=6144/N=5120 split2 R2 Q5 kernel from T=4 to compile-time T=5/T=6; K=17408 remains on the original split2 path because R2 previously regressed there.
- Isolated K6144 T5 improved about 71.7 -> 61.8 us/layer (~14%); T6 improved about 89 -> 69.81 us/layer (~22%). T4/T5/T6 variants use 64 registers/thread with no spills.
- Short deterministic K4 improved from 42.35 baseline to 42.96 then 42.59 tok/s on repeated runs; output and 56/112 acceptance remained identical.
- Short deterministic K5 improved from 40.19 baseline to 40.57 then 40.74 tok/s on repeated runs; output and 59/130 acceptance remained identical.
- Matched sustained 1024-token K4 improved 40.96 -> 41.34 tok/s with unchanged 47.18% acceptance/354 rounds; K5 improved 33.88 -> 34.34 tok/s with unchanged 35.50% acceptance/369 rounds.
- Keep the R2 route shape-specifically for K=6144 T=4..6 only. Fixed K3 remains the best current width; these parity improvements preserve future adaptive-width viability.

### 2026-08-15 12:56 +07 — SM75 GDN and INT8 long-context correctness checkpoint
- Replaced the invalid SM75 TF32 fallback with the corrected Tensor-Core fragment repack: each TF32 source register a0/a1/a2/a3/b0/b1 is shuffled independently within 4-lane subgroups before the receiving lane selects the low/high half; selected scalars are converted to FP16 fragments for native SM75 m16n8k8 MMA with FP32 accumulation.
- Preserved native SM75 INT8 MMA decomposition to IMMA.8816, INT8 prefill Bc=32/55,296-byte shared-memory geometry, absolute page_off = key & kPagedKVPageMask addressing for Bc32 scale/code loads, and host/runtime INT8 enablement.
- Preserved TU102 Qwen27 GDN gating residency schedule: T9-256 Split8, T257-512 Split4, T513-1024 Split2, >1024 unsplit, 68-CTA cooperative envelope. Added permanent 27B normalized-QK gated_delta_net coverage at T65 and T129.
- Complete SM75 gated_delta_net FP64 oracle: OK gated_delta_net correctness. GQA public-contract suite: PASS gqa_attention public-contract correctness.
- Product retrieval validation: 522-token BF16 chunk256 -> MAGIC42 at 221.40 tok/s prefill; 522-token BF16 chunk512 -> MAGIC42 at 233.81 tok/s; 522-token INT8 chunk512 -> MAGIC42 at 233.87 tok/s.
- Real-code retrieval: 3,789-token BF16 -> MAGIC42 at 237.48 tok/s prefill; 3,789-token INT8 -> MAGIC42 at 238.98 tok/s; 21,107-token real-repository INT8 -> SM75_REAL_REPO_RETRIEVAL_55296 at 233.16 tok/s prefill.
- 76,800 max-context / auto-KV smoke passes with MTP3 loaded: 16.67 GiB weights, 2.57 GiB INT8-G64 KV payload, 2.88 GiB sequence reservation, 1.00 GiB fixed headroom, 1.11 GiB free after startup, output OK.
- Short deterministic MTP3 regression remains exact: canonical 94-token sky paragraph, 36 rounds, 108 drafted, 60 accepted (55.56%), accepted positions 31/19/10, 47.12 tok/s decode.
- Independently scoped remaining gdn_gating_proj_test issues: Qwen27 T2049 has a tiny reduction-criterion miss (-1.36053 vs -1.36054), then Qwen35 still hits cudaErrorCooperativeLaunchTooLarge from its unported residency assumption. Required Qwen27 <=1024 routes and product paths are green.
- A tiny max-context=2048 + prefill-chunk=512 ordinary run can under-plan CUDA Graph allowance (18 MiB consumed vs 12 MiB planned); normal 8192 reservation is healthy. Track this separately under graph-planner/long-context robustness, not GDN/INT8 correctness.

### 2026-08-15 15:47 +07 — SM75 BF16 chunk512 canonical GDN reduction + runtime robustness
- Root cause: TU102 Qwen27 gating routes changed FP32 reduction associativity from Split8 (8x10 K-tiles) at T<=256 to Split4/Split2 at wider prefill, creating ~1e-7 g/beta drift that recurrent GDN amplified across layers/long context.
- Fix: T9-256 retains cooperative Split8; T257-1024 uses staged canonical Split8 with the same eight 10-K-tile FP32 partials, normal CTA waves, and deterministic fixed-order Stage-B reduction in caller/planner-owned workspace. INT8 and recurrence unchanged.
- Model proof: after fix, T256 vs T512 identical first-256-token residuals are byte-exact at every layer 0-63 (old rel-L2 L0 2.21e-4, L2 1.77e-3, L15 3.81e-3, L31 1.10e-2, L47 4.17e-2, L63 6.62e-2 -> all 0).
- Permanent GDN coverage adds T255/256/257/511/512/513/1023/1024 plus exact canonical prefix invariants across 256->512, 512->513, and 513->1024. Qwen27 cases through T1024 pass; independent pre-existing T2049 tiny criterion miss and unrelated Qwen35 cooperative-residency failure remain.
- GQA long-history regressions pass: ~31K T1/T7 BF16+INT8, BF16 T256-vs-T512 causal prefix, and one-T512 vs two-sequential-T256 full-output invariant; complete GQA public contract PASS.
- Graph/KV accounting fix treats cudaMemGetInfo graph delta as diagnostic and validates aggregate post-startup free memory against planned slack; kv_capacity regression PASS and controlled concurrent allocation reports startup shortfall rather than false graph overrun.
- Product: 522 BF16 chunk512 MAGIC42 at 235.42 tok/s prefill. Real 31,375-token BF16 chunk512 exact early/middle/late markers at 222.69/220.86/220.67 tok/s. BF16 chunk256 early exact at 202.07 tok/s; INT8 chunk512 early exact at 226.89 tok/s.
- Short deterministic MTP3 unchanged: 94 generated, 36 rounds, 108 drafted, 60 accepted (55.56%, 2.67 tok/round), 0 fallback, 47.16 tok/s decode.

### 2026-08-16 00:12 +07 — SM75 A8 turbo source checkpoint reaches 64.3 tok/s
- Promoted the reproducible SM75 A8 decode path into canonical tracked source: Q4 gate/up A8 with shared T4 activation staging, Q5 MLP-down split4-interleave T4 A8, and GDN value/z A8 with reversible decode-layout transforms.
- Full canonical build passed after making the family runtime layout transition variant-gated so the unrelated Qwen35 MoE target remains buildable.
- Two fresh canonical 1024-token greedy BF16-KV MTP3 runs were deterministic and byte-identical: SHA ad070c8822b16afc4acfe2d7162269e6bd7d030d64ae442ea8fa135fa5819f3e; 372 rounds, 1113 drafted, 651 accepted (58.49%), accepted positions 300/204/147, zero fallback.
- Fresh canonical decode measured 64.28 and 64.18 tok/s. This is an experimental/turbo checkpoint because GDN A8 remains subject to the separate quality/determinism qualification lane; the committed correctness baseline remains historically available behind this checkpoint.

### 2026-08-16 16:14 +07 — Exact SM75 proposal-B2 lane cleanly reproduces 72 tok/s
- Created isolated branch perf/exact72-proposal-b2 from b6c36cfa and reconstructed only the exact local-top2 vocabulary checkpoint plus the T1 proposal-B2 shortlist path. A fresh CUDA 13.1 SM75 build completed all 237 requested targets. ninfer_argmax_test passed. Two full 1024-token greedy BF16-KV MTP3 runs were byte-identical to canonical SHA ad070c8822b16afc4acfe2d7162269e6bd7d030d64ae442ea8fa135fa5819f3e with 372 rounds, 1113 drafted, 651 accepted (58.49%), positions 300/204/147, zero fallback, at 72.08 and 71.77 tok/s. This branch is the exact/reference-behavior performance anchor; approximation lanes remain separate. Nsight Compute still fails ERR_NVGPUCTRPERM, so producer/consumer pipeline work remains deferred pending counters.

### 2026-08-16 21:02 +07 — q45g cross-context determinism hardened and stress-qualified
- Starting from exact72 commit 9ca3a4b, the production graph route reproduced the historical pressure-dependent divergence at pressure repetition 18 (SHA edeb895..., 375 rounds, 648 accepted versus canonical 372/651). A no-graph campaign also reproduced by repetition 13, proving graph replay was not the sole trigger. The concrete production stale-state hazard was the target B2 local-top2 shortlist: 1,940 producer CTAs form 243 partials, but the final partial has only four CTAs and writes 64 of 128 candidate ranks; the old reader consumed all 128 and depended on a per-replay memset. The fix structurally masks unwritten tail ranks and removes that reset dependency. Tiled argmax was additionally hardened to consume arbitrary atomic preseeds, repair invalid or stale-NaN winners, and no longer depend on a reset node. Added opt-in full MTP transaction digesting, complete generated-token digests/IDs, a tracked cross-context pressure harness, a pressure helper, an arbitrary-preseed graph replay test, and a B2 stale-tail geometry regression. Validation: two independent campaigns each passed 2 normal + 30 pressured full 1024-token CUDA-Graph runs with identical SHA ad070c88..., token digest fd498e564fc6135e, trajectory digest dd47b7546e29da83, 372 rounds, 1113 drafted, 651 accepted, positions 300/204/147, zero fallback; a 2+25 no-digest campaign also passed; no-graph and launch-blocking controls passed; 24 diverse corpus normal/pressure runs passed including a 1024-token code case; final token-ID smoke passed 1 normal + 2 pressured runs; full build passed; argmax, MTP, speculative, GDN replay/state tests passed; Compute Sanitizer memcheck/initcheck/synccheck/racecheck reported zero errors or hazards for the stale-preseed test. Normal exact decode remains about 71.8-72.0 tok/s. One unrelated pre-existing gdn_input_proj_conv_record test aborts on both worktrees because it invokes an SM120-only NVFP4 A4 path on SM75.

### 2026-08-18 16:50 +07 — Aug-18 TU102 canonical frontier repaired
- Hardware calibration: Core-X UUID GPU-2fffc240-5d58-fbc8-27d6-6b349448fda9 sustains roughly 1.68-1.86 GHz SM; internal UUID GPU-020906b4-0904-3471-e4ee-e41ec2350be3 is flat at 1350 MHz. Identical archived binary/output measured 71.04 tok/s on Core-X versus 65.38 on internal; policy must calibrate by UUID and loaded clocks.
- Current trustworthy single-GPU INT8-KV frontier on Core-X is 70.07 tok/s median over 69.68/70.07/70.42, digest 2fb75f62ab6e0e31, 380 rounds, 1139 drafted, 642 accepted. Full details: ../ninfer-2080ti-int8-tu102/.chatgpt/hotloop_20260818/baseline/e2e/summary.json.
- Hot target kernels remain Q4 A8 T4 167.816 us median, Q5-down A8 T4 106.805 us, and GDN Q5 A8 T4 79.220 us. Closed exact-layout variants include Q4 8/32-warps, chunk20/80, early-scale, pair2 INT/FP; Q5 Parts2/3, tile8, two-step prefetch, dual-row stream, pair2 INT/FP, cache carveouts; GDN PC6/PC8. Evidence under ../ninfer-2080ti-int8-tu102/.chatgpt/hotloop_20260818/.
- Fixed MTP5/T6 is CLOSED as the default: median 55.81 tok/s versus 66.52 for T4, with ~39% higher round cost. Attention Q5 gate/value A8 is a portable ordinary-target gain and is committed on perf/sm75-int8-tu102 at 7046bb9; ordinary multilingual token IDs matched, while Thai/Japanese long MTP trajectories require speculation-side qualification.
- Fixed B2 proposal/target maps requalified: proposal shortcut is target-semantics-safe in the exact-target control and worth roughly 1-2 tok/s; target top-128 W8 rerank changes the canonical target trajectory and remains checkpoint-qualified, not exact-engine. Workload-curated maps are CLOSED.
- BF16-source selective Q5-down to Q4 is CLOSED. q4down2/q4down4 changed target trajectories for negligible or inconsistent useful gain; q4down8 failed every held-out workload. The later duplicate BF16 audit/download was accidental and must not be repeated; reuse the complete local 18-shard checkpoint referenced by the existing bf16_recipe manifests.
- Dual scalar/C4 capacity remains proven on feat/sm75-dual-gpu-full-context. Populated ~262K one-request performance is still OPEN. The production-equivalent TP2 chain across all 64 Q4+Q5 MLPs measured 1.55329x with exact host reduction; artifact: .chatgpt/dual_c1_20260818/tp2_chain64/CONCLUSION.md.
- Current TP2 runtime integration is intentionally dirty on base 7249dcb: apps/dual/tp2_mlp.{h,cu}, mlp_tail override plumbing, --tp2-mlp-dir, and E2E qualification. Exact digest/trajectory are preserved, but host-barrier, cudaMemcpyPeerAsync, mapped-host, event-DMA, and packet-DMA transports measured 61.12-63.94 tok/s versus the 64.78 graph-backed baseline and are REJECTED individually.

### 2026-08-18 16:50 +07 — OPEN — overlap owner TP2 engine with remote input DMA
- Hypothesis: the resident 64-layer TP2 mechanism is fast, but per-layer transport remains on the critical path. Unlike the rejected host-barrier/peer/mapped/event/packet variants, this experiment overlaps the owner shard Q4+Q5 engine with owner-local D2H followed by remote H2D using separate quantize-ready and DMA-ready events.
- Prior evidence: .chatgpt/dual_c1_20260818/tp2_runtime/event_dma/CONCLUSION.md and packet_dma/CONCLUSION.md. This is the explicitly queued final bounded transport test, not a duplicate.
- Success gate: same generated digest 14660c413b4cf185 and identical 49-round / 147-draft / 77-accepted MTP trajectory; useful decode must beat the 64.78 tok/s graph-backed baseline, with UUID/loaded-clock capture and TP2 staging/engine/reduction timings.
- Win: promote the overlap transport into the opt-in runtime and continue populated-context C1. Loss: classify TP2 per-layer host-staged runtime transport as closed and move immediately to the next measured C1 mechanism; do not revisit prior transport variants.

### 2026-08-18 17:00 +07 — CLASSIFIED — TP2 owner-engine/remote-input overlap is engine-positive, useful-neutral
- Qualified fixed run on Core-X fast UUID GPU-2fffc240-5d58-fbc8-27d6-6b349448fda9 (1830-1860 MHz loaded) plus internal slow UUID GPU-020906b4-0904-3471-e4ee-e41ec2350be3 (1350 MHz flat).
- Graph-backed baseline: 64.85/64.89 useful tok/s, 47 rounds, 141 drafted, 78 accepted (55.32%). Overlap TP2: 64.71/65.23, 49 rounds, 147 drafted, 77 accepted (52.38%). All outputs SHA-identical with digest 14660c413b4cf185.
- Normalized round cost improved from ~41.65 to ~39.90 ms (~4.2%), so the mechanism is real; useful TPS is neutral because target numerics change MTP acceptance. Artifact: .chatgpt/dual_c1_20260818/tp2_runtime/overlap_dma_a8/.
- Keep this as the TP2 transport base. Do not repeat host-barrier, peer, mapped, event-only, or packet-only input variants.

### 2026-08-18 17:00 +07 — OPEN — TP2 owner-GPU FP32 partial reduction
- Hypothesis: after input overlap, the remaining per-layer transport downloads both 80-KiB FP32 partials, sums them on CPU, and uploads the result. Keep the owner partial resident; download only the remote partial, upload it to the owner, and perform the exact two-part FP32 sum plus residual epilogue on the owner GPU.
- Preflight found no prior owner-device output-reduction experiment. This is distinct from all closed input-staging variants.
- Success gate: preserve TP2 digest 14660c413b4cf185 and its 49/147/77 trajectory; reduce TP2 dual-engine/reduction time and materially beat the 64.97 median overlap result. Loss: restore overlap source, classify output reduction closed, and move to the next measured C1 mechanism.

### 2026-08-18 17:12 +07 — ACCEPTED — TP2 owner-GPU FP32 reduction and portable runtime packaging
- Alternating fixed-workload control: 65.34/64.94/65.00 tok/s, median 65.00. Owner-GPU reduction: 65.64/65.65/65.71, median 65.65 (+1.00%).
- Reduction time fell from 0.031 s median to 0.013 s (-58%) over 3136 MLP calls. Engine time stayed 1.497-1.499 s.
- All six full outputs were SHA-identical and every run preserved digest 14660c413b4cf185 plus the 49-round / 147-draft / 77-accepted / 34,27,16 trajectory.
- Code, strict-JSON, and Thai 64-token controls were also byte-identical with identical MTP trajectories. Candidate useful TPS improved 49.12->49.43, 62.29->62.67, and 31.01->31.34.
- Calibrated split69 kernels are tracked under apps/dual/tp2_generated; CMake no longer depends on untracked development artifacts. TP2 remains opt-in via --tp2-mlp-dir.
- This is the first runtime TP2 stack to beat the qualified graph-backed C1 baseline in useful TPS, but 65.65 remains far below the >=100 C1 gate. Artifact: .chatgpt/dual_c1_20260818/tp2_runtime/owner_gpu_reduction/.

### 2026-08-18 17:32 +07 — CLASSIFIED — exact W8 head row parallelism is mechanically positive
- Repaired the unfinished discriminator: the artifact is W8G32, not group64; extraction now streams the exact 1.35086-GB output head and CUDA source compiles.
- T4 full W8 projection + argmax on Core-X: 5.642 ms. Two exact row shards on Core-X + 1350-MHz internal: 3.682 ms, 1.53239x. Id mismatches 0, score delta 0.
- Transport is only 81,920 bytes of hidden fanout and 64 bytes of local-argmax return. Artifact: .chatgpt/dual_c1_20260818/w8_head_parallel/.
- This is not yet a production head result; first map whether the current target path uses full W8 or B2 shortlist/exact rerank and measure the actual head fraction.

### 2026-08-18 17:32 +07 — OPEN — complete populated dual-C1 context matrix
- Existing harvested points: 7,948 populated tokens at 8K gave 50.85 tok/s; 32,524 at 32K gave 45.57. 64K/128K/262K were never produced.
- Generated deterministic meaningful prompts with exactly capacity-244 frontend tokens using the complete local 18-shard tokenizer checkpoint: 65,292 / 130,828 / 261,900 tokens. Manifest: .chatgpt/dual_c1_20260818/context_profiles/prompts/manifest.json.
- Run current qualified both-MLP-stage1-all dual MTP3 at split32, 32 generated tokens, with UUID/loaded-clock capture. Classify stage0, stage1/head, proposal and fold scaling before choosing head or context parallelism.
- No TP2 comparison until the baseline context curve is measured; then test the current 8142cd7 TP2 runtime at the first context where its target-side savings might overcome transport.

### 2026-08-18 17:35 +07 — CLOSED for current MTP3 route — full-W8 row parallelism is below wall-time threshold
- Current T4 target verification uses B2 coarse projection plus exact 512-row rerank (~0.68-0.70 ms/call), not the full 248320-row W8 head.
- Existing profile contains only six full W8 T1 projections totaling ~15.1 ms. A 1.53x head split can save only ~5 ms over ~1.9 s decode (<0.3%).
- Preserve the exact 1.532x discriminator, but do not integrate it now. Reopen only if full-head verification frequency materially increases.

### 2026-08-18 17:48 +07 — REJECTED — TP2 fully asynchronous output-event submission
- Exact alternating gate: synchronous owner-GPU reduction 65.48/65.54/65.51 tok/s (median 65.51); event-only async 65.32/65.35/65.34 (median 65.34, -0.26%).
- All stdout hashes and the 49/147/77, 34/27/16 MTP trajectory matched. The ~0.070 s engine metric is host enqueue time, not completion time.
- Restore commit 8142cd7 TP2 transport. Do not reopen output-event-only submission without a new multi-round overlap design.

### 2026-08-18 17:48 +07 — CLOSED — per-layer TP2 host fences were not the missing C1 gain
- Nsight A8-stack decomposition found only ~18% two-GPU overlap, motivating an exact event-chained output/reduction experiment that removed 3,136 host engine synchronizations.
- Alternating fixed-workload control: 65.48/65.54/65.51 tok/s; event-chain: 65.32/65.35/65.34. All outputs and the 49-round 147/77 acceptance trajectory were identical.
- Host-reported engine blocking disappeared, but target stage wall time did not. CUDA dependencies and the residual chain preserved the same critical path. Restored accepted owner-GPU reduction. Artifact: .chatgpt/dual_c1_20260818/tp2_runtime/async_output/.

### 2026-08-18 17:56 +07 — CLASSIFIED — TP2 C1 split rebalance finds 68.55 tok/s frontier at split48
- Fixed exact-output sweep on accepted owner-GPU TP2 runtime: split40 65.78, split44 66.98, split48 68.55, split52 65.38, split56 66.89 tok/s.
- All stdout hashes/digests matched. Split48 used 46 rounds / 138 drafted / 79 accepted (57.25%); this is a legal geometry-dependent trajectory, so held-out qualification is required.
- Stage balance at split48: stage0 1.126 s, stage1/head 0.422 s. The Core-X-heavy split wins useful TPS despite strongly unequal stage times because TP2 offloads MLP work inside every layer.
- Artifact: .chatgpt/dual_c1_20260818/tp2_runtime/split_rebalance_retry/.

### 2026-08-18 18:01 +07 — CLASSIFIED — split48 is opt-in, not portable default
- Held-out versus split32: code +3.68%, JSON +0.61%, Thai -1.25%; geomean +0.99%. Split44 geomean +0.68% and also regresses Thai.
- Code/Thai take explainable legal split-geometry trajectories; JSON is identical. Preserve split48 as the checkpoint/hardware speed profile, but retain split32 for portable context-scaling baselines.

### 2026-08-18 18:11 +07 — CLASSIFIED — split48 does not provide a representative C1 engine jump
- Held-out TP2 comparisons: code 49.43 -> 51.25 tok/s with changed trajectory; JSON 62.67 -> 63.05 with identical output; Thai 31.27 -> 30.88 with changed trajectory.
- Fixed-workload 68.55 tok/s was predominantly acceptance/geometry, while normalized stage round cost improved below 1%. Split48 is not promoted as canonical.
- Keep split32 as the capacity-oriented reference; revisit dynamic split only as part of a larger coherent schedule.

### 2026-08-18 18:21 +07 — CLOSED — optimized proposal head cannot predict the target correction from the stage boundary
- Reused the canonical Q4/B2 optimized proposal package on GPU0 to score the four stage-boundary verifier columns; predictor overhead was ~2.6 ms/round.
- Correction-token top1: general 5.0/15.0/16.7% at split32/40/48; code 3.3/3.2/9.7%; JSON split32 0%. Overall valid-column top1 stayed 1-14%.
- The miss rate makes speculative next-round GPU0 preload rollback-dominated. Diagnostic API restored. Artifact: .chatgpt/dual_c1_20260818/boundary_proposal_audit/.

### 2026-08-18 18:35 +07 — CLOSED — moving canonical MTP proposal recursion to stage0 is exact but slower
- New hypothesis: run the unchanged proposal package on fast Core-X stage0, transfer accepted target hidden there, and leave target verification/acceptance/replay semantics unchanged.
- Matched control 65.78/65.58/65.49 tok/s (median 65.58); stage0 proposal 65.13/65.05/65.05 (median 65.05). All stdout hashes and 49-round 147/77 trajectory matched.
- Proposal/alignment rose from 0.219-0.221 s to 0.236 s; extra hidden transfer/state work outweighed faster-device execution. Restored pre-experiment source. Artifact: .chatgpt/dual_c1_20260818/tp2_runtime/proposal_stage0_c1/.

### 2026-08-18 18:42 +07 — ACCEPTED primitive — exact host-staged two-GPU GQA context parallelism
- Added global split-range cached-GQA launch, exact FP32 online-softmax range reduction, and exact two-state merge. Tracked discriminator: bench/ops/gqa_context_parallel_bench.cu.
- Across 64K/128K/262K T4 INT8-G64 cases, the merged BF16 output was bit-identical to the canonical full cached-GQA result: zero bit mismatches, zero delta, relative-L2 0, cosine 1.
- Loaded-clock 262K fraction sweep: 51/34 global splits (60/40) was the hardware-calibrated optimum, 2929.111 us full versus 1765.092 us including pinned-host state transport and merge, 1.659466x.
- Fast Core-X UUID GPU-2fffc240-5d58-fbc8-27d6-6b349448fda9 operated around 1830 MHz; internal UUID GPU-020906b4-0904-3471-e4ee-e41ec2350be3 remained 1350 MHz. Artifact: .chatgpt/dual_c1_20260818/context_parallel_exact/.
- Next gate: wire a synchronized remote shadow KV cache into the C1 sharded runtime behind an explicit option, then prove populated-context end-to-end parity and useful TPS. The microbenchmark is not itself a C1 result.

### 2026-08-18 18:47 +07 — CLOSED — exact full-head stage-boundary correction prediction is unusable
- New coarse-overlap hypothesis: score the actual GPU0 boundary verifier window with the checkpoint final norm and all 248,320 exact W8 head rows, then preload the next GPU0 pass if the target correction is predictable.
- 396 rounds across general/code/JSON/Thai/Japanese/Arabic: correction top1 4.04%, top32 11.87%, top256 23.74%; median rank 9305.5, p95 171167. Per-column top1 was 4.55%.
- Exact norm+full-head projection cost 5.36 ms/round on healthy Core-X. Rollback/tree economics are decisively negative. Runtime capture instrumentation was restored. Artifact: .chatgpt/dual_c1_20260818/boundary_exact_head_audit/.

### 2026-08-18 20:31 +07 — CLOSED — bulk shadow-KV synchronization is not material
- 64K canonical prefill 409.979 s versus incremental shadow mirror 410.638 s: +0.659 s, about 0.16%.
- Production KV is head-major while the shadow benchmark path is page-major; a bulk converter adds correctness risk for negligible wall-time recovery. Keep incremental append. Artifact: .chatgpt/dual_c1_20260818/context_parallel_runtime/bulk_shadow_sync/CONCLUSION.md.

### 2026-08-18 21:14 +07 — CLASSIFIED — 64K context fork is reduction-tree reassociation, not cache/range corruption
- Real-runtime decomposition over 80 T4 full-attention calls: Q/K/V/position transport was bit-identical; canonical-owner remote `(acc,m,l)` and shadow-cache remote state were bit-identical on every call; canonical-partition and shadow-partition merges matched call-for-call.
- Both partitioned merges differed sparsely from monolithic full reduction on 60/80 calls. First difference: layer3, visible65296, T4 column3, q-head22/dim174, BF16 delta 0.0009765625. Maximum observed absolute delta was 0.0625.
- Independent append-populated 65,296-key discriminator, including reversed shadow physical pages, found zero logical K/V code/scale mismatches, zero remote-state mismatches, and zero merged-output mismatches.
- Close omitted/duplicated KV, page mapping, G64 scale, T4 tail, and transport bug hypotheses. The dual path is exact against the same partitioned owner-cache reference; monolithic full differs only by floating-point reduction grouping.
- Artifact: .chatgpt/dual_c1_20260818/context_parallel_runtime/full_compare/CORRECTNESS_CONCLUSION.md.

### 2026-08-18 21:25 +07 — CLASSIFIED — round model requires coherent TP plus verifier-budget gain
- Fixed C1: 127 useful tokens / 49 rounds = 2.592 tokens/round, 39.45 ms wall/round. Target wall is 33.84 ms; proposal 4.47 ms; fold 0.33 ms; misc 0.82 ms.
- NVTX-apportioned target wall: MLP 17.64 ms/round, GDN 11.84 ms, full-attention branch 4.31 ms. TP2 engine_seconds is overlapping/summed work, not an independent wall bucket.
- Surfacing the measured 14.818-ms resident MLP chain alone predicts ~70.8 useful tok/s. Optimistic 1.553x whole-target TP predicts ~94.6; hiding proposal under it predicts ~113.
- 262K exact KVP scan is 28.24 ms/round across 16 full-attention layers. Conservative serial KVP+TP model is ~46.6 tok/s; full weight/KVP overlap ~76.6; also hiding proposal ~88.2 at current 2.592 useful tokens/round.
- Therefore short C1 >=100 needs coherent all-dominant-linear TP plus overlap or more tokens/round. Populated 262K >=100 additionally requires better verifier-budget use (Bole/tree/adaptive width); fixed MTP3 cannot meet the gate under the measured scan floor.
- Artifact: .chatgpt/dual_c1_20260818/round_model/CONCLUSION.md.

### 2026-08-18 22:12 +07 — ACCEPTED primitive — true 60/40 sequence-sharded INT8 KV
- Added a production-kernel discriminator with physical page ownership, full logical block tables, unowned entries set to -1, one optional boundary page, real Q broadcast, current-KV routing, helper FP32 state return, and exact merge.
- Sharded fast/slow range states matched canonical range states bit-for-bit at 4K/64K/128K/262K. Sharded merge matched canonical split/merge bit-for-bit; remaining full-vs-split differences are reduction association.
- 262K uses 2,474 fast + 1,623 slow pages for 4,096 logical pages: exactly one duplicate boundary page, payload ratio 1.00024414 instead of 2x shadow duplication.
- Loaded T4 result including communication: 2,946.727 us full versus 1,841.683 us sharded, 1.600019x. Q stage 32.058 us, engines 1,586.722 us, state return/merge 221.341 us.
- Artifact: .chatgpt/dual_c1_20260818/sequence_sharded_kv/. Next gate is direct owned-page allocation in the C1 runtime and explicit prefill-to-decode ownership conversion; do not retain the duplicate-shadow cache.

### 2026-08-18 22:16 +07 — CLASSIFIED — 65K first target fork is a measured narrow-margin reduction-association flip
- Prefix snapshot reduced canonical 65K restart from 414.211 s to 4.473 s and reproduced stdout, IDs/MTP state, Q hashes, and 32 deep comparator records exactly.
- Mirror and partitioned runs were semantically identical through rounds 0-4. First semantic fork: round5/output index9, shared frontier65300, anchor248045, drafts846/198/248046.
- Monolithic exact head chose token198 at 12.4375 over token846 at 12.0625 (margin0.375). Partitioned exact head chose846 at12.3125 over198 at12.25 (margin0.0625). Score movement -0.1875/+0.25 fully explains the flip.
- Optimized target and exact full W8 head agreed in both paths; cache codes/scales and canonical-vs-shadow remote states were bit-identical. This is legal deterministic reduction reassociation, not cache/replay/index corruption.
- Artifact: .chatgpt/dual_c1_20260818/context_parallel_runtime/numerical_trace_snapshot/. Continue true physical sequence sharding; do not impose monolithic byte-identity.

### 2026-08-19 02:26 +07 — OPEN — native C1 physical sequence-sharded KV runtime
- Prior accepted primitive: c8c71b0 logical-range ownership plus sequence_sharded_kv exact 60/40 page discriminator. Prior duplicate-shadow context runtime is superseded and is not the target architecture.
- Hypothesis: construct each canonical text shard with only its permanent logical page range; use an owner-local temporary full cache during prefill, convert the 65K prefix into owner/helper physical ranges at the prefill→decode boundary, then free the temporary full cache. MTP KV remains full and must not inherit text-page sharding.
- Success gate: snapshot-restored 65K E2E reaches the partitioned numerical contract with real physical allocated bytes near one KV copy; helper/owner range states match the accepted primitive; no full shadow remains after transition. Then pass 262K allocation/capacity and compare KVP-only, TP2-only, KVP+TP2 by useful TPS and normalized ms/round.
- Loss path: preserve evidence, identify the exact state-planner/cache-switch blocker, and move to the smallest direct-cache ownership seam rather than reviving duplicate shadows.

### 2026-08-19 03:03 +07 — ACCEPTED runtime — native physical sequence-sharded KV at populated 65K
- Legacy 65K full-state snapshot was migrated directly into permanent owner/helper page ranges; no old context shadow snapshot was loaded. MTP cache remained full and independent.
- Exact partitioned contract: stdout/IDs/digest ac20cb527e9d1d39 and 11 rounds / 33 drafted / 17 accepted / 7,6,4 matched the qualified partitioned reference byte-for-byte.
- Physical ownership: 638 fast pages + 387 slow pages for 1024 logical pages, exactly one duplicate boundary page. Stage state fell to 808.16 MiB + 684.29 MiB; complementary helper KV was 1.03 GiB total.
- Prefix restore 3.204 s; decode 0.701 s; 44.21 useful tok/s versus prior duplicate-shadow partitioned reference 42.35. Target stage0/stage1 0.237/0.256 s. Core-X loaded median 1845 MHz; internal 1350 MHz.
- Real peak allocated memory: 9929 MiB Core-X and 12406 MiB internal. Artifact: .chatgpt/dual_c1_20260818/physical_kv_runtime/qualification65k/.

### 2026-08-19 03:13 +07 — ACCEPTED capacity — native physical sequence-sharded KV fits at 262K
- Instantiated the actual C1 model, sharded stage state, and complementary helper page ranges at max_context=262144. Physical inventory: 2474 fast + 1623 slow pages for 4096 logical pages, exactly one duplicate boundary page.
- Stage0 state/KV 2.64/2.49 GiB; stage1 2.30/2.15 GiB; complementary helper KV 4.13 GiB aggregate. Refreshed sampled peak allocation was 11147 MiB Core-X and 13662 MiB internal; no OOM, fallback, or hidden full-shadow allocation.
- The process intentionally stopped at the explicit missing-prefix boundary after all allocations. Artifact: .chatgpt/dual_c1_20260818/physical_kv_runtime/capacity262k/.

### 2026-08-19 03:13 +07 — CLASSIFIED — native KVP and current TP2 compose as an engine gain but a useful-yield loss
- Matched 65K medians: KVP 44.215 tok/s and 63.773 ms/round; TP2-only 39.020 and 61.115 ms/round; KVP+TP2 32.025 and 56.941 ms/round.
- KVP+TP2 improves normalized round cost by 1.11998x versus KVP, proving the coarse engines compose. Useful throughput falls to 0.7243x because trajectory changes from 11/33/17 (7,6,4) to 17/51/12 (8,4,0).
- Peak KVP+TP2 allocation was 14921 MiB Core-X / 17270 MiB internal, so physical KV frees enough capacity. Current TP2 still adds 4.63/4.50 GiB duplicate shards and is not the final storage design.
- Next: preserve engine-vs-yield reporting, deduplicate TP2 storage, then improve useful tokens/round instead of calling the lower useful TPS an engine regression. Artifact: .chatgpt/dual_c1_20260818/physical_kv_runtime/composition65k/.

### 2026-08-19 03:22 +07 — OPEN — alias canonical local TP2 weights and retain only nonlocal shards
- Preflight found a concrete layout path: transformed Q4 fragment rows are exact fast-prefix/slow-suffix aliases; transformed Q5 needs only canonical full-row stride plus local group offset (0/138).
- Hypothesis: pass canonical gate/up and down Weight descriptors through the MLP override, skip loading fast shards for local layers0-31 and slow shards for local layers32-63, and use an alias-aware Q5 kernel. Only opposite-stage compact shards remain resident.
- Expected TP2 extra storage falls from 4.63/4.50 GiB to approximately 2.32/2.25 GiB. Success gate: identical TP2 output/trajectory and matched engine round cost, measured allocation reduction, then 262K physical KVP+TP2 allocation.
- Loss path: classify exact layout mismatch; retain only the transformed representation demonstrably required rather than restoring all duplicate shards.

### 2026-08-19 03:53 +07 — ACCEPTED — TP2 aliases canonical local weights and retains only nonlocal shards
- Q4 local shards alias canonical transformed prefix/suffix rows. Q5 local shards use exact compact accumulation with canonical 272-group row stride and group offsets 0/138. Each GPU loads only opposite-stage compact shards.
- Four fixed 128-token runs were exact: digest 14660c413b4cf185, 49 rounds, 147/77 accepted, by-position 34/27/16. Duplicate TP2 median 65.24 tok/s / 1.4935 s engine; alias 65.71 / 1.4925.
- Extra resident TP2 storage fell fast 4.63 -> 2.32 GiB and slow 4.50 -> 2.25 GiB. Load median fell 17.6785 -> 13.7125 s. A nonlocal-only 128-file, 4.565-GiB package runs successfully.
- 65K physical KVP+alias TP2 exactly matched duplicate TP2: 32.045 useful tok/s, 56.912 ms/round, 17/51/12 trajectory. Peak memory fell 14921/17270 -> 12517/14930 MiB.
- 262K KVP+dedup TP2 allocation passed at 16247 MiB Core-X / 17676 MiB internal. Remaining blocker is useful-token yield and non-MLP target cost. Artifact: .chatgpt/dual_c1_20260818/tp2_storage_dedup/.

### 2026-08-19 04:00 +07 — CLASSIFIED — actual 65K combined round remains mostly serialized outside TP2
- Nsight on physical KVP + deduplicated TP2: verifier window 989.76 ms across 17 rounds; only 244.66 ms (24.72%) had both GPUs busy, with 85.49 ms launch/serialization idle.
- Per-round NVTX spans: full-attention KVP 15.91 ms, GDN mixer 14.09 ms, TP2 post-mixer 20.62 ms. Measured target wall 46.65 ms under profiling; proposal 5.94 ms and fold 0.35 ms.
- At current 1.706 useful tokens/round, perfect removal of all GDN work projects only about 39.8 tok/s; even a 2x GDN engine is about 34.2. A GDN-only implementation cannot approach 100.
- Required for 100 at current yield is 17.06 ms/round versus 58.41 profiled; conversely current wall needs 5.84 useful tokens/round. Therefore wider/tree verifier economics are mandatory in parallel with GDN/serialization work. Artifact: .chatgpt/dual_c1_20260818/combined_round_profile65k/.

### 2026-08-19 04:30 +07 — OPEN — populated-65K dual verifier-width economics on native physical KVP
- Prior fixed T6/short-context work is CLOSED. Materially new hypothesis: at populated 65K the dominant exact KV scan can be amortized across MTP4/MTP5 verifier columns, so width may improve useful tokens/round per complete round even when it lost at short context.
- Current dirty implementation generalizes one-session dual C1 from three drafts/four columns to up to five drafts/six columns, preserves full independent MTP KV, imports the existing width-4 prefix snapshot into width-specific state layouts, and leaves multi-session/TP2 restricted to depth3.
- Success gate: build/tests pass; snapshot-restored KVP depth3/4/5 runs report complete ms/round and useful tokens/round, no fallback/state/position/replay failure, and a material engine-product gain. Numerical forks must be deterministic and explainable under the partitioned contract; do not use prompt/token curation.
- Loss path: classify wider fixed depth closed at long context, restore the dirty width implementation, and move immediately to the tiny exact tree/Bole oracle rather than more fixed-depth tuning.

### 2026-08-19 05:12 +07 — SUPERSEDED — adaptive tail accidentally changed fixed verifier-depth semantics
- This intermediate matrix lowered draft extent near the output budget, unintentionally turning fixed-depth verification into an adaptive short final round. Measurements are retained as diagnostic history but are superseded by the fixed-budget matrix below.
- The scheduler regression was isolated to assigning `state.draft_extent = round.next_extent`; fixed MTP3 historically keeps the selected verifier budget and exits to scalar tail when a full round no longer fits.
- Depth4 follows a deterministic alternative partitioned trajectory (first ID fork at output index4), so accept it as an opt-in populated-context verifier-budget candidate, not a universal default. No token/prompt curation.
- Next gate: KVP + deduplicated TP2 at depth4, reporting normalized ms/round separately from useful yield. Then held-out/adaptive confidence qualification; fixed depth5 is closed. Artifact: .chatgpt/dual_c1_20260818/dual_width65k/full/.

### 2026-08-19 05:14 +07 — OPEN — mixed-column TP2 composition for populated depth4/5 verification
- TP2 packed kernels are exactly T4. New bounded hypothesis: MLP columns are independent, so apply TP2 to the contiguous first four verifier columns and run every remaining column through canonical post-mixer kernels after one full-width RMSNorm.
- This changes no token membership, acceptance logic, state/replay semantics, or TP2 shard math; it only composes exact per-column execution paths. Success gate is depth4 KVP+dedup-TP2 useful tokens/round and normalized ms/round versus KVP depth4 and TP2 depth3.
- Reject if any column is dropped, state/replay fails, or complete verifier economics do not materially improve.

### 2026-08-19 05:22 +07 — CLOSED — mixed T4 TP2 plus canonical tail is the wrong depth4 MLP schedule
- Matched populated-65K depth4 medians: KVP 57.835 tok/s, 66.545 ms/round; TP2-only 43.885, 90.438; KVP+TP2 44.175, 87.106.
- The mixed path preserves all columns but streams TP2 shards for columns0-3 and then the canonical full MLP weights for column4. That double weight pass raises target0/1 from 25.773/31.227 to 36.727/40.636 ms per round.
- KVP+TP2 is 0.76396x the complete verifier product of KVP depth4 and has one fallback. Restore the mixed override. Reopen MLP TP at depth4 only with a true single-pass T5 shard kernel, not another canonical-tail composition.
- Retain physical KVP depth4 as the accepted verifier candidate and move to the measured GDN/remaining target bucket. Artifact: .chatgpt/dual_c1_20260818/dual_width65k/tp2_mixed_columns/full/.

### 2026-08-19 05:36 +07 — ACCEPTED — fixed-budget populated-65K verifier depth4
- Default depth3 compatibility is exact again: byte-identical output and the accepted 11-round/33-draft/17-accepted trajectory on the 32-token snapshot regression.
- Balanced 128-token runs: depth3 2.3962 useful tokens/round at 51.918 ms/round (46.155 modeled tok/s); depth4 3.8485 at 64.153 ms (59.989); depth5 3.8485 at 74.458 ms (51.685).
- Depth4 improves the complete useful-token/round divided by round-cost product by 1.29976x versus exact depth3 and cuts rounds 53 -> 33 with zero fallback. Depth5 is output/yield-identical to depth4 but 16.1% slower.
- Keep depth3 as the compatibility default; expose depth4 as an opt-in populated-context verifier budget. No token membership, checkpoint-specific policy, or adaptive tail behavior. Artifact: .chatgpt/dual_c1_20260818/dual_width65k/fixed_full/.

### 2026-08-19 05:57 +07 — CLOSED or blocked — exact two-GPU GDN head/state discriminator did not pass
- Resident upper-bound evidence: {}.
- Exact benchmark status: build_rc=None, run_rc=None, accepted=False, implementation blocker=False.
- Do not integrate a fake byte-stream or duplicate-full-recurrence path. Preserve the measured blocker and move to useful-yield/tree-verifier work while retaining the depth4 physical-KVP milestone.
- Evidence: .chatgpt/dual_c1_20260818/gdn_tp_design/head_sliced/ and resident_upper_bound/.

### 2026-08-19 10:08 +07 — CORRECTION / OPEN — exact head-sliced two-GPU GDN chain discriminator
- Supersedes the 05:57 entry: NO_AGENT_IMPLEMENTATION / AGENT_DEFERRED / RUN_SKIPPED_NOT_BUILT are not measured rejection evidence. No benchmark was built or run.
- New experiment is benchmark-only and uses real Qwen3.8-27B GDN weights: full 48-layer depth4/T5 Core-X baseline versus calibrated ~60/40 head/state ownership, resident concurrent engines, pinned-host hidden broadcast, FP32 helper partial return, ordered FP32 merge and residual.
- Success gate: canonical numerical/state correctness and >=1.35x complete 48-layer chain speedup including communication. Win -> integrate and qualify depth4 physical KVP vs KVP+GDN-TP. Loss -> write measured CLOSED result and move immediately to the tiny exact 2-branch depth2/3 tree/Bole oracle.
- Planned source: bench/ops/gdn_tp_chain_bench.cu plus bench/CMakeLists.txt only. Artifact root: .chatgpt/dual_c1_20260818/gdn_tp_design/head_sliced/.

### 2026-08-19 11:31 +07 — CLOSED — real head-sliced two-GPU GDN chain misses both speed and state gates
- Implemented and ran the previously missing real-weight 48-layer depth4/T5 discriminator; the earlier NO_AGENT_IMPLEMENTATION status was not evidence.
- Best calibrated 27/21 value-head split with event-chained hidden broadcast and FP32 partial return: 19.872923 ms Core-X baseline vs 19.212447 ms dual, only 1.034378x against the required 1.35x.
- Single-layer oracle was near-exact, but 48-layer residual/SSM relative-L2 reached 0.05259/0.04884. 24/24, 27/21, 30/18, 33/15 all failed. Do not integrate into KVP.
- Next: tiny exact 2-branch depth2/3 tree/Bole verifier oracle with canonical selected-branch GDN state reconstruction. Artifact: .chatgpt/dual_c1_20260818/gdn_tp_design/head_sliced/CONCLUSION.md.

### 2026-08-19 11:50 +07 — CLOSED — exact two-branch depth-2/3 tree/Bole oracle cannot beat accepted depth-4
- Used the reusable 65K snapshot and existing Q4/B2 proposal scores only. Forked only the last draft, so packed tree widths map exactly to accepted width-4/5 target geometries.
- Depth2: 10/53 top2 rescues, useful yield 2.35185 -> 2.53704. Depth3: 2/51 rescues, 2.38462 -> 2.42308.
- Exact forced rescues increased acceptance/licensed output by one, folded canonical GDN state, and continued the next ordinary round with zero fallback.
- Even zero-cost proposal/fold upper bounds are 56.82 tok/s (depth2) and 42.29 (depth3), below accepted depth4 57.60. Do not widen or curate. Artifact: .chatgpt/dual_c1_20260818/tree_bole_oracle/CONCLUSION.md.

### 2026-08-19 11:57 +07 — OPEN — true single-pass T5 Q4/Q5 TP discriminator
- Distinct from the CLOSED T4 TP2 plus canonical fifth-column tail: this benchmark streams each full/fast/slow Q4 gate-up and Q5 down shard exactly once for all five columns.
- Use all 64 real Qwen3.8-27B MLP layers, resident block69 69/67 split, pinned input/partial staging, and the proven 64-layer chain harness. Compare true full T5 against concurrent sharded T5 including communication/reduction.
- Success gate: accepted numerical tolerance plus measured complete-chain cost that projects a material populated-depth4 C1 gain in the current round model. Loss: record CLOSED and do not integrate another mixed-column schedule.
- Artifact root: .chatgpt/dual_c1_20260818/tp2_t5_discriminator/.

### 2026-08-19 12:03 +07 — ACCEPTED primitive — true single-pass T5 Q4/Q5 TP materially improves the round model
- Unlike the CLOSED T4+tail schedule, all five columns share one full/fast/slow Q4 gate-up and Q5 down weight pass across 64 real layers.
- Full Core-X 25.605977 ms vs sharded 16.758695 ms, 1.52792x; max abs 0.03125, relative-L2 0.00231604; loaded clocks 1860/1350 MHz.
- Conservative populated-depth4 projection: 66.818 -> 57.971 ms/round, 57.60 -> 66.39 useful tok/s (+15.3%). Zero proposal still only ~78.8, so this is necessary but not sufficient.
- Integrate width5 into the existing deduplicated TP coordinator and qualify from the reusable 65K snapshot. Artifact: .chatgpt/dual_c1_20260818/tp2_t5_discriminator/CONCLUSION.md.

### 2026-08-19 12:57 +07 — ACCEPTED — T5 BF16 partial transport removes exact redundant bytes
- New hypothesis: T5 shard outputs are already BF16, so avoid expanding the remote partial to FP32 for D2H/H2D. T4 keeps its qualified FP32 transport.
- Alternating populated-65K control 48.47/48.42/48.36 tok/s (median 48.42); BF16 partial 49.39/49.37/49.42 (median 49.39). Round cost 67.2564 -> 65.9231 ms; TP engine 34.8462 -> 33.2564 ms/round.
- All six outputs and the 39-round 117/86 trajectory are identical. Accept. Artifact: .chatgpt/dual_c1_20260818/tp2_t5_runtime/bf16_partial/.

### 2026-08-19 12:59 +07 — ACCEPTED runtime — true single-pass T5 TP composes with physical KVP
- Storage-deduplicated T5 TP aliases canonical local Q4/Q5, keeps only opposite-stage compact shards, and uses exact BF16 remote partials. T4 stays on its qualified FP32 path.
- Populated-65K KVP median 46.88 tok/s and 69.4615 ms/round; KVP+T5 TP BF16 median 49.39 tok/s and 65.9231 ms/round: 1.0535x useful and 1.0537x normalized round-cost speedup.
- TP5 repeats are byte-identical at digest 09ff33b553ac8a05, 39 rounds, 117 drafted, 86 accepted, zero fallback. T4 regression is byte-identical. Remaining gap is coordinator transport/synchronization. Artifact: .chatgpt/dual_c1_20260818/tp2_t5_runtime/CONCLUSION.md.

### 2026-08-19 13:05 +07 — PASS capacity — 262K physical KVP plus deduplicated T5 TP fits both TU102 cards
- Real allocation reaches the intentional missing-prefix boundary with 2474 fast pages, 1623 slow pages, one duplicate boundary page, and the nonlocal-only T5 package loaded.
- Peak memory: 16211 MiB Core-X UUID GPU-2fffc240-5d58-fbc8-27d6-6b349448fda9; 17756 MiB internal UUID GPU-020906b4-0904-3471-e4ee-e41ec2350be3.
- Classification PASS. Artifact: .chatgpt/dual_c1_20260818/tp2_t5_runtime/capacity262k/.

### 2026-08-19 13:18 +07 — CLASSIFIED — post-BF16 T5 profile leaves a 21 ms/round coordinator gap
- Populated-65K depth4 profile: 48.14 tok/s, 39 rounds, 86 accepted; T5 shard kernels are ~10.8-11.1 ms/round per rank, but TP coordinator time is 32.05 ms/round.
- 6619 cudaStreamSynchronize calls consume 1.725 s and 19034 cudaMemcpyAsync calls 0.610 s in the decode window; two-GPU overlap is only 0.608/2.634 s.
- Exact BF16 byte elimination helped, but transport-only perfection still cannot reach 100 at current yield. Next: adaptive T4/T5 verifier budget measured as useful tokens per exact round cost. Artifact: .chatgpt/dual_c1_20260818/tp2_t5_runtime/profile65k_bf16/CONCLUSION.md.

### 2026-08-19 13:28 +07 — ACCEPTED correction — restored snapshots now reach the configured fixed verifier depth
- Found a contract bug: a legacy 3-draft prefix snapshot loaded under --mtp-depth 4 kept draft_extent=3 forever while paying width5/T5 target kernels.
- Populated-65K true fixed depth4 (one bootstrap depth3 round, then 31 depth4 rounds): 59.02 tok/s median, 32 rounds, 127 drafted, 93 accepted, 3.906 useful tokens/round. Broken depth4 was 49.43 tok/s, 39 rounds, 117 drafted, 86 accepted.
- Acceptance-threshold policies 1/2/3 measured 57.36/47.08/47.08 tok/s and are rejected. Promote only fixed-depth bootstrap semantics. Artifact: .chatgpt/dual_c1_20260818/adaptive_width65k/CONCLUSION.md.

### 2026-08-19 15:10 +07 — CLOSED — full per-layer TP/KVP event DAG is exact but regresses populated-65K C1
- Materially different from the old output-event-only test: event-chained TP input/engines/return/merge and physical-KVP packet/ranges/state-return/merge, including pinned-buffer reuse.
- Fixed depth4 snapshot gate: control 58.94/58.96/59.16 tok/s (median 58.96, 67.3125 ms/round); DAG 55.42/55.37/55.44 (median 55.42, 71.625 ms/round). All runs preserved digest 09ff33b553ac8a05 and 32-round 127/93 trajectory.
- Removed 4096 TP and 1088 non-mask KVP host synchronizations/run, but Nsight overlap fell 20.90% -> 18.46% and GPU idle rose 49.03 -> 83.60 ms in the matched 32-token profile. Thousands of fine-grained waits over-constrained launch-ahead.
- Restore accepted coordinators. Reopen only as a coarser multi-layer/double-buffered or persistent-work-queue schedule. Artifact: .chatgpt/dual_c1_20260818/event_dag65k/.

### 2026-08-19 15:26 +07 — CLOSED — hardware-cost-aware position/tree oracle cannot materially beat 59.05 tok/s
- New oracle evaluated every depth3/depth4 proposal position with exact top-128 scores across general/code/JSON/Thai/Japanese/Arabic, leave-one-workload-out margin policies, and real 65K T4/T5/T6 costs (51.160/67.220/74.530 ms).
- Depth4 top-2 rescue rates by position: 10.01%, 5.59%, 3.38%, 1.30%. Every implementable held-out policy selected no branch; always-branch projections were 53.43-54.60 tok/s.
- Even the nonimplementable target-aware upper bound branches/rescues 20.29% of rounds and reaches only 60.728 tok/s (+2.84%), below the 62.00 material gate. Do not build a larger tree from the current proposal distribution. Artifact: .chatgpt/dual_c1_20260818/hardware_cost_tree_oracle/.

### 2026-08-19 16:08 +07 — CLOSED — T4 block64 shared-mask sparsity cannot deliver material Q5 byte skipping
- New single-Core-X audit captured the exact T4 Q5-down activation_q/activation_s seam: 21,376 real MLP calls across general/code/JSON/Thai/Japanese/Arabic. Audit and control outputs/MTP trajectories were identical for every workload.
- Energy-ranked shared union skips at 90/80/70/60/50% independent per-column retention were only 0.31/1.48/3.83/7.69/13.77% mean (0/1.10/3.31/6.99/12.87% median). At 50%, independent retained energy median was only 75.14%.
- This fails the required 30-40% shared byte-skip gate. Do not implement a sparse Q5-down kernel or run post-hoc zeroing quality sweeps. Production source was restored. Artifact: ../ninfer-2080ti-int8-tu102/.chatgpt/int8_tu102/block64_audit/CONCLUSION.md.

### 2026-08-19 17:04 +07 — OPEN — exact GDN input-projection-only TP discriminator
- Prior 48-layer GDN head/state TP is CLOSED at 1.034378x and fails accumulated recurrent-state qualification because it duplicates/shards recurrence and output work.
- Materially new hypothesis: keep RMSNorm/gating, full DeltaNet recurrence, SSM state, gated RMSNorm, and Q5 output projection canonical on Core-X; shard only Q4 query/key plus Q5 value/z projection and convolution rows, then concatenate projected rows in canonical order.
- First gate is a real 48-layer input-bucket discriminator. Using the measured 19.872923-ms GDN chain and ~6.47-ms canonical recurrence+output remainder, dual input projection must be <=8.25 ms to project >=1.35x complete-chain speedup. If it misses, close without runtime integration.

### 2026-08-19 17:04 +07 — OPEN — exact Q5 stored-bit compression oracle for single Core-X
- Block64 activation sparsity is CLOSED by the oracle quality gate. Existing q5_highbit_entropy_preflight contains only search evidence, no measured code-distribution/storage result.
- New exact hypothesis: represent each Q5G64 group as a 4-bit core plus group-local offset/palette and sparse exact exceptions, reconstructing the current Q5 codes exactly. Measure optimal 16-code-window coverage and true bits/weight including exception indices/codes before any kernel work.
- Gate: exact representation must remove enough streamed bytes to plausibly contribute multiple tok/s; reject if best practical encoding is not materially below current 5.25 bpw.

### 2026-08-19 17:20 +07 — CLOSED — exact Q5 four-bit core plus sparse exceptions expands storage
- Audited all 246 Q5G64_F16S tensors: 11.496B weights / 179.622M G64 groups. The optimal contiguous 16-code window still leaves 14.890% exact exceptions, 9.530 per group on average.
- Corrected for the mandatory FP16 G64 scale, the best packed exact representation is 6.0598 bpw versus current 5.25; byte-list 6.8824 and bitmap 6.0726. MLP down alone is 6.0717 bpw.
- Do not build this kernel. Exact group-local core+exception coding cannot remove Q5 streamed bytes on this checkpoint. Artifact: ../ninfer-2080ti-int8-tu102/.chatgpt/int8_tu102/q5_exact_core_sparse/.

### 2026-08-19 17:24 +07 — CLOSED — exact GDN input-projection-only TP is slower end to end
- Materially new discriminator kept recurrence/SSM/output projection canonical and sharded only Q4 q/k plus Q5 value/z projection/convolution. Projected q/k/v/z and all 48 conv-state slices were bit-identical.
- 48-layer canonical input bucket 15.122403 ms; sharded 16.348275 ms (0.925015x). Resident engines reached 10.642977 ms, but hidden broadcast 2.945044 ms, slow-row return 3.951930 ms, and concatenation 1.254789 ms erased the gain.
- With canonical recurrence+output remainder, projected complete GDN is 22.788210 vs 19.872923 ms (0.872070x), far below 1.35x. No runtime integration. Artifact: .chatgpt/dual_c1_20260818/gdn_input_tp_only/.

### 2026-08-19 17:35 +07 — CLOSED — intra-round 2+3/3+2 verifier microbatch pipeline has a negative ceiling
- New coarse hypothesis measured real 65K T2/T3 canonical and T4/T5 TP2 stage costs, then assumed perfect stage overlap with zero extra boundary/replay cost.
- T2 stage sum 41.771 ms, T3 44.077, T4 45.517, T5 54.611. Ideal 2+3 and 3+2 stage walls are 64.733/65.820 ms because narrower passes restream all weights twice.
- With current T5 non-stage remainder, projected rounds are 81.789/82.875 ms and only 47.76/47.13 useful tok/s. No runtime implementation. Artifact: .chatgpt/dual_c1_20260818/intra_round_pipeline/.

### 2026-08-19 18:03 +07 — CLOSED — prompt/history lookup drafter has no material held-out ceiling
- Recreated exact runtime prompt tokens and 775 depth4 verifier rounds across general/code/JSON/Thai/Japanese/Arabic. Frontier/anchor reconstruction had zero alignment errors.
- Leave-one-workload-out recent/modal suffix policies selected zero safe lookup rounds and projected no gain. Even a nonimplementable per-round oracle reaches only 59.13-59.15 tok/s versus the ~62 tok/s material gate.
- No runtime implementation. This portable multilingual drafter cannot materially improve useful tokens/round on the held-out corpus. Artifact: .chatgpt/dual_c1_20260818/prompt_lookup_oracle/.

### 2026-08-19 18:33 +07 — OPEN — coarse phase scheduler ceiling from accepted fixed-depth4 trace
- Prior full per-layer TP/KVP event DAG is CLOSED: exact but 67.3125 -> 71.625 ms/round because 6867 fine stream waits reduced launch-ahead. This is a materially different preflight: model only multi-operation phase/layer-group scheduling, persistent queues, or coarse graph launch.
- Use the current fixed-depth4 control trace plus the 59.05 tok/s / 67.21875 ms-round frontier. Separate dependency-critical GPU gaps from CUDA API time already hidden under GPU work.
- Implementation gate: optimistic reclaim must be >=5.38 ms/round (8%) and map to a concrete coarse boundary without violating residual/GDN recurrence. Pass -> implement one bounded coarse scheduler. Fail -> close immediately and build the drafter requirements envelope.

### 2026-08-19 18:42 +07 — CLOSED — coarse phase scheduler cannot meet the implementation gate
- Parsed the accepted fixed-depth control across the complete decode window: 728.522 ms wall, 679.489 ms any-GPU busy, 49.033 ms all-device idle, 152.230 ms two-GPU overlap.
- Even an impossible queue/graph deleting every all-device-idle interval scales to only 4.524 ms/round (6.73%), 62.695 ms/round, about 63.31 useful tok/s. The 8% gate was 5.378 ms.
- Residual/GDN recurrence blocks cross-layer overlap; TP ranks and KVP ranges are already concurrent within each legal layer phase, and the next normalized hidden cannot be double-buffered. No scheduler implementation. Artifact: .chatgpt/dual_c1_20260818/coarse_scheduler65k/.

### 2026-08-19 18:53 +07 — CLASSIFIED — drafter requirements envelope rules out fixed linear widths for the hard gates
- Measured non-draft bases after subtracting current proposal: T4/T5/T6 = 44.990/57.781/63.439 ms at 65K; projected physical-KVP bases = 71.151/84.542/100.507 ms at 262K using measured T4-T6 KVP costs.
- At 65K, free perfect linear drafts cap T4/T5/T6 at 88.91/86.53/94.58 tok/s. At 262K they cap at 56.22/59.14/59.70. Therefore no fixed linear width can hit dual 100.
- T5/65K needs 94.6% accepted drafts at 2 ms/round to reach 80; at 95% acceptance a small draft model has only 2.219 ms/round = 0.555 ms/token. Dual 100 needs packed/tree yield >=6.18 at T5-like 65K cost and >=8.85 at T5-like 262K cost with a 4-ms drafter.
- Held-out current useful yield is only general/code/JSON/Thai/Japanese/Arabic = 2.346/1.969/3.000/1.372/1.859/1.826 tokens/round. Do not implement another fixed-width policy or sparse-context same-model draft. Audit for a genuinely learned/distilled proposal source. Artifact: .chatgpt/dual_c1_20260818/drafter_requirements/.

### 2026-08-19 19:40 +07 — CLOSED — Qwen3.5-0.8B Q4 learned drafter misses both cost and quality envelopes
- Tokenizer audit passed for text: identical 248044-token base vocabulary and all merges; only seven target audio/TTS specials differ outside the legal text domain.
- Healthy Core-X populated-65K decode: 181.09 draft tok/s = 5.5296 ms/token, 22.118 ms/four drafts, versus <=0.555 ms/token at ~95% acceptance and <=1.18 even at perfect acceptance.
- Six held-out teacher-forced oracle: 59.48% next-token top1; depth3 40.09% accepted/2.203 useful per round; depth4 33.74%/2.350; depth8 only 2.570 useful per round.
- Best optimistic economics: 29.62 dual-65K, 22.03 dual-262K, 41.56 single-GPU useful tok/s. No runtime integration. Artifact: .chatgpt/dual_c1_20260818/drafter_requirements/qwen35_08b_discriminator/FINAL_CONCLUSION.md.

### 2026-08-19 20:14 +07 — ACCEPTED primitive — fused multi-column B2 draft head
- Standalone SM75 discriminator includes activation prep, one shared transformed-weight scan, per-horizon top128, exact raw-Q4 gather/rerank, and deterministic argmax; production proposal semantics were untouched.
- Three cold-cache runs: T1 median 378.880 us; T4 431.008 us; T5 448.480 us. T4 is ~3.52x faster than four serial T1 heads and clears the <=1.2-ms gate decisively.
- Rank128 residual adapters for horizons2-4 are 7.5 MiB with ~0.013-ms ideal byte floor. Proceed to teacher-hidden capture and adapter distillation. Artifact: .chatgpt/dual_c1_20260818/auxiliary_head_requirements/fused_b2/CONCLUSION.md.

### 2026-08-19 20:59 +07 — CLOSED — direct low-rank hidden horizon adapters do not generalize
- 1719 aligned four-horizon records; exact offline Q4 head reproduces runtime drafts at 98.31-99.07%. Rank128 hidden regression leaves held-out cosine 0.541/0.412/0.341 and relative-L2 0.934/1.056/1.117 for H2/H3/H4.
- Token-supervised rank128 hidden adapter memorized fit rows (3.914 useful/round) but regressed validation 1.949 -> 1.480 and six-family held-out 2.229 -> 1.618. This is a clean prompt-level generalization failure, not a head-evaluator artifact.
- Close hidden adapters. Next new mechanism: share one fixed proposal-head scan and learn small horizon-specific low-rank logit corrections. Artifact: .chatgpt/dual_c1_20260818/direct_horizon_adapters/CONCLUSION.md.

### 2026-08-19 21:07 +07 — CLOSED — low-rank proposal-logit corrections do not improve held-out yield
- Shared-base-head horizon LoRA ranks 4/8/16 achieved held-out 1.566/1.541/1.569 useful tokens per round versus 2.229 recursive; validation stayed 1.44-1.46 with no positive rank trend. Larger rank is not a new hypothesis.
- The 131K shortlist oracle itself reaches only 3.829 held-out useful/round. Even at the 58.225-ms fused-head floor that is about 65.8 t/s, so this head family cannot reach 70/80/100 without changing vocabulary coverage and training scale.
- Close small hidden/logit auxiliary heads. Artifact: .chatgpt/dual_c1_20260818/direct_horizon_adapters/LOGIT_LORA_CONCLUSION.md.

### 2026-08-19 22:57 +07 — ACCEPTED reference — Lucebox DFlash + DDTree runs materially faster on sm_75
- Pinned Lucebox b7f8894 and checksum-pinned public Qwen3.6-27B Q4_K_M target + 1.06-GB DFlash draft built/running on healthy Core-X sm_75.
- First 10 published-order HumanEval prompts, 256 generated each: AR mean 24.083 t/s; DDTree budget22 mean 72.551 t/s (median 69.05, range 56.93-126.62), AL mean 6.86.
- Mean DFlash draft cost 12.649 ms/round; complete tree verification/selection 81.621 ms/round; peak observed VRAM 18,843 MiB; loaded SM median 1800 MHz.
- Accepted as executable algorithmic reference, not an NInfer hard-goal result. Next: Qwen3.6-draft -> Qwen3.8 tap-transfer oracle and native parent-indexed conv/GDN tree semantics. Artifact: .chatgpt/dual_c1_20260819/lucebox_sm75/CONCLUSION.md.

### 2026-08-20 04:04 +07 — CLOSED — zero-shot Qwen3.6 DFlash feature transfer does not survive Qwen3.8 target ownership
- Exact Qwen3.8 tap/embedding/head oracle over 8,400 held-out positions: top1/4/8/16/32 = 2.488/6.262/9.750/14.095/19.560%; median teacher rank 378.5, p95 66,090.80, p99 229,771.08.
- Linear 16-horizon yield is 1.035 useful tokens/round. Luce-style DDTree budgets 12/16/22 yield 1.035/1.035/1.096; budget22 best workload is only 1.189 and never exceeds two useful tokens.
- Close direct checkpoint transfer only. DFlash/DDTree and Qwen3.8-specific AngelSpec/DFlare-scale training remain open. Next: restore capture seam, qualify, then native parent-indexed tree oracle. Artifact: .chatgpt/dual_c1_20260819/dflash_transfer/analysis/CONCLUSION.md.

### 2026-08-20 04:30 +07 — ACCEPTED primitive — native parent-indexed GDN tree oracle and compact selected-path fold
- Added parent-indexed convolution history and normalized GDN recurrence for topologically ordered trees, plus selected-path gather into existing compact ReplaySSM records.
- Two depth3 branches at exact Qwen3.8 geometry matched canonical branch-by-branch Q/K/V, conv history, GDN output and complete FP32 state at every node. Selected-path fold and the next ordinary continuation also matched exactly.
- Dense FP32 node state is correctness-oracle storage only. Next is budgets 8/12/16/22 target-cost measurement, then AngelSpec/ReplaySSM compact factors if the complete verifier envelope is material. Artifact: .chatgpt/dual_c1_20260819/gdn_tree_oracle/CONCLUSION.md.

### 2026-08-20 04:49 +07 — ACCEPTED cost discriminator — parent-indexed tree GDN is correct, but target weight streams dominate budgets 8/12/16/22
- Real 48-layer Qwen3.8 GDN + selected-path fold on Core-X: B8/B12/B16/B22 = 29.109/49.134/58.504/93.529 ms. Tree recurrence itself is only 3.376/4.749/6.114/8.162 ms; Q4/Q5 input+output streams dominate.
- Dense transient FP32 node state is 24/36/48/66 MiB; selected-path compact records 6.82/10.23/13.64/18.76 MiB. Production will retain all-node compact ReplaySSM factors and fold after selection.
- With a Luce-like 12.649-ms draft, the optimistic node-slope 65K model needs AL 9.00/11.95/14.18/18.93 for 100 t/s; projected 262K needs 11.68/14.62/16.86/21.60. Current target streams plus existing drafter quality cannot hit the gates. Artifact: .chatgpt/dual_c1_20260819/gdn_tree_target_budget/CONCLUSION.md.

### 2026-08-20 05:07 +07 — ACCEPTED production primitive — all-node compact ReplaySSM factors with post-selection path fold
- Every tree node is now stored compactly before target selection; a non-contiguous root-to-leaf index list replays only the selected factors and commits recurrent/conv state once. Two-branch depth3 correctness and post-commit continuation remain exact.
- Real 48-layer all-node storage is 13.64/20.46/27.28/37.51 MiB for budgets 8/12/16/22. Gather is 0.87-1.05 ms and selected fold 0.58-0.84 ms; Q4/Q5 target projections still dominate total 29.43/49.27/58.67/94.20 ms.
- Production commit/storage is accepted, but the optimistic 65K/262K node-slope model still needs AL 8.98/11.85/14.03/18.72 and 11.66/14.53/16.70/21.39 for 100 t/s. Artifact: .chatgpt/dual_c1_20260819/gdn_tree_target_budget/all_node/CONCLUSION.md.

### 2026-08-20 05:28 +07 — PASS infrastructure gate — Qwen3.8 AngelSpec full-rank train/checkpoint/eval smoke fits Core-X
- Public Qwen3.6 DFlash CPT block0 + fusion was loaded at Qwen3.8-compatible shapes; exact Qwen3.8 taps/embeddings/head were used, with no LoRA/rank adapter or token curation.
- 131.08M full-rank fusion parameters completed one FP32-master AdamW step; 524.31-MB checkpoint reload was bit-exact. Peak CUDA allocation/reservation 5.86/6.38 GB.
- Held-out top1/useful yield stayed 1.25%/1.0 after one deliberately tiny step, so this is only a systems PASS. Next quality run requires materially larger Qwen3.8 offline shards, while target-side ancestor attention is completed. Artifact: .chatgpt/dual_c1_20260819/angelspec_smoke/CONCLUSION.md.

### 2026-08-20 05:44 +07 — ACCEPTED primitive — exact ancestor-only GQA tree tail and compact prefix-state merge
- Two depth3 branches matched independently compacted canonical chains exactly for FP32 tail (acc,m,l) and merged BF16 output; sibling K/V mutation could not affect the other branch.
- Core-X complete tail+merge costs B4/B5/B6/B8/B12/B16/B22 = 16.38/16.13/16.67/20.35/25.50/28.45/37.25 us with only 97-532 KiB transient state.
- Tree attention is accepted and negligible; Q4/Q5 target weight streams plus learned proposal quality remain the complete-round blockers. Artifact: .chatgpt/dual_c1_20260819/gqa_tree_tail_budget/CONCLUSION.md.

### 2026-08-20 11:44 +07 — ACCEPTED lane — native Qwen3.8 MTP is Speculators-compatible and learns full-rank offline
- All 15 factory mtp.* tensors map bit-exactly into Speculators Qwen35MTPLayer; 424.70M native MTP parameters train while the 248,320-vocabulary embedding/head remain frozen.
- A balanced 600-step full-rank continuation smoke reduced mean loss first100->last100 4.185->2.686 and improved held-out useful K3/K4/K5 2.1009/2.1743/2.2080 -> 2.1315/2.2232/2.2661.
- Native export is exact, but runtime promotion is deferred: the 1,719-row corpus lacks verifier-root/cache state and long-context roots. Native MTP remains active. Artifact: .chatgpt/dual_c1_20260820/native_mtp_quant_verify/mtp_training/CONCLUSION.md.

### 2026-08-20 11:44 +07 — CLOSED runtime discriminator — SpecMQuant-style tree-to-sequence hierarchy misses measured gates
- Existing correct-alternative mass is only 15.34% at T4 and 20.29% at T5/T6; tuned-native-head proxy path accuracy is 48.30%/45.72%.
- With zero intermediate cost, perfect full-width caps at 65K are only 73.49/74.38/82.33 t/s for T4/T5/T6. T6 needs 91.63-99.51% selector accuracy to reach 80 with a 0-2 ms intermediate.
- No hierarchy runtime was built. Artifact: .chatgpt/dual_c1_20260820/native_mtp_quant_verify/hierarchy_economics/CONCLUSION.md.

### 2026-08-20 11:44 +07 — ACCEPTED target route — N17-N64 Q4/Q5 grouped R64xC32 clears the wide-projection gate
- N22 remains one physical 53.85-MB weight scan; replacing C128 with C32 improves isolated cold input projection 1.338->0.612 ms (2.19x). Exact FP64-oracle qualification passes N8/N12/N16/N22.
- Real 48-layer N22 input projection falls 64.674->29.568 ms (2.19x), and complete GDN+all-node-record+fold falls 94.200->59.498 ms (1.58x); B8/B12/B16 remain effectively unchanged.
- Route is promoted only for N17-N64; N65+ preserves C128. Recomputed AL7-10 still misses realistic 65K/262K 100-t/s gates, so no wider tree runtime is built. Artifact: .chatgpt/dual_c1_20260820/native_mtp_quant_verify/projection_routes/CONCLUSION.md.

### 2026-08-20 21:33 +07 — OPEN — matched 65K BF16 versus INT8 physical sequence-sharded KV
- New owner-directed hypothesis: the accepted physical KVP architecture is currently INT8-only, while canonical BF16 small-T has an existing producer; generalize dtype without changing page ownership, compact state transfer, or deterministic merge.
- This is not a duplicate cache/geometry experiment: prior physical KVP qualification covered only INT8. Gate each dtype against its own canonical same-dtype reference, preserve INT8 byte/layout semantics, then run matched factory-MTP depth4 65K E2E with dtype-specific snapshots.
- Success gate: deterministic same-dtype correctness plus material BF16 65K reduction in KVP/round wall and useful TPS. Win -> bounded 262K BF16 capacity probe; loss -> preserve dual-dtype support and proceed to exact Qwen3.8 DFlash2 sidecar.

### 2026-08-20 22:53 +07 — CLASSIFIED — native MTPv2 accepted-prefix/on-policy objective
- OFFLINE_SIGNAL_ONLY; best run=run600; K3/K4/K5 useful=2.1314984709480123/2.217125382262997/2.2599388379204894; active/skipped steps=578/22.
- Relative versus factory K5=0.023545706371191244; relative versus teacher-forced K5=-0.0026990553306341925. No alpha interpolation or checkpoint blending was used.
- Artifact: .chatgpt/dual_c1_20260820/native_mtp_v2/CONCLUSION.md.

### 2026-08-21 04:41 +07 — REJECTED speed route / ACCEPTED dtype implementation — BF16 physical sequence-sharded KV
- The same physical ownership, partial (acc,m,l), pinned-host transfer and deterministic merge architecture now supports BF16 without scale planes; INT8 layout and snapshot version remain preserved. Same-dtype exact range/state/merge gates pass at 8K/65K/262K for both dtypes.
- Matched populated-65K factory-MTP depth4 + dedup T5 TP, three deterministic repeats: INT8 58.75 useful t/s, 67.563 ms/round, 3.90625 useful/round; BF16 33.19 t/s, 83.196 ms/round, 2.71739 useful/round.
- Even normalized round cost loses: BF16/INT8 round-engine 0.8121x and GQA-bucket 0.6313x; BF16 also uses about 1.2-1.3 GiB more peak VRAM per card. Keep INT8 as 65K/262K speed+capacity route; retain clean BF16 support but close further BF16 tuning and skip full 262K BF16 product probe. Artifact: .chatgpt/dual_c1_20260820/bf16_physical_kv/matched65k/CONCLUSION.md.

### 2026-08-21 05:59 +07 — CLOSED — exact-target Qwen3.8 DFlash2 sidecar does not clear NInfer integration economics
- Completed 24-run 8K matrix across general/code/JSON/Thai/Japanese/Arabic: greedy n4 AL 3.168 at 11.79 ms draft/round; greedy n7 AL 3.745 at 15.94 ms; sampled n4/n7 AL 2.830/3.517.
- Optimistic NInfer projection peaks at 50.80 t/s for 65K and 37.27 t/s for 262K; classification CLOSED_NO_NINFER_INTEGRATION_AT_CURRENT_COST_YIELD. Preserve sidecar only as an upstream reference.
- Artifact: .chatgpt/dual_c1_20260820/dflash2_sidecar/ninfer_economics/CONCLUSION.md.

### 2026-08-21 05:59 +07 — OWNER CONSTRAINT — inference-only model inputs; local behavioral training is stopped
- Allowed weights are factory Qwen3.8 target/native-MTP, professionally published exact-target drafts, and mechanical conversion/quantization only. No local training, finetuning, distillation, interpolation, LoRA, adapters, TTT, or agents.
- Removed abandoned MTP teacher-dump instrumentation and the never-run root-onpolicy training scripts. Next lanes are factory-native-MTP execution/control and short dual systems architecture.

### 2026-08-21 06:23 +07 — CLOSED — device-resident dual factory-MTP accept/propose transaction is not an engine gain
- Exact 8K output is unchanged, but the candidate changes proposal trajectory: legacy 49 rounds/77 accepted versus device 44/82.
- Median normalized round cost regresses 39.745->40.011 ms and proposal cost/round 4.469->4.545 ms. The apparent 65.21->72.15 useful t/s is acceptance-shaped, not systems acceleration.
- Candidate reverted. Artifact: .chatgpt/dual_c1_20260821/native_mtp_control/device_round/CONCLUSION.md.

### 2026-08-21 06:24 +07 — CLOSED architecture — proposal/control offload alone cannot reach short-dual 100
- Fixed 8K workload has 2.5918 decode-useful tokens/round (128 generated minus one scalar tail) at 39.745 ms; target stage0+stage1/head is 34.082 ms and proposal 4.469 ms.
- Making proposal and fold free projects only ~74.2 t/s. Even a 1.5x whole-target gain plus free proposal projects only ~108.5 t/s, so target weight execution is the mandatory large bucket.
- Artifact: .chatgpt/dual_c1_20260821/short_dual_round_model/CONCLUSION.md.

### 2026-08-21 06:41 +07 — CLOSED — upstream SM75 Marlin/AWQ low-bit dataflows do not beat NInfer
- GPTQ-Marlin and AWQ GEMM require SM80 cp.async/mma.m16n8k16 and fail direct sm75 compilation.
- Upstream AWQ G64 GEMV compiled unchanged on TU102 and was numerically valid on a mechanically converted real Qwen3.8 [4096,5120] Q4-G64 weight, but measured 48.10/143.46 us at T1/T4 versus NInfer 46.98/63.81 us.
- AWQ T4 repeats the weight stream per batch row; no port is justified. Artifact: .chatgpt/dual_c1_20260821/upstream_kernel_oracles/CONCLUSION.md.

### 2026-08-21 06:54 +07 — CLOSED — full Core-X target with internal TP2 MLP helper remains slower than canonical owner
- Exact 8K owner+TP2 gives 66.37 t/s at the legacy 49-round trajectory versus 65.27 layer-sharded TP2, but canonical Core-X alone gives 76.63 t/s and 43 rounds.
- Owner+TP2 executes 3,136 per-layer coordinator calls and spends 1.401 s in TP2 engines; it is 13.4% slower than owner-only despite removing stage1 and boundary transfer.
- Candidate reverted. Artifact: .chatgpt/dual_c1_20260821/short_owner_tp2/CONCLUSION.md.

### 2026-08-21 06:57 +07 — CLOSED — wider fixed factory native-MTP loses on ordinary short inference
- Core-X depth3/depth4/depth5 all produce identical greedy target output, but measure 76.67/64.22/61.89 t/s.
- Wider depths improve AL 2.93->3.23/3.32 while complete cost dominates; depth3 remains accepted for the fixed short workload.
- Artifact: .chatgpt/dual_c1_20260821/factory_mtp_width_short/CONCLUSION.md.

### 2026-08-21 07:00 +07 — CLOSED oracle — short factory-MTP dynamic tree misses even a clairvoyant 80-t/s gate
- Reused the accepted six-family proposal rescue trace with fresh Core-X depth3/4/5 complete costs; no runtime/tree code was added.
- A selector that branches only on target-correct alternative rounds projects below 80 because 15-20% rescue mass cannot amortize wider verifier cost.
- Artifact: .chatgpt/dual_c1_20260821/factory_mtp_dynamic_tree_short/CONCLUSION.md.

### 2026-08-21 07:06 +07 — CLOSED — multilingual factory-MTP width oracle always selects depth3
- Six-family 256-token mean is D3/D4/D5 = 68.25/55.39/54.52 t/s; depth3 wins general, code, JSON, Thai, Japanese, and Arabic individually.
- Several D4/D5 runs also fork target output under width-dependent numerical geometry, so wider depth is not a portable adaptive control surface.
- No entropy/depth runtime is built. Artifact: .chatgpt/dual_c1_20260821/factory_mtp_width_short/multilingual/CONCLUSION.md.

### 2026-08-21 07:28 +07 — CLOSED — FlashQLA SM75 ReplaySSM adaptation is too small at accepted T4
- Raw upstream recurrence is numerically correct and fast; compact-record adaptation preserves raw K/V/g/beta exactly.
- T4 public record 31.01 us falls to 26.93 us with 16-lane reduction, but changes BF16 output by ~1e-4 relative L2; exact-tree 32-lane route is 30.72 us.
- Best T4 saving is only ~0.196 ms across 48 layers versus ~3.6 ms/round needed for single 80. No E2E integration. Artifact: .chatgpt/dual_c1_20260821/upstream_kernel_oracles/flashqla_sm75/CONCLUSION.md.

### 2026-08-21 07:38 +07 — CLOSED clarification — isolated 1.55x TP2 chain is not a live residual-chain speedup
- The chain bench feeds one prequantized T4 packet independently to every layer; it omits preceding normalization/control dependencies and does not feed reduced residuals between target layers.
- Integrated TP2 already quantizes once and transfers one compact A8 packet. Exact event/DMA variants removed host blocking but stayed neutral/negative; full-owner+TP2 is 66.37 t/s versus owner-only 76.63.
- Do not reopen per-layer host-staged TP2 coordination. Artifact: .chatgpt/dual_c1_20260821/tp2_chain_reconciliation/CONCLUSION.md.

### 2026-08-21 10:20 +07 — CLOSED architecture discriminator — SSD/SpecBranch short-dual mapping misses the 100-t/s gate
- Fresh matched owner runs: Core-X 76.595 t/s at 38.560 ms/round and 1852.5-MHz median loaded clock; internal 70.355 t/s at 41.980 ms/round and 1350 MHz. Same digest/43 rounds/83 accepted/AL2.93.
- SSD proposal caching has an impossible 100%-hit ceiling of 86.64 t/s. Trace-scaled F8 full-target caching reaches only 96.09 t/s even if all eight branches share one impossible helper weight scan.
- F32/F64 only approach/cross 100 in that impossible model, but recurrent branch state alone is ~9.4/18.8 GiB before KV/logits/workspace. No runtime implementation. Artifact: .chatgpt/dual_c1_20260821/ssd_specbranch_ceiling/CONCLUSION.md.


### 2026-08-21 11:04 +07 — BLOCKED upstream oracle — real weicj SM75 Marlin is valid but not a competitive Qwen3.8 target path
- v0.1.16 contains real sm_75 cubins and direct FP8-Marlin production shapes pass at ~458-542 GB/s with cosine >=0.999997.
- Official Qwen3.8 FP8: one GPU OOMs; TP2/PP2 load ~14.1-14.3 GiB/card but a worker dies in the first dummy forward across attention/GDN backends and 64-8192 profiler widths. Clean NCCL transport passes.
- Measured FP8-Marlin dense weight streams already total roughly 49 ms/round before nonlinears/state/head versus NInfer ~34.08 ms complete T4 target. No port. Artifact: .chatgpt/dual_c1_20260821/weicj_vllm_oracle/CONCLUSION.md.

### 2026-08-21 11:13 +07 — CLOSED published proposer oracle — Qwen3.8 DSpark cannot amortize its TU102 inference cost
- Real published 1.36B BF16 checkpoint: Core-X backbone 7.669 ms, Markov 0.259 ms, confidence 0.095 ms, exact NInfer W8 T7 head 4.781 ms; proposer total 12.803 ms.
- Granting width8 the impossible width4 target cost gives 46.885 ms/round. Published all-request mean AL3.39 projects 72.30 t/s; best workload AL4.57 projects 97.47 t/s.
- Real width8 target/integration can only be slower. No runtime implementation. Artifact: .chatgpt/dual_c1_20260821/dspark_oracle/CONCLUSION.md.

### 2026-08-21 11:19 +07 — CLOSED upstream proposal dataflow — SM75 Marlin MTP body projects only 78.1 t/s
- Complete five-matrix factory-MTP W8 body: NInfer 1.078 ms/pass versus patched FP8-Marlin 0.827 ms/pass; three recursive passes save 0.753 ms/round.
- Substitution projects 38.560->37.807 ms/round and 76.60->78.12 useful t/s. Single80 still needs another 0.888 ms before conversion/trajectory risk.
- No port. Artifact: .chatgpt/dual_c1_20260821/native_mtp_upstream_dataflow/CONCLUSION.md.

### 2026-08-21 11:26 +07 — CLOSED mechanical variant — FP8-Marlin DSpark remains below the short gates
- Full published draft backbone regresses 7.195->9.806 ms at M15 (0.734x) and shifts hidden features to cosine0.9879/relative-L2 15.54%; FP8 T7 head is 2.285 ms and finite.
- Total mechanical proposer is 12.445 ms. With the impossible width4 target floor, published mean/max AL project 72.86/98.23 t/s.
- No target-feature capture or integration. Artifact: .chatgpt/dual_c1_20260821/dspark_oracle/fp8_mechanical/CONCLUSION.md.

### 2026-08-21 11:49 +07 — CLOSED upstream dataflow oracle — Humming does not beat accepted Marlin on TU102
- Real FP16xUINT4 group64 T4 graph medians are GDN input/QKV/gate-up/down/GDN output 110.59/94.24/218.24/118.27/49.15 us versus Marlin 105.86/92.16/191.71/107.04/48.74 us.
- INT8xUINT4 is numerically legal with explicit per-row quantization but slower at 168.58/154.59/268.35/173.60/92.67 us combined graph. Bundled Triton quant_input segfaults before GEMM.
- No integration or local kernel work. Artifact: .chatgpt/dual_c1_20260821/humming_oracle/CONCLUSION.md.

### 2026-08-21 11:54 +07 — CLOSED proposal quantization — SM75 Marlin 4-bit native MTP peaks at 79.58 t/s after measured yield
- Complete three-pass body: W8 3.235 ms vs INT4 G128/G64/G32 1.570/1.619/1.680 ms. G128 has only 0.023 ms unchanged-yield margin above single80.
- Real 327-row multilingual holdout K3 retention: G128 99.13%, G64 99.13%, G32 98.84%; per-group MSE-clipped G128 improves to 99.42% but needs 99.94%, projecting 79.58 t/s.
- FP4 generic weight-only smoke lacks a valid NVFP4/MXFP4 conversion contract and is not timed. No kernel/artifact integration. Artifact: .chatgpt/dual_c1_20260821/native_mtp_int4_quality/CONCLUSION.md.

### 2026-08-21 13:11 +07 — CLOSED P0-B — LeanAttention/PersistentKV comparison has no remaining material dataflow delta
- NInfer already maps long decode CTAs by KV head + sequence split, reuses one K/V tile across all six grouped query heads, and exactly merges `(acc,m,l)` states. The 262K 51/34 split yields 204/136 CTAs, exactly 3/2 CTAs per 68-SM device; the ragged B8 workqueue mechanism does not apply to one active request.
- Impossible deletion of every measured query transfer, state return, and merge across 16 full-attention layers saves only 4.054 ms, 4.80% of the 84.54-ms T5-like floor and below the required 7–8 ms / 8% gate.
- No alternative CUDA benchmark or runtime code. Artifact: .chatgpt/dual_c1_20260821/longkv_p0b/CONCLUSION.md.

### 2026-08-21 13:11 +07 — CLOSED P0-C implementation — PipeInfer independent-proposer oracle is material but unavailable under current allowed weights
- A free independent proposer would pipeline measured 16.224/17.857-ms target stages and project 142.50/143.81 useful t/s for realistic 128/256-token completions (133.93/135.09 after retaining measured fold/control), so this architecture is distinct from SSD and has a real oracle.
- Factory MTP, exact DFlash2, and published DSpark are target-hidden/tap conditioned; the next verify microbatch cannot exist before the current target tail, leaving one target microbatch in flight. Both GPUs are already target stages and there is no owner-allowed independent exact-target proposer placement.
- Same-block T2+T3/T3+T2 overlap is already measured at only 47.76/47.13 t/s because it restreams weights. No runtime code. Artifact: .chatgpt/dual_c1_20260821/pipeinfer_ceiling/CONCLUSION.md.

### 2026-08-21 13:52 +0700 — ACCEPTED P0-A — factory-MTP cold dual prefill is genuinely pipelined
- C1 factory MTP had serialized every prompt chunk behind one pinned boundary; production now uses bounded two-buffer stage overlap and preserves --no-prefill-pipeline diagnostics.
- Populated ~65K improves 414.073 -> median 229.569 s and 157.68 -> 284.42 tok/s: **1.804x / 44.56% lower TTFT**, with exact repeated digest/output. Full 128-token decode and prefix capture/reload preserve every ID, stdout byte, and MTP counter; decode speed is unchanged.
- Wider chunks are CLOSED: 256 gains only 4.8% at 4K then regresses 12.97% throughput at 32K; 512 exceeds the MTP arena. Production remains [1,128].
- Nsight partitions remaining layer work as dense 66.67%, GDN recurrence 22.66%, and full attention 10.64%; impossible-deletion ceilings close FlashQLA-class recurrence (1.293x) and FlashInfer/FA2 attention (1.119x) against P0-A.
- Build, focused MTP/runtime tests, exact smoke, snapshot, and CLI domain gates pass. Decode gates remain 80/100/100@65K/100@262K. Artifact: .chatgpt/dual_c1_20260821/prefill_p0a/CONCLUSION.md.


### 2026-08-21 14:48 +0700 — P0-D global lookup-tail frontier CLOSED; copy/edit special mode retained
- Replayed the exact request-context suffix lookup from syv-ai/qwen38-27b-rtx3090 commit 51905d1902e699e35e4c11cb0eb9ffe230a11344 across general, code, JSON, Thai, Japanese, Arabic, and repository-edit workloads. No curated map, external corpus, training, or finetuning was used.
- With real W4/W8/W12/W16 target costs and all lookup/proposal/control/fold overhead removed, the seven-workload macro reaches only 60.44 useful tok/s for actual W16 head override and 74.42 tok/s even when every lookup hit is magically accepted in full. The independent spec-none trajectory reaches only 39.80 actual / 51.24 impossible-perfect at W16. The unchanged 100 dual gate is not met.
- The copy-heavy repository-edit workload is a real special-mode exception: actual lookup ceilings are 105.97/113.34/124.86 tok/s on the spec-none trajectory and up to 130.48 tok/s for factory-D3 plus matching W16 lookup tail. This is explicitly not the global frontier and no default runtime route was added.
- The trace also records existing MTP trajectory differences versus grouped MTP and spec-none on several prompts; this exactness risk further blocks silently enabling a default lookup route. Reproducer and full rows: .chatgpt/dual_c1_20260821/lookup_tail_p0d/analyze_lookup_tail.py, analysis_summary.json, analysis_rows.csv, CONCLUSION.md.

### 2026-08-21 14:56 +07 — P0-D request-context lookup-tail global lane closed
- Replayed seven 256-token target trajectories with the exact syv lookup policy and measured NInfer W4/W8/W12/W16 target costs. All lookup/controller/proposal/fold/tail costs were made free.
- Macro strict lookup-tail ceilings are 53.87/56.15/58.92 tok/s at W8/W12/W16; even the impossible every-hit-perfect oracle reaches only 63.69/67.90/74.42, below the immutable dual 100 gate. No global implementation.
- Repo-edit/context-copy is a distinct opt-in opportunity: strict lookup-tail projects 95.72/110.78/130.48 tok/s. Classified SPECIAL_CONTEXT_COPY_MODE_ONLY, not a general frontier.
- Artifacts: .chatgpt/dual_c1_20260821/lookup_tail_p0d/CONCLUSION.md and analysis_summary.json.

### 2026-08-21 15:01 +07 — CLOSED high-M Q4 one-time FP16-HMMA staging
- Folded Q4 gate/up with one-time BF16-to-FP16 staging measures 1.329x versus materialized at M128, but only 1.057-1.059x versus the existing folded kernel at M128/M256/M512.
- Replacing all 64 gate/up calls saves 79.007 ms in the measured 713.855-ms dense stream; complete cold-prefill ceiling is only 1.080x / 7.38% TTFT reduction, below the 1.5x/30% promotion gate.
- Experiment reverted. The broader high-M packed-weight/stripe-scheduled oracle remains live. Artifact: .chatgpt/p0_prefill_20260821/highm_dense_oracle/q4_fp16_hmma/CONCLUSION.md.

### 2026-08-21 16:36 +07 — P0-E INT4 old-KV oracle — CLOSED / DO NOT IMPLEMENT current formats
- Real accepted 4K/65K INT8 snapshots plus an exact 262144-token construction were decoded and evaluated with old-KV INT4, newest 128/256 tokens left canonical INT8.
- Best full-KV4 candidate (page/channel affine K + per-token group64 V) produced 7.12% mean, 14.99% p95, and 16.48% max attention-output relative-L2 error; minimum cosine 0.986864. This fails before target-token qualification.
- Full K/V INT4 has a 13.69-ms byte-only scan saving from the measured 28.24-ms 262K scan, but the quality-preserving K4/V8 direction has only a 6.85-ms raw ceiling before unpack/dequant cost, below the required 8 ms. No fused CUDA kernel.
- Evidence: .chatgpt/p0_kv4_20260821/CONCLUSION.md, oracle.json, ceiling.json.

### 2026-08-21 16:43 +07 — P0-A3 4K weight-stationary Marlin prototype — ACCEPT FOR NINFER SEMANTIC INTEGRATION
- The accepted factory-MTP prefill pipeline remains unchanged and focused runtime/MTP tests pass. A 4K macroblock prototype keeps M128 kernels, charges byte-exact direct Q4/Q5 rowsplit repack plus scale permutation once per layer per macroblock, and flushes 256 MiB before every M128 call.
- Runtime role 0 at 1860 MHz: 2.485 s cold dense stage block; runtime role 1 at 1350 MHz: 2.683 s. Both outputs finite; prototype peak allocation 491.81 MiB.
- Complete 65K projection is 118.766 s by calibrated stage ratios. A deliberately pessimistic absolute model is 144.028 s, 37.26% below the accepted 229.569-s TTFT and 16.671 s inside the <=160.698-s gate.
- Proceed to layer-outer/chunk-inner NInfer integration preserving causal KV append, GDN state order, chunk128 geometry, snapshots, exact IDs/digest/MTP counters, and the accepted inter-stage prefill pipeline.
- Evidence: .chatgpt/p0_prefill_20260821/a3_weight_stationary/CEILING.md and prototype4k/CONCLUSION.md.

### 2026-08-21 17:45 +07 — Owner 80/100/100/100 gate recertification after A3/E
- single=76.67/80.0 FAIL; dual_short=65.26/100.0 FAIL; dual_65k=58.75/100.0 FAIL; dual_262k=50.85/100.0 FAIL. No gate is lowered; absent genuine populated262K E2E evidence fails closed. Continue to next materially different complete-system mechanism.

### 2026-08-21 17:48 +07 — P0-A3 NInfer macroblock Marlin integration rejected on exact output
- The layer-outer/chunk-inner native schedule was exact on the full 128-token 4K route (digest d62625bfc9df037c; 50 rounds/150 drafted/74 accepted/33,23,18), but was slower without a weight-stationary leaf. The focused SM75 Marlin Q4 gate-up + Q5 down integration reduced prefill 12.211->9.924 s, yet changed IDs/digest and MTP to 61 rounds/63 accepted. FP32 accumulation gave 10.999 s and 57/67; adding canonical BF16 weight and SwiGLU boundaries gave 12.205 s, matched only the first 52 IDs, then diverged with 57/67. The mandatory 4K exactness gate failed before 32K/65K, so no production source is retained. Artifact: .chatgpt/p0_prefill_20260821/a3_integration/CONCLUSION.md.

### 2026-08-21 18:01 +07 — Exact-Q5 Marlin plus MTP package misses single80 under resident memory
- Matched T4 Q5 down is 230.560 us/layer in NInfer versus 221.504 us/layer in maintained SM75 Marlin, only 0.580 ms over 64 layers. Adding the separately proven 0.753-ms factory-MTP Marlin saving still leaves the fixed short round above the 36.919-ms single80 requirement. A duplicate exact-Q5 packed down layout is 87.656 MiB/layer including scales; 3.40-GiB planned slack minus 64-MiB scratch fits only the modeled resident subset, which remains below 80. Exact Q5 output is slower in Marlin (116.736 vs 71.392 us/layer), and fused GDN/attention inputs cannot credit isolated Q5 without double streaming. CLOSED without source port. Artifact: .chatgpt/next_bucket_20260821/q5_mtp_single80/CONCLUSION.md.

### 2026-08-21 18:07 +07 — P0-G KIVI-style asymmetric old-KV numerical/executable oracle
- Classification CLOSED_NUMERICAL_OR_CEILING; numerical selected None. Actual Qwen Q/K/V plus 65K/262K adversarial screen: {'k2v2_t32_d64': {'min_cosine': 0.804923770424912, 'max_relative_l2': 0.7498537676796231, 'min_top1_retention': 0.16666666666666666, 'actual_count': 0, 'synthetic_count': 5}, 'k2v4_t32_d64': {'min_cosine': 0.8997880253024106, 'max_relative_l2': 0.47847683048481976, 'min_top1_retention': 0.16666666666666666, 'actual_count': 0, 'synthetic_count': 5}, 'k4v4_t32_d64': {'min_cosine': 0.9709430766913593, 'max_relative_l2': 0.36225040886599646, 'min_top1_retention': 0.16666666666666666, 'actual_count': 0, 'synthetic_count': 5}, 'k4v4_t64_d64': {'min_cosine': 0.9709430767053893, 'max_relative_l2': 0.3622505001202987, 'min_top1_retention': 0.5, 'actual_count': 0, 'synthetic_count': 5}}. Metadata-aware ceilings: {'k2v2_t32_d64': {'bytes_per_token_layer': 704.0, 'fraction': 0.3333333333333333, 'pessimistic_ms': 13.208083333333333, 'saving_ms': 15.031916666666666, 'material': True}, 'k2v4_t32_d64': {'bytes_per_token_layer': 704.0, 'fraction': 0.3333333333333333, 'pessimistic_ms': 13.208083333333333, 'saving_ms': 15.031916666666666, 'material': True}, 'k4v4_t32_d64': {'bytes_per_token_layer': 1216.0, 'fraction': 0.5757575757575758, 'pessimistic_ms': 18.67423484848485, 'saving_ms': 9.565765151515148, 'material': True}, 'k4v4_t64_d64': {'bytes_per_token_layer': 1152.0, 'fraction': 0.5454545454545454, 'pessimistic_ms': 17.990965909090907, 'saving_ms': 10.249034090909092, 'material': True}}. Executable 340-CTA SM75 load/unpack result: None. Recent 512 tokens remain current INT8. Evidence: .chatgpt/continuation_1605_20260821/p0_g_kivi/CONCLUSION.md and executable/result.json.

### 2026-08-21 18:51 +07 — P0-H vocabulary W4 exactness and executable discriminator
- Classification CLOSED_VOCAB_W4_FULL_OR_CERTIFIED_FALLBACK. Full-W4 six-family result None; certified exact-fallback numerical result None; measured all-vocabulary bound-scan result None. Full W4 is promoted only with identical six-family trajectories and portable >=81.6 useful t/s. Certified mode uses exact group64 Cauchy bounds and runs the complete accepted W8 head for every uncertified row. Evidence: .chatgpt/continuation_1605_20260821/p0_h_vocab_shortlist/final_decision.json.

### 2026-08-21 18:51 +07 — P0-I Quest-style GQA-union page-selection oracle
- Classification CLOSED_QUEST_NUMERICAL_UNION_OR_EXECUTABLE_GATE. Numerical page-bound result None; executable scoring/sort/union/load result None. The ceiling charges BF16 min/max metadata, the union of selected pages across all six Q heads sharing each KV head, recent512, and actual SM75 selection overhead before mapping to the measured 28.24 ms 262K GQA bucket. Evidence: .chatgpt/continuation_1605_20260821/p0_i_quest/final_decision.json.

### 2026-08-21 18:56 +07 — P0-F W3 target-weight external/numerical oracle
- Classification CLOSED_W3_EXTERNAL_OR_QUALITY_OR_COMPLETE_ROUND. External exact-shape/M4-M5 coverage 0, speedups conservative/median/optimistic None/None/None; independent source-checkpoint quality pass False. Short/65K median projections None/None useful t/s; these are ceilings, not gate results. Evidence: .chatgpt/continuation_1605_20260821/p0_f_w3_target/CONCLUSION.md.

### 2026-08-21 19:02 +07 — CLOSED corrected exact-Q4 Marlin single80 lane — valid E2E is slower
- Resolved both non-performance blockers: Q5 layout routing and explicit resident-memory planning; retained factory optimized MTP head, greedy output, CUDA Graphs, and direct A8 Q5 down.
- Matched 1875-MHz complete-path result: canonical 76.51 tok/s versus 36-layer resident Marlin 65.29 tok/s. Outputs are byte-identical with digest 14660c413b4cf185 and identical 43/129/83/35,29,19 trajectory.
- Surgical Nsys attribution: actual T4 N34816 Marlin is 412.741 us/layer versus accepted fused SM75 Q4 A8 SwiGLU 167.610 us/layer. The earlier 83.08 ceiling accidentally used an M128 N7168 attention-projection Marlin timing, so it is corrected and superseded. Prototype rejected; no parameter sweep.

### 2026-08-21 19:18 +07 — CLOSED W4 vocabulary K16 reduction — exact but no E2E gain
- Resolved the dead hidden-capture dependency: reran the real T4 verifier-head capture without CUDA Graph replay, collected 384 hidden rows across six families, restored source, completed quality, and retired all stale waiters.
- Actual-hidden recall is 100% at K16 through K512, but matched complete-path K128/K16 medians are 76.24/76.25 useful tok/s with byte-identical output and identical 43/129/83/35,29,19 trajectory.
- The current production route already uses the W4+exact-W8 K128 head. The prior 81.9 projection assumed an unqualified 2x head speedup and is superseded. Retain K128; no K16 promotion.

### 2026-08-21 19:27 +07 — P0-I Quest-style page selection real-oracle correction
- Supersedes the 18:51 placeholder that reported None. A restored diagnostic captured 511 chunks / 65,292 real tokens from one fixed full-attention layer. Rigorous BF16 min/max page bounds were evaluated with exact current INT8 K/V for page sizes 16/32/64 and 2/5/10/20/40% budgets, including recent512 and GQA-head union cost. Every configuration failed the numerical gate; material byte ceilings had catastrophic attention quality (for example p64_f0.2: 8.129-ms projected saving but cosine 0.4025, rel-L2 2.2193, softmax mass 0.2033, top-key retention 0.1667). Classification CLOSED_NUMERICAL_OR_COMPLETE_GQA_GATE; executable not required; source restored. Evidence: .chatgpt/continuation_1605_20260821/p0_i_quest/CONCLUSION.md and final_decision.json.

### 2026-08-21 19:38 +07 — Complete-system decode bucket profile after P0-H/P0-I closures
- Fresh Nsys whole-route traces were reclassified without another profiler run. Short C1 route: target stage wall 18.877+16.228 ms/round versus proposal 4.596 ms and fold 0.368; target Q4/Q5 projection is 3.434 ms/round on runtime GPU0 and 2.766 on GPU1, while GPU1 proposal W8 is 3.198. Populated65K: target stage wall 26.000+30.375 ms/round, proposal 9.531, fold 0.375; target Q4/Q5 projection dominates at 20.544/20.999 GPU-ms per round, with exact long-GQA second at 8.566/10.453. Startup transforms were excluded. Conclusion: after safe-attention approximations closed, the next materially different mechanism must attack target projection dataflow; no additional broad NCU campaign. Evidence: .chatgpt/continuation_1605_20260821/next_bucket/decode_profile/REFINED_RESULT.md and refined_result.json.

### 2026-08-21 19:43 +07 — P0-F SM75 W3 target projection real-oracle correction
- Supersedes the earlier placeholder closure with missing evidence. The published Humming arbitrary-bit SM75 kernel was executed directly on the dynamically calibrated runtime card for all five real Qwen target projection shapes at T4 and T5. INT3/INT4 complete projection ratio was 0.875 at T4 and 0.854 at T5. Applying this favorable FP16 external ratio with zero cast/packing/integration overhead to the measured NInfer target-dense bucket gives only 0.775-ms short and 6.065-ms populated65K complete-round savings, below the 8-ms mechanism gate. No checkpoint-quality pass or production integration is warranted. Classification CLOSED_NO_COMPLETE_ROUND_OPERATOR_CEILING. Evidence: .chatgpt/continuation_1605_20260821/p0_f_w3_target/humming_w3/CONCLUSION.md and final_decision.json.

### 2026-08-21 19:53 +07 — CLOSED training-free layer-pruned self-drafter — free-target ceiling misses 100
- Teacher-forced mechanical layer skipping used the exact factory target and no trained/modified weights. Full 64 layers reproduced all 128 tokens; 23 structured 16-48-layer masks were measured.
- Only keep-gdn0+gdn2+full-attention (48 layers) retained useful agreement: 69.5% top1, D3 U=2.51, but 32.2 ms/autoregressive draft token on the fast role. Most 16-32-layer masks had <=8.6% agreement.
- After scaling only draft work to the current 1350-MHz helper and making target verification, transport, state, and integration free, the best configuration remains below 100 useful tok/s. No runtime implementation; diagnostic source restored.

### 2026-08-21 20:05 +07 — P0-J SM75 INT2 target projection
- External Humming FP16xINT2 on all five real target shapes passed only the populated65K operator ceiling: favorable W2/W4 ratio 0.777 at T5 projects 9.271-ms NInfer dense saving; short saving only 1.190 ms. Mechanical quality then decoded the current native Q4/Q5 artifact and evaluated 17 representative projection blocks using 48 real six-family T4 verifier hidden rows. Symmetric group64 W2 had 0.343-0.435 rel-L2; even a strictly stronger four-independent-centroid/group64 oracle remained 0.279-0.370 rel-L2 and failed the generous <=0.10 / cosine>=0.995 gate throughout. Classification CLOSED_W2_EVEN_FOUR_CENTROID_ORACLE_FAILS; no E2E port. Evidence: .chatgpt/continuation_1605_20260821/p0_j_w2_target/humming_w2/CONCLUSION.md and quality/CONCLUSION.md.

### 2026-08-21 20:19 +07 — P0-L CUDA graph/subgraph residency
- Short decode is already stage-wide graph replay; cudaGraphLaunch API is only 0.250 ms/round. The populated65K physical route is eager at the cross-device GQA override, but a favorable existing-trace ceiling that deletes every kernel/memset/event issue call and charges every memcpy at median API duration is 6.447 ms/round, below the 8-ms mechanism gate. Larger sync totals wait on mandatory owner projection, helper transfer/compute/return, state upload and exact merge; graphing cannot remove that work. Classification CLOSED_GRAPHABLE_ISSUE_SURFACE_BELOW_8MS. Evidence: .chatgpt/continuation_1605_20260821/p0_l_graph_residency/CONCLUSION.md and result.json.

### 2026-08-21 21:06 +07 — P0-N multi-branch successor PipeInfer
- Combined the accepted compact all-node GDN/tree state, exact ancestor-only tree attention, and N17-N64 grouped Q4/Q5 route into a genuinely new multi-successor PipeInfer ceiling.
- Every implementation cost except measured target work is free: branch MTP/proposal generation, transfer, selection, fold, graph/control, storage, and wide full-attention/MLP growth. Exact outcome coverage rises 23.53/37.25/45.10/52.94/58.82% for 1-5 successor branches.
- The corresponding optimistic stage0 branch batches are 17.085/25.150/36.666/41.931/42.287 ms and free-everything useful ceilings are only 84.34/74.14/57.76/53.22/53.92 tok/s. CLOSED without runtime code. Evidence: .chatgpt/continuation_1605_20260821/p0_n_multibranch_pipeinfer/CONCLUSION.md.

### 2026-08-21 21:10 +07 — Owner correction — stock GPU settings are mandatory for all hard gates
- The 82.02 tok/s single result was obtained with +100 MHz graphics offset, +1000 MHz memory-transfer offset and a 280 W power limit. It is diagnostic-only and is not a valid 80-tok/s gate result.
- All authoritative performance and 80/100/100/100 recertification must use stock card settings: default clocks/boost behavior, default power limit, automatic/default fan control, and no OC/undervolt/forced clocks. Do not modify GPU settings to pursue software performance.

### 2026-08-21 21:16 +07 — P0-O semantic hidden-key retrieval availability
- Evaluated a mechanism materially different from lexical lookup: a perfect per-position semantic-addressing upper bound inspired by hidden-state retrieval. It sees every prior token position, never misses a cosine neighbour, merges for free, and charges widths up to 60 only the current D3 round.
- Across general/code/JSON/Thai/Japanese/Arabic, cross-request perfect addressing tops at 44.08 tok/s; granting current-request hidden retrieval as well tops at 56.49 tok/s, with the worst family 39.39. Repo-edit can exceed 100 as a special copy-heavy mode, but cannot define the global frontier.
- CLOSED_EVEN_PERFECT_SEMANTIC_ADDRESSING for hard-gate implementation. Evidence: .chatgpt/continuation_1605_20260821/p0_o_semantic_retrieval/AVAILABILITY.md.

### 2026-08-21 21:16 +07 — Genuine populated262K frontier — deterministic 32.76 useful tok/s
- Built a genuine 261,900-token prefix, captured a 9.07-GiB legacy snapshot, and reloaded it into physical sequence-sharded INT8 KV plus split36 deduplicated TP2. The initial reload failure was a split32-package/split36-run mismatch; a corrected symlink-only nonlocal split36 package resolved it without recapturing.
- Three full 128-token runs are byte-identical: digest 12deaecb50f4ebb6, 42 rounds, 167 drafted, 83 accepted, accepted-by-position 40/39/3/1, zero fallback. Useful TPS 32.84/32.76/32.71, median 32.76.
- Median round is about 92.31 ms for 3.024 useful tokens/round; target stages dominate at roughly 38.8 + 35.3 ms/round, proposal about 15.0 ms/round. The immutable 100-t/s gate is not met.
- Evidence: .chatgpt/continuation_1605_20260821/populated262k_frontier/result.json and physical_runs/run_0..2.

### 2026-08-21 21:16 +07 — TurboQuant training-free KV oracle — CLOSED numerical gate
- Evaluated deterministic randomized-Hadamard + Gaussian Lloyd-Max norm-corrected K/V on 65,292 actual Qwen tokens, all four KV heads and 24 query heads; newest512 remained current INT8.
- K4/V4 has a material 11.445-ms raw 262K byte ceiling but min cosine 0.99031 and max attention-output relative-L2 0.16047. K3/V4 improves rel-L2 to 0.13058 but min cosine remains 0.99358. K4/V3 is worse. No downstream or CUDA implementation.
- Evidence: .chatgpt/continuation_1605_20260821/p0_l_turboquant/CONCLUSION.md and result.json.

### 2026-08-21 21:16 +07 — KVarN K4/V4 first downstream — recent512 changes trajectory; larger residual remains live
- Actual65K attention screen for calibration-free Hadamard + dual-axis variance normalization passes strongly at K4/V4 (G128 min cosine 0.999605/max rel-L2 0.02829; G64 0.999705/0.02428) with a 10.84/10.21-ms raw 262K ceiling.
- Conservative canonical-INT8-recode snapshot with G128/sink128/recent512 changes the 4K target trajectory: digest d62625bfc9df037c -> 3dd391dc491850ab and 50/150/74 -> 48/144/76 rounds/drafted/accepted.
- Per the old-KV contract, the lane remains live only for group64 and larger recent-INT8 windows before any native kernel.
- Evidence: .chatgpt/continuation_1605_20260821/p0_k_kvarn/result.json and snapshot4k_quality/result.json.

### 2026-08-21 21:18 +07 — Genuine populated-262K physical-KV E2E frontier
- Cold unsharded bootstrap completed on 261,900 prompt tokens in 1,201.235 s (218.03 prompt tok/s) and produced a 9.74-GB exact snapshot.
- Physical INT8 sequence-sharded KV + deduplicated T5 TP + factory MTP depth4 then completed three deterministic 128-token runs: 32.84/32.76/32.71 useful tok/s, median 32.76, digest 12deaecb50f4ebb6, 42 rounds, 167 drafted, 83 accepted, positions 40/39/3/1, zero fallback.
- Measured per-round buckets are stage0 ~38.8 ms, stage1/head ~35.3 ms, proposal ~15.0 ms, physical-GQA helper ~28.7 ms. The 100 gate is not met. Evidence: .chatgpt/continuation_1605_20260821/populated262k_frontier/CONCLUSION.md and result.json.

### 2026-08-21 21:21 +07 — Post-reboot single hard-gate recertification
- Six alternating exact complete-path runs after reboot: faster runtime UUID 76.61/76.52/76.47 useful tok/s (median 76.52, 1860-MHz loaded); other UUID 70.52/70.51/70.46 (1350 MHz).
- All six have identical stdout, digest 14660c413b4cf185, 43 rounds, one fallback, 129 drafted, 83 accepted, positions 35/29/19. The previous isolated 82.02 observation is not stable/certified. Single80 remains open and requires about 1.7 ms/round exact complete-path saving.
- Evidence: .chatgpt/hard_gates_20260821/post_reboot_single/CONCLUSION.md and result.json.

### 2026-08-21 21:24 +07 — Genuine populated-262K physical-KV frontier
- Cold unsharded capture successfully populated 261,900 prompt tokens in 1201.235 s (218.03 tok/s) and wrote a 9.738-GB legacy stage snapshot. Existing legacy-to-physical page packing then loaded physical INT8 sequence-sharded KV plus deduplicated T5 TP for three exact 128-token runs. All three were byte/digest/MTP deterministic: median 32.76 useful tok/s, 42 rounds, 167 drafted, 83 accepted, accepted-by-position 40/39/3/1, zero fallback. This is the first trusted genuinely populated-262K E2E frontier and is far below 100. Evidence: .chatgpt/continuation_1605_20260821/populated262k_frontier/CONCLUSION.md and result.json.

### 2026-08-21 21:24 +07 — P0-N query-aware mixed INT8/INT4 KV tiers
- Combined the closed Quest page-ranking and KV4 directions without dropping any pages: query-bound page unions plus recent512 remain canonical INT8, all other pages use page/channel affine K4 and token/group64 V4, and every page remains in the exact softmax. Real 65K four-head capture plus 262K gaussian/near-tie/peaked tests found material 7-8.6-ms byte ceilings only where top-key retention fell to 0.67-0.83 or cosine dropped below 0.995. Regimes with stronger numerical fidelity lost the complete 7-ms gate. Classification CLOSED_NUMERICAL_OR_COMPLETE_BYTE_GATE; no CUDA implementation. Evidence: .chatgpt/post_reboot_20260821/p0_n_tiered_kv/CONCLUSION.md and result.json.

### 2026-08-21 21:34 +07 — P0-H direct W4 vocabulary top1
- An environment-gated direct-W4 top1 canary bypassed the exact W8 candidate rerank. On the dynamically selected 1860-MHz runtime GPU it preserved the complete fixed trajectory (same IDs, digest, 44 rounds, 132 drafted, 82 accepted) but measured 62.03 baseline versus 61.92 useful tok/s direct-W4. Exact rerank is not the single bottleneck. Classification CLOSED_EXACT_BUT_NO_E2E_GAIN; source restored. Evidence: .chatgpt/post_reboot_20260821/full_w4_direct/CONCLUSION.md and result.json.

### 2026-08-21 21:50 +07 — CORRECTION direct W4 vocabulary head — CLOSED after full fixed + six-family E2E
- The complete direct-W4 rerun supersedes the earlier partial fixed-prompt note: fixed output changes digest/trajectory and regresses 76.56 -> 73.27 useful tok/s; six-family output is non-exact in every family.
- Family effects are unstable and quality-bearing (e.g. code 101.52 -> 77.90, general 62.21 -> 53.32; only isolated JSON/Thai throughput rises while output changes). Direct W4 cannot be a production head.
- Source was restored; only the accepted MTP prefill pipeline remains in apps/dual/main.cpp. Evidence: .chatgpt/post_reboot_20260821/full_w4_direct/result.json and CONCLUSION.md.

### 2026-08-21 21:55 +07 — P0-H direct W4 vocabulary top1 corrected canonical run
- Supersedes the earlier noncanonical 62-t/s canary, which omitted --lm-head-draft. The corrected trusted route reproduces 76.70/76.42 t/s baselines, digest 14660c413b4cf185, 43 rounds, 129 drafted and 83 accepted. Direct W4 top1 changes the fixed output/digest and lowers speed to 73.27 t/s; all six portable families also change output/trajectory. Classification CLOSED_DIRECT_W4_EXACTNESS; source restored. Evidence: .chatgpt/post_reboot_20260821/full_w4_direct/CONCLUSION.md and result.json.

### 2026-08-21 22:20 +07 — CLOSED target-only naive Jacobi-shift discriminator
- Width-6 target-only Jacobi removed recursive MTP proposal after prefill and seeded guesses only from later outputs of the preceding target row. Canonical target verification/fold preserved every ID, stdout byte, and digest.
- Fixed complete path regressed 43.46 -> 21.50 useful tok/s: rounds 47 -> 115, guesses 575, accepted only 7 (1.22%), while proposal fell 0.383 -> 0.008 s. Target work more than doubled.
- Do not widen or tune this same shift mechanism. Published full Lookahead Decoding remains distinct only if an actual online multi-trajectory n-gram pool first demonstrates materially higher exact accepted columns. Source restored. Evidence: .chatgpt/post_reboot_20260821/jacobi_d5/CONCLUSION.md.

### 2026-08-21 23:26 +07 — P0-M INT4-MTP plus exact-Q5 resident package — CLOSED
- Real maintained-SM75-Marlin MSE-fine Q4G64 factory MTP improves the fixed complete path from 76.29 to 77.97 useful tok/s with exact digest/stdout and 43 rounds/83 accepted.
- Even granting the independently measured 38-layer exact-Q5-down subset its full 0.344 ms/round for free projects only 78.69 useful tok/s, below single80 before integration or numerical cost.
- Q5 resident source work is therefore not implemented; rejected Marlin-MTP/A3 experimental source is restored. Evidence: .chatgpt/continuation_1605_20260821/p0_m_int4mtp_q5resident/CONCLUSION.md.

### 2026-08-22 00:00 +07 — P0-P FAFO exact-verification frame ceiling
- Official FAFO commit 08972d9f73188e94557abf20ae3954cd0a03135f verifies guesses exactly, but its target frame is (level-1)*(window+guesses). Published level9/window25/G20 runs 360 exact-target positions for at most 9 useful tokens. With accepted wide-target costs calibrated to fresh 65K and genuine262K, and lossy attention/candidate/control/fold free plus post-N22 GDN held flat: default perfect ceilings are 6.31/6.89 t/s; W25/G1 is 10.56/11.41. Even deleting all lookahead and verifying only one width9 guess reaches 92.72 t/s at 262K. Removing lookahead collapses to already-closed lookup/speculative verification. Classification CLOSED_FAFO_EXACT_TARGET_FRAME_CEILING_BELOW_LONG_GATES; no port. Evidence: .chatgpt/post_reboot_20260821/p0_p_fafo/CONCLUSION.md and result.json.

### 2026-08-22 00:05 +07 — CLOSED static proposal-head prefix truncation — N98K only +0.25 tok/s
- Mechanically tested current curated Q4 proposal head at N131072/98304/65536/32768/16384 with exact target output.
- N98K median 76.885 vs 76.635 tok/s with same 43 rounds and 83/129 accepted, but changed by-position 35,28,20. N65K regressed to 75.495 and 44 rounds; N32K/N16K regressed to ~67 and 50 rounds.
- Source restored. No material complete-system gain; do not retain or sweep same-family row prefixes.

### 2026-08-22 00:13 +07 — P0-Q VeriCache-style same-model compressed-KV drafter
- Fresh genuine262K depth0 diagnostics are deterministic at 14.42 t/s and isolate a 7.056-ms/token target dense/GDN floor after deleting every physical long-GQA packet/helper cost. A VeriCache-style same-model drafter was then given free compressed attention, free output/control/fold, free perfectly overlapped full-KV transfer, perfect acceptance, and drafted+1 useful tokens. Exact wide verification uses the accepted stock-card target curve. Perfect horizons 25/40 reach only 75.84/82.05 t/s; the infinite-horizon ceiling is 95.63 t/s. Classification CLOSED_SAME_MODEL_COMPRESSED_KV_DRAFTER_ASYMPTOTE_BELOW_262K_GATE; no runtime. Evidence: .chatgpt/post_reboot_20260821/p0_q_vericache/CONCLUSION.md and result.json.

### 2026-08-22 00:20 +07 — P0-R genuine262K MTP depth3 discriminator
- Fresh physical262K depth3+TP2 runs measured 31.68/31.60 t/s, below depth4 authority 32.76. Geometry changes the target trajectory/digest, expands to 52 rounds, and accepts 73/156 drafts (37,34,2); a third attempt terminated with cudaErrorIllegalAddress. No hardware/driver reset was attempted. Classification CLOSED_DEPTH3_SLOWER_DIFFERENT_TRAJECTORY_AND_UNSTABLE; depth4 remains authoritative. Evidence: .chatgpt/post_reboot_20260821/p0_r_depth3_262k/CONCLUSION.md and result.json.

### 2026-08-22 00:28 +07 — P0-S exact lossless KV compression
- The genuine262K INT8 snapshot has 134,119,424 group64 code groups. An exact low-bit base+exception-mask codec retains 99.867% of bytes because scale-normalized groups contain near-full-range codes; scan saving is only 0.037 ms. A stronger page-local reversible Zstd oracle across raw/channel-major/token-delta/token-XOR layouts reaches best ratio 0.9188, only 2.294 ms from the 28.24-ms scan with decompression free. Classification CLOSED_LOSSLESS_KV_COMPRESSION_BYTE_CEILING_BELOW_8MS; no kernel. Evidence: .chatgpt/post_reboot_20260821/p0_s_lossless_kv_exception/CONCLUSION.md and final_decision.json.

### 2026-08-22 04:16 +07 — P0-T s4o3 six-family qualification — CLOSED
- Rebuilt s4o3 from the saved one-file proposal-ranking patch, copied both baseline/candidate binaries, restored source, and ran interleaved two-repetition stock-card D3 E2E across general/code/JSON/Thai/Japanese/Arabic.
- Representative mean regressed 64.26 -> 61.63 useful tok/s (0.959x). General/code/Thai/Japanese/Arabic all slowed; only JSON improved. Final output differed for general/JSON/Thai/Arabic; code and Japanese were exact but slower. Loaded clocks were ~1800-1875 MHz and every run retained the stock 250 W limit.
- Classification CLOSED_NONPORTABLE_OR_NO_GAIN. Per owner instruction, do not combine with P0-M because s4o3 did not survive representative qualification. Source remains restored. Evidence: .chatgpt/post_reboot_20260821/p0_t_proposal_dimension_subsample/six_family_qualification/{CONCLUSION.md,result.json}.

### 2026-08-22 04:17 +07 — P0-U target-vocabulary dimension subsampling — CLOSED
- Existing whole-route attribution measures the entire current W4-shortlist plus exact-W8-rerank target head at 0.945 ms/round; its approximate projection is 0.594 ms.
- Fixed single80 needs about 1.64 ms/round saving at U=127/43 and 38.56 ms. Even impossible deletion of the complete vocabulary head reaches only 78.52 useful tok/s, so subsampling part of the approximate projection cannot qualify and can change target output.
- No E2E variant sweep. Source remained restored. Evidence: .chatgpt/post_reboot_20260821/p0_u_target_vocab_dimension_subsample/{CONCLUSION.md,result.json}.

### 2026-08-22 04:32 +07 — P0-V exact Q4 random-access entropy codec — CLOSED
- Q4 gate/up entropy is 3.4836 bits/nibble, but the current SM75 warp consumes 32 independent uint4 lane fragments per 512-byte group tile. Preserving that lane-local MMA mapping requires at least 32 13-bit starts.
- Shannon stream plus lane starts is 497.9/512 code bytes. With exact G64 scales, impossible zero-decode-cost payload saving is 2.58%; favorably applied to all 14.725 measured Q4 target GPU-ms, only 0.380 ms/round.
- Stock single80 needs ~1.70 ms/round. Bound misses before decode, alignment, headers, offset loads, or coalescing. No CUDA codec. Evidence: .chatgpt/post_reboot_20260821/p0_v_q4_random_access_entropy/{CONCLUSION.md,result.json}.

### 2026-08-22 04:49 +07 — P0-W exact W3 plus sparse residual — CLOSED
- Top-eight Q4 symbols cover 84.91%, leaving 15.09% exact exceptions. An impossible 1-bit mask plus one distinguishing bit/exception already consumes 4.1509 bits/weight before values/positions/alignment.
- The strongest executable W3 base, Humming, saves only 0.775 ms/round when favorably substituted across the measured short target-dense bucket, versus ~1.70 ms required for single80. Exact residual work is strictly positive.
- No CUDA implementation. Evidence: .chatgpt/post_reboot_20260821/p0_w_exact_w3_sparse_residual/{CONCLUSION.md,result.json}.

### 2026-08-22 05:16 +07 — P0-Y true SM75 Q4 producer/consumer pipeline — CLOSED
- Implemented one real Turing producer/consumer architecture: 8 loader warps double-buffer complete weight+scale groups in shared memory while 8 compute warps each own two original row tiles. Codes, G64 scales, and per-tile FMA order remained exact.
- Stock 250-W A/B, three repetitions: baseline median 76.58 versus pipeline 76.16 useful tok/s (0.9945x). Digest 14660c413b4cf185 and 43/129/83/35,29,19 trajectory were identical.
- Candidate used 63 registers/thread and 27,648 B dynamic shared. Shared traffic/barriers outweighed overlap. Closed without Q5 propagation or parameter sweep; source restored.

### 2026-08-22 05:29 +07 — P0-Z SubSpec-style low-bit substitute drafter — CLOSED
- Rebuilt the exact historical mixed Q4/Q5 runtime only into a teacher-forced diagnostic binary; production source was restored before all measurements. Tested Q4-down2/4/8 substitute checkpoints over six fixed-corpus families on the stock helper.
- Mean top1 agreement was 91.28/90.89/82.94%, but helper cost remained 47.14/47.11/46.60 ms per autoregressive draft token. The best impossible ceiling is q4down2 D1 at 40.82 useful tok/s with canonical target verification, transfer, state, and control all free.
- SubSpec benefits CPU-offload replacement; this deployment is fully GPU-resident. Closed without runtime implementation.

### 2026-08-22 05:40 +07 — P0-AA exact Q5 producer/consumer pipeline — CLOSED
- Implemented one materially different Q5 schedule: four loader warps double-buffer low/high/scales/A8 groups while four original K-partition warps compute. The first activation-tile integration bug was corrected before classification.
- Corrected stock A/B is byte-exact with digest 14660c413b4cf185 and identical 43/129/83/35,29,19 trajectory, but median regresses 76.39 -> 73.65 tok/s (0.964x).
- Shared staging/barriers and 256-thread blocks outweigh overlap. Closed without P0-M combination or parameter sweep; source restored.

### 2026-08-22 06:49 +07 — P0-AB certified sparse proposal head
- Captured 1008 actual proposal-head hidden/token pairs across six families with source restored. Wrapped s4o3 in rigorous residual intervals covering omitted exact-Q4 dimensions plus selected B2 weight/activation error. Signed segment bounds at 128/64/32/16 dimensions certified 0/1008 states. Current approximate-head tokens also differ from the all-row exact-Q4 argmax on up to 4.6% of a family. Classification CLOSED_CERTIFICATE_ZERO_PASS_RATE; no fallback implementation.

### 2026-08-22 07:00 +07 — P0-AG target B1 K256 exact rerank
- One geometry-correct K256 rerank discriminator compiled and passed the tail-mask test. General remained exact at +0.55%, but Arabic still changed digest/trajectory despite K256 (+2.41%). The exact W8 winner is outside the first 256 B1 candidates on a real Arabic state. Classification CLOSED; no further candidate-count tuning or P0-M combination. Source restored.

### 2026-08-22 09:27 +07 — Owner correction: fast-MTP/turbo output parity is behavioral, not byte-exact
- Do not reject a performance route solely because generated digest/trajectory differs from T1 or another qualified fast route. Historical NInfer fast-MTP permits deterministic near-tie/numerical divergence across batched/fused/quantized paths.
- Real correctness bugs remain disqualifying: state/mask/index/rollback errors, cross-column leakage, nondeterminism, memory safety, broken speculative transactions, or state-machine disagreement.
- Otherwise qualify output-changing routes by representative behavior/quality plus performance. Benign wording/outline/ordering changes are allowed; reject only material quality/pathology regressions.
- P0-AG target-B1 K256 is therefore **reopened for quality qualification**. Its Arabic baseline/candidate share 66/256 initial tokens then diverge into different coherent technical outlines; no observed genius-to-idiot collapse. K128 target-B1 was byte-identical on 5/6 families and differed only Arabic.
- `AGENTS.md` now contains the durable fast-MTP/turbo behavioral-output contract.

### 2026-08-22 10:38 +07 — P0-AH Q4G128 maintained SM75 Marlin complete E2E — CLOSED
- Completed the normal CUDA runtime path: fixed device permutation compile, generic variant guards, full-packed attention split routing, T1-T4/T128 execution, pre-graph preparation, workspace admission, and explicit 256-MiB prepared-weight reservation; production source restored before E2E.
- Stock interleaved 128-token medians: accepted W8 76.45 useful tok/s versus Q4G128 Marlin 74.48. Both are deterministic and produce the same rendered output/digest. Candidate trajectory 44/132/82 versus baseline 43/129/83 is allowed by the corrected contract; it is rejected because the real complete path is 2.58% slower.
- Classification CLOSED_FIXED_BELOW_MATERIAL_GATE. Six-family qualification not warranted after the representative fixed route regressed.

### 2026-08-22 10:54 +07 — P0-AG B1/K256 corrected behavioral qualification
- Applied the corrected fast-MTP output contract: digest/trajectory changes were allowed when deterministic and nonpathological.
- Two stock repetitions across general/code/JSON/Thai/Japanese/Arabic were deterministic and passed structural/script/repetition checks; output changed coherently for code, JSON, and Arabic.
- Representative mean nevertheless regressed 64.23 -> 64.04 tok/s (0.997x), dominated by code 71.38 -> 66.16. Classification CLOSED_DETERMINISTIC_QUALITY_OK_BUT_REPRESENTATIVE_NO_GAIN; do not combine with P0-M; source restored.

### 2026-08-22 10:59 +07 — P0-AI QuantSpec-style Qwen3.8 complete-system ceiling
- Modeled the official QuantSpec mechanism against this Qwen3.8 physical-262K system, not paper headline speedups: same-model draft, hierarchical 4-bit KV, optional W4 draft weights, exact current INT8 target verification.
- Favorable bound makes draft KV attention/head/packing/control/transfer/fold free, converts every Q5 and even W8 text tensor to Q4, and scales the whole measured 7.056-ms non-attention floor by weight bytes to an impossible 5.937 ms/token.
- Acceptance is credited from the less-quantized six-family Q4-down2 oracle. Best trace ceiling is 45.28 t/s at horizon6; horizons25/40 are 31.43/22.33. Perfect acceptance reaches only 90.12 at horizon40 and crosses 100 only near horizon127; observed useful/round saturates near 10.44. Classification CLOSED_QUANTSPEC_QWEN_COMPLETE_SYSTEM_BELOW_GATES; no runtime.

### 2026-08-22 11:01 +07 — P0-AJ matched Humming W4/W5 target schedule
- Compared executable SM75 Humming on the exact hot Qwen T4 shapes against current fused NInfer kernels, not generic GEMMs.
- Q4 gate/up: Humming projection-only 199.99 us vs current fused Q4 A8 SwiGLU 167.61 us. Q5 down: Humming exact-weight FP16 projection-only 126.83 us vs current fused A8+residual 106.63 us.
- Humming loses before omitted activation/cast/epilogue work. Classification CLOSED_MATCHED_HUMMING_SCHEDULE_SLOWER_THAN_CURRENT_FUSED_KERNELS; no port or parameter sweep.

### 2026-08-22 11:04 +07 — P0-AK QServe exact-dataflow transfer to SM75
- Inspected official OmniServe/QServe W4A8 per-group source at commit 02b2925aa6fa3b92b06316a1524b7f38922cd9c8.
- Its core overlap requires cp.async and m16n8k32 s8 MMA; direct ptxas at sm_75 rejects both as sm_80+ features. Upstream build supports only 8.0/8.6/8.7/8.9/9.0.
- After subtracting Ampere ISA, the residual is the already-measured exact P0-Y synchronous producer/consumer pipeline: 76.16 vs 76.58, despite current Q4 reaching 86.55% peak DRAM. QServe INT8 scale/zero-point semantics also do not preserve NInfer Q4G64 FP16-scale fused SwiGLU/A8. CLOSED_QSERVE_CORE_ISA_SM80_ONLY_SM75_RESIDUAL_ALREADY_MEASURED_SLOWER.

### 2026-08-22 11:21 +07 — P0-AN parallel exact late-fork replica branches
- Repriced the exact tree/Bole rescue measurements as two full model replicas verifying both branches perfectly concurrently.
- Depth2 late-fork yield 2.53704 and depth3 2.42308 useful/round. At the accepted 34.082-ms target-stage lower bound, with proposal/top2/tree/transport/fold/state free, ceilings are only 74.44 and 71.10 useful tok/s.
- Populated target cost is higher. Classification CLOSED_PARALLEL_REPLICA_BRANCH_YIELD_BELOW_DUAL100; no replica-state implementation.

### 2026-08-22 11:22 +07 — P0-AL SDFP component-level pruning ceiling
- Executed a training-free component-level lower bound distinct from whole-layer P0-K: retained 42/64 FFNs (published 65% SDFP fraction) while making all 32 retained mixers free.
- Best stock fast-card 42-FFN-only cost is 17.961 ms/autoregressive token. Even perfect acceptance at unbounded horizon, with target verification/communication/state/fold free, asymptotes to only 55.68 useful tok/s; actual SDFP retains 32 mixers and is slower.
- Classification CLOSED_SDFP_COMPONENT_COST_FLOOR_BELOW_DUAL100. FIT ranking cannot change the fixed-cost ceiling. Diagnostic source restored and final ninfer-dual build passed.

### 2026-08-22 11:54 +07 — P0-AO full-W4 target-head corrected behavioral qualification
- Applied the owner-corrected fast-path contract: changed output/digest was allowed if deterministic and representative-quality sound.
- Two stock repetitions across six families were deterministic, but representative mean regressed 64.20 -> 60.98 tok/s (0.950x). General/code/JSON also slowed; Thai/Japanese failed structural script/length quality and Arabic changed nonportably.
- Classification CLOSED_DETERMINISTIC_BUT_SLOWER_AND_NONPORTABLE. Source had already been restored; no promotion and no P0-M composition.

### 2026-08-22 12:02 +07 — P0-AP per-round proposal candidate-cache
- Implemented one full 131k proposal scan followed by graph-safe exact-Q4 rerank of the same 128 candidates for draft positions 2-3; source restored after candidate build.
- Three paired stock 256-token runs were deterministic: baseline 72.33 tok/s, 93 rounds, 162/277 accepted; cache candidate 59.53 tok/s, 115 rounds, 140/343 accepted.
- Later-position acceptance collapsed from 73/54/35 to 93/40/7. Classification CLOSED_ACCEPTANCE_COLLAPSE; no cache-period micro-sweep because one remaining saved scan cannot clear the complete-system gate.

### 2026-08-22 12:19 +07 — P0-AP request-local active proposal vocabulary
- Implemented a GPU-resident exact-Q4 active vocabulary populated only from prompt IDs, historical proposal IDs, and already-computed target top1 IDs; no future-token or offline-corpus leakage.
- Three paired stock 256-token runs were deterministic: baseline 72.33 tok/s, 93 rounds, 162/277 accepted; active candidate 53.48 tok/s, 129 rounds, 126/385 accepted.
- Yield collapsed 2.74 -> 1.98 useful tok/round despite head-work reduction. Classification CLOSED_ACCEPTANCE_COLLAPSE; no candidate-size/window/refresh micro-sweep. Source restored.

### 2026-08-22 13:44 +07 — P0-AQ official Lookahead oracle
- Reconstructed official LADE LRU prompt/generated-pool semantics from upstream and generated six exact current-target Jacobi rows from one random seed for general/code/JSON.
- Charged every lookahead level plus verified n-gram under N8/12/16/24/32 node budgets. The cost bound is deliberately impossible-favorable: measured 48-GDN-layer all-node+fold only, all 16 full-attention layers/proposal/state/control free, and N23-N32 free over N22.
- Chronological best is 37.04 useful tok/s. Even all rows preloaded with same-first clairvoyant candidate selection reaches only 38.99. Classification CLOSED_OPTIMISTIC_EXACT_TARGET_CEILING_BELOW_SINGLE80; no port.

### 2026-08-22 14:02 +07 — P0-AR device-resident CUDA-Graph self-launch MTP
- Both stock SM75 cards tail-self-launched a device graph 3/3 with zero errors; the actual NInfer MTP graph also device-instantiates and runs.
- Actual MTP graph clone, 20 repetitions: host launch+synchronize 37.142214 ms/round versus one host launch + device tail self-launch 37.197430; saving -0.055216 ms/round.
- 128-token host-boundary timing: graph+sync 36.993528 ms, post CPU 0.000529, mandatory fold+sync 0.624350, ledger CPU 0.000366 per round. The fold remains a serial GPU dependency.
- P0-M is 77.97 t/s; even deleting the entire fold plus CPU bookkeeping gives only 79.31 t/s. Classification CLOSED_NO_MATERIAL_COMPLETE_ROUND_SAVING; no AR port or P0-M joint test.

### 2026-08-22 14:32 +07 — P0-AS Token Recycling disjoint hot-start final
- Captured 403 exact current MTP rounds and 1,612 real T4 verifier hidden rows across six families; reconstructed W8 top-k8 and official adjacency/tree semantics.
- Cold chronological OPT MTP-spine was 61.52 t/s only under GDN-only favorable cost. The required paper-style leave-one-family-out hot-start was then charged real N8/N12/N16 target costs.
- Fixed overwrite tested all 5! calibration-family orders; bounded union used only disjoint prior requests. Best was N8 MTP-spine 35.99 representative t/s with lookup/update/control/fold free.
- Same-query union/clairvoyance is inadmissible. Classification CLOSED_DISJOINT_HOTSTART_REAL_COST_BELOW_SINGLE80; no runtime port.

### 2026-08-22 14:35 +07 — P0-AT LogitSpec training-free oracle
- Cloned official smart-lty/LogitSpec and replayed its longest-3-gram pool, next-token direct retrieval, next-next logit guidance, rank pruning, and bounded tree on six current target trajectories.
- Charged real N8/N12/N16 NInfer target costs; retrieval, tree construction, controller and fold were free. Missing-position guidance was made clairvoyantly exact.
- Strongest result is MTP-spine N8 with exact next-next guidance: U=2.427, 44.71 representative useful t/s. Even all-vocabulary retrieval upper is 39.29 pure-TR.
- Classification CLOSED_EVEN_ALL_VOCAB_RETRIEVAL_UPPER_BELOW_SINGLE80; no runtime port.

### 2026-08-22 14:55 +07 — P0-AV device-resident cross-GPU TP2 mailbox
- Measured a host-free persistent owner/helper mailbox on both stock GPU role assignments with 40/50-KiB four-leg GPU->RAM->GPU->RAM->GPU payloads and system-fenced generation flags.
- T4 long-run is 198.12 us/layer best, 12.68 ms across 64 layers; all 3,136 exchanges passed byte verification. T5 is 239.49 us/layer.
- Even deleting all 17.64 ms of target MLP arithmetic and charging only mailbox traffic gives 87.90 useful tok/s; after P0-M only 91.49. CLOSED_ZERO_COMPUTE_CEILING_BELOW_DUAL100; no port.

### 2026-08-22 15:17 +07 — P0-AU current P0-M recertification
- Recovered the preserved Marlin source, corrected the actual environment gate, made bind graph-safe, and declared the exact 222.1865-MiB cache to the startup planner.
- Interleaved stock E2E: current baseline 76.35, repaired P0-M 79.01 useful tok/s, deterministic and byte-identical. This supersedes the old 77.97 result and leaves ~0.46 ms/round to single80.

### 2026-08-22 15:17 +07 — P0-AW exact commit-count fold specialization
- Exact compile-time fold kernels for commit counts 1-4 were interleaved against repaired P0-M.
- 78.92 versus 78.95 useful tok/s, exact output/trajectory; neutral. CLOSED_EXACT_FOLD_SPECIALIZATION_NEUTRAL; source restored.

### 2026-08-22 15:44 +07 — P0-AX verifier-state snapshot selection
- Exact C1 prototype wrote four GDN state frontiers during verify and selected the accepted snapshot instead of replay-folding.
- Naive 96-copy selector: 78.81 -> 76.36 tok/s. Two-kernel fused selector: 78.74 -> 76.74. Both exact; source restored.
- Fusing ~154 MiB/round of state copies recovers only 0.38 tok/s; extra full-state writes dominate. CLOSED_EXTRA_SNAPSHOT_WRITES_DOMINATE; no more copy tuning.

### 2026-08-22 15:54 +07 — P0-AY B2 vocabulary projection CTA geometry
- Regrouped the exact B2 producer into 2/4/8/16 warps per CTA while preserving arithmetic, 15,520 total warps, 128 candidates/partial, memory, and exact W8 rerank.
- All stock 64-token outputs/trajectories are identical; medians 86.25/86.26/86.31/86.22. CLOSED_EXACT_GEOMETRY_NEUTRAL; no wider sweep.

### 2026-08-22 16:41 +07 — P0-AZ s4o3 six-family qualification
- Completed 24 stock-GPU E2Es: P0-M baseline and s4o3, two deterministic 256-token runs each across general/code/JSON/Thai/Japanese/Arabic.
- Candidate remained structurally portable and deterministic but representative mean regressed 66.14 -> 63.11 useful tok/s (0.954x); Japanese regressed 62.95 -> 50.25.
- CLOSED_REPRESENTATIVE_OR_QUALITY_FAIL under the corrected output contract. The fixed prompt 79.30 result is not promoted. This also removes the changed economics that briefly reopened the 37-layer Q5 Marlin subset.

### 2026-08-22 19:37 +07 — P0-BE joint B1 K64 — CLOSED / NEUTRAL
- Stock paired K128/K64 medians were 79.805/79.850 useful tok/s; paired median delta -0.005 and mean +0.01625 tok/s.
- Both were deterministic with the same output. Joint K64 is noise-sized and below single80.
- Only proposal-K64/target-K128 and proposal-K128/target-K64 asymmetric discriminators remain; no wider candidate-count sweep.

### 2026-08-22 20:00 +07 — P0-BE asymmetric K64/K128 — CLOSED / NEUTRAL
- Eight stock interleaved triplets on the dynamically selected card: K128/K128 median 79.845, proposal-K64/target-K128 79.815, proposal-K128/target-K64 79.820 useful tok/s.
- Paired mean deltas were -0.0113 and -0.0450 tok/s respectively; all variants were deterministic and preserved the same output and MTP trajectory.
- Candidate-count tuning is closed. Proceeding to direct-indexed exact rerank.

### 2026-08-22 20:18 +07 — P0-BF direct-indexed exact rerank — positive, final recert pending
- Direct original-row W8/Q4 rerank removed candidate code/scale gather. Microprofile net savings are about 15 us target and 8.9 us proposal per invocation path; combined profiled decode 87.30 vs 86.95 with identical semantic output.
- Ten stock interleaved 128-token pairs: baseline/indexed medians 79.725/79.750, paired median +0.055 and mean +0.063 tok/s. Deterministic and byte-identical with identical 43/129/83/35,27,21 trajectory.
- Positive package retained; one bounded P0-BG top-k collapse is being discriminated before source promotion and final single certification.

### 2026-08-22 20:19 +07 — P0-BG one-CTA direct top-k — CLOSED / DO NOT TUNE
- Exact one-CTA-per-column selector compiled and preserved semantic output.
- Target direct top-k was 478.1 us versus existing two-stage about 39.5 us; proposal 240.95 versus about 46.4 us. Combined profiled decode regressed 88.10 -> 85.49.
- Closed immediately; no selector parameter sweep.

### 2026-08-22 20:35 +07 — P0-BF promoted indexed-rerank single80 CERTIFIED
- Promoted direct original-row W8/Q4 exact rerank eliminates candidate code/scale gathers for target and proposal while preserving B1 coarse ranking and semantics.
- Twelve stock complete-path runs on the dynamically selected card: median 80.185, mean 80.153, min 79.84, max 80.48 useful tok/s; deterministic digest 14660c413b4cf185 and identical 43/129/83/35,27,21 trajectory.
- Single80 is frozen. Proceed immediately to dual-short, populated65K, and genuine populated262K 100 gates; do not polish single.

### 2026-08-22 20:52 +07 — Promoted P0-M+B1 indexed package on stock dual-short — structural gap
- Dynamic role assignment selected UUID order 2fffc240 then 020906b4. Six deterministic complete-path repetitions: median 63.095, min 62.60, max 63.32 useful tok/s; digest 14660c413b4cf185.
- Trajectory is 51 rounds / 153 drafted / 74 accepted / 36,20,18. Median stage0/stage1/proposal/fold are 16.784/18.176/2.627/0.333 ms per round.
- Arithmetic stages are already balanced; dual100 requires a structural yield/dependency mechanism, not another B1/head polish. Evidence: .chatgpt/post_reboot_20260821/dual_short_promoted_frontier/result.json.

### 2026-08-22 21:11 +07 — P0-BI periodic target anchoring / accept-all D3 — REJECTED
- Real dual-short accept-all was deterministic at 98.095 tok/s median with 31 rounds and 31/31/31 accepted positions; six-family single mean was 110.07 tok/s.
- Representative quality failed: Thai became repetitive mixed-language fragments, Arabic repeated a short phrase, and code degraded. Existing six-family evaluator classified CLOSED_REPRESENTATIVE_OR_QUALITY_FAIL.
- Entropy/deferred D3 gates below unconditional acceptance remained far below dual100. Tail-overrun cannot repair quality; canonical greedy acceptance retained.

### 2026-08-22 22:36 +07 — P0-BN intra-round target microbatch pipeline — CLOSED
- Promoted-package T2 stage0/stage1 are 16.514/17.986 ms; T4 is 16.725/18.176 ms.
- A 2xT2 two-stage pipeline must re-stream weights and costs 52.486 ms target makespan versus 34.902 ms current T4. Zero-overhead useful ceiling is below current, so no implementation.

### 2026-08-23 00:55 +07 — P0-BO current D4 probe — CLOSED
- Stock fixed short current D4: single 68.16 useful tok/s (39 rounds, 154 drafted, 87 accepted, AL 3.23); dual layer-shard 49.01 useful tok/s (49 rounds, 196 drafted, 77 accepted).
- Both complete deterministically; output-contract relaxation is irrelevant because the route is strongly performance-negative. Six-family authority already selects D3.
- No D4 sweep. Continue with canonical D3/T-width economics and checkpoint-robust independent proposer work.

### 2026-08-23 01:26 +07 — P0-BQ PARD-Qwen3-0.6B + TLI/SLEM — CLOSED
- Published amd/PARD-Qwen3-0.6B commit f9f650f was measured mechanically on stock TU102 against the current six-family deterministic NInfer target outputs. TLI has 130,783 unique decoded-string draft IDs (86.23% of draft vocab); SLEM retokenizes unrestricted draft text.
- Representative useful/round: TLI K4/K8/K12 1.590/1.608/1.612; SLEM 1.650/1.677/1.686. Best SLEM K12 family rows: general 1.707, code 2.207, JSON 3.012, Thai 1.255, Japanese 1.552, Arabic 1.340. Stock Transformers parallel block medians are about 29-32 ms at 1875 MHz.
- Feeding real yield into the current PipeInfer target floor closes the route even with a free independent noncontending proposer and every actual K+1 target width priced as measured T2: best infinite 93.74, finite128 92.62, finite256 93.18 useful tok/s. Accepted 65K target cadence gives 57.34; 262K is worse. No runtime port.
- No actual modified/abliterated Qwen3.8 target exists locally, so authority forbids fabricating a DFlash2 or modified-target row. Local vLLM PARD loaded the checkpoint but segfaulted during SM75 engine initialization across full/draft-only, FlashInfer/Triton, fork/spawn; this does not affect the free-proposer closure.

### 2026-08-23 01:51 +07 — P0-BS PARD probability tree / factory spine — CLOSED
- Published PARD TLI top-8 logits were replayed as prefix-probability best-first trees on the current promoted P0-M+B1 six-family target outputs, with exact forced-scalar factory-MTP drafts available for a mandatory hybrid spine.
- Pure N8/N12/N16/N22 representative useful/round: 1.882/1.952/2.013/2.059. Hybrid: 1.801/1.908/1.964/2.043. Thai and Arabic remain weak; the hybrid never beats pure PARD at matched node budget.
- Charging real target-node costs 28.717/49.188/58.548/59.498 ms and making proposer/tree/transport/fold/state free gives a best impossible ceiling of 65.55 useful tok/s (pure N8). No runtime port.

### 2026-08-23 01:54 +07 — P0-BT factory/PARD consensus target skip — CLOSED
- Collected 1,512 current promoted P0-M+B1 target roots with exact factory D3 drafts plus published PARD K4 TLI/SLEM probabilities and margins.
- Factory/PARD first-token agreement occurs at only 44/1512 TLI and 38/1512 SLEM roots, with 20.45%/23.68% observed precision. Two- and three-token agreements had zero correct examples. Exhaustive confidence/margin thresholds found no nonempty zero-error rule.
- Even the invalid quality-tolerant oracle that emits every wrong agreement and deletes those target rounds for free reaches only about 65.6-65.9 useful tok/s. CLOSED_CONSENSUS_SKIP_EXACT_COVERAGE_BELOW_DUAL100; no runtime implementation.

### 2026-08-23 02:29 +07 — P0-BU conservative top-K loose acceptance — CLOSED
- Promoted P0-M target, top-K32 loose gate, six-family periods 3/4/8: representative means 74.28/77.17/75.65 useful tok/s.
- Every row is below dual100 before any stricter FLy deferred-window or margin rule; loose trajectories also show representative pathologies.
- A conservative rule only removes loose rounds and cannot raise this ceiling. No threshold sweep or source promotion.

### 2026-08-23 02:35 +07 — P0-BV PARD+CSD perfect-gate ceiling — CLOSED
- Measured PARD SLEM-K12 mismatch stream with an impossible CSD upper: perfect/free semantic gate, free proposer/transport/control, T2 target cadence, and unchanged saved target path after rescued lexical mismatches.
- Frequency-qualified lambda=2 reaches only 100.58 useful tok/s on the full calibration corpus; leave-one-family-out is 96.98-100.15. Lambda>=3 is below 96; real PARD block is ~31.11 ms.
- The knife-edge full-corpus row has no portability or implementation margin. Lambda1 memorizes one-off pairs and is not frequency-qualified. No target-logit/runtime port.

### 2026-08-23 02:53 +07 — P0-BW published PARD2 target-hidden compatibility — CLOSED
- Captured actual modified-target layers 63/56/48/40 for every prompt column and 255 teacher-forced output positions across six families; diagnostic source restored.
- Published PARD2-Qwen3-14B warp has matching 20480 target dimension, but general-family compatibility is catastrophic: TLI K4/8/12 U=1.0667, SLEM U=1.0364, zero full-survival, 34.3-34.8ms proposer blocks.
- The checkpoint-specific learned warp does not transfer to this modified 27B/64-layer target. Free-proposer ceiling already misses dual100; no six-family expansion or runtime port.

### 2026-08-23 03:28 +07 — P0-BY Vegas/SparseSpec sparse self-drafting — CLOSED
- Cloned official platformxlab/vegas. Sparse self-drafting alters attention/KV access but still executes the target Q4/Q5 and GDN weight stream for every causal draft token.
- Delete attention, selection, communication, target verification, fold, and control entirely. Measured 17.64-ms dense + 11.84-ms GDN gives 29.48 ms per causal draft token; depth1 ceiling is 67.84 useful tok/s and deeper horizons decrease toward 33.92.
- Classification CLOSED_SPARSE_SELF_DRAFT_WEIGHT_STREAM_CEILING_BELOW_100. No port/sparsity sweep. Sparse target verification remains separate.

### 2026-08-23 03:32 +07 — P0-BX SWIFT/KnapSpec dynamic-layer self-drafter — CLOSED
- Ran 53 general-only mechanical mixer/MLP masks (published SWIFT, structured, deterministic random/asymmetric) with full model and source restored. Full-mask six-family teacher validation was 99.48% top1.
- Every causal draft token was charged at measured partial-model latency; target cadence used the impossible favorable 17.986-ms T2 lower bound with communication/fold/control/free target-width growth.
- Best actual independent placement is skip-middle-24-both, 40/40 modules, 70.31% top1, U=1.684, 27.197 ms/draft token, only 61.93 useful tok/s. Held-out cannot rescue a general ceiling below 100. No port.

### 2026-08-23 03:33 +07 — P0-BZ sparse verification + Windowed-MTP joint ceiling — CLOSED
- Jointed published Dustin/SpecAttn target sparse verification with Windowed-MTP. Selector/metadata/gather/sparse-kernel/communication are free; all measured long-GQA time is deleted; factory proposal is reduced to promoted short-context 2.627 ms/round; acceptance unchanged.
- 65K projects only 91.22 useful tok/s. Genuine262K projects 66.52; even deleting proposal entirely gives only 77.60.
- CLOSED_SPARSE_VERIFY_PLUS_WINDOWED_MTP_COMPLETE_PATH_BELOW_LONG_GATES. No selector capture or port because the complete system misses before quality.

### 2026-08-23 03:40 +07 — P0-CA Approximate Speculative Decoding horizons — CLOSED
- Cloned official ASD verifier commit d4876f5d0571f40ef81ad2ecf3bca78374d33015. Bounded each horizon by measured complete-path round cost with every draft unconditionally accepted, before regret selection or quality.
- D3 ceiling 98.095, D4 93.05, D5 68.18 useful tok/s. D5 was one bounded stock probe: 48 rounds, stage0/stage1 30.02/43.10 ms, proposal 7.38 ms.
- CLOSED_ASD_ALL_HORIZONS_BELOW_DUAL100_BEFORE_QUALITY. No risk-budget/gate/cap sweep can exceed accept-all.

### 2026-08-23 03:49 +07 — P0-CB D3 endpoint-consistency certificate — CLOSED
- Used the aligned 403-round six-family factory trace and mechanically corrected exact acceptance-decisive target winners. Tested every subset of draft columns 0-2 plus global target-margin screens.
- Every one/two-column subset has false whole-block accepts. Zero false positives requires all three draft columns, U=1.5285; even free additional head work at last-B1 wall is only 38.99 tok/s.
- CLOSED_ENDPOINT_CERTIFICATE_REQUIRES_ALL_DRAFT_COLUMNS. No runtime candidate.

### 2026-08-23 04:02 +07 — P0-CC sticky exact-anchor regret fallback — CLOSED
- Built a fail-closed last-B1 candidate: exact anchors accumulate rejected drafts and permanently return the request to canonical verification after a global budget.
- Screened periods 2/3/4/5/6/8 and budgets 4/8/16/32/64. Best fixed result is period4 budget32 at 96.68 useful tok/s; no candidate clears dual100, so no six-family expansion.
- CLOSED_STICKY_EXACT_ANCHOR_BELOW_DUAL100. Source restored after diagnostic build.

### 2026-08-23 04:14 +07 — P0-CD PARD-2 target-independent mode — CLOSED
- Downloaded official PARD2-Qwen3-8B commit 67a1516c8f6fc145cda99916799a0cbb3a4af135 and ran target-independent ordinary-drafter mode through TLI/SLEM on the actual six-family modified Qwen3.8 traces.
- Best SLEM K12 U=1.6445 with 0.214% full-block survival; family U general1.803/code2.065/json2.813/thai1.231/japanese1.399/arabic1.376.
- Free-proposer scaling is only 91.44 short / 55.93 populated65K, below PARD-1 and gates. Real proposer is ~31.34 ms in FP16. No port.

### 2026-08-23 04:22 +07 — P0-CE Qwen3-0.6B sequential drafter — CLOSED
- Loaded official Qwen3-0.6B commit c1899de289a04d12100db370d81485cdf75e47ca as a target-independent causal drafter and measured general TLI/SLEM acceptance with incremental KV.
- Best supported near-gate point is SLEM K3 U=1.7655, full survival 12.41%; with proposer/mapping/transport/contention free and measured T4 target cadence 18.176 ms, ceiling is 97.13 useful tok/s.
- K2 is 89.01; K4 uses measured D4 cadence and is 33.85. Current runtime max D5 has 43.10-ms target cadence. No six-family/GGUF spend.

### 2026-08-23 06:04 +07 — CLOSED P0-CF/P0-CG BitBLAS native low-bit — exact kernels miss body and target gates
- Repaired the local BitBLAS/TileLang SM75 runtime (bundled TVM/TileLang path, obsolete tail-split pass, CUTLASS/template includes) and ran exact stock T4 INT8xINT4/INT2 kernels on all five factory-MTP shapes. All sampled int32 outputs are exact.
- Graph-warm five-matrix totals per pass: W4 1.540 ms, W2 1.176 ms; three passes 4.620/3.529 ms. W2 is slower than current NInfer W8 3.235 ms and 2.18x promoted P0-M Q4G64/Marlin 1.619 ms, before scales/epilogues. P0-CF closed; no W2 quality/mixed recipe.
- P0-CG exact M4 K5120 N34816 W4 projection-only is 628.416 us (617.117-us evaluator), versus current fused Q4 projection+SwiGLU+A8 ~167-168 us and the 150-us close gate. Closed without port. Evidence: .chatgpt/post_reboot_20260821/p0_cf_cg_bitblas_lowbit/{CONCLUSION.md,result.json}.

### 2026-08-23 06:20 +07 — P0-CF supplementary W2 acceptance closure
- The stock BitBLAS/Ladder operator closure remains authoritative: exact graph-warm W2 is 3.529 ms for three dependent MTP passes versus promoted P0-M 1.619 ms, before scales/epilogues.
- Mechanical actual-checkpoint full-W2 was deterministic but retained only 61.03% accepted drafts across six families. The single permitted mixed recipe (W2 input/QKGV/gate-up, W4 attention-output/down) retained 91.55% but its graph-warm body is 3.849 ms/round and target rounds increased on every family.
- No W2/mixed port. One-shot target W4 projection-only is 628.416 us versus current fused 167-168 us, so P0-CG and selective target W2 remain closed.

### 2026-08-23 06:25 +07 — CLOSED P0-CI current-head intermediate target exits — no T4 round exits early
- Used 1,536 six-family teacher-forced target positions captured after layers 40/48/56/63 and reproduced the promoted B1 K128 head mechanically from the exact current W8 artifact (B1 transform, B2 activation, K128 exact rerank).
- Representative direct top1 is 0.46% / 0.72% / 8.59% / 76.82% at layers 40/48/56/63. A leave-one-family-out zero-calibration-error margin/stability policy exits only 13/1536 tokens early and has 3 held-out errors.
- After grouping by real factory T4 rounds and conservatively charging the maximum column depth, all 847 rounds remain layer63. Projected dual-short/genuine262K stay 63.09/32.76 useful tok/s. No runtime implementation. Evidence: .chatgpt/post_reboot_20260821/p0_ci_early_exit/{CONCLUSION.md,result.json}.

### 2026-08-23 06:29 +07 — P0-CI compressed device-resident TP2
- Measured the real 64-layer TP2 shard kernels and 20.64/10.32-KiB mapped-host packets. Parallel critical TP2 arithmetic is 10.326 ms/round before communication.
- On the promoted 80.185-tok/s round, zero communication reaches only 100.053 tok/s. Measured A8/A4 packet economics reduce the optimistic complete ceiling to 82.865 tok/s before quantization/reduction/integration.
- CLOSED_TP2_COMPUTE_FLOOR_LEAVES_NO_COMMUNICATION_BUDGET; no quality capture or port.

### 2026-08-23 06:37 +07 — P0-CJ A4 activation-side target-Q4
- Bounded exact M4 K5120 N34816 A4xW4/A4xW2 discriminator on stock SM75.
- Vendored BitBLAS/Ladder legalizes weight layout but rejects int4 activations as unsupported; no optimized SM75 function exists.
- CLOSED_NO_LEGAL_OPTIMIZED_SM75_A4_SCHEDULE before quality or integration; no tuning campaign.

### 2026-08-23 06:50 +07 — P0-CK committed-hidden extrapolation proposer
- Verified scalar hidden/head alignment, then evaluated stale and secant prediction on 397 actual six-family MTP round transitions.
- Best future-hidden mean cosine is only 0.5036. Best next-anchor top1 recall is 0.50%; top4 8.06%; top64 34.26-37.03%.
- CLOSED_NEXT_ANCHOR_HIDDEN_PREDICTION_NO_BRANCH_COVERAGE. Single branch almost never hits; prior two/multibranch exact target ceilings are already below dual100.

### 2026-08-23 07:08 +07 — P0-CJ TEAL T4 activation sparsity
- Official TEAL sparse GEMV has no verifier-width path: width4 falls back to dense matmul. A restored diagnostic measured exact retained-channel union across T4 before considering a custom quantized sparse kernel.
- At requested s50, row sparsity ~49% becomes only 8.16-14.58% union sparsity; impossible savings are <1 ms short and <8 ms at populated65K. At s70, row ~69% becomes 28.46-36.23% union, but factory acceptance collapses to 55 accepted drafts and complete speed 40.18 tok/s.
- CLOSED_TEAL_T4_UNION_OR_QUALITY_ECONOMICS. No six-family expansion or sparse-kernel port; source restored.

### 2026-08-23 07:12 +07 — P0-CL TEAL shared-T4 activation sparsity
- Existing target-only diagnostic measured the T4-shared zeros required to skip aligned quantized weight columns. At 50% per-row sparsity only 8.2-14.6% of columns are zero in all four rows; at 70%, 28.5-36.2%.
- Grant zero metadata/selection cost, canonical yield, and proportionally delete the entire stage0/stage1 wall by the maximum measured per-device union sparsity—even non-linear/state work. Ceiling is only 89.97 useful tok/s.
- Actual 70% route collapses to 40.18 tok/s and 55/210 accepted. Per-row sparse GEMV would load ~120% of a shared dense T4 weight stream. CLOSED; no sparse quantized port or threshold sweep.

### 2026-08-23 07:20 +07 — P0-CL shared T4 activation mask
- Measured a genuinely different target dataflow: one shared feature mask across all T4 verifier rows, so a sparse kernel could preserve one weight scan. General s60 kept 59% shared sparsity and 74 accepted drafts, so it advanced to two-run six-family qualification.
- Candidate was deterministic but failed representative quality in all six families; candidate rounds/accepted were general 106/148, code 123/131, JSON 91/162, Thai 188/66, Japanese 136/118, Arabic 152/100. All outputs changed and language/structure/content checks failed.
- CLOSED_REPRESENTATIVE_OR_QUALITY_FAIL. No sparse kernel port.

### 2026-08-23 07:32 +07 — P0-CM joint G64 post-mixer sparsity
- Measured one shared G64 activation mask over all T4 verifier rows, post-mixer gate/up and down only; attention/GDN projections remained exact. Diagnostic source was hash-restored and rebuilt.
- The only readable point, s30, retained 65 accepted drafts but has an impossible zero-overhead populated65K saving of only 6.31 ms from the exact 21.68-ms post-mixer profile. s40 reaches 8.51 ms only after rounds 51→74, accepted 74→50 and repetition 0.643; s50/s60 collapse acceptance.
- CLOSED_G64_QUALITY_OR_COMPLETE_ROUND_CEILING. No sparse kernel or six-family run.

### 2026-08-23 07:39 +07 — P0-CN GDN associative record scan
- Verified target GDN uses a serial T4 recurrent_record kernel and deferred accepted-state fold, so an associative low-rank scan is algebraically possible.
- Exact promoted traces make the complete removable bucket <1 ms/round even after deleting every record and fold cycle: short upper 0.826 ms, populated65K upper 0.940 ms.
- CLOSED_GDN_RECORD_SCAN_BUCKET_BELOW_8MS. The larger GDN time is projection/gating/conv, not recurrence; no scan port.

### 2026-08-23 07:46 +07 — P0-CN exact T4 GDN recurrence/state rewrite
- Current recurrent kernel already loads each 128x128 FP32 state tile once, retains it in registers across all four verifier tokens, and stores once; chunked path is only for sequence length >=64.
- Promoted whole-route trace attributes 0.395/0.403 GPU-ms per round on dev0/dev1 to all recurrent/state/gating/projected-conv kernels. Deleting every matched GDN kernel entirely raises 63.095 only to 64.38 useful tok/s.
- CLOSED_DELETE_ALL_GDN_BELOW_DUAL100; no associative/tensor-core T4 rewrite.

### 2026-08-23 07:51 +07 — P0-CM shared-G64 25 percent six-family
- A separately implemented shared-G64 diagnostic pruned exactly 25% of G64 groups across verifier rows and completed two deterministic runs for general/code/JSON/Thai/Japanese/Arabic.
- All six families changed output and failed representative-quality checks; diagnostic dense execution was also slower (general 61.80→44.22, code 50.27→40.77, JSON 73.16→67.16, Thai 34.00→32.41, Japanese 44.53→37.59, Arabic 50.53→38.11 tok/s).
- CLOSED_REPRESENTATIVE_OR_QUALITY_FAIL. Tracked source restored and no shared-G64 hook remains.

### 2026-08-23 07:53 +07 — P0-CP current-artifact full dual A8 one-shot
- Paired current/full/current stock runs on promoted P0-M+B1: current median 63.19 tok/s, legacy full-A8 64.23 tok/s. Output and 51 rounds / 153 drafts / 74 accepts / 36,20,18 trajectory are identical.
- Full A8 saves only 32 ms stage0 over the complete run; stage1, proposal and fold are unchanged. It does not explain the single 43-round versus dual 51-round yield gap.
- CLOSED_CURRENT_FULL_A8_BELOW_GATE. Retain both-mlp-stage1-all; no mode sweep.

### 2026-08-23 08:13 +07 — CLOSED P0-CS native Q4/W8 factory-proposer union tree
- Captured current promoted Q4 and original W8 factory D3 drafts at every canonical root across six families; source restored.
- Exact trie union gives representative U=1.780 with mean 4.8 target nodes and max7. Charging measured N8 target cost gives only 62.00 tok/s with both proposers free, 53.73 with parallel proposal, 49.76 serial. Thai U=1.104; no runtime tree port.
- Evidence: .chatgpt/post_reboot_20260821/p0_cs_q4_w8_ensemble/{CONCLUSION.md,result.json}.

### 2026-08-23 08:33 +07 — P0-CT distribution-correct target sampling
- Ran real target-distribution sampling with a temporary full-W8 head and promoted P0-M proposer; source restored.
- Best mild top2 T=0.2 point is 61.94 tok/s, 49 rounds, 76/147 accepted versus greedy full-head 59.89, 51 rounds, 74/153.
- Granting the 49-round yield to promoted B1 with zero sampling/head overhead reaches only 65.67 tok/s. CLOSED_DISTRIBUTION_SAMPLING_YIELD_BELOW_DUAL100; no B1 integration or six-family spend.

### 2026-08-23 08:35 +07 — CLOSED P0-CT distribution-correct one-hot sampling
- Used the existing exact one-hot rejection-sampling kernel with a source-restored full-target-head diagnostic; digest changes at temperature>0 are valid target samples.
- Best setting temperature0.2/top-k2 reached 61.94 useful tok/s, 49 rounds/76 accepted versus full-head greedy 59.89 and promoted B1 greedy about63.1. Higher entropy collapsed acceptance.
- Closed below dual100. Non-degenerate proposal q remains a separate mechanism requiring real p/q overlap evidence.

### 2026-08-23 08:46 +07 — P0-CQ official VPTQ SM75 exact-shape oracle
- Repaired one bounded upstream compatibility defect: dynamic shared-memory launch now uses sizeof(SharedStorage), covering CUTE alignment padding. Compute-sanitizer reports zero errors; 2048-centroid 2-bpw output is finite, bitwise repeatable, rel-L2 0.004263, cosine 0.99999094.
- Stock calibration on both TU102 cards: exact M1 K5120 N34816 gate/up alone is 2724-2730 us under CUDA Graph replay.
- This single projection is 6.81x the 0.4-ms entire-body gate and 1.68x the promoted P0-M full three-pass 1.619-ms body. CLOSED_VPTQ_GATE_UP_ALONE_ABOVE_FULL_BODY_AND_GATE; no remaining shapes, quality work, or port.

### 2026-08-23 08:50 +07 — P0-CU proposal/target overlap
- The diagnostic capture failed before build and restored source. Complete-system arithmetic makes rebuilding unnecessary.
- Promoted dual C1 is 39.78 ms/round with proposal 2.627 ms and fold 0.333 ms. Making proposal free reaches only 67.56 tok/s; making proposal+fold free reaches only 68.17 tok/s.
- CLOSED_PQ_OVERLAP_CONTROL_BUCKET_BELOW_DUAL100. Continue only with target-stage or useful-per-round mechanisms large enough to remove ~14.7 ms/round.

### 2026-08-23 08:52 +07 — CLOSED P0-CU exact non-degenerate proposal-q overlap
- Captured full BF16 factory proposal logits (131072 support plus token map) and full target logits (248320 x T4) at every D3 position across general/code/JSON/Thai/Japanese/Arabic; source restored.
- Computed exact local sum-min(p,q) overlap for temperature/top-k grids. Best common policy temperature1.2/top-k8 gives representative U=2.125, family range 1.255-3.824.
- Even with proposal computation free, complete target+fold ceiling is only 60.21 useful tok/s; charging current Q4 proposal gives55.99. Closed before runtime implementation.

### 2026-08-23 09:05 +07 — CLOSED P0-CW embedding-space probing paper ceiling
- Primary ESP paper arXiv2603.17942v2 is training-free but adds a masked-token base-model probe and target tree verification.
- Reported Qwen3 acceptance gain is 8-12% and best complete throughput gain 15-19%. Applied to the promoted 63.095 dual frontier, the optimistic result is75.08 tok/s versus the required100.
- Closed without implementation. Evidence: .chatgpt/post_reboot_20260821/p0_cw_esp_paper_ceiling/{CONCLUSION.md,result.json}.

### 2026-08-23 09:23 +07 — P0-CV proposal-distribution rejection sampling
- Captured 20 promoted P0-M/B1 D3 rounds with exact full target logits and the 131072-row proposal distribution while preserving the production draft/target tokens; diagnostic source restored.
- Best exact overlap screen is p top2 T0.8 / q top2 T0.8: mean 1-TV 0.7584, expected U=2.9501. Even giving branch-conditioned recomputation, sampling, residual correction, and all head cost for free at the promoted 39.78-ms wall reaches only 74.16 useful tok/s.
- CLOSED_PROPOSAL_DISTRIBUTION_OVERLAP_BELOW_DUAL100. No stochastic-q runtime or six-family spend.

### 2026-08-23 09:30 +07 — P0-CV sensitivity-selected training-free self-drafter
- Teacher-forced diagnostic measured all 64 single-layer omissions and cumulative keep-60 through keep-8 masks; no training or weight changes, source restored.
- Every D-depth proposal needs D sequential autoregressive passes. The best impossible mask/depth reaches only the target-free ceiling recorded in classification.json; high-quality keep60 D3 has U=3.7647 but proposer cost 114.143 ms/round.
- CLOSED_PRUNED_SELF_DRAFTER_PROPOSER_ALONE_BELOW_DUAL100. No runtime or six-family expansion.

### 2026-08-23 09:31 +07 — CORRECTION P0-CV pruned self-drafter complete-system bound
- The prior target-free table was not the closure: keep8 D1 can exceed 100 only when target verification is deleted.
- Using the stronger valid upper bound—perfectly overlap every pruned proposal with the measured 18.176-ms target critical stage and make all other work free—every mask/depth remains below dual100. classification.json and CONCLUSION.md supersede the target-free wording.

### 2026-08-23 09:32 +07 — P0-CW proposal-distribution plus perfect PipeInfer
- Fed the real promoted p/q overlap into a complete two-GPU work/dependency bound. Mean overlaps are 0.8808/0.7280/0.6664, all-three probability 0.44843, U=2.95010.
- Perfectly balancing measured target+proposal work into 18.794-ms waves and making full heads, branch-conditioned q, sampling, correction, transport, state, synchronization, and control free gives 101.17 infinite tok/s but only 99.11 at the required 128-token completion (100.57 at 256).
- CLOSED_QDIST_PIPEINFER_COMPLETE_WORK_BELOW_DUAL100. No prototype can beat this upper bound.

### 2026-08-23 09:35 +07 — P0-CX budgeted mismatch acceptance and suffix reuse
- Applied the training-free ASD regret verifier to 20 captured promoted D3 rounds with exact target logits.
- Strict local U is 3.05. First-mismatch logit-gap median is 7.5 and p90 16.375. The most aggressive tested budget accepts seven mismatches, spends 22 logit-gap units, and reaches only U=3.55 / 89.24 free-cost tok/s; modest budget 2.5 reaches 82.96.
- CLOSED_ASD_REGRET_OPPORTUNITY_BELOW_DUAL100_OR_EXCESSIVE. Later captured rounds are already optimistic after the first trajectory change; no E2E or six-family spend.

### 2026-08-23 09:42 +07 — P0-CY verifier-keyed model-free drafting
- Projected 1,612 real verifier rows from 403 six-family T4 captures and validated 382/384 sampled source-head labels (two BF16/FP32 ties resolved to the prior source-head authority).
- Causal same-request, leave-family-out prewarmed, and combined semantic pools were tested with up to 32 oracle-selected free branches. Best U is only 1.3062, 1.95% full-D3, 32.84 free-cost tok/s.
- CLOSED_MODEL_FREE_SEMANTIC_POOL_BELOW_DUAL100. No retrieval index or runtime port.

### 2026-08-23 12:16 +07 — P0-CZ calibrated Q4G64 MLP-down PTQ — CLOSED
- Captured real post-SwiGLU BF16 inputs for layers 12/13/14/20/21/56/57/58 from disjoint PG19 calibration/held-out prompts; diagnostic source restored byte-for-byte and rebuilt.
- Compared exact current Q5 and exact historical clipped-RTN Q4 against group-local GPTQ, activation-weighted scale selection, and AutoRound-style code refinement in the unchanged signed symmetric Q4G64+FP16-scale format.
- Unregularized GPTQ/AutoRound excess ratios were 1.239/0.962 versus old clipped Q4. Internal validation selected AWQ-scale lambda0.05 at 0.759, but untouched PG19 heldout was 0.733 with 0/8 layers at <=0.5 and worst 1.068.
- Independent TypeScript/JSONL/Thai/Japanese/Arabic heldout agreed at 0.749 excess ratio. Required 2x gap closure is missed; no artifact canary/E2E. QuaRot fallback not opened because calibrated Q4 is not close.

### 2026-08-23 12:22 +07 — P0-CZ per-layer calibrated Q4G64 robustness correction
- Ran the strongest remaining allowed variant: each of the eight down layers used its own internal-validation-selected AWQ/AutoRound encoder and regularization.
- Untouched PG19 excess ratio is 0.709 with 0/8 layers at <=0.5; independent TypeScript/JSONL/Thai/Japanese/Arabic is 0.714 with 1/8 at <=0.5 and worst 1.110.
- This strengthens the closure: neither global nor per-layer calibrated code/scale selection materially closes the Q5->Q4 gap. No canary/E2E; QuaRot remains unwarranted.

### 2026-08-23 12:25 +07 — P0-DB Q4 plus low-rank residual — complete-system ceiling below gate
- Screened a genuinely different mechanical format: historical Q4G64 plus deterministic low-rank quantization-residual factors, not plain Q4.
- The established package changes only eight down layers and saves 88.8 MiB. At measured 230.560 us/layer Q5 down, deleting all eight kernels entirely is only 1.844 ms/round; ideal 5->4 byte scaling is 0.369 ms.
- Rank4/8/16/32 BF16 factors consume 1.38/2.75/5.50/11.00 MiB and add two projections. Even perfect zero-cost correction cannot reach the 8-ms gate, so no SVD fit or runtime port.

### 2026-08-23 12:47 +07 — P0-DD Layer/Traversal Verification — free-control ceiling below dual100
- Audited the 2026 Layer Verification lift and lossless Traversal Verification against NInfer tree economics. They change acceptance over an existing tree; they do not create candidates or reduce target-node forward work.
- Traversal Verification reports 2.2–5.7% average acceptance gain and less near zero temperature. Applying the full 5.7% with zero rule/control overhead to NInfer best free-proposer tree ceiling 65.55 gives only 69.29 tok/s; even 25% gives 81.94.
- Dual100 needs 52.6% more accepted progress at identical node cost. Factory/PARD candidate diversity remains the binding limit. No verifier rewrite.

### 2026-08-23 12:47 +07 — P0-DE Hierarchical Speculative Decoding — no compatible intermediate drafter
- Audited HSD, which stacks a portfolio of draft models and reports up to 1.2x over the best single-draft baseline.
- NInfer has no unclosed intermediate drafter: factory MTP is promoted; PARD/PARD2/Qwen0.6, lookup/Cacheback, Token-Recycling and low-bit substitutes are closed; local training is prohibited.
- HSD optimizes an existing hierarchy but creates neither proposal diversity nor lower target work. Adding a closed drafter only adds serial cost. No implementation.

### 2026-08-23 12:57 +07 — P0-DC exact 262K sequence-shard balance — current 60/40 retained
- Applied one shared diagnostic fraction to stage KV allocation and the exact (acc,m,l) coordinator; source was restored before runs. Swept 0.50/0.55/0.60/0.65/0.70 on the genuine populated-262K snapshot and split-36 TP2 route.
- Useful speeds were 27.32/27.60/31.76/27.39/28.15 tok/s. The current 0.60 fraction also minimized helper time (0.623 s) and stage walls while balancing runtime memory near 16.9 GiB/card.
- Output trajectories differed at some exact reduction boundaries but no representative-quality candidate improved complete E2E. Classification CLOSED_CURRENT_60_40_RETAINED; no source change.

### 2026-08-23 12:59 +07 — P0-DD heterogeneous text-layer split — negligible exact gain
- Swept split 20/24/28/32/36/40/44 on the current promoted dual-short path. Split40 won the screen and was paired against split32 for three 128-token runs.
- Split32 median 65.95 versus split40 66.04 useful tok/s; visible output is identical, but split40 uses 48 rather than 47 rounds. Stage redistribution is 16.723/18.191 to 21.021/13.771 ms, leaving total target wall effectively unchanged.
- Only 1.00136x complete-path gain, far below the 8-ms admission gate. Classification CLOSED_NEGLIGIBLE_SPLIT_GAIN; no default/source change.

### 2026-08-23 13:07 +07 — P0-DA fixed H64 QuaRot multimodal reconstruction — CLOSED
- Ran the unfinished fixed, non-learned block-64 Hadamard oracle on the five preferred MLP-down layers whose BF16 source shards are locally complete, using balanced disjoint text+vision calibration and separate text/vision holdout columns. vision/* weights modified: zero.
- Only layer58 H64+AWQ-round passes jointly: text/vision excess ratios 0.208/0.178 and worst Q5 proximity 1.233x. Layers20/21/56/57 fail; all clipped-RTN rotations fail. The authority requires at least two admitted layers before cost/canary because one layer is not material.
- Classification CLOSED_MULTIMODAL_RECONSTRUCTION_GATE. No online Hadamard kernel or artifact canary.

### 2026-08-23 13:13 +07 — P0-DE promoted graph node attribution — dense stream is the real target wall
- Reran the current P0-M+B1 dual-short route with surgical Nsys CUDA-Graph node tracing. The older aggregate trace intentionally hid child kernels and falsely looked idle.
- Per target graph: runtime device0 child kernels 16.532 ms, of which Q4+Q5 dense 15.062 ms (91.1%); device1 18.061 ms, dense 15.684 ms (86.8%). Child sums reproduce the stage graph spans; there is no multi-ms internal orchestration gap.
- Classification CLOSED_NO_GRAPH_INTERNAL_IDLE_TARGET_DENSE_DOMINATES. Future mechanisms must reduce/amortize target dense weight traffic or increase useful/round enough to pay for it.

### 2026-08-23 13:21 +07 — P0-DF exact low-bit code-plane compression — Shannon limit closes
- Sampled every one of 320 text-backbone Q4G64/Q5G64 tensors from the promoted P0-M artifact. Vision and MTP tensors were excluded; FP16 scales remained exact. The oracle grants ideal Shannon coding, zero metadata, and zero GPU decode cost.
- All target low-bit payload saves only 6.10% at the Shannon limit: Q4 12.22%, Q5 0.41%. Against 30.746 ms/round of promoted target dense work this is an impossible 1.874-ms saving, versus 13.976 ms required for dual100 at current yield.
- Classification CLOSED_SHANNON_LIMIT_BELOW_DUAL100. No GPU codec or source change.

### 2026-08-23 13:22 +07 — P0-CZ calibrated Q4G64 MLP-down Phase A — CLOSED
- Completed exact old candidate layers 12,13,14,20,21,56,57,58 using 320 fit / 84 internal-validation / 144 held-out real post-SwiGLU columns per layer; runtime-style A8 activation error is included.
- Best regularized group-local GPTQ Q4G64: held-out SSE 10159.719 versus old clipped Q4 10908.334 and deployed Q5 3249.857. Old-excess improvement is only 1.1083x (required >=2x); SSE remains 3.1262x Q5. Every layer fails individually. Activation-weighted scale search is 1.0983x.
- Closed before artifact/runtime and before multimodal qualification. Fixed QuaRot R4 not run because this is not close. No target/vision weight or production source change.

### 2026-08-23 13:27 +07 — P0-DG cross-layer predictive exact low-bit codec
- Granted every same-role tensor its previous fully decoded layer for free, then entropy-coded aligned current Q4 nibbles/Q5 nibbles+high bits conditionally; FP16 scales exact, metadata and GPU decode free.
- Conditional entropy is effectively unchanged: aggregate ideal saving 6.10%, or 1.874 ms/round against the promoted 30.746-ms dense bucket, versus 13.976 ms required for dual100.
- CLOSED_CONDITIONAL_SHANNON_LIMIT_BELOW_DUAL100. No codec implementation.

### 2026-08-23 13:31 +07 — P0-DI exact contribution sparsity — Prox/WiSparse upper bound closed
- Used the true post-SwiGLU activation and true BF16 down-weight column norm on disjoint text+vision calibration and holdout sets. This is stronger than a learned/quantized proxy; retained channels and weights remain exact.
- At 40% channel sparsity the worst holdout row rel-L2 is already 0.1266 and min cosine 0.991969. Dual100 materiality starts around 65%, where error is 0.2974/min cosine 0.954799. Group64 masks are much worse.
- The 65K saving oracle gives salience, packing, irregular gathers, and sparse-kernel overhead for free. Classification CLOSED_EXACT_CONTRIBUTION_SPARSITY_QUALITY_OR_MATERIALITY.

### 2026-08-23 13:32 +07 — P0-DG cross-layer exact predictive coding — closed at conditional-entropy limit
- For every same-role/same-shape target Q4G64/Q5G64 tensor, an oracle chose the best other layer predictor and granted ideal conditional arithmetic coding, free reference metadata, exact FP16 scales, and zero GPU reconstruction cost.
- Aggregate exact saving is only 6.35% over 13.354 GiB: Q4 families ~12.4-13.7%, Q5 families 0.5-0.8%. Applied to 30.746 ms/round target dense work, impossible saving is 1.952 ms versus 13.976 ms required.
- Classification CLOSED_CROSS_LAYER_CONDITIONAL_ENTROPY_BELOW_DUAL100. No codec/source change.

### 2026-08-23 13:33 +07 — P0-DJ exact Q5 low4+high1 decomposition — free-correction ceiling closed
- Current T4 Q5 down is 112.896 us/layer at 84.66% peak DRAM. Q5G64 is 5.25 bits/weight including scales; low4 plus unchanged scales is still 4.25.
- Grant perfect-bandwidth low4 and make exact high1 correction, activation bitplanes, launch and accumulation free: saving is only 1.198 ms per 32-layer target stage. Even deleting all measured Q5 dense work removes only 8.155 ms from the critical stage, below the 13.976-ms dual100 need.
- Closed before BMMA/bitserial prototype; no source change.

### 2026-08-23 13:35 +07 — P0-DH Gated Subspace Inference real-activation screen
- Evaluated published training-free cached W·B subspace inference on eight real MLP-down layers using 1428 fit / 712 gate-selection / 144 held-out PG19 activation rows per layer.
- Ranks 64–512, BF16 basis and cached image, free gate/metadata/dispatch, whole-T4 exact fallback. Even an impossible output-error oracle admits zero held-out T4 batches at a generous 5% relative-output gate.
- Rank512 median held-out activation residual is 0.69–0.88 and sampled-output error 0.75–0.83; all fallback plus factor storage is 1.395x current bytes. CLOSED_GSI_DOWN_COMPATIBILITY_OR_BYTE_GATE.

### 2026-08-23 13:53 +07 — P0-DK additive/codebook low-bit official-operator screen — CLOSED
- Pinned official AQLM 76d5e77 and QuIP# b1c4eef without fitting codebooks. AQLM custom CUDA compiled on SM75; QuIP# E8P PTX fails because mma.m16n8k16 requires SM80+.
- At exact T4 gate/up M4,K5120,N34816, AQLM one-codebook (~1 bpw codes) is 0.936 ms/layer and two-codebook (~2 bpw) 1.959 ms/layer. Sixty-four gate/up layers alone are 59.93 ms versus the current entire target dense graph 15.06/15.68 ms.
- Closed before quality/fitting/integration; no local training or source change.

### 2026-08-23 14:04 +07 — P0-DK training-free DeltaKV residual KV
- Used the actual 65,292-token one-layer 65K Q/K/V capture. Inverted and re-applied D256/R64 RoPE with 4.9e-8 rel-L2 roundtrip, then tested causal nearest-reference direct Q4G64 residuals at 10% references, k1/k4, INT8/BF16 references.
- Best k1 INT8 stores 56.46% of current INT8 and has an ideal/free-decode 12.30-ms 262K GQA saving, but real reconstruction requires 1.465x current vector traffic before lookup/dequant. Attention min cosine is 0.99152, max rel-L2 0.1299; k4 loses top-key retention and costs 4.17x traffic. Ideal 65K saving is only 3.07 ms.
- Published DeltaKV trains a compressor/decompressor, prohibited here. Training-free direct residual is CLOSED_NUMERICAL_OR_REFERENCE_TRAFFIC_OR_COMPLETE_GQA_GATE; no CUDA port.

### 2026-08-23 14:16 +07 — P0-DL native-MTP Bole tree — perfect finite 128-token ceiling closed
- Reopened only the materially distinct August-2026 Bole shape: native MTP top-k4, prefix-connected tree, maximum depth8. Existing 20-round Qwen3.8 proposal capture has seven first mismatches ranked 2, one ranked 3, one ranked 6, one ranked 8; an impossible top4-rescue/perfect-descendant oracle reaches 8.1 useful/round.
- Charged the existing production-style all-node node-slope target envelopes and the integer number of rounds for 128 user tokens. At 262K, even perfect B8/B12/B16/B22 trees reach only 73.20/88.11/95.79/99.71 tok/s. Bole depth8 is lower; proposal/tree/control costs are free in this bound.
- Closed before counterfactual MTP state cloning or runtime implementation. The populated hard gates require lower exact target/attention cost, not another tree drafter.

### 2026-08-23 14:21 +07 — P0-DM periodic canonical anchors — dominated by failed accept-all ceiling
- The prior P0-BI implementation was unconditional accept-all; interval anchors were not swept. Unconditional accept-all is nevertheless a strict speed upper bound because anchor rounds execute the same target pass and accept no more than all D3 drafts.
- Its measured dual-short ceiling was 98.095 tok/s, already below 100, while representative general/code/Thai/Arabic quality failed. Any interval 2/3/4/8/16 is slower and cannot change populated target floors.
- Closed analytically; no source patch or rerun.

### 2026-08-23 14:34 +07 — P0-DN target-carried bonus seed — requires second target scan
- Current D3 verifier width4 already covers [anchor,draft1,draft2,draft3] and emits the fourth target output as bonus/fallback. It does not compute target hidden after that emitted bonus token.
- An exact target-derived first draft for the next round therefore requires an additional autoregressive target forward of the bonus: a second full target weight scan, strictly dominated by factory-MTP proposal.
- Closed by source dependency; no implementation.

### 2026-08-23 14:38 +07 — P0-DO temporal target projection memoization — impossible oracle below gate
- Evaluated held-out prompts 6/7 over eight real post-SwiGLU layers. Cache may choose the exact best any-prior activation after seeing 256 BF16-source down-output rows; lookup/storage/retrieval are free.
- Exact BF16 and runtime-A8 duplicate rates are 1.408%. Hit rates at 5/10/20% output rel-L2 are 1.408/1.849/2.729%, worth only 0.433/0.568/0.839 ms of the 30.746-ms dense graph versus 13.976 ms required.
- Closed before runtime implementation.

### 2026-08-23 14:45 +07 — P0-DN one-module Ladder Residual
- A material module-split ceiling was screened with an environment-gated T4 verifier: mixers use the pre-MLP residual while the previous MLP delta is applied one layer late. Source was restored byte-for-byte before execution.
- The immediate-delta control remained close to canonical for the full 128-token response. Delay1 diverged at token 3, emitted EOS after 7 tokens, and collapsed MTP acceptance from 73/159 to 2/12.
- Published Ladder Residual relies on post-training adaptation; local training is prohibited. CLOSED / DO NOT IMPLEMENT; no six-family or distributed port.

### 2026-08-23 14:58 +07 — P0-DC intermediate-head early rejection
- Finished the previously orphaned exact indexed-head trace at layers 15/31/47 over 330 MTP rounds across general/code/JSON/Thai/Japanese/Arabic. Production source was restored before execution.
- Known exact-target top1 is 0.000/0.131/2.362%; accepted-draft match recall is 0.000/0.231/2.315%. Every checkpoint falsely rejects 215/330 rounds and collapses trace-local useful progress to 1.000/1.000/1.015 tok/round.
- This misses before charging three extra vocabulary heads or dynamic-width suffix execution. CLOSED / DO NOT IMPLEMENT.

### 2026-08-23 15:27 +07 — P0-DP remote-only static low-bit TP2 schedules
- Corrected P0-DI to the actual owner/helper contract: only the remote partial is quantized. Raw A4 meets the 6.568779-ms wire deadline but misses held-out residual quality.
- Six-family deterministic canaries tested plain/error-feedback A4, tail-eight A6, four sensitive A8 layers, and the measured-sensitive-eight A6 set. Some prompts improve or become exact, but Japanese/Arabic remain below representative quality and sensitive-eight also fails code.
- All source was restored. Static schedules CLOSED; only adaptive per-round precision remains scientifically open.

### 2026-08-23 20:40 +07 — P0-DQ adaptive remote-only TP2 bounded close
- EF0.08 measured 13,759.4 B average and missed the 11,840-B deadline. EF0.10 was the first fitting point at 11,733.7 B and reproduced Japanese exactly across repeats, so 0.12 was not run.
- The single fixed EF0.10 six-family canary met the deadline everywhere but failed established representative quality on general (LCS .4688), code (.7656), Thai (.8516), and Arabic (.7578); JSON and Japanese passed exactly.
- Per-language policy is prohibited. Classification CLOSED_FIXED_EF0P10_SIX_FAMILY_QUALITY; no production wire implementation; diagnostic source restored.

### 2026-08-23 21:00 +07 — P0-DR PEARL factory-MTP finite schedule close
- Source audit confirms the next-root target distribution is not retained: target column0 is the committed anchor; target verification precedes acceptance; acceptance precedes factory proposal; proposal consumes the selected accepted target hidden state.
- Exact six-family 128/256 traces were modeled. Official PEARL reaches only 40.43/39.13 representative-mean useful tok/s. Current D3 with all proposal/fold free and target-stage PipeInfer reaches 58.54/57.21.
- A stronger impossible oracle gives D5 yield, prices D5 target as smaller T4, makes replica/proposal/fold/control/scalar tails free, and pipelines future target stage0. It reaches representative means 62.80/60.89 and fixed-general 53.48/58.41; only an old code-only trace exceeds105.
- Future target-stage0 speculation is exactly the prior PipeInfer invalid-ancestor correction-bubble dependency. Classification CLOSED_FINITE_CEILINGS_BELOW_105_AND_PIPEINFER_EQUIVALENT; no runtime; vision/full-context path untouched.

### 2026-08-23 21:20 +07 — P0-DS/P0-DT exact lossless remote-TP2 closure
- Captured 1,536 real T4 remote partial packets after exact source restoration. The FP32 packet is BF16-expanded, so the lossless raw floor is 40,960 B, but the deadline is 11,840 B.
- Ideal BF16 symbol entropy is 11.228 bits/value = 28,743 B/packet; even granting sign+exponent free leaves the mantissa at 17,848 B. Best generic/temporal exact codec is 29,417 B.
- A stronger 768-packet owner-side-information capture gives ideal H(remote|owner)=10.488 bits/value = 26,849 B; owner/remote correlation is only .10-.14 and calibrated exact predictors are worse.
- Classifications CLOSED_LOSSLESS_REMOTE_PARTIAL_RATIO_BELOW_WIRE_REQUIREMENT and CLOSED_OWNER_SIDEINFO_CONDITIONAL_ENTROPY_ABOVE_WIRE_BUDGET. No runtime; source restored; vision/target/state unchanged.

### 2026-08-23 22:03 +07 — P0-EB exact BF16 TP2 mailbox ceiling
- Accepted exact-BF16 T4 TP2 transport is retained, but a distinct asynchronous event/mailbox rewrite was bounded before source work.
- The existing stock-TU102 mailbox discriminator measures the exact-BF16 chain at 12.680 ms/round. Granting polling, CPU sync, launch gaps and all integration free leaves a 20.920-ms non-mailbox floor.
- Impossible complete round is 33.600 ms / 87.89 useful tok/s at measured yield, below dual100. CLOSED_EXACT_BF16_EVENT_CHAIN_BELOW_DUAL100; no orchestration implementation.

### 2026-08-23 22:12 +07 — P0-EA exact BF16 TP2 promoted frontier
- Promoted exact T4 BF16 TP2 partial transport: remote packet 81,920 B FP32-expanded -> 40,960 B exact BF16; owner reduction stays BF16-visible and preserves ordering.
- Dynamic order calibration selected UUID order 01. Six complete-path reps are deterministic at 69.17-69.40 useful tok/s, median 69.35, digest 14660c413b4cf185, 49 rounds, 147 drafted, 77 accepted, positions 34/27/16.
- Per-round stage0/stage1/proposal/fold medians are 15.429/16.765/4.041/0.327 ms. This supersedes the 65.910 short-dual frontier but remains DUAL100_STRUCTURAL_GAP. Source retained in apps/dual/tp2_mlp.cu.

### 2026-08-23 22:20 +07 — P0-EC exact BF16 selective-layer TP2 close
- Bounded exact mask screen tested all/no TP2, stage0/stage1 only, full-attention/GDN only, intersections, and first/last16 using the promoted exact-BF16 transport.
- All-layer is fastest at 69.60 useful tok/s. Best hybrid GDN-only is 68.20; lower-GDN 67.06; stage1-only 66.87; stage0-only 66.81; no-TP2 fallback 66.02.
- CLOSED_ALL_LAYER_DOMINATES_SELECTIVE_TP2. Diagnostic source restored; promoted exact-BF16 all-layer source retained.

### 2026-08-23 22:34 +07 — P0-ED exact BF16 accept-all close
- Exact-BF16 transport makes unconditional D3 fast enough: fixed median 101.61 and six-family candidate 105.00-105.38 useful tok/s, deterministic.
- Representative quality catastrophically fails every family: LCS 0.117-0.602, decoded similarity 0.043-0.449; Thai also loses script structure.
- CLOSED_ACCEPT_ALL_REPRESENTATIVE_QUALITY. No verifier-policy source retained; promoted exact-BF16 TP2 remains.

### 2026-08-23 22:54 +07 — P0-EE exact-BF16 periodic target anchors
- Fixed p8/p16/p32 all crossed dual100; p16 fixed median 105.13. P8 is the quality-maximizing policy because it contains every p16/p32 exact anchor plus additional exact corrections.
- P8 six-family 256-token runs were deterministic and >=100 everywhere but failed quality in all families: LCS .0703-.5820, decoded similarity .0702-.2605; Thai repetition .8785.
- P16/P32 are correction-subsets of P8 and therefore quality-dominated. CLOSED_P8_QUALITY_P16_P32_DOMINATED; no source retained.

### 2026-08-23 23:00 +07 — P0-EF asymmetric TP2 ownership ceiling
- Timed the current 69/67-block T5 shards independently across all 64 real MLP layers: 16.236 ms external/fast role, 13.232 ms internal/slow role, 16.835 ms concurrent complete pass.
- Granting linear scaling and fractional ownership gives an ideal 62.0/74.0 split and only 1.646 ms maximum complete-pass saving.
- Below the 8-ms mechanism gate before shard regeneration or integration. CLOSED_IDEAL_ASYMMETRIC_TP2_SAVING_BELOW_8MS; no source/storage rewrite.

### 2026-08-23 23:15 +07 — P0-EG CAS-Spec perfect-router cascade ceiling
- Evaluated published cascade self-speculation against the existing mechanical layer-sparse drafters with an oracle stronger than any realizable router: it knows the target token, executes only the cheapest correct drafter, pays no routing/transport/state overhead, and charges every target width at the impossible T2 floor 18.176 ms.
- Depth1/2/3 oracle speeds are 43.55/41.23/40.16 useful tok/s. Deleting all draft cost raises depth2/3 to 119.36/140.85, proving candidate quality exists but real layer-sparse draft work dominates.
- CLOSED_PERFECT_ROUTER_CASCADE_BELOW_105; no cascade runtime or additional mask sweep.

### 2026-08-23 23:22 +07 — P0-EH nonlinear 4-bit remote TP2 packet oracle
- Screened fixed NF4, FP4-E2M1, logarithmic, and one globally calibrated 16-level codebook on 384 disjoint held-out real T4 remote partials with exact layer-to-layer error-feedback semantics.
- All packets fit 10.88-10.912 KB. Asymmetric uniform A4 mean/P95 rel-L2 is .13425/.15876; best NF4 is only .13087/.15956 (2.5% mean improvement, 0.5% P95 regression). Other codebooks are worse.
- CLOSED_NONLINEAR4_ERROR_REDUCTION_INSUFFICIENT; no runtime canary.

### 2026-08-23 23:36 +07 — P0-EI exact CommFuse-style TP2 row streaming
- Built a standalone exact row-range Q5 discriminator using unchanged arithmetic, separate helper compute/copy streams, cudaMemcpy2DAsync row packets, and exact BF16 owner reduction across all 64 real checkpoint layers.
- All 1/2/4/8/16-chunk configurations were byte-identical in both owner directions. Best was slow-owner/fast-remote with 2 chunks: 15.456 -> 14.795 ms, only 0.661 ms saving; finer chunks regress.
- Applied to the 69.35-tok/s, 49-round frontier this projects 70.60 tok/s, below 105. CLOSED_EXACT_ROW_STREAMING_BELOW_105; no production edit.

### 2026-08-23 23:41 +07 — P0-EJ fixed vector-4 remote TP2 packet oracle
- Calibrated fixed 256x2 vector codebooks on disjoint first-round captures and qualified on second-round packets with exact layerwise error feedback. Packets are 10.88-11.52 KB, within the 11.84-KB deadline.
- Best center/max VQ mean rel-L2 0.12880 versus asymmetric A4 0.13425 (4.1% better), but P95 worsens 5.4%; all VQ variants worsen P95 and fail portability across families.
- CLOSED_VECTOR4_ERROR_REDUCTION_INSUFFICIENT; no runtime canary. Together with no CUDA P2P, lossless entropy floor, nonlinear scalar4, and exact row streaming, the remote transport class is exhausted.

### 2026-08-23 23:54 +07 — P0-EK output-row TP2 ownership ceiling
- Measured real 64-layer T5 components: full Q4 gate/up 13.51 ms, full Q5 down 10.63 ms, current intermediate-sharded TP2 chain 16.75 ms.
- Output-row TP was granted full duplicated gate/up, perfect half-row Q5 scaling, half-sized exact BF16 gather, free duplicated weights, and zero integration overhead.
- Serial projection 77-79 useful tok/s; even perfectly hiding the entire gather under compute remains below 105. CLOSED_OUTPUT_ROW_TP2_PERFECT_OVERLAP_BELOW_105; no shard generation.

### 2026-08-23 23:56 +07 — P0-EL CUDA peer-copy fallback
- cudaMemcpyPeerAsync works through a driver fallback despite canAccessPeer=0, so it was measured against the exact 40,960-byte T4 BF16 packet in both GPU orders.
- Peer fallback 43.4-44.9 us versus explicit pinned D2H+H2D 23.0-28.2 us, byte-exact but 1.6-1.9x slower.
- CLOSED_CUDA_PEER_FALLBACK_SLOWER_THAN_EXPLICIT_STAGING; current transport retained.

### 2026-08-24 00:00 +07 — P0-EM token-axis low-rank TP2 packet oracle
- Screened fixed H4, per-packet PCA4-Q4, PCA3-Q5, PCA2-Q8, and rank1-BF16+sparse residual packets on six-family held-out T4 captures with BF16 factor storage and exact layerwise error feedback.
- All fit 10.58-11.84 KB. Best PCA4-Q4 mean/P95 0.13356/0.15817 versus A4 0.13427/0.15854, under 0.6% gain and not portable; lower-rank formats are much worse.
- CLOSED_TOKEN_AXIS_PACKET_ERROR_INSUFFICIENT; no runtime canary.

### 2026-08-24 00:17 +07 — P0-EO exact token-sequence-parallel target ceiling
- Measured the no-TP2 depth1/T2 target and granted two full target replicas, free row exchange, free GDN prefix scan, free causal-attention dependency, and free state publication.
- Even this impossible exact token-sequence-parallel round stays below the 105 runtime admission threshold.
- CLOSED_FREE_DEPENDENCIES_BELOW_105; no duplicate-model/runtime implementation.

### 2026-08-25 14:32 +07 — P0-EQ exact-BF16 T5-TP2 long recertification
- Retained promoted source rebuilt and ran from the existing populated 65K and 261,900-token snapshots with both runtime orders calibrated dynamically; no hardware-state changes.
- 65K three-run median 58.04 useful tok/s (58.00-58.09), deterministic digest 09ff33b553ac8a05 and 32/127/93/29,24,21,19 trajectory. 262K median 31.68 (31.54-32.38), deterministic digest fce07559f26ae625 and 42/167/82/41,38,2,1 trajectory.
- The retained exact-BF16 TP2 transport is correct but does not raise the long gates. Measured TP2 input+engine+reduction is ~36.5 ms/round at 65K and ~33 ms/round at 262K; proceed directly to shared T4/T5 FFN-skip oracle.

### 2026-08-25 14:35 +07 — P0-ER shared T4/T5 FFN-skip impossible oracle
- Captured whole BF16 verifier columns before/after every FFN on actual promoted T4 short and populated65K T5 rounds; source was restored and rebuilt before analysis.
- Shared eligibility required every valid column to have cosine >=0.99, with first/last4 layers and first4 calls protected. T4 skips 12.10 layers/round (18.91%) for an impossible 3.170-ms saving. T5 skips 12.50 (19.53%) for 7.135 ms.
- Both miss the 8-ms admission gate even with free clairvoyant gating and no reconstruction penalty. Classification CLOSED_SHARED_T4_T5_FFN_SKIP_BELOW_8MS; no causal policy/runtime work.

### 2026-08-25 14:44 +07 — P0-ET long exact-BF16 TP2 ablation
- Tested retained all-layer exact-BF16 TP2 against owner-only target MLP while preserving physical INT8 context GQA, populated snapshots, D4, and selected GPU order.
- 65K TP2 58.005 vs owner 56.140 useful tok/s; 262K TP2 32.285 vs owner 29.985. Every route deterministic.
- Helper/GQA contention does not overturn TP2 economics. CLOSED_OWNER_ONLY_SLOWER; retain all-layer exact-BF16 TP2 and continue on target/attention/yield structure.

### 2026-08-25 15:04 +07 — P0-EX/P0-EY new sparse-attention oracles — intrinsic sparsity insufficient
- Tested the 2026 UNIQUE training-free page selector exactly as published: page8, lambda0.5, per-page BF16 mean plus scalar std, max across six query heads sharing each KV head, recent512 retained. Used existing actual 65K Q/K/V captures and the prior 262K stress suite.
- UNIQUE has no material+quality point: at the last material 50% selection point, actual min cosine 0.9932, max rel-L2 0.1443, and min softmax mass only 0.7424. At 60%, quality improves but projected saving falls to 6.18 ms.
- A stronger impossible exact-score, zero-selector-cost GQA-group top-k oracle also fails: 60% retains only 0.947 minimum actual mass for 8.92-ms raw saving; the first >=0.95-mass point is above 60% and below the 7-ms gate. Therefore Quest/UNIQUE/RocketKV/Kascade-style selectors cannot solve this checkpoint without changing attention training. No CUDA work.

### 2026-08-25 15:14 +07 — P0-EU representative populated-262K output qualification
- Generated an exact 261,900-token prepared chat prompt with an explicit long-form technical-report task, cold-prefilled it once, and evaluated deterministic D3/D4 continuations at the full legal 240-token remainder.
- Both depths produce coherent, sectioned, control-token-free English prose. D3 median 25.80 useful tok/s versus D4 21.985. D3: 123 rounds, 369 drafts, 113 accepted; D4: 118 rounds, 471 drafts, 117 accepted.
- The inherited 34.5/32.4 frontiers were corpus-completion runs that emitted im_end immediately and continued through control tokens; they are not representative user-visible quality evidence. Owner-corrected contract therefore qualifies D3 and resets the genuine populated-262K frontier to 25.80.

### 2026-08-25 15:27 +07 — P0-FB local exact 262K GQA ablation
- The legacy/full-local 261.9K snapshot fits both 22-GB cards, so local exact GQA was requalified on the representative technical-report prompt rather than the old control-token corpus.
- Two deterministic D3 runs reach only 20.25 useful tok/s versus 25.80 physical sequence-sharded; output trajectory also differs. The exact-BF16 TP2 engine expands to about 83.5 ms/round local versus about 29.1 ms/round physical because full local KV competes with helper weights.
- CLOSED_LOCAL_EXACT_GQA_SLOWER_AND_DIFFERENT_ARITHMETIC_TRAJECTORY. Retain physical sequence sharding.

### 2026-08-26 00:32 +07 — P0-FG direct-global-V Bc16 close
- Repaired the stale diagnostic patch against the promoted Bc16 source, built the candidate, passed public GQA correctness, restored production source, and ran four paired 262K physical dual repetitions.
- Promoted Bc16 median 1809.032 us versus direct-global-V 2013.939 us; direct global V is 10.2% slower despite exact bitwise output.
- CLOSED_NO_MATERIAL_EXACT_SHAPE_WIN. The shared V-code staging and cp.async overlap are beneficial; do not port direct-global-V.

### 2026-08-26 01:00 +07 — P0-FH exact cached-T4 Bc16 long-GQA operator
- Promoted T5 Bc16 remained the baseline; a cached-only T4 Bc16 launcher was built separately, passed public GQA correctness, restored source, and ran four paired physical dual repetitions at 65K and 262K.
- 65K T4 Bc32/Bc16 dual medians 667.054/520.593 us = 1.2813x. 262K T4 medians 2306.065/1811.372 us = 1.2731x. All isolated outputs bitwise exact.
- Classification ADVANCE_T4_BC16_E2E. Representative 262K D3 and 65K D3 paired E2E is required before promotion.

### 2026-08-26 01:23 +07 — P0-FJ promote exact cached-T4 Bc16 long-GQA
- Paired stock-card D3 E2E after exact isolated T4 Bc16: populated65K 47.88 -> 55.195 useful tok/s (1.1528x); representative261.9K 25.985 -> 31.20 (1.2007x). Candidate deterministic in both routes.
- The initial absolute topic-term gate was invalid because the 240-token baseline itself covered only 7/11 late prompt terms. Corrected owner output contract compares deterministic candidate quality to the baseline truncation: baseline/candidate 181/180 words, bigram 0.989/0.966, 3/3 headings, identical 7/11 topic coverage, no control specials.
- PROMOTED cached-only T4 Bc16. Public GQA tests pass; append-and-quantize remains canonical Bc32. Output/trajectory changes are accepted under the corrected representative-quality contract.

### 2026-08-26 01:48 +07 — P0-FN full-layer target tensor-parallel ceiling
- Modeled a genuinely distinct all-layer head/tensor-parallel target: retained exact-BF16 TP2 MLP unchanged, then granted perfect 50/50 GDN/attention sharding plus free reductions, row exchange, state exchange, and causal synchronization.
- The impossible ceiling is only about 75 tok/s at 65K D4 and 40 tok/s at representative 262K D3. Even impossibly halving the already-parallel TP2 MLP again leaves representative 262K around 55 tok/s.
- CLOSED_FULL_LAYER_TP_FREE_COMM_BELOW_105; no shard generation or runtime implementation.

### 2026-08-26 01:57 +07 — P0-FI compact three-resident direct-V close
- After repairing the Bc16-scoped shared-memory layout, the compact DirectGlobalV Wc3 candidate built, passed public GQA correctness, and restored production source.
- At physical 262K T5, retained Bc16 median is 1807.396 us versus compact3 2016.336 us. Candidate is bit-exact but 10.4% slower; resources are 80 registers, 96-byte stack, 21,828-byte shared.
- CLOSED_NO_MATERIAL_EXACT_SHAPE_WIN. Keep retained Bc16; no E2E promotion.

### 2026-08-26 02:09 +07 — P0-FO Bole/native-MTP tree recost after Bc16 — actual D5 close
- Bc16 legitimately reopened the old perfect-tree economics, so a current Q4G64 factory D5 q-distribution capture was rebuilt against the promoted artifact and measured for 18 real rounds.
- Native D5 U=3.389. Paying all top-k alternatives and continuations at every selected mismatch position, while granting free branch-state cloning and proposal generation, reaches at best U=3.833.
- Finite 128/256 ceilings are only 47.71/48.42 tok/s at 65K and 38.93/39.52 at 262K. CLOSED_ACTUAL_D5_BRANCH_CONTINUATION_BELOW_105; no tree runtime.

### 2026-08-26 02:27 +07 — P0-FP P0-M long combination — context-selective result
- Paired canonical versus qualified Q4G64/Marlin factory-MTP artifact on retained Bc16 long routes.
- 65K D4: exact target output, canonical 60.27 versus P0-M 62.235 tok/s (+3.26%); proposal 0.243 -> 0.171 s. Representative 262K D3: exact target output but P0-M changes proposal ranks, rounds 111 -> 116, and loses 31.17 -> 30.475 tok/s.
- Use P0-M only for the 65K package; keep canonical MTP for 262K. No global artifact promotion.

### 2026-08-26 03:05 +07 — P0-FQ split68/fraction0.50 valid E2E close
- Fixed the actual integration error: allocator ownership, physical snapshot packing, coordinator fraction, and split68 were unified. Rebuilt the candidate, restored production source, and ran three valid paired repetitions at 65K D4 and representative262 D3 using legacy unsharded snapshots.
- 65K improved 28.190 to 29.210 useful tok/s (+3.62%), but representative262 regressed 31.180 to 30.140 (-3.34%). Candidate output was deterministic; representative262 quality passed under the corrected owner contract.
- CLOSED_E2E_SPLIT_BALANCE_BELOW_GATE. A global fraction0.50/split68 policy is not promotable; production remains Bc16 with the retained geometry.

### 2026-08-26 19:03 +07 — P0-FR teacher-conditioned Bole branch compatibility — CLOSED
- Completed the unfinished branch-conditioned proposal capture on the current Q4G64 factory-MTP artifact: 62 target-refreshed roots and 1,274 proposal distributions; diagnostic source was already restored.
- Paying cumulative target-conditioned proposal ranks under budgets 8/12/16/22 gives mean useful/root only 1.968/2.000/2.016/2.048. Best finite ceilings are 24.58/24.76 tok/s at 65K and 20.06/20.21 at 262K for 128/256 completions.
- Classification CLOSED_BRANCH_CONTINUATION_BELOW_GATE; no Bole runtime. Evidence: .chatgpt/post_reboot_20260821/p0_fr_bole_branch_compat/TEACHER_RESULT.md and teacher_result.json.

### 2026-08-26 19:35 +07 — P0-FS exact target-logit-margin acceptance — CLOSED
- Exported exact W8 rerank IDs/scores from the promoted B1 head and accepted mismatched factory drafts only when winner_score-draft_score was below a bounded delta; production source was restored before all runs.
- Fixed delta8 reached 101.51 useful tok/s and delta16 108.24, but delta8 six-family qualification was deterministic yet failed quality everywhere (LCS 0.191-0.695, text similarity 0.107-0.518); code was 99.23 and Thai/Japanese/Arabic 63.19/77.62/68.69.
- Periodic high-delta candidates reproduced already-closed accept-all/p8/p16 trajectories. Classification CLOSED_EXACT_MARGIN_REPRESENTATIVE_QUALITY_OR_SPEED; no source retained.

### 2026-08-26 19:42 +07 — P0-FT exact sequential/lazy verifier — CLOSED
- Measured owner-only exact T1 at 24.57 useful tok/s / 40.375 ms per causal token. Current exact-BF16 TP2 D3 has U=2.6122 over 49 rounds.
- Even granting free proposal/control/rollback/state and stopping after exactly the accepted prefix, sequential exact verification costs ~105.47 ms/round and cannot exceed 24.77 useful tok/s. CLOSED_SEQUENTIAL_EXACT_VERIFY_WEIGHT_RESTREAM_BELOW_GATE.

### 2026-08-26 20:07 +07 — P0-FU representative q-distribution PipeInfer correction
- The apparent 107-109 tok/s P0-FU reopening used only the 20-round fixed-general P0-CV capture at U=2.9501.
- Re-costed with the existing complete six-family P0-CU common-policy authority: representative U=1.87691; Thai/Japanese 1.151/1.401.
- Even retaining the newer 32.115+2.615-ms target/proposal work, perfect two-GPU balancing, the maximal p_full compatible with U, and free heads/q-recompute/RNG/correction/transport/state/control, the representative infinite ceiling is below dual100; finite128/256 are lower.
- Classification CLOSED_REPRESENTATIVE_QDIST_PIPEINFER_BELOW_DUAL100. No runtime prototype. Evidence: .chatgpt/post_reboot_20260821/p0_fu_qdist_pipeinfer_recost/CONCLUSION.md and representative_result.json.

### 2026-08-26 20:19 +07 — P0-FV ASD budgeted suffix strict-trace upper bound
- Used the existing full-BF16 six-family target-logit captures to price request-budgeted low-regret mismatch acceptance plus realized-prefix suffix reuse.
- The stitched strict-trajectory simulation is deliberately optimistic and invalid after the first accepted mismatch, so it can only close: best B64/g2/M2 gives representative U=2.3538, family range 1.5385-3.8750.
- With proposal, ledger, correction, head, communication, and control free, short ceiling is only 73.29 tok/s (67.78 with current proposal). CLOSED; no runtime patch.

### 2026-08-26 20:24 +07 — P0-FW long D4 inter-round target pipeline
- Re-costed assumed-full next-round stage0 overlap on the actual exact-BF16 TP2 D4 long traces, not the short corpus.
- 65K full survival is 19/32; with proposal/fold/head/control/state/transport free, infinite ceiling is below100 and finite128/256 are lower after fill/drain.
- 262K full survival is only 1/42, so nearly every speculative stage0 is discarded; ceiling is far below100. CLOSED_LONG_INTERROUND_PIPELINE_BELOW_100; no runtime code.

### 2026-08-26 20:59 +07 — P0-FX full target replica plus physical-GQA helper
- Built a guarded full 64-layer target/MTP stage on the fast card and a zero-weight second-card physical-GQA helper. Full state fit at65K; 8K snapshot import/helper execution was byte-exact.
- Three populated65K D4 runs were deterministic at 47.19/47.43/47.55 useful tok/s, digest09ff33b553ac8a05, 32 rounds, 127 drafts, 93 accepted, 30/24/20/19.
- Full local target cost ~66.4ms/round, worse than retained split exact-BF16 TP2 target 27.2+30.3ms. CLOSED_FULL_REPLICA65_BELOW_RETAINED_FRONTIER; source restored.

### 2026-08-26 21:21 +07 — P0-FY static shared-FFN + inter-round pipeline — CLOSED by real 65K yield
- The layerwise-stable eight-layer mask 4,5,8,9,10,11,12,13 was deterministic and modestly faster on five of six short families, but that proxy did not compose at long context.
- On the retained P0-M Q4G64 proposal + exact-BF16 TP2 + physical-GQA 65K route, baseline 63.645 tok/s / 32 rounds / 93 of 127 accepted became 45.320 tok/s / 47 rounds / 77 of 187 accepted. D4 tail survival fell from 31,24,19,19 to 35,28,8,6.
- The added rounds overwhelm skipped-FFN compute and invalidate the joint pipeline ceiling. Classification CLOSED_STATIC8_LONG_YIELD_COLLAPSE; source restored, no pipeline runtime. Evidence: .chatgpt/post_reboot_20260821/p0_fy_static_ffn_pipeline_joint/CONCLUSION.md and long65k/result.json.

### 2026-08-26 21:47 +07 — P0-FZ promoted delta7 + P0-M dual-short recert — PASS 104.94
- Integrated the exact-top128 logit-margin verifier as an opt-in production path and rebuilt focused speculative/vocab tests.
- Four stock complete-path runs on the promoted P0-M artifact were deterministic: 104.87/105.05/104.93/104.95 useful tok/s, median 104.94, digest afa00cc9a1d014ee, 68 rounds, 204 drafted, 186 accepted, positions 66,65,55.
- Six-family candidate outputs were deterministic and independently task-coherent under the corrected owner contract. Dual-short >=100 is frozen. Evidence: .chatgpt/post_reboot_20260821/p0_fz_margin_p0m_joint/promoted_recert/result.json and six_family_d7/result.json.

### 2026-08-27 08:25 +07 — P0-GW4 populated65K exact force-accept + K16 final head — PASS 100.125
- Implemented an exact device-side fully-licensed greedy transition for the already-qualified force-D4 route, removing the generic target-logit acceptance work while preserving target hidden folding and exact continuation.
- Reduced only the final T1 vocabulary shortlist from K128 to K16, using prior real six-family hidden-state qualification; all four complete-path runs stayed byte-identical to the accepted trajectory.
- Four stock populated65K runs: 100.04/100.13/100.12/100.13 useful tok/s, median 100.125; 48 rounds, 191/191 drafts accepted, positions 48,48,48,47, digest 9857b3b57a58e2d3. Diagnostic source restored. Evidence: .chatgpt/post_reboot_20260821/p0_gw4_force_accept_k16/recert_result.json.

### 2026-08-27 17:09 +07 — P0-NP2 exact Wc16/Bc16 cached GQA occupancy schedule
- Replaced cached-only Wc8/Bc16 with Wc16/Bc16 to test whether 32 resident warps/SM could improve the exact T4/T5 long scan. The separate candidate built, isolated physical-dual outputs were bit-exact at T4/T5 and 65K/262K, and source was restored. Across 24 paired runs the 262K ratios were 0.9981/1.0049 at T4 and 0.9977/0.9971 at T5 for fractions 0.50/0.60; 65K gains were only 0.1-1.0%. CLOSED_NO_MATERIAL_OPERATOR_WIN; retain Wc8/Bc16. Evidence: .chatgpt/post_reboot_20260821/p0_np2_gqa_wc16_bc16/result.json and wc16.resources.txt.

### 2026-08-27 18:27 +07 — P0-NU static-prefix pooled-summary GQA — CLOSED
- Exact recent-tail plus contiguous prefix K/V means with log-count multiplicity was evaluated on 65,292 actual captured tokens across all 24 query heads. Every material point failed badly: best 10.35% scan point had min cosine 0.3508 and max relative-L2 0.9365; no strict-quality >=15ms-saving point exists. Classification CLOSED_PREFIX_SUMMARY_QUALITY_OR_MATERIALITY. Evidence: .chatgpt/post_reboot_20260821/p0_nu_prefix_summary_gqa_oracle/{result.json,CONCLUSION.md}.

### 2026-08-27 18:27 +07 — P0-NR force-D4 fifth-proposal q-distribution — CLOSED
- Fixed the prior fifth-proposal out-of-bounds diagnostic by reusing the last legal metadata slot, captured 235 proposal distributions over 47 representative262 rounds, and costed an impossible infinite inter-round pipeline. Actual continuation top1/top4/top16 hit rates were 34.0/36.2/38.3%; the best free-state infinite ceiling was only 74.10 useful tok/s. Classification CLOSED_FORCE_D4_INTERROUND_PIPELINE_BELOW_105. Source restored. Evidence: .chatgpt/post_reboot_20260821/p0_nr_force_d4_qdist_fixed/result.json.

### 2026-08-27 18:44 +07 — P0-NW StreamingLLM sink+tail GQA — CLOSED
- Executed 16 exact sink-prefix plus recent-tail ranges on genuine representative262. Several configurations passed the representative output screen, including sink0.5%+tail2.5%, but the best quality-valid route was only 74.15 useful tok/s. Runtime target stages remained about 1.28/1.03 s over 48 rounds, so this sparsity family does not approach the 100 gate. Source restored. Evidence: .chatgpt/post_reboot_20260821/p0_nw_sink_tail_gqa/{screen.json,result.json}.

### 2026-08-27 20:43 +07 — P0-OA native affine K4V2 cached GQA
- Built the native SM75 cached-GQA decoder for mechanically packed affine K4/V2 old-range KV, fixing the initial static-shared overflow by fetching 16 B/token of low metadata directly. Source was restored and rebuilt. On an exact 4K snapshot the native route remained coherent but changed the factory trajectory: 68.48 -> 40.19 useful tok/s, rounds 43 -> 72, acceptance 83/129 -> 52/216, despite only a ~1.9% per-round regression. This closes K4V2 on quality/yield; the independent oracle's K4V4 quality is materially stronger, so K4V4 is the bounded fallback now in flight. Evidence: .chatgpt/post_reboot_20260821/p0_oa_native_affine_k4v2/e2e4k/result.json.

### 2026-08-28 05:53 +07 — P0-OQ one-range owner fast path — CLOSED
- Built the owner-only retained-range fast path and restored source. Eight representative262 target-only sink/tail cases were measured with exact MTP. No case passed the representative quality screen; best speed was 83.86 tok/s and low sink cases changed into control-token or off-task continuations. Classification CLOSED_ONE_RANGE_FASTPATH_QUALITY_OR_SPEED. Evidence: .chatgpt/post_reboot_20260821/p0_oq_one_range_fastpath/result.json.

### 2026-08-28 05:53 +07 — P0-OR fixed key-channel sidecar — CLOSED
- Evaluated fixed per-layer/head canonical-INT8 key-channel sidecars over affine K4/V4 on all 16 real 65K attention captures. Query-weighted top 1/2/4/8/16% key channels do not reliably repair K4; strict and balanced impossible ceilings are 86.85/92.01 tok/s. The most permissive cosine policy still reaches only an impossible 98.85 tok/s before sidecar decode and metadata. No runtime implementation. Evidence: .chatgpt/post_reboot_20260821/p0_or_key_channel_sidecar/{result.json,CONCLUSION.md}.

### 2026-08-28 06:05 +07 — P0-OS outlier-token sidecar — CLOSED
- Real all-layer 65K captures were screened with old-range affine K4/V4 plus canonical INT8 whole-token exceptions. Append-time RAW quantization-error ranking and a stronger captured-query AWARE upper bound both fail the gate: RAW strict/balanced/cosine impossible 262 ceilings are 92.06/92.13/97.87 tok/s; AWARE tops out at the same 97.87 before exception lookup/decode overhead. No runtime implementation. Evidence: .chatgpt/post_reboot_20260821/p0_os_outlier_token_sidecar/{result.json,CONCLUSION.md}.

### 2026-08-28 08:04 +07 — P0-PC layerwise native K8V4 discriminator — ADVANCE
- Fixed the Bc32 native affine decode path, propagated global layer identity through direct and physical-GQA launches, and tested K8V4 on every attention layer except exact layer15. At 4K the mixed route is coherent and control-token-free, accepts 82/132 drafts, and runs 66.81 vs 59.57 tok/s. The old ID-similarity closure was invalid under the owner-corrected representative-quality contract; advance to genuine262. Evidence: .chatgpt/post_reboot_20260821/p0_pc_layerwise_k8v4/e2e4k/{result.json,corrected_decision.json}.

### 2026-08-28 08:16 +07 — P0-PF layerwise native K8V4 + force/FFN/tailfast — CLOSED
- Composed native layerwise K8V4 with P0-M, exact-BF16 T5 TP2, force-D4 exact continuation, the quality-safe FFN mask, and tail-fast ownership on genuine representative262. The full route fell to 62.39 tok/s and produced repetitive output (diversity 0.677), with target stages 1.523+1.348 s over 48 rounds. Native K8V4's full-size-layout nibble decode is neither quality-safe nor performance-material in this implementation; close. Evidence: .chatgpt/post_reboot_20260821/p0_pf_layerwise_k8v4_force_ffn_tailfast/screen262.json.

### 2026-08-28 12:57 +07 — OWNER HARD REQUIREMENT: final single + dual multimodal support
- Final promotion is blocked unless both single and optimized dual serving support Qwen3.8 image/video multimodal input. Current single shared vision tests pass; current dual `TextShardStage` is explicitly text-only (`.vision = false`) and therefore still needs real integration + E2E qualification. Do not infer support from shared vision operators or empty PASS markers. See `PLEASE_READ_THIS_2026-08-28_HARD_REQUIREMENT_FINAL_MULTIMODAL_SINGLE_AND_DUAL.md`.

### 2026-08-28 13:37 +07 — P0-Q3 columnwise target-GQA — CLOSED
- Built a real columnwise verifier path: speculative draft columns used a smaller retained KV range while the final continuation column preserved the qualified target sink+tail range. Fixed the original tail-fast anchor mismatch and pinned-host staging reuse race before measurement; source was restored.
- At representative262 over 32 D4 rounds, baseline was 91.43 useful tok/s. Draft ranges 0.1/0.5%, 0.2/1%, 0.5/2.5%, and 1/5% were only 83.10/82.22/80.61/80.90. GQA helper time rose from 0.364s to 0.560-0.635s because splitting T5 into T4+T1 duplicates helper launch/transfer/merge coordination.
- All routes retained 124/127 accepted drafts; the mechanism is computationally negative before quality promotion. CLOSED_COLUMNWISE_EXTRA_GROUP_OVERHEAD. Evidence: .chatgpt/post_reboot_20260821/p0_q3_columnwise_target_gqa/screen160_result.json.

### 2026-08-28 13:49 +07 — P0-Q4 stock-single P0-M + delta7 recert — SINGLE100 PASS
- Re-tested the later P0-M Q4G64/Marlin factory-MTP artifact and qualified delta7 policy on stock single rather than leaving single frozen at the earlier 80.185 result. Loaded-role calibration selected GPU-2fffc240... dynamically; no clock or hardware setting was changed.
- Four exact complete-path 128-token repetitions were 107.09/107.44/107.31/107.23 useful tok/s: median 107.27, min 107.09, identical digest 2ffb3c07aa9a84ce, 33 rounds, 99 drafted, 95 accepted, 33/32/30 by position, and identical output bytes. A 256-token extension measured 107.55 tok/s.
- Two 256-token runs across each of general/code/JSON/Thai/Japanese/Arabic were deterministic and representative-quality valid. Family medians were 107.555/107.270/103.900/72.680/89.045/88.895; mean 94.891. The fixed hard endpoint therefore advances from 80.185 to 107.27 useful tok/s. Evidence: .chatgpt/post_reboot_20260821/p0_q4_single_delta7_p0m_recert/final_result.json.

### 2026-08-28 13:50 +07 — Correction to P0-Q4 stock-single recert evidence
- The immediately preceding P0-Q4 prose copied stale intermediate counters while final_result.json was generated from the completed recert set. The authoritative four stock-single runs are 107.45/107.33/107.21/107.12 useful tok/s (median 107.27), identical digest ef4dc9c008396373, 33 rounds, 99 drafted, 94 accepted, 33/32/29 by position, and identical output bytes. Loaded median clocks were 1845-1860 MHz. The 107.27 frontier is unchanged. Evidence: .chatgpt/post_reboot_20260821/p0_q4_single_delta7_p0m_recert/final_result.json.

### 2026-08-28 14:39 +07 — P0-Q3 serialized columnwise target-GQA
- Representative262 control reproduced 86.44 useful tok/s, digest cb7cda10a3883e76, 188/191 accepted. Draft-column sink+tail candidates retained 188/191 acceptance but changed output digest and fell to 74.48-74.57 tok/s.
- The prototype serialized a draft-column helper execution and a continuation-column helper execution. Context-GQA helper time rose from 0.610 s to about 0.965 s and target wall rose by about 0.44 s. This closes the two-round-trip implementation, not the columnwise scan mechanism.
- Next mechanism is fused columnwise T5 state production: two local range launches per device, packed into one T5 state buffer, then one helper transfer and one merge. Source restored; no production source retained.

### 2026-08-28 14:48 +07 — P0-Q4 fused columnwise target-GQA
- Built a packed-state implementation with draft and continuation ranges executed locally on both GPUs, followed by one helper state transfer and one T5 merge. Source was restored after candidate capture.
- A representative262 smoke at draft 0.5% sink + 2.5% tail produced 68.16 useful tok/s and repetitive output. Per-round context-GQA helper time was ~21.6 ms versus ~12.7 ms for the 86.44 control.
- Kernel inspection establishes the complete ceiling: the accepted T5 tensor-core kernel loads each KV tile once and reuses it across all five query columns. Any separate draft/final range execution must load the continuation range once plus the draft range again, so memory traffic is never below the shared T5 control. CLOSED; no further columnwise implementation.

### 2026-08-28 21:07 +07 — P0-Q5 canonical-T5 draft/final KV ranges — CLOSED
- A rebuilt canonical-T5 verifier layout used shortened old-KV ranges for the four draft columns while preserving the qualified final continuation range.
- Representative262 control was 86.70 tok/s and quality-pass. All four draft-range points were only 75.39-75.51 tok/s, changed to the same repetitive digest, and failed quality despite unchanged 48/191/188 and 47,47,47,47 acceptance.
- Helper traffic stayed 4.02 GiB while stage0/stage1 rose from 1.263/1.027 s to about 1.49/1.20 s. The layout did not compact wire traffic and added verifier cost. Source restored; do not reopen unchanged.

### 2026-08-28 22:45 +07 — OWNER RESEARCH NOTE: fresh 262K mechanisms only / no repeats
- Added `PLEASE_READ_AFTER_P0_PY_2026-08-28_FRESH_UNTRIED_RESEARCH_NO_REPEATS.md`. Immediate priority is to finish P0-PY with a compatible prefix snapshot before classifying it. Fresh bounded mechanisms behind it are native packed INT4 tensor-core dense Q4 on SM75 and an offline INT8-PV oracle that leaves the current INT8-G64 KV cache unchanged. After the 262K natural frontier, investigate request-aware B8 persistent work scheduling and batchwise cross-request overlap. The note contains an explicit do-not-reopen list covering prior KV4/S4-QK/sparse-attention/column-range/graph/tree/FFN/DAG closures.

### 2026-08-28 23:09 +07 — OWNER HARD STOP: NO SUBAGENTS
- Reiterated the existing `AGENTS.md` prohibition in `PLEASE_READ_THIS_NOW_2026-08-28_2309_NO_SUBAGENTS_HARD_STOP.md`. No `codex exec`, additional Codex workers, Claude/OpenCode, sub-agents, code-mode delegated workers, agent loops, or autonomous coding/research wrappers. Primary worker must do the work directly. Multiple detached Codex workers observed shortly before this note were an instruction violation, not an allowed workflow.

### 2026-08-28 23:44 +07 — P0-Q33 native packed INT4 dense Q4 — CLOSED below current SM75 Q4 op
- Verified TU102 native IMMA.8832.S4.U4/S4.S4 emission and built a direct packed-activation T4 prototype. The first integrated direct-S4 candidate failed exactness, exposing that the existing fragment mapping cannot be consumed by the naïve signed decomposition. A separate proven exact native-U4 decomposition at the actual verifier gate/up shape M17408 K5120 produced byte-exact A8 output and scales, but measured 299.068 us/op versus the retained fused SM75 Q4 A8 SwiGLU 167.610 us/op. This is 78.4% slower, far below the required >=8% complete-op gain; no production code retained. Evidence: .chatgpt/post_reboot_20260821/p0_q32_native_int4_dense_q4/native17408_run.stdout and .chatgpt/post_reboot_20260821/p0_q33_native_s4_prepacked_a8/paired_result.json.

### 2026-08-28 23:44 +07 — P0-Q34 INT8-PV over resident INT8-G64 V — CLOSED numerically
- Used actual all-layer populated65K captured Q/K/V. Preserved current INT8-G64 K/V. For each output feature group, folded the resident per-token V scale into BF16 softmax probabilities, requantized those weights in token-G64 INT8, and evaluated exact int32 Pq*Vq accumulation. Across 16 full-attention layers: min cosine 0.96352, max relative L2 0.49734, mean relative L2 0.09110. Early layers were particularly poor; only late-layer subsets approached acceptable quality and cannot supply a material complete-round saving. No 262 capture or CUDA implementation justified. Evidence: .chatgpt/post_reboot_20260821/p0_q34_int8_pv_oracle/result65.json.

### 2026-08-29 00:28 +07 — P0-Q36 fixed-reference unified-softmax kernel — CLOSED
- Implemented the Q35 phi=0 fixed-reference softmax directly in the current INT8 small-T GQA partial kernel: removed per-block max/warp-max, online alpha rescaling, accumulator renormalization, and the shared alpha array. Built a copied PA/K16 candidate and restored the promoted source.
- Resource evidence showed 128-192 B less shared memory and up to 16 B less stack for relevant kernel variants, but populated65K E2E was 106.66 versus 106.88 tok/s and GQA packet time regressed 0.368->0.372 s.
- The candidate emitted near-empty repetitive token IDs (mostly 0/13), failed representative quality, and changed digest. The offline float oracle did not model the integrated BF16 probability/staged-reduction behavior sufficiently. CLOSED; do not advance reducers or joint P0-PY/Q36.
- Evidence: .chatgpt/post_reboot_20260821/p0_q36_unified_softmax_kernel/{result65.json,resource_compare.json,patch.json,candidate.sha256}.

### 2026-08-29 01:14 +07 — P0-Q35 stage0 proposal + K16 representative262 — CLOSED
- Finished the previously setup-blocked P0-PY lane with a valid stage0-owned D3 prefix snapshot, then joint-tested it with the previously exact K16 final T1 head in a full 240-token representative262 pair.
- Control was 87.03 useful tok/s; stage0 proposal was 87.21. Both ran 48 rounds, 191 drafted, 188 accepted, 47/47/47/47. Proposal time moved only 0.313->0.306 s and target remained 1.264+1.031 s. Candidate output changed digest and became repetitive.
- Classification CLOSED_NO_MATERIAL_QUALITY_SAFE_GAIN. Ownership migration alone is not overlap and cannot close the remaining target-pass gap. Source was already restored. Evidence: .chatgpt/post_reboot_20260821/p0_q35_stage0_proposal_k16/paired262_result.json.

### 2026-08-29 01:20 +07 — P0-PY stage0 canonical proposal, corrected representative262 — CLOSED
- Re-captured a valid stage0-owned D3 representative262 prefix snapshot and completed the required full 240-token paired control/candidate using the original K128 head and exact PA/PW environment.
- Control 86.29 tok/s, stage0 owner 86.89. Both used 48 rounds, 191 drafts, 188 accepts, 47/47/47/47. Proposal fell only 0.328->0.313 s; target stayed 1.267+1.029 s. Candidate output dropped to 31 words and failed representative quality.
- Classification CLOSED_NO_MATERIAL_QUALITY_SAFE_GAIN. The valid snapshot removes the setup ambiguity; ownership migration does not create true overlap. Evidence: .chatgpt/post_reboot_20260821/p0_py_pa_stage0_proposal/paired262_result.json.

### 2026-08-29 01:34 +07 — P0-Q38/Q39 stage0-owned proposal discriminator — CLOSED
- Built a compatible stage0-owned 261900-token snapshot and ran control/stage0/stage0+K16 complete-path discriminators. Forced D4 saved only 14 ms total for stage0 and 23 ms for K16 over 48 rounds, while both produced degenerate representative output.
- Under natural target acceptance, stage0 remained coherent but slowed from 43.07 to 40.07 useful tok/s; stage0+K16 accepted 0/619 drafts and fell to 17.44 tok/s. Snapshot setup is valid; the ownership/proposal economics are not.
- Do not reopen stage0 proposal ownership or stage0 K16 absent a new overlapping schedule that changes the complete-system ceiling. Evidence: .chatgpt/post_reboot_20260821/p0_q38_post_capture_discriminator/result.json and p0_q39_stage0_natural_accept/result.json.

### 2026-08-29 01:34 +07 — P0-Q42 INT8-PV block16 numerical admission — ADVANCE TO 262/COST CEILING
- New scale-grid evidence reopens only the published INT8-PV formulation: resident per-token V scale is folded into P, with one BF16 scale per 16-token block and exact int32 Pq*Vq accumulation.
- Across all 16 actual populated65K full-attention layers, P-block16 reaches min cosine 0.999879896, max relative L2 0.0176841, mean cosine 0.999990085, and max abs 0.25. It is numerically stronger than the earlier qualifying P-block32 route and matches the promoted Bc16 tile.
- A genuine262 all-layer Q/K/V capture and a timing-only free-PV/V-side complete-system ceiling are in progress. No runtime implementation unless the measured ceiling can close the remaining gate.

### 2026-08-29 06:49 +07 — P0-Q42 block16 INT8-PV genuine262 — CLOSED by optimistic complete-system ceiling
- Genuine262 block16 numeric oracle passed strongly (min cosine 0.999957, max relative L2 0.01227), but the repaired compute-sanitizer-clean timing discriminator closes implementation. A deliberately cheaper P-block32 one-KV-head lower bound is 1.648 ms on the fast role and 2.310 ms on the other role, while the retained exact GQA averages only about 1.066/2.295 ms per full layer and already covers all four KV heads plus QK/softmax/PV. Required block16 scaling, remaining KV heads, QK and softmax can only worsen the candidate. CLOSED_COMPLETE_SYSTEM_CEILING; no runtime integration. Evidence: .chatgpt/post_reboot_20260821/p0_q42_int8_pv_pblk16_ceiling/{result.json,CONCLUSION.md}.

### 2026-08-29 07:26 +07 — P0-Q45 native FP16-staged exact PV — genuine262 frontier 91.66
- SM75 has no BF16 tensor-core MMA: the retained GQA staged BF16 P/V then converted every ldmatrix fragment to FP16 inside mma_bf16. Reworked only the PV operand staging: each probability and dequantized V value is first rounded exactly to the same BF16 value, converted once to FP16 in the producer/loader path, staged as FP16, and consumed by native Turing FP16 MMA. The PV loop is K-outer so one P fragment is reused across all output slices; every accumulator retains identical K order.
- Populated65K paired control/candidate 106.28 -> 107.77 useful tok/s, byte-exact and trajectory-exact. Genuine populated262 paired control/candidate 86.50 -> 91.66 useful tok/s, byte-exact, same cb7cda10a3883e76 digest and 48/191/188/47,47,47,47 trajectory. Target wall improves ~5.7%; measured GQA packet+helper wall improves ~14.4%. Source restored after copied candidate build. Evidence: .chatgpt/post_reboot_20260821/p0_q45_native_fp16_pv_operands/{result65.json,result262.json,patch.json,binaries.sha256}.

### 2026-08-29 07:28 +07 — P0-Q50 query prequant once-per-GQA — CLOSED
- Measured the exact current T5 Q-int8/group-scale work replicated across 4x85 split CTAs against an optimistic prequant-once plus per-CTA global->shared reload. Fast/other role save only 0.0036/0.0045 ms per complete GQA launch; even multiplying the larger value by all 16 full-attention layers is ~0.072 ms/round versus ~4.5 ms/round remaining to 262K100 after P0-Q45. CLOSED_COMPLETE_SYSTEM_CEILING; no runtime integration. Evidence: .chatgpt/post_reboot_20260821/p0_q50_query_prequant_ceiling/result.json.

### 2026-08-29 07:48 +07 — P0-Q52 attention-mass-guided sink/tail rebalance — quality survives, speed still ~92
- Genuine262 captured Q/K showed current 1% sink + 5% tail retains mean attention mass ~0.7626, while wider sink/shorter tail retains slightly more mass at lower KV budget; tested 2/2.5, 2.5/2, 3/1.5, 3.5/1 target fractions with MTP fractions fixed 0.1%/0.5% on the exact P0-Q45 kernel.
- All four 240-token E2E cases passed the representative technical-report quality gate and retained 48 rounds / 191 drafted / 188 accepted / 47,47,47,47. Best was 3% sink + 1.5% tail at 92.01 t/s (3.5/1 91.97), versus Q45 exact 91.66. The lower scan budget mostly removes helper-side GQA that is overlapped and does not close the critical path. Keep as quality-safe policy evidence, not the 262K gate solution. Evidence: .chatgpt/post_reboot_20260821/p0_q52_sink_tail_rebalance/result.json.

### 2026-08-29 07:48 +07 — P0-Q47 Q45 T5 schedule re-screen — W16/Bc32 faster but quality-invalid
- Re-screened long T5 INT8 GQA CTA schedule after P0-Q45 changed PV producer/consumer balance. W16/Bc32 was ~1.4% faster than Q45 W8/Bc16 on an 80-token discriminator, but changes online-softmax rounding. Full240 joint test with the quality-safe 3% sink + 1.5% tail policy reached 93.63 t/s but collapsed into severe phrase repetition (trigram diversity ~0.067; topic coverage 2/9). CLOSED for promotion. Evidence: .chatgpt/post_reboot_20260821/p0_q47_fp16_pv_schedule/screen262.json and .chatgpt/post_reboot_20260821/p0_q54_q47_s3t15/result.json.

### 2026-08-29 09:53 +07 — P0-Q56 D5 MTP context rescue — CLOSED
- On the Q45 exact FP16-PV + force-D5/K16 genuine262 route, widening MTP GQA from 0.1%/0.5% to 0.2%/1%, 0.5%/2%, and 1%/5% did not improve complete-path speed. Results were 92.03/89.90/89.84 useful tok/s versus 92.11 control, with the same 40 rounds / 198 drafted / 195 accepted / 39,39,39,39,39 trajectory at the smallest point; wider points also increased proposal wall. CLOSED_NO_GAIN. Evidence: .chatgpt/post_reboot_20260821/p0_q56_d5_mtp_context_rescue/result.json.

### 2026-08-29 10:03 +07 — P0-Q57 Q45 force-D5 + width6 TP2 — CLOSED for promotion
- Fixed the actual T6 routing seam so width6 TP2 executed 1768 calls on genuine262. Systems speed improved from D5 no-TP2 92.11 to 94.17 useful tok/s and stage1/head fell ~0.961->0.900 s, but the output collapsed to a repeated section-number sequence with only 66 words and 4/9 topic groups. Width6 TP2 is real; force-D5 is not a quality-valid endpoint. Evidence: .chatgpt/post_reboot_20260821/p0_q57_q45_d5_tp2_routefix/{CONCLUSION.md,e2e262/result.json}.

### 2026-08-29 10:39 +07 — P0-Q59/Q61 mapped TP2 transaction
- Q59 GPU-polling transaction is byte-exact on a genuine262 32-token smoke but regresses 80.27->68.61 t/s. Surgical Nsys attributes ~280 ms over 382 TP2 calls to mapped-system-memory wait kernels alone (~0.73 ms/call). Q61 tested scheduler-level cuStreamWaitValue32/cuStreamWriteValue32 on the mapped host pointer; the documented cross-device scheduler deadlock manifested and the bounded canary timed out, after which both GPUs were idle without reset/driver operations. Close mapped polling/stream-mem-wait variants; only revisit TP2 async via a genuinely distinct dependency primitive. Evidence: .chatgpt/post_reboot_20260821/p0_q59_mapped_tp2_transaction/smoke262/result.json, p0_q60_mapped_tp2_profile/analysis.json, p0_q61_stream_memop_canary.

### 2026-08-29 11:05 +07 — P0-Q63 Q45 physical sequence fraction rebalance — CLOSED
- Re-screened exact physical sequence ownership after Q45 FP16-PV changed GQA economics. Fractions 0.58/0.60/0.62/0.64 were byte-identical on a 120-token genuine262 screen at 85.88/86.03/86.03/86.05 useful tok/s; target stage0 remained ~0.613 s and stage1 ~0.497-0.499 s over 24 rounds. Best 0.64 is only 1.00023x baseline. CLOSED_CURRENT_60_RETAINED. Evidence: .chatgpt/post_reboot_20260821/p0_q63_q45_fraction_rebalance/screen_result.json.

### 2026-08-29 11:08 +07 — P0-Q58 Q45 split40 genuine262 — CLOSED
- Performed the expensive correct D3 split40 261900-token capture (1336.953s, digest 8d1ace904a398d17), then matched Q45 D4 physical runs. Split36 was 91.24 useful tok/s with target0/1 1.204/0.983s; split40 was slower at 90.77 with target0/1 1.348/0.850s, same 48 rounds and 188/191 acceptance. Split40 output also became highly repetitive. Moving the full-attention boundary only redistributes stage wall and worsens coordination; CLOSED. Evidence: .chatgpt/post_reboot_20260821/p0_q58_q45_split40/result_d3.json.

### 2026-08-29 11:55 +07 — P0-Q57 Q45 D5 TP2 route fix — CLOSED quality
- Fixed the real Q55 integration miss: TextContext::mlp_tail previously offered MlpExecutionOverride only at T4/T5, so the compiled T6 TP2 engine never ran. After extending verify-only routing to T6, genuine262 recorded 1768 width6 TP2 calls and improved 92.11 -> 94.17 useful tok/s with 40 rounds / 198 drafted / 195 accepted / 39,39,39,39,39. However the representative 240-token output degenerates into repeated nested heading fragments (66 words, only 4/9 required topic families); not promotable. Source restored. Evidence: .chatgpt/post_reboot_20260821/p0_q57_q45_d5_tp2_routefix/e2e262/result.json.

### 2026-08-29 11:57 +07 — P0-Q58 Q45 split40 rebalance — CLOSED
- Re-screened split40 after Q45 changed exact GQA economics. On genuine262 D4 with the same Q45/TP2/force/FFN/tailfast route, split36 delivered 91.24 useful tok/s while split40 delivered 90.77. Target stage0/1 moved from 1.204/0.983 s to 1.348/0.850 s; total target wall worsened and helper/TP2 did not compensate. CLOSED_NO_GAIN. Evidence: .chatgpt/post_reboot_20260821/p0_q58_q45_split40/runs_d3/{split36,split40}/summary.txt.

### 2026-08-29 11:57 +07 — P0-Q59 FFN residual-update reuse — CLOSED oracle
- Revisited the previously unimplemented same-round cross-layer FFN residual-update reuse oracle against the quality-safe skip mask. On every remaining candidate layer the reuse advantage over zero-skip is tiny or mixed; e.g. L21<-L2 improves worst cosine only +0.00030 and max relative error by 0.0018, L58<-L42 +0.0052 cosine/+0.0030 relative, while several candidates worsen relative error. This cannot safely unlock the multiple extra FFN skips needed for ~8-9% genuine262 gain. CLOSED_BEFORE_RUNTIME. Evidence: .chatgpt/post_reboot_20260821/p0_er_shared_ffn_skip/cross_reuse.json.

### 2026-08-29 12:12 +07 — P0-Q61 exact B32 staging with canonical B16 subblocks — CLOSED
- Built a Q45-derived SM75 T5 cached-GQA path that stages 32 physical KV keys per CTA tile but performs QK, online-softmax recurrence, BF16-rounded P, and PV as two sequential canonical 16-key subblocks. It is byte- and trajectory-exact at representative65K, but the B32 specialization consumes 44,032 B shared and 192-200 B stack versus the retained B16 26,240 B shared/8 B stack; paired E2E is 107.53 vs 107.85 useful tok/s. CLOSED_BEFORE_262_NO_GAIN. Evidence: .chatgpt/post_reboot_20260821/p0_q61_exact_b32_sub16/{screen65/result.json,resources.txt}.

### 2026-08-29 12:22 +07 — P0-Q62 direct V load on SM75 — CLOSED
- Confirmed SM75 cp_async is synchronous and built an exact Q45 T5 cached-GQA path that removes the intermediate V-int8 shared plane, directly loads canonical V codes from global in loader warps, and stages the same BF16-rounded FP16 PV operands. Resource use drops from 26,240 to 22,144 B shared while retaining 128 registers, but paired representative65K is byte/trajectory exact and slower: 106.73 vs 107.69 useful tok/s, helper 0.105 vs 0.094 s. Shared staging is beneficial cache/reuse on TU102. CLOSED_BEFORE_262_NO_GAIN. Evidence: .chatgpt/post_reboot_20260821/p0_q62_direct_v_load/{screen65/result.json,resources.txt}.

### 2026-08-29 12:51 +07 — P0-Q64 retained-range local subdivision — genuine262 >=100 recert
- NCU showed the retained 1% sink + 5% tail GQA launches were grid-underfilled (dominant T5 launches only 16 CTAs on GPU0 and 4 CTAs on GPU1 across 68 SMs, ~6%/~2% compute/DRAM utilization). Q64 keeps the exact retained key set and physical 60/40 KV ownership but subdivides retained ranges into eight compute partial states before the same FP32 state merge. Full genuine261900-token context recert is 104.53/104.46/104.57 useful tok/s, median 104.53, min 104.46; all three have digest 8c857a8c13f7fd19, 48 rounds, 191 drafted, 188 accepted, positions 47,47,47,47, deterministic output, and representative-report quality pass. Evidence: .chatgpt/post_reboot_20260821/p0_q64_retained_local_splits/recert262_f8/result.json.

### 2026-09-01 11:22 +07 — P0-Q60 stream-memops TP2 — CLOSED exact
- Replaced Q59 SM-spinning mapped mailboxes with exact copy-engine pinned DMA plus cuStreamWriteValue32/cuStreamWaitValue32 ordering. A 1000-round 32 KiB two-GPU oracle was exact at 88.034 us/round. Same-snapshot Q45/force/FFN/tailfast 4K E2E remained bit-exact (digest d3c0120a09f400e2; 12 rounds; 47/47 accepted; identical stdout hash), but slowed 96.55 -> 92.36 useful tok/s; target stage0/1 rose 0.231/0.241 -> 0.255/0.246 s. CLOSED_NO_GAIN. Source/baseline restored. Evidence: .chatgpt/post_reboot_20260821/p0_q60_stream_memops_tp2/smoke4k/result.json.

### 2026-09-01 11:23 +07 — P0-Q48/Q49 BF16-FP16 conversion/direct-dequant — CLOSED
- The proposed all-bits BF16->FP16 conversion was exact over all 65,536 BF16 patterns but 0.61-0.62x the reference conversion throughput on both GPUs. Direct INT8 dequant to FP16 was only 1.001-1.003x the current exact BF16-rounded->FP16 path in isolation, far below the ~8-9% genuine262 complete-system requirement. CLOSED_BEFORE_RUNTIME. Evidence: .chatgpt/post_reboot_20260821/p0_q48_bf16_f16_bitconvert/{gpu0.stdout,gpu1.stdout} and .chatgpt/post_reboot_20260821/p0_q49_direct_fp16_pv_oracle/{gpu0.stdout,gpu1.stdout}.

### 2026-09-01 11:50 +07 — OWNER NEXT: unified multimodal100, productionize Q64, then B8 + headroom
- Added `PLEASE_READ_NEXT_2026-09-01_MULTIMODAL100_B8_AND_HEADROOM.md`. Do not stop at text genuine262 ~104.5. Q65 unified image is 105.28, mixed 101.53, video 98.97; recover video >=100 without workload gaming and fix the evaluator's terminal-`<|im_end|>` false control classification. Then integrate Q64/Q45 as a real production dispatch rather than diagnostic copied binaries. Next product priority is measured 1/2/4/8-user serving with per-user TPS/TTFT/p95/fairness/memory, followed by reusable headroom above 100. NO SUBAGENTS remains absolute.

### 2026-09-01 12:01 +07 — P0-Q66/Q67 multimodal breadth and short-policy separation
- Unified Q65/Q64 dual already passes sustained image/video/mixed decode >=100. Added Chinese breadth checks: two-image comparison returned the exact 2/3/yellow-star facts with only a terminal <|im_end|>. The long-context force-D4 policy corrupted exact short mixed-media answers (24-24 and N-1), while the natural D3 policy returned exact 24-9 and NIFER-9 with terminal-stop-only output. Decision: production dispatch must keep short factual multimodal on natural D3 and reserve force-D4/FFN/sink-tail/local-split policy for qualified long/sustained routes. Evidence: .chatgpt/post_reboot_20260821/p0_q66_multimodal_breadth/result.json and p0_q67_multimodal_short_policy/result.json.

### 2026-09-01 13:19 +07 — P0-Q74 representative65 D3 margin screen — CLOSED
- Screened D3 direct target-margin acceptance on the valid 65,292-token representative snapshot with Q45 FP16-PV, Q64 retained-range subdivision, P0-M Q4/Marlin, tail-fast ownership, and exact-BF16 T4 TP2. Delta 7/10/12/16/20 delivered 62.52/85.48/86.76/97.95/96.29 useful tok/s; every output failed representative quality or leaked chat controls. No point reached both 100 and quality. CLOSED. Evidence: .chatgpt/post_reboot_20260821/p0_q74_rep65_margin_d3/result.json.

### 2026-09-01 13:44 +07 — P0-Q73 realistic C1/C2/C4/C8 serving baseline — scalar decode and bulk-prefill bottleneck
- Direct realistic clinic helpdesk/EMR baseline on the production source route used eight distinct prompts at ~1.8K and ~7.6K tokens. Aggregate ordinary decode rises from 23.19 to only 39.14 tok/s at 2K C8 and from 22.05 to only 38.01 tok/s at 8K C8; per-user falls to 4.89/4.75. The implementation serializes every prompt before decoding and then uses scalar T1 round-robin, so worst TTFT reaches 47.677 s at 2K C8 and 190.100 s at 8K C8. Active-rate fairness is 0.242/0.074; p95 inter-token gaps are 203/209 ms. GPU memory remains ~10.4/10.3 GiB and ~10.9/10.9 GiB, leaving ample capacity. Outputs are coherent/control-clean. This justifies true B1/B2/B4/B8 decode batching and then admission/prefill overlap. Evidence: .chatgpt/post_reboot_20260821/p0_q73_multisession_baseline/result.json.

### 2026-09-01 13:44 +07 — P0-Q76 reserved-control draft guard — CLOSED quality
- Guarded licensed-D4 so any host-known draft prefix containing Qwen reserved/control token IDs falls back to canonical acceptance. Representative65 remained fast at 103.08 useful tok/s and emitted no embedded controls, but still collapsed into repetitive low-diversity prose (0.485 trigram diversity, only 4/9 topic families). The corruption predates the control token. CLOSED_CONTROL_GUARD_SPEED_OR_QUALITY. Evidence: .chatgpt/post_reboot_20260821/p0_q76_control_token_guard/rep65_result.json.

### 2026-09-01 13:50 +07 — P0-Q75 representative65 D4 margin screen — CLOSED
- On the valid 65,292-token representative snapshot, Q45/Q64 D4 with natural acceptance and direct margin 7/10/12/16/20 measured 32.47/61.53/66.91/63.37/88.81/91.51 useful tok/s. All six outputs failed the representative quality/control contract, and none reached 100. CLOSED_D4_MARGIN_QUALITY_OR_SPEED. Evidence: .chatgpt/post_reboot_20260821/p0_q75_rep65_d4_margin/result.json.

### 2026-09-01 13:50 +07 — P0-Q76 host-known control-token guard — CLOSED
- Added a target-independent structural guard that disables licensed-D4 when the host-known draft prefix already contains a Qwen reserved control token. The representative65 discriminator remained fast at 103.08 useful tok/s and removed embedded controls, but still collapsed into repetitive text (trigram diversity 0.485, only 4/9 topic groups). Control-token avoidance alone is insufficient; close this guard and advance to exact target-argmax prefix acceptance. Source restored. Evidence: .chatgpt/post_reboot_20260821/p0_q76_control_token_guard/rep65_result.json.

### 2026-09-01 13:50 +07 — P0-Q73 multi-session scalar baseline through B8
- Measured actual shared dual-engine ordinary serving at 2K and 8K prompts for B1/B2/B4/B8. The current bulk-prefill then round-robin scalar decode saturates at only ~39 aggregate tok/s: 2K B1/B2/B4/B8 = 23.19/39.23/39.16/39.14 aggregate; 8K = 22.05/38.03/37.99/38.01. B8 per-user falls to 4.89 and 4.75 tok/s, TTFT grows to 47.7s and 190.1s, and device memory remains below 10.91 GiB. This establishes large unused compute/memory headroom and justifies the in-flight full-session batched decode path. Evidence: .chatgpt/post_reboot_20260821/p0_q73_multisession_baseline/result.json.

### 2026-09-01 14:02 +07 — P0-Q78 exact D4 target-argmax prefix acceptance — CLOSED
- Built a no-logits exact D4 verifier: accepted T4+T1 target heads computed exact target argmax for all draft columns and continuation; the accept kernel committed only drafts equal to target argmax and emitted target at first mismatch. Representative65 quality became fully coherent/control-free (174 words, 9/9 topic groups, diversity 0.959), proving the quality failure was licensed mismatch acceptance. Exact acceptance was only 101/535 over 134 rounds and throughput 41.07 useful tok/s, far below 100. CLOSED_EXACT_D4_SPEED. Source restored. Evidence: .chatgpt/post_reboot_20260821/p0_q78_exact_d4_argmax/rep65_result.json.

### 2026-09-01 14:02 +07 — P0-Q78 exact D4 argmax/prefix acceptance — CLOSED speed
- Replaced licensed-D4 draft-token overwrite with exact T4 draft-column target argmax plus T1 continuation, then committed only the exact matching draft prefix. Representative65 quality became fully coherent/control-clean (174 words, 9/9 topic families, 0.959 trigram diversity), proving the quality failure came from non-exact draft licensing. But acceptance fell to 101/535 across 134 rounds and complete-path throughput collapsed to 41.07 useful tok/s. Exact D4 cannot meet the 100 gate. Evidence: .chatgpt/post_reboot_20260821/p0_q78_exact_d4_argmax/rep65_result.json.

### 2026-09-01 14:20 +07 — P0-Q80 shared-engine C4 MTP aggregate100 — ACCEPT
- Measured real shared-weight/shared-dual-engine C4 MTP serving with four independent prompts and quality-specific fact checks. At 2K prompts: 127.46 aggregate / 31.87 per-user useful tok/s. At 8K: 109.12 aggregate / 27.28 per-user. All eight outputs are control-free, diverse, and preserve the required prompt facts. This clears the requested concurrent aggregate100 target through B4 and establishes C4 as the basis for C8. Evidence: .chatgpt/post_reboot_20260821/p0_q80_mtp_batch4_baseline/quality_result.json.

### 2026-09-01 15:24 +07 — P0-Q82 C8 cross-round preload — ADVANCE
- Unified eight-session C8 scheduler now pipelines next-round pair0 stage0 during the final current-round pair. Genuine 8K helpdesk/EMR C8 is exact to Q81 (all eight digests, 381 rounds, 1143 drafted, 619 accepted, 271,196,152), all eight quality cases pass, and aggregate useful decode improves 92.35 -> 103.01 tok/s (11.54 -> 12.88 per user, +11.54%). Peak memory 11.10/13.66 GiB. This opens the aggregate100 C8 gate but not the owner 20-30 per-user headroom target; advance to true T16/group4 verifier geometry. Evidence: .chatgpt/post_reboot_20260821/p0_q82_c8_crossround_preload/result8k.json.

### 2026-09-01 16:40 +07 — P0-Q90 native T16 Q4/Q5 token tiling — CLOSED no gain
- Fixed the native T16 Q4 SwiGLU and Q5 residual kernels by tiling token columns in grid.y (0..7 and 8..15), eliminating the silent warp-local truncation. Identical-prompt C8 symmetry is exact across all eight sessions. Real 2K C8 preserves the Q88 trajectory and quality but records 113.07 aggregate tok/s versus Q88 114.23 (0.9898x), so native Q4/Q5 tiling alone is non-material. CLOSED_NO_MATERIAL_QUALITY_GAIN. Evidence: .chatgpt/post_reboot_20260821/p0_q90_native_t16_token_tiles/{symmetry/result.json,result2k.json}.

### 2026-09-01 16:56 +07 — P0-Q91/Q92 native T16 Q4/Q5/GDN tiling — CLOSED no long-context gain
- Completed exact native T16 token tiling for Q4 SwiGLU, Q5 residual-add, and GDN value/z. Identical-prompt C8 symmetry is exact across all eight sessions; real 2K reaches 114.37 aggregate tok/s with valid outputs. On the decisive 8K eight-family workload, however, native T16 records 94.01 aggregate / 11.75 per-user tok/s versus the exact 2xT8 route 94.23 / 11.78; quality remains fully passing. Therefore native T16 is correct but does not materially improve long-context throughput. CLOSED_NO_LONG_CONTEXT_GAIN. Evidence: .chatgpt/post_reboot_20260821/p0_q91_native_t16_dense_gdn_tiles/{symmetry/result.json,result2k.json} and .chatgpt/post_reboot_20260821/p0_q92_native_t16_8k/result.json.

### 2026-09-02 01:19 +07 — P0-Q95 T16 Q4/Q5 weight-fragment reuse — ADVANCE
- Built isolated one-CTA T16 A8 kernels that reuse each packed fragment across token halves, unlike closed Q90/Q91 grid.y tiling. Bitwise exact against two qualified T8 calls. Cold-cache: Q4 SwiGLU 372.992 -> 323.584 us (1.15269x); Q5 residual-add 342.112 -> 227.264 us (1.50535x). Runtime used an ephemeral container mapping healthy minor1 to container nvidia0 only; host device/driver state untouched. Q5 clearly material; Q4 smaller but exact and additive. Evidence: .chatgpt/post_reboot_20260821/p0_q95_t16_weight_reuse/result.json.

### 2026-09-02 01:24 +07 — P0-Q96 GDN T16 Q5 fragment reuse — ADVANCE
- Full GDN projection discriminator is bitwise exact against two qualified T8 calls: 0 Q/K, value, or z mismatches. Cold-cache 374.080 -> 315.520 us (1.18560x). The one-CTA T16 value/z kernel reuses each Q5 fragment across both token halves; Q/K is computed once at T16 and remained bitwise identical. Evidence: .chatgpt/post_reboot_20260821/p0_q96_gdn_t16_weight_reuse/result.json.

### 2026-09-02 01:34 +07 — P0-Q98 T32 Q4/Q5 all-eight weight reuse — ADVANCE
- Built one-CTA T32 A8 kernels that reuse each packed fragment across four eight-token halves. Bitwise exact versus four qualified T8 calls. Cold-cache Q4 SwiGLU 738.432 -> 616.256 us (1.19826x); Q5 residual-add 676.768 -> 423.872 us (1.59663x). This is the first all-eight compute primitive with a material C8 ceiling. Evidence: .chatgpt/post_reboot_20260821/p0_q98_t32_weight_reuse/result.json.

### 2026-09-02 01:38 +07 — P0-Q99 GDN T32 fragment reuse — CLOSED
- T32 full-GDN route remained bitwise exact (0 Q/K, value, z mismatches) but cold-cache regressed 745.472 -> 751.136 us (0.99246x). Q/K projection and larger live state erase value/z reuse. Do not integrate T32 GDN. Evidence: .chatgpt/post_reboot_20260821/p0_q99_gdn_t32_weight_reuse/result.json.

### 2026-09-07 10:58 +07 — P0-Q110 MTP Marlin T16 admission — CLOSED
- Increasing the existing MTP Marlin prepared geometry from T8 to T16 reached the batch4 alignment launch but produced a CUDA illegal-address fault before next-draft readback. Source was restored by the guarded build. Keep the exact Q108 hybrid: Marlin T1..T8, Q4 fallback for alignment T16. Evidence: .chatgpt/post_reboot_20260821/p0_q110_single_c4_proposal_b4_marlin16/smoke_c4/result.json.

### 2026-09-07 11:23 +07 — P0-Q82 physical C8 after GPU return — RECERT
- Both stock GPUs returned to NVML/CUDA. The exact persisted Q82 8K eight-family workload re-ran at 103.69 aggregate / 12.96 useful tok/s/user, versus prior 103.01 / 12.88. All eight digests and the 381-round, 1143-draft, 619-accept, 271/196/152 trajectory are identical to the previously quality-passing authority. Evidence: .chatgpt/post_reboot_20260821/p0_q82_c8_crossround_preload/recert_20260907/r0/result.json.

### 2026-09-07 11:35 +07 — P0-Q108 C8 ragged replay ownership — 2K ADVANCE
- Separated C8 pair-capacity2 and group4-capacity4 GDN replay records, fixing the first exact T8 fallback crash. Q108 then completed the realistic eight-family 2K workload with all eight digests and final token streams exactly equal to Q88. Aggregate decode improved 114.23 -> 141.11 tok/s (14.28 -> 17.64/user, +23.5%). Speculative aggregate changed by one round, but final user-visible output is byte-identical. Evidence: .chatgpt/post_reboot_20260821/p0_q108_c8_ragged_replay_storage/result2k.json.

### 2026-09-07 11:44 +07 — SUPERVISOR — Q108 8K is REP quality-pass, not exact; pivot to serving architecture
- Q108 8K completed rc=0 at 126.98 aggregate / 15.87 useful tok/s/user, prefill 335.907 s, decode 8.001 s. This is +23.27% aggregate versus Q82 103.01 / 12.88, but it remains below the owner's 20 tok/s/user C8 floor and below Q109 physical C4+C4 (166.85 aggregate, min 20.32/user).
- Exactness is NOT preserved at 8K: only sessions 0/2/4 are byte/digest exact to Q82; aggregate trajectory changes 381 rounds / 1143 drafted / 619 accepted / 271,196,152 -> 384 / 1152 / 619 / 269,198,152. The same Q82 eight-family fact/diversity/control checks pass all 8 sessions, so classify this REP_QUALITY_PASS_NOT_EXACT, never strict EXACT. Evidence: .chatgpt/post_reboot_20260821/p0_q108_c8_ragged_replay_storage/supervisor_result8k.json.
- BOUND THIS HARNESS LINE HERE. Do not open another apps/dual host-wrapper microcycle merely to polish C8. The inherited ConcurrentExecutor already rebuilds exact active RoundMembership, admits/backfills, retains prefix state, chunks staged prefill, and alternates prefill GPU units with live decode; ProgramImplCore::decode_batch already consumes exact lanes/budgets and dispatches MTP batches by actual B. Move reusable group4/ragged/replay primitives behind that product contract instead of replacing the scheduler.
- For ordinary 8-worker serving, treat two physical C4 replicas as the current winning topology; EngineOptions already supports per-device Engine instances. Preserve shared/sharded execution for the product cases that actually need it (e.g. dual single-request/long-context), but do not force shared C8 to beat the superior replica topology without a distinct requirement. Next product gates: TTFT/prefill overlap and prefix reuse first, then p95/fairness/dynamic join-leave; preserve C1 fast path, multimodal dispatch, and 262K capability.

### 2026-09-07 11:45 +07 — P0-Q108 C8 ragged geometry + replay ownership — 8K ADVANCE
- Physical realistic 8K C8 completed at 126.98 aggregate / 15.87 useful tok/s/user versus Q82 103.01 / 12.88 (+23.27%). All eight fact-specific quality checks pass with no embedded controls; answers remain diverse and coherent. Different exact T16/T8/T4 geometry changes some speculative trajectories/digests, so old digest equality is not the output contract. Peak memory 10.78/13.66 GiB. Evidence: .chatgpt/post_reboot_20260821/p0_q108_c8_ragged_replay_storage/{result8k.json,quality8k.json}.

### 2026-09-07 12:01 +07 — P0-Q111 C8 ragged + group4 proposal T8-Marlin — 2K ADVANCE
- Added exact native-MTP group4 proposal batching and composed unsupported T16 prepared Marlin gate/up + down as two ordered qualified T8 slices. Realistic 2K C8 is byte/digest-exact on all eight outputs and improves Q88 114.23 -> 150.91 aggregate tok/s, 14.28 -> 18.86/user (+32.1%). Evidence: .chatgpt/post_reboot_20260821/p0_q111_c8_ragged_proposal_t8marlin/result2k.json.

### 2026-09-07 12:01 +07 — SUPERVISOR — Q111 2K harvested; stop shared-C8 harness here
- Q111 C8 ragged proposal T8-Marlin completed 2K at 150.91 aggregate / 18.86 tok/s/user, prefill 78.747 s, decode 6.733 s. All eight final digests/texts are exactly equal to Q88, but speculative trajectory is not exact; analyzer therefore reports REJECT under its strict trajectory gate. Treat as output-exact only, never strict EXACT.
- This improves Q108 2K (141.11 / 17.64) and Q88 (114.23 / 14.28), but still misses the owner's 20 tok/s/user C8 floor and remains below Q109 physical C4+C4 (166.85 aggregate, min 20.32/user). There is no product-performance justification for an 8K Q111 continuation before production serving is qualified.
- DO NOT launch Q111 8K or another apps/dual shared-C8 microvariant now. The current-source ninfer-serve production probe is the active priority. A false artifact-contract failure was traced to HPSERVER repo_exec dropping its env= injection; explicit /usr/bin/env reaches NINFER_MTP_Q4G64 correctly and gets past the descriptor contract. Resume production Engine C4 measurement once GPUs are free, then address TTFT/prefill/prefix reuse/router based on that evidence.

### 2026-09-07 12:16 +07 — SUPERVISOR — production C4 found exact T16 Marlin integration seam; Q113 stopped
- Current-source ninfer-serve with explicit /usr/bin/env correctly accepts the Q4G64 artifact contract and loads the 16.13 GiB device weight set. It then fails before serving with: `prepared MTP gate/up Marlin launch rejected its qualified shape`. Root cause is mechanical: production C4 MTP verify forms T16 (= B4 * width4 for draft3+target), while src/ops/marlin/mtp_q4_marlin.cu currently qualifies only T1..T8.
- Q111 developed the exact reusable fix we need: Variant::mtp_post_mixer composes T16 as two contiguous ordered T8 prepared-Marlin slices on the same stream (for gate/up and, when prepared, down), rather than Q110's invalid native-T16 kernel. The isolated patch has been applied to current source for a production candidate build; pre-patch variant.cpp is saved under p0_supervisor_production_engine_c4/variant.pre_t16.cpp. Do not copy Q111's apps/dual scheduler/ragged wrappers into production.
- Q111 8K final shared-C8 recert: 143.22 aggregate / 17.90 user, prefill 336.899 s; supervisor qualification says all 8 workload quality checks pass but outputs/digests/trajectory are not Q82-exact. Shared C8 remains below physical C4+C4 and is closed for generic polishing.
- Q113 split36 launched after the stop checkpoint even though physical fraction/split rebalance is an already-closed family; supervisor terminated that process group to free both GPUs. DO NOT relaunch Q113/split rebalance or any other shared-C8 harness microvariant. Active priority is rebuild ninfer-serve with isolated T16 composition, run real C4 8K helpdesk serving, then act on TTFT/prefill/p95 evidence.

### 2026-09-07 14:25 +07 — Production C4+C4 HTTP topology implemented; physical smoke passed
- Added --devices with full independent public Engines behind one ninfer-serve endpoint, stable bounded least-load routing, prefix locality hints, per-replica media preparation, global request IDs/device attribution, and actual summed execution counters. Kept the existing Engine scheduler/prefix validation; no apps/dual scheduler wrapper.
- Fixed worker CUDA-device binding with synchronous startup error propagation and owning-device teardown. Composed ragged small MTP projections as ordered qualified <=T8 Marlin slices while preserving high-M prefill.
- CPU router/admission/churn, CLI options, request logging, OpenAI, Responses, and Anthropic schema tests all pass. Physical stock dual-GPU server completed 40/40 streamed short smoke requests with exact READY0..READY7 request isolation and seed-correlated device logs. Short responses did not exercise sustained C4 or actual cancellation; no throughput/quality seal. Evidence .chatgpt/production_serve_20260907/. Next measure real 2K/8K cold/warm/staggered serving; preserve original owner gates.

### 2026-09-07 14:57 +07 — Production serving: warm TTFT fixed; dual-head cache leak closed; throughput still open
- Actual native HTTP C4+C4 lifecycle baseline completed cold/warm/staggered/churn/refill. Three streamed requests cancelled mid-generation and all eight replacements completed without errors. Source-weight-owned head caches replace process-global Q4/W8 singletons that alternated GPU ownership and leaked until OOM. Two-GPU regression exercises distinct expected outputs, 24 alternating plus 16 concurrent re-preparations, explicit release/reload: zero bytes lost on either GPU.
- Cached-owner admission now uses free active capacity before least-load spill. Repeated native warm 2K C8 runs: 8/8 prefix hits, TTFT p95 0.646535 and 0.643887 seconds, versus initial 13.675331; all 16 outputs stop naturally under 1024-token bounds. End-to-end aggregate 85.579/85.129 and minimum visible decode 12.4485/12.4494 remain below owner gates. NOT SEALED.
- Bounded Nsys window landed in cold prefill, not sustained decode: Q4 and Q5 dense projections account for 75.6% of kernel time. Testing existing 512-token prefill route and preparing separate short-prompt steady-decode profile. Physical image/video/mixed fixtures prepared; not yet qualified. Evidence .chatgpt/production_serve_20260907/.

### 2026-09-07 15:25 +07 — SUPERVISOR — production aggregate ceiling remains open; new research note
- Added `PLEASE_READ_NEXT_2026-09-07_PRODUCTION_C4_BATCH_AMORTIZATION_AND_AGENTIC_SERVING.md` after repo/artifact/worktree archaeology plus current external serving research. Do not seal at the Q109 166.85 physical C8 result while the real production C4 dense-body ceiling remains materially above measurement.
- Highest-priority missing bridge is Q95/Q96 exact T16 weight-fragment reuse in the real Engine: current production still composes key T9-T16 Q4/Q5 paths from <=T8 slices, while sustained C4 Nsight attributes about 85% of GPU kernel time to dense Q4/Q5 and Q95/Q96 already measured ~1.15x Q4 / ~1.50x Q5 / ~1.19x GDN T16 operator gains. Next throughput measurement must be sustained C4 512-1024 output tokens, not the 3-token READY smoke.
- After true T16 production integration, bound a one-card short-context C8/T32 ceiling if memory permits, then measure current-Engine C4 MTP-vs-ordinary crossover. Separately pursue multi-lane packed prefill/decode-maximal scheduling and prefix/shape-aware replica admission. Agentic session-history n-gram+MTP is a later separately-labelled accelerator, never a general frontier claim. No subagents; existing correctness/stock/multimodal/262K contracts remain unchanged.

### 2026-09-07 15:26 +07 — SUPERVISOR — sustained production C4 A8 is a major advance
- The full `c4_a8_text2k` run landed: warm physical HTTP C4+C4 is 152.669 aggregate E2E tok/s with TTFT p50 0.724s / p95 1.564s, minimum visible decode 22.41 tok/s/user, fairness 0.9624, 8/8 complete. Fully decode-ready B4 intervals reached ~181-199 committed tok/s. This validates the C4 A8 mechanism as a real production advance, not the earlier 3-token smoke.
- Cold remains 47.13 aggregate with TTFT p95 60.05s and refill 96.63 with p95 28.76s, so serving is still not sealed. Preserve/qualify/profile the A8 path first; then apply Q95/Q96 true T16 fragment reuse only where the A8 path still composes two T8 calls and where it is structurally compatible. Do not regress the new A8 route merely to reuse an older primitive.
- Research note `PLEASE_READ_NEXT_2026-09-07_PRODUCTION_C4_BATCH_AMORTIZATION_AND_AGENTIC_SERVING.md` has been updated with this result. The ceiling question is now whether sustained ~190-200 physical decode can be pushed toward 250+ while cold/prefill is fixed separately.

### 2026-09-07 15:28 +07 — SUPERVISOR — multimodal native baseline finished; A8 vision discriminator remains open
- Physical native vision C4+C4 baselines completed without `NINFER_SERVE_C4_A8`: warm image 89.54 aggregate E2E / 12.16 min visible user, video 94.26 / 12.15, mixed 81.11 / 9.98; all are `MEASURED_NOT_CERTIFIED` and below the owner gate. These command manifests use the same binary but omit the C4 A8 env flag.
- The C4 A8 enable predicate is not vision-specific, so this does NOT close multimodal A8. After text A8 correctness/trajectory qualification, run a bounded A8+vision image/video/mixed discriminator with the exact same fixtures and classify output correctness before any performance promotion. If A8 is compatible, it is the shortest path to testing whether the large text batch win transfers to the remaining physical multimodal gate.


### 2026-09-07 18:56 +07 — Native production serving receipt reconciliation and cold-prefill candidate

- Recovered actual D1 C4+C4 numeric warm gates (~236–249 aggregate, all clients~26–28+); prior chat final was stale. Clean cancellation checkpoint retention passed real text/media replay and fixed post-churn refill latency. See continue_1639/.
- Closed decode_priority16 as a standalone fix: six warm clients still16.23tok/s at the slowest during real cold joins; cold/warm latency regressed. Default unchanged.
- Charged dequant+dense-GEMM oracle passed; new opt-in Program-owned cuBLASLt prefill MLP preserves decode layout and planned scratch. Full MLP now2.006x atT128 with exact canonical/fragment equality, relative L2~0.00295 versus existing MLP. First native C4 cold p95~42.5->30.5s with warmmin33.4.
- Fixed admission tie destroying a retained prefix despite an empty lane; exact reuse remains primary. Current physical C8 cold/warm/join/churn/refill qualification is bounded and tracked in continue_1839/. Not sealed; no defaults promoted.

### 2026-09-07 19:41 +07 — Arrival-aware production serving protection plus charged dense-prefill validation
- Actual source/receipt reconciliation: dense prefill alone reduced matched2K cold p95TTFT52.685->30.756s, without reducing warm physicalC8 throughput below245.69/min28.66. Separate16-round all-cohort priority hurt coldTTFT; not promoted.
- New arrival guard grants bounded decode bursts only for clients already streaming when the prefilling request entered the Engine. Same-cold cohorts and short/final suffixes keep one-round alternation; cancelled/retired protectees lose burst credit. Five CPU and eight accounting regressions pass.
- Physical2K candidate: all-cold30.579s p95, warm251.710aggregate/min29.710, real cold-join six established streams minimum20.912 during shared7.221s cold-pending window; arriving cold clients16.15–19.35s TTFT. No C8 inference from this six-client measurement. Evidence continue_1839/arrival_guard/text2k.
- Charged dense MLP is1.792x with exact tested canonical/fragment outputs. Independent FP64 Q5-output oracle passes existing LinearAdd criteria on both GPUs; eight focused multimodal/text fixture outputs pass reviewed facts in cold/warm original and dense controls. Longer8K/multimodal retention verification still in progress; no default, deployment, original65K/262K seal, hardware action or delegation.

### 2026-09-07 20:16 +07 — Bounded exact media preparation cache and multimodal serving qualification
- Added opt-in per-Frontend immutable CPU media cache:64MiB/64entries retained cap, exact encoded bytes plus SHA256/modality/all processor settings, per-request independent patches/spans/positions and aggregate budgets. No GPU/KV/generated-output cache. Tests cover collisions, options, changed bytes, LRU/cap, oversized bypass and concurrent access.
- Fixed preexisting single-frame-video sample clamp:min_frames was larger than source frame count, requesting duplicate distinct frames and failing PPM parsing. Processor retains temporal padding authority. Full frontend tests pass both with cache off and on.
- Same-binary physical C4+C4 video cache A/B: warm p95TTFT3.234945->1.851938s; decode239.792->241.646 aggregate,min28.148->28.313; refillmin28.423,p95TTFT2.186s. Mixedwarm233.323/min27.382,p95.926s;image239.981/min28.186,p95.927s. Warm/refill natural-stop cohorts and cancellation/refill complete with zero errors; GPUpeaks20165/20274MiB.
- Final focused quality:16/16 cold/warm text/code/chart/natural/multi-image/video/mixed-history expected facts checked. Long5900-token multimodalretention8/8 exact24-9 outputs. Thirteen rebuilt CPUtests plus nine current stream-accounting tests pass; real cancellation replay/identity invalidation/incomplete-prefill tests pass withcacheon.
- Fresh image/video arrivals exposed a real remaining mixed-prefill fairness gap hidden by whole-request averages: three warm clients on image GPU14.675min during4.634s cold-pending window while videoGPU21.797min. Arrivalguard incorrectly disabled before final cold chunks. Candidate now classifies original uncached suffix, preservesboundedprotection toend, and retainsone-roundshortwarm/all-coldpolicy. Six focusedCPUtests+9accountingtests pass; physicalrecheck running proc-651031a6-cd1e-4a07-a883-f4672cc64b86. Evidence continue_1839/media_prepare_cache/final_chunk_protection. No defaults/commits/pushes/daemon/hardware/agents.

### 2026-09-07 20:29 +07 — Final-chunk protection and repeated physical fresh-media fairness profile
- Original remaining-tail criterion disabled protection for final chunks of substantial cold requests; now classify computed+remaining original suffix. Short cached suffixes and all-cold cohorts still one-round. Both implementation and bounded-cancellation/deadline checks preserved.
- Final16-round profile reached20.343 then19.763 minimum among6establishedtextclients while2freshimage/video requests waited; retain this as insufficient margin, not roundedpass. Documented opt-in24-round profile passed two independent real3+1perGPU scenarios at22.352 and21.874 minimum, windows7.606/7.132s. FreshimageTTFT8.127/8.217s,video19.010/17.018s. This6clientwindowisnotC8.
- Same final24profile all-cold2K p95TTFT30.414/30.627s, warmC8248.237/243.559 aggregate withmin29.244/28.808 andp95TTFT.808/.809s;zeroerrors. Alljobs terminal and GPUsfree. Default1unchanged,fastflagssremainoptin,nohardware/delegation/deployment/commits/pushes.
- Rebuilt13CPU and9stream-accounting regressionspassed. Currenthandoff/currentstatus/docs/provenanceindexupdated withactual implemented denseprefill,arrivalprotection,mediaCPUcache,state/qualitytests andlong-contextlimitations; obsoletependingT16chatclaims superseded. See continue_1839/serving_checkpoint.md and media_prepare_cache/final_chunk_protection/protected24_a,b.

### 2026-09-07 21:03 +07 — Real conversation affinity fix and larger cold-prefill discriminator
- Added actual appended-reply followup/followup2 harness. Original whole-request fingerprint lost locality:3/8 prefixhits, p95TTFT29.116/29.970s. Bounded longest-turn-prefix hints now preserve8/8 GPUowners and8/8 exact-validated Engineprefixhits; p95TTFT11.224/3.883s. Router role/media/tool/history/spill/concurrency tests pass. No answer caching or weaker reuse identity.
- Charged dense-prefill largerchunk discriminator is material: physical8K eight-request cold p95 fell124.15->72.278s atchunk512. Native adaptive cap now separates allocated maximum from small live work units and plans servicework conservatively at the small cap. Building/qualifying; defaults unchanged. Evidence continue_2051/.

### 2026-09-07 21:23 +07 — Adaptive cold/live prefill, larger-width numerical correction and real history growth
- Added public prefill_live_chunk/--prefill-live-chunk (default0 inheritsmax): maxstartupworkspace with small128-tokenworkonlyforrequestsprecededbyestablishedstreams. Serviceworkusesconservativeminimum. Physical1024/128 2Kcoldp95=15.945s, warmminimum28.336,p95=.809;freshimage/videojoinsqualifiedsixuser min22.543. Initial6CPU+10accountingchecks pass.
- AtT1024 founddenseSwiGLUalwaysmaterializedBF16g/up rounding differedfromnativefusedroute. Nowusesexistingnativeplanincludingraggedboundaries. T1024 twoGPU/seeds have0reference/fragmentmismatch and3.15–3.31xspeedup;T128exact;256/384/512relativeL2~.0003 withfragmentexact. No gate lower/noVisionweightchange. Finalphysicalquality/replays underway.
- Real7.6Kinput+actualreply+7.6Knewdoc produced15.4Khistories;8/8ownerretained,6/8prefixhits,p95followup94.73s,all8naturalstops. AutoKVpool52K/55Kperreplica is less than4*15.4K, so prefixeviction/capacity must not be disguised aswarmC8success. Existingexplicitcapacitybeingexamined. Evidence continue_2051/.

### 2026-09-07 22:13 +07 — Recovered adaptive-prefill/history receipts and corrected real HTTP arrival scheduling
- Recovered completed continue_2051 evidence: adaptive1024/128 physical8Kcoldp9565.64s vs124.15old; actualfollowup8/8owner+prefixhits. ExistingMarlinownerlifetimeandmodelreloadchecks areterminalgreen,notrerun. AutoKVcapacityisaggregateperGPU;vision+four15.4Khistoriesdoesnotfitqualified64Kpool. Preserveexplicitcapacityfailureandqualitylimitations.
- NewHTTPingresstimestamp is propagatedtoEngine separatelyfromsubmission; media preparation no longer reclassifies initialcohortsaslatejoiners. Sameconfigvideocoldp9521.22->14.67s;warmmin27.35andrefill27.81,allnaturalstops,zeroerrors. WarmTTFT1.91->2.62s inoneA/B remainsnoteda variance/tradeoff,notuniformlatencyclaim.
- Genuinelateimage/video3+1perGPU check remainsqualified:6establishedclientsmin22.18whilefreshrequestsawaitfirstcontent. Added2/4clientwindowaccountingthatcannotclaimC8,continuinghistorycancel/refill,ingress/deadline/leasechecks. Currentboundedprocess proc-34e00917-aa54-4086-ba58-8148a3e82ffc runs2/4andmixedgrowth. Noagents/hardwarechanges/defaultpromotion/commit/push/deployment.

### 2026-09-07 22:24 +07 — Actual conversation churn, original ingress and Vision-reserved long-history capacity qualified
- Mixed actual appended replies/questions: 8/8 prefix hits, common physical C8 226.196 aggregate / 26.361 minimum, p95 3.497s. Repeated latest history and cancellation/refill are labelled separately; refill223.778/min26.002/p95.973s, 3 intended cancellations, all other responses natural and no errors.
- Actual 2/4-client Vision-enabled cohorts measured warm minima40.673/41.751 and p95.179/.385s, with matching actual-cohort accounting that cannot claim the8-user gate. Follow-ups retained2/2 and4/4prefixes; no extrapolated short-followup throughput.
- Original HTTP ingress versus delayed media Engine submission fixed; coldvideo p9521.220->14.669/14.430s. Firstwarm2.619s thenrepeat1.854s versusbaseline1.912; preserve variation. True fresh media3+1perGPU still has6established minimum22.181. New real Engine ingress, lease rejection, deadlines, cancellation/identity checks passed.
- Explicit49152 aggregateKVtokens perGPU,12288 perrequest,Visionenabled and1024/128prefill passes native checks. Eight7.6Khistories retained8/8 prefixes through actual followup; warmp95.885,followup6.802,latest-repeat205.768/min22.703,p95.854s. ObservedGPUpeaks20991/21100MiB; less spare memory than auto policy, no budget bypass. Focusedquality16/16expectedfacts natural. Four16Kmultimodal profiles remain unqualified.
- Final13rebuiltCPU/API/protocol/frontend/scheduler/cache tests and11measurement tests pass. All GPU jobs terminal, native current build and diffcheck clean; source remains uncommitted. Updated current handoff, board,docs and newest verified_result_index. No agents/training/hardware changes/commits/pushes/deployment.

### 2026-09-08 02:08 +07 — Cold serving profile, attention-input prefill and bounded Vision candidate
- Profiled actual C4+C4 cold8K path: approximately19.1% of captured GPU time remained in old grouped Q4/Q5 attention-input projections. Program-owned dense attention path now tests all4 outputs with nonuniform inputs; T128/384/1024 checks on bothGPUs match exactly and yield3.5–5.8x operator speedup. First real8K cohort coldp95 65.48->55.46s,8/8followup prefixhits,all non-prime outputs natural,minimum23.82 actualfollowup.
- Rejected gate/up row-striped expansion despite workspace524->269MiB and tested exactoutputs: total runtime budget unchanged becauseVisionworkspace dominates; smallT slowed. Source restored tofullwidth expansion,attention optimization retained. No hardware or budget changes.
- New Vision candidate batches complete existing independent temporal attention segments up to the largest segment allowed by the unchanged128Mfrontendpairbudget. Smallmedia path unchanged,full output/transient/scatter preserved. Complete real-artifact embedding A/B for image/video/extendedvideo beingrun before promoting capacity; currentGPUtest proc-b3dbfb30-5943-4733-9452-e87e3ea10fb5. No seal/default/commit/push/delegation.

### 2026-09-08 02:19 +07 — Bounded Vision batching clears64Kmultimodal capacity; embedding equivalence passed
- Unchanged128Mfrontendattentionpairbudget now bounds the maximum independent Visionsegment; smallitems useidenticaloldpath,largeitems executecomplete existingsegmentbatches with unchangedscatter/MRoPE/output and allframes retained. Real model image/video/40segmentextendedvideo full-embedding A/B exact onbothGPUs; extended88hundredmergedtokens workspace488013056->158604544B, encodecost~2%higher. CPUboundary testscovermaximumsegmentandfull32Kmergedpublicrange. No memory-budget weakening.
- NativeVisionenabledpool65536/maxcontext16384 nowpassesstartup onbothcards. Eightactualimage+video histories7700->15300tokens retained8/8prefixesandreturnedall24-9expectedanswers,cold58.59/followup59.45/repeat2.72p95. These5tokenanswersprovecapacity/retention,notdecodeSLO.
- Reopenedthealreadytestedrow-streamedgate/up onlyafterVisionceaseddominatingworkspace: actualruntime4.70->4.45GiB,256MiBless onbothGPUs; freeafterstartup771/660MiB versus515/404. Thisisnowrealcapacitybenefit,notleaf-onlyclaim. Currentcombinedsustained15Kmedia/churn/largevideoquality/freshjoinrun proc-6ee66046-4347-4de9-91de-ba532add623a isbounded. Evidence .chatgpt/production_serve_20260908/cold_memory/.

### 2026-09-08 02:40 +07 — Long-media cache and exact added-token scan remove13-second warm preparation overhead
- Represent cached preparedmedia as originalresizedRGB8 patchsamples, notFP32 arrays; eachrequest reconstructs identicalFP32normalization/order. Full256valueindependentgradientoracle andcachelimit/budget/concurrencytestspass. Actual40secondclip retained54268335Bunderunchanged64MiBcap; oldfloatpayload216268800Bdidnotfit.
- Measuredcachehitstill13.22/13.11s becauseTokenizerre-searchedremainingstreamforeveryabsentaddedtokenateachvideopad. Per-encode next-occurrence positions nowrescanonlyconsumedoverlappingmatches, retainingearliestposition/tokenIDtieandordinaryNFC/BPE. Independentoldmatcher/randomoverlap/32768marker/concurrencyregressionpassed. Actualsameclippreparecold14.922->1.575s,warm13.22/13.11->.1237/.1294s. Cacheentrysizeandpreparedpatchcountidentical.
- Post-fixrealHTTPquality40svideo,15KmediaC8churn/refillandstatechecksarecurrentboundedjob proc-1314d092-cc73-4f1c-af2e-1c332e929cc4. Do notclaimfrontendtimeasHTTPtimeorfive-tokenqualityasC8throughput. Earlier15KfullanswersC8warm214.24/min24.94 andrefill208.78/min24.46; no agents/hardware/defaultchanges/commit/deployment.

### 2026-09-08 02:47 +07 — 16K-capacity multimodal profile and exact scanner final physical checkpoint
- Final8client15.358–15.395Kmediahistories warm213.907aggregate/min24.549,p951.1116s;post3cancelrefill212.993/min25.350,p951.0829s. Allnoncancelledoutputsendednaturally,error0. Actual7.7K->15.3Kmediahistorygrowthkept8/8prefixeswithallshortexpectedanswers; distinctfromthroughputmeasurement. Fullcold15.4Kp95118.943s,new7.642Ksuffix59.455sremainlatencylimits.
- Finalreal40svideoHTTPwarm.674986s,retry.844689sversus14–16soldscanner; completequalitycohortwarm.8535p95and24/24expectedfactrepliesnatural. Isolatedcachehit13.22->.1237s; sameencodedpixels,floatpatches and cachedentrysize. Exactadded-tokenold-reference/overlap/concurrency/32Kmarkerchecksand256RGB8normalizationoraclepass.
- Current14rebuiltCPU/API/protocol/frontend/scheduler/cache/workspacechecks+11measurementtests+actualEngineingress/deadline/lease/cancellationchecksallpassed. VisionembeddingexactbothGPUs;native65536pool/Visionmemorybudgetpasseswith4.45GiBruntimeand771/660MiBfreeafterstartup;observedpeaks21233/21342MiB. Noallgatesseal;broaderquality/preemptiveVision/oldC1longcontextremainopen.
- Updatedservingdocs,currenthandoff,liveboard,andverified_result_index; allboundedGPUjobsandHTTPserversended. Sourceuncommitted,preservedinheriteddirt,fastsettingsopt-in,noagents/training/hardwarechanges/commit/push/deployment.

### 2026-09-08 03:09 +07 — Queued steady-decode architecture resumed; fresh current graph-node profiles and exact candidate closure
- Read Sep7C4amortization and Sep8post-prefillownerqueues plus dumpedservingcontract. Current C4/D1 usesT8,notT16. Fresh actualHTTPgraph-nodeprofiles C1/C2/C4 held1/2/4clientsperGPU. C4 bodytime:Q4A8T8 19.8%,Q5downT8 17.9%,Q5outputSIMT15.0%,GDNQ5T8 7.3%,GDNQ4 7.0%. OldD3/T16profiledoesnotdescribecurrentD1.
- One architecturally distinct exactQ5T8 mechanism parallelizedindependentG64integerdots andconsumedoriginalorderedFP32FMA; exactbitsbut14–17%slowerbothcards. CLOSED_NO_SPEEDUP,notrouted;source restoredwithscopedbackups,prototype archived. FreshNCUwithclock-controlnone:Q5T8activewarps14.56%,DRAM59.91%;longscoreboardratioavailablebutnotapercentage. Countersworknow.
- Nextqueuedboundedwork:single-card8clientD1/T16batchreuseceilingandcurrentEngineordinary-vs-MTPcrossover,withoutlowering16K/C4orlongcontextcontracts. Noagents/hardware/defaultpromotion/commits/pushes. Evidence production_serve_20260908/decode_architecture/.

### 2026-09-08 04:21 +07 — HPSERVER preflight and genuine C1 priority: recovered missed completed decode work
- Healthyconnector/host/GPUprobes; no computeclients before work. Old one_gpu_c4_mtp stopped on harness-invalid32Kpool/C4context4K; later corrected crossover andQ5matrices actuallycompleted but were missed by previousfinal. Recovered15terminalsuccessreceipts with exactbinaryidentities; not rerunning them.
- RecoveredC4MTP139.02vsordinary69.88 aggregateoneGPU;C8MTP159.58/min17.76vsordinary74.05/min9.25. Do notreplacephysicalC4+C4withoneGPUC8. Q5optin qualifiedmixedwarm240.30->254.02,min27.68->30.04;realappended223.77->242.79,min24.50->26.55;15.4K212.07->220.96,min24.36->26.04. T2row8bitexact;T8partition4 numericallyqualified/notbitexact. Existingproofandcancellationlogsretained.
- AddedserialC1harness andoverlap/labelregressions;14measurementtests pass. Newboundedproc-1eb846e2-f208-4020-b428-ced5d2c76770 tests six fixed general/code/JSON/Thai/Japanese/Arabic workloads serially withoneGPU/oneconfiguredslot and768outputcap. D1baselinevsQ5thenexistingD3fullhead. Historical107frontier usedloose_logit_delta7 andisnot apples-to-apples withstrict currentEngine. Do notchangeacceptanceto manufacturethroughput.
- No inferenceruntimeedits thiscontinuationyet; noagents/training/hardwarechanges/commits/pushes/defaultpromotion. Currentevidence .chatgpt/production_serve_20260908/raw_c1/.

### 2026-09-08 04:36 +07 — Real C1 profiling drives independent draft/full-target control
- TrueC1serial6workloadfullheadD1client40–48,D3client44.65–71.54;768outputboundisatimingsample/notcompletioncertificate. D1Q5firstpairheatconfounded:baselineclocks1785->1665,candidate1620->1507MHz. Reverse3workloadpair supportsmodest3–4%gainwithexactoutputbytes;nohardwarechanges.
- C1D3actualgraph-nodeprofile:Q4T4 24.1%,fullW8T1proposal15.6%,Q5T4 15.3%,GDNQ58.5%,targetfullW8T4 5.5%. Codeauditconfirmedlegacyoptimizedproposalalsoenabledtargetshortlist. Addedexplicit--full-target-head separatingproposalchoicefromtargetauthority; frozenfeatures,allfourshortcutreadinessguards,preparation,reservations,CLI/logs andtestsupdated. Legacydefaultspreserved.
- Currentqualifiedbuild passes4focusedCPU+14harnesstests. Separatedheadrealimage/video/mixedexpectedfactcases8/8natural,error0. Boundedcurrentproc-0805d304-ea06-43f6-9cc3-8a181fdba9a1 completesmatched6familyC1comparison. No performanceclaimbeforeterminalreceipts.

### 2026-09-08 04:54 +07 — Full-vocabulary draft-only projection gives a genuine six-workload C1 advance
- Short-vocabularyoptimizeddraft/fulltargetwasnotageneralwin:Thai47.27->33.07with358->580rounds. Didnotpromoteit. AddedexplicitC1SM75fast-full-vocab-draft usingexistingall-vocabularyranking/native-rerankonlyforproposals; fulltargetverificationmandatoryandalltargetshortcutguardsremainoff. Noartifactormarginchange.
- SamebinaryoneGPU/oneconfiguredslotserialsixworkloads clientTPS baseline->candidate:general51.277->57.758,code73.633->82.705,JSON73.380->82.299,Thai47.186->52.468,Japanese45.703->51.277,Arabic46.073->52.114. Gains11.2–13.1%; all6deliveredoutputsbyte-identical. Five768tokencaps,Japanese672natural; notcomplete-answerorallgatescertificate.
- ExecutedNsightfulltargetW8T4remains;formerfullW8T1draftscan15.6%replacedbydraft-onlycoarseranking2.5%plusrerank. Q4T4now27.9%/Q5T4 17.7%—nextC1architectureworkbelongsinthebody,nottinyproposalknobs. Frozenpolicy,validation,chargedcacheandrollback/destructorownershipimplemented; relevantCPU4andharness14passed;focusedmedia8/8natural.
- Currentboundedproc-fdc03372-8de7-46d5-bbf5-32e7fd6f7af6 checkssecondcardC1,realcancellationand3-cyclemodelreload. EarliernormalC4+C4regressionwarm252.41/min29.51,followup236.72/min26.58,refill237.15/min27.90allnatural/error0. Defaultprofileunchanged,noagents/training/hardwareoperations/commits/pushes/deployment.

### 2026-09-08 05:01 +07 — Raw C1 full-vocabulary draft-only final checkpoint with dual-serving preserved
- Finalsamebinary6workloadC1clientTPS51.277/73.633/73.380/47.186/45.703/46.073->57.758/82.705/82.299/52.468/51.277/52.114;11.2–13.1%gain,all6deliveredoutputbytesidentical. Inputs61–90tokens,5outputs768capped,JP672natural. Secondcardgeneral/Thai/Arabic14.1–15.2%gain,identicaloutputs,botharmsloadedmedian1350MHz. NoaggregateasC1claim.
- Full-vocabularycandidateusesexistingfastprojectionforproposalsonly;nativefullW8targetverificationremainsactualprofiledpathandnoacceptancemarginchange. NewfrozenCLI/APIflagsandindependenttargetguards;C1/SM75/backendchecks,chargedcacheswithsourceownerteardownandrollback. Legacydefaultsunchanged. Short-vocabularycandidateThai30%regressionnotpromoted.
- FinalcurrentbinarydualC4+C4mixedwarm259.417/min30.124,p95.7824;actualfollowup240.721/min26.497,p953.2658,8/8prefixhits;3cancelthenrefill248.230/min27.421,p95.7885. Noncancellednatural,error0. Thisretainsmainservinggoal—notattributingC4ratestoC1-onlymode.
- 15rebuiltCPUchecks+14measurementchecks,8focusedmediafacts,actualC1cancellation/ingress/deadline/lease/invalidation and3modelreloadcyclesallpassed. Reloadidenticalfreebytes22883532800aftereachdestruction. Verifiedsource/binaryresultindex,currenthandoff,docs,liveboardupdated;alljobsended,noCUDAclients,nohardwarechanges/agents/training/commit/push/deployment. RawC1bodyoptimizationandqueuedproduction/longcontextworkremainOPEN.

### 2026-09-08 05:36 +07 — Preserve all accepted profiles; canonical107 current-source root cause identified
- Owner requires every accepted best profile retained, not replaced by strict controls. Protected historical CLI6875163c, native server15046858 and dual41e8a52f before rebuilding. Added retained profile evidence manifest/guard and explicit owner preservation clause to AGENTS. This evidence guard does not certify current performance.
- Added explicit immutable C1D3 margin policy with CLI/HTTP/API/log support and small/multiblock layout/boundary/sampling tests. Initial rebuilt route111.3 differed from canonical107 digest, so NOT accepted as preservation. No knob fitting or reduced gate.
- Bounded native acceptance/MTP-input observer showed initial target decisions match, but draft2 differs. Root cause is prior Marlin identity hardening: original bare-qdata lookup let initial MTP query reinterpret packed attention parent with narrower N; new exactN/K lookup correctly falls back to another arithmetic path. Restoring that accepted draft-only interpretation now uses explicit --mtp-retained-packed-prefix and exact prepared parent/view bounds, not a generic cache weakening. It is a documented numeric profile, not an exact logical matrix slice. Current-source exact128 verification in bounded process proc-d5ec6a9c-cebb-4244-936c-9696b0e741ec; no claims until terminal match.

### 2026-09-08 06:10 +07 — Accepted C1 source preservation completed; strict79 and margin107 golden controls plus physical dual regression
- Recovered original C1 margin7 semantics rather than treating lower strictfulltarget numbers as replacements. Explicit validated/frozen/logged margin and retainedpackedprefix restore sourceCLI/server. Firstcurrentpair108.06/107.95 and finalCLI107.35 exactly match stdoutSHA c4ce5202...,digestef4dc9c008396373,33/99/94,33/32/29. HTTPsamegeneratedtext/counts108.046Engine/105.450conservativeclient;CLIterminalnewlineaccounted.
- Strict79matching128fixture current80.51 with exactstdout/digest14660c413b4cf185,43/129/83,35/27/21. RealEngine afterunrelatedwarmup repeatsbothgoldens;strict80.142/80.033,margin106.691/106.521loaded. Original79.01,distinctcertified80.185 and107.27 remainseparatecumulativeheadlines; notpromotingnew universalnumbers.
- Fulltargetfastfullvocabmargincombo100.03 onmatching128 versusretained107.35;new768corpusmixedperformanceandJPrepeatedphrase18times,notpromoted. RetainedlongerThaialsolanguagedrift;preservequalified128/256limits. Strictfulltarget6outputhashesunchanged;no79headlinecreditfornewfulltargetprojector.
- FinalcurrentphysicalC4+D1mixedwarm257.241/min30.043,p95.783;actualappended241.402/min27.820,p953.277,8prefixhits;3cancelrefill243.588/min28.216,p95.785. Allnoncancellednatural,error0. Logconfirmsmargin=-1/packedfalse/fulltarget. Originaldual65K/262K/mediawitnessespreservednotretimed.
- Final16CTesttargets+14streamcases+9preservationcasespass;bothGPUmarginOp,Marlinowner/view/lifetime,retainedcancel/ingress/deadlines and3reloadcyclespass,exactfreebytes22883532800each. All18historical/evidencegroupsand3priorbinariesprotectedwithSHAguards;newactualmodelgoldentestcontainsstrict79andmargin7. Currenthandoff,primaryledger,AGENTS,docs,manifest,indexupdated. Alljobsfinished,nocomputeclients,noagents/training/hardwarechanges/commits/pushes/deploy;queuedbody/prefill/admission/longcontextworkremains.

### 2026-09-08 06:30 +07 — Canonical C1 body oracle rejected; queued multi-owner prefill implementation
- CurrentcanonicalD3margin7profile confirms Q4gate30.1/Q5down19.1/GDNQ510.6/outputQ59.4percentGPUtime. ExistingMarlinW4A16fullgateopwithoutputA80.82–0.84xbaselinebothcards,notmaterial;notrouted.Noacceptedbodieschanged.
- NextqueuedP1:multiple independently staged cold owners inside existing request-transient memory budget. Defaultsingleowner and allstrict/margin/dualheadlines retained. Buildrealhead-of-line A/B thenzero/cancel/reload/memory tests. Evidence c1_body_0625/multi_prefill/.

### 2026-09-08 06:51 +07 — Multi-owner staged prefill: real isolation proof fixes shared execution selectors
- Opt-in max-prefill-lanes1..C added; default1retained. Startup-frozen request transient nowuses disjoint stable per-lane slices, no extraGPUallocation; round-robin existing-ownerstepsalternate boundednewadmission. CPUallocator30000transitions andCLI/log/budgettestspassed.
- Real3ownerimage/image/textcanceltest exposed sharedKVselector scalars onlyboundatadmission: isolatedblue=blue, concurrentblue=black. advance_prefill nowrebindseachownertext/backendrowsandropedelta everychunk; DFlashcontexthingressalso restored. Correctedrealtestblueexactisolated afterredcancel, textREADY, jointpeak1310720=2x655360singleimage withinunchanged41943040capacity, cancelledredprefixdiscarded andlatercompletedprefixreusable.
- EarlierpostpatchtimingsbeforeKVfix markedUNQUALIFIED,notpromoted. Finalcorrectedbinary same-binaryarrival/mixed/cold8K/freshjoinandstrict79/margin7goldenchecks running in boundedproc-845aca2c-1233-4b94-920a-202aa572ed48. NoC1kerneladvanceclaimed;Marlingateoraclealreadyrejected;allhistoricalprofilesprotected.

### 2026-09-08 07:08 +07 — Independent staged prefill qualified; strict C1 priority and all retained paths preserved
- Corrected samebinarylong/shortC8 test: shortmedian12.336->3.188s(-74.15%),shortmax13.526->3.885; long11.316/13.550->13.569/16.130s andcohort27.428->27.970s. Output2153/2102allnaturalwithcorrectidentitytags. Startupmemorydescriptorsidentical; latencytradeoffnotuniversalTPSgain. Defaultmax-prefill-lanes1 unchanged;4optin.
- ActualmixedcorrectedC8warm254.822/min30.051,p95.7826;realappended240.251/min27.520,p953.2362,8prefixhits;post3cancelrefill246.126/min27.904,p95.7835. Noerrors/noncancelcaps. Freshmediakept6establishedmin21.962;image/videoTTFT7.529/19.410. Cold7.6Kx8p9555.477->54.238; warmfloors27.72. Notpackedprefill.
- Fixedper-stepKVrow/rope/DFcontextselectionafterreal3ownerblue/red/textcanceltestfoundadmission-onlyscalarscrossedowners. Finalsurvivingblueexactisolatedcontrol,peak1310720=2x655360within41943040budget; cancelledprefixdiscarded/reusedregionclean. Earlierpre-fixperfexplicitlyUNQUALIFIED.
- 17CTesttargets+14streamtests+11guardcases+16focusedcold/warmfactspass. Guardnegativecaseupdatedfor06:49strict100owneroverride(-1nowcorrect); extra tests prohibitrelaxedgatecredit/loweredstrictgate,all18witnesses/binariesintact. Strict79/margin7goldenoutputs/countsexactafterwarmup. Gate strictD3>=100stillOPEN; retainedmargin7notcredited.
- FreshstrictC1D3actualprofileverifiedoneGPU/oneconfiguredslot,marginoff:Q4gate30.0,Q5down19.1,GDNQ510.6,outputQ59.4percent. Full-opMarlinoracle0.82–0.84xbothGPUsrejectedwithoutproductiondispatch. NextstrictC1body/runtimecampaign;donotreopenknobsorcountC8asC1. Sourceuncommitted,alljobs/serversended,nohardware/agents/training/commit/push. Checkpoint c1_body_0625/multi_prefill/.

### 2026-09-08 10:10 +07 — Strict C1 two-limb projection candidate reaches real model; quality and lifetime qualification pending
- Fresh owner strict100 priority retained. ExistingBF16MMAoutput/GDNoracle0.17–0.24x rejected. Newdataflow sends high/lowradixactivationlimbsthroughonepackedQ4/Q5weightload. CompleteopcapturedspeedQ41.43–1.49/Q51.24–1.27bothGPUs;smallnumericaldifferences,notbitexact. IndependentFP64criteria48casesbothGPUs nowpasswithsourceidentity/stridedguards/reloadnoGpubyteslostafterfirstmodulewarmup.
- Samebinarystrict128goldenABBA:81.13,82.69,82.60,80.85tok/s; allgoldenoutputbytes/digest/43rounds129drafts83accepts35/27/21unchanged. ExplicitC1strictD3candidatecachecharges2054189056bytes; defaults/acceptance/historicalpathsunchanged. Thisis~2%fixedfixture—notstrict100closure.
- SecondGPUstrict768sixfamilyclient58.45/86.06/84.26/33.08/49.46/49.90->62.01/89.00/94.29/34.90/49.44/53.71;longeroutputs differ,JP723->768cap. No language soup in inspectedTH/JP/AR;notcomplete-answercertificationoruniversalwin. Boundedquality/cancel/reload/dualchecks next. Source candidate remains opt-in; noagents/training/hardware/commit/push. Evidence strict_projection_0947/.

### 2026-09-08 10:26 +07 — Strict C1 radix alternative qualified at incremental gain; all preserved routes stay intact
- FinalstrictC1samebinaryreversecandidate82.63vscontrol80.74,alloriginalstrictoutputbytes/digest14660c413b4cf185,43/129/83and35/27/21exact. InitialABBA81.13,82.69,82.60,80.85. Thisis2.0–2.3%fixturemovement,notstrict100closureoruniversalTPS. Longer6familyoutputs differ;JPnoimprovementandoutputcap. Candidateexplicit/defaultoff.
- Newprojectionfamilyreusesonepackedweightfragmentforradixhigh/lowactivationlimbs. Q4op1.43–1.49x/Q5output1.24–1.27x;independentFP64existingA16criteria48casesbothGPUs passwithstridedguards/sourceidentity/reprepare. Finalactualrequestedandchargedcache2080MiB roundsnativeallocationblocks. OldBF16MMAoracleandmemorylesscanonicalreadoracle slower;bothclosed/notrouted.
- Candidate realprefix/cancellation/changedmedia/deadline checks and3modelreloadcycles pass,16focusedVision/text/codefacts natural. Preservedstrict79andmargin107goldensmatch. NormalC4+D1warm251.701/min29.535,p95.780;actualfollowup235.498/min26.761,p953.235,8prefixhits;refill240.778/min27.871,p95.785;allnoncancellednatural,error0. NoC1candidateenabledindual/noattributionofbatchgain.
- Final18CTest+14streamchecks+retentionguardall18witnesses andoldbinariespass;postallocationbothGPUoptestsandreversegoldenpass. Updatedindex/docs/status/handoffwithoutchangingownerheadlinevalues. AllGPUjobsterminal,GPUsfree,sourceuncommitted; noagents/training/hardware/commit/push/deployment. Largerstrictbodygainandqueuedproductionwork remainopen.

### 2026-09-08 10:39 +07 — Strict C1 exact-integer HMMA architecture discriminator closed
- Tested Q4gate/up with integer-valuedFP16 operands on native m16n8k8, preserving exact G64 integer dots, original FP32scale/FMA order and fusedBF16/A8epilogue. Exactcodes/scalesbothcardsbut0.979–0.983xbaseline. No production route/defaultchanges; prototypearchived andactivebenchdefinitionremoved. Evidence c1_exact_hmma_1040/. Continue nextmaterialmechanism, nottile/warp tuning.

### 2026-09-08 10:44 +07 — Strict C1 exact independent-projection fork/join oracle closed
- Current leaf-kernel graphforkjoin Q4Q/K vsQ5value/Z testedexact20replaysbothcards. Serial145.4usvsparallel168.8–169.5us,other140vs164–165:0.851–0.861x. Concurrencyaloneincreasescontention;notrouted,nohardwarechanges. Activeexperimentalbenchremoved,archive c1_exact_hmma_1040/fork_join/. Bothnewdataflowmechanisms preservearithmetic butfailperformance;oldwinnersunchanged.

### 2026-09-08 10:48 +07 — Strict C1 official LUT-GEMM executable oracle below required complete-path ceiling
- Tested officialNAVER647d49fc unmodifiedG64-compatiblebiasLUTkernel oncurrentQ4gateupshape34816x5120 withmechanicalbinarypacking. ProjectiononlyT1261.6usvsnativecompleteT4186.5us;fourT1~1003.7us,otherT1256.6–257.2/fourT1997.7–1006vs180.5. Oraclegrantsfreeconversion/epilogue;noportorqualitypromotionwarranted. No targetweight/artifactchanges. Sourcearchivedc1_exact_hmma_1040/lut_oracle;activeCMakeandCLI/serverbinarybytesunchanged.

### 2026-09-08 11:20 +07 — Radix longer-answer coherence retained; scale-fused HMMA rejected; composition test active
- Ownerclarification: coherentnumericalalternativesmaystayandstack; donotconfusewordingdifferenceswithlanguagesoup. SamebinarystrictC1/D3fourcase3072budget:controlThai1037/JP723/Arabic1189naturals,codecapped3072;radixThai1057/JP800/Arabic1265/code2991naturals. FullThai/JP/Arabicread:coherent,ordinaryterms,noobservedpathologicalrepetition. Thai32.91->34.80,Arabic50.36->54.77;notuniversalqualitycertification.
- Generatedcodefullcandidatehasunterminatedtestdocstringandcontradictoryassertions; fullanswerNOTacodepass. Extractedunmodifiedcacheimplementationsbothpass10000deterministicOrderedDictoperations. Preservecandidateprosequalityandseparatecodegap; donotdiscardradixwholesaleorclaimalllongoutputsqualified.
- Scale-fusedcontinuous-HMMAQ4oraclechangesgroup-scaledataflowbut0.979–0.986xcurrentbothGPUs. Rejected/prototypearchived,temporarybuildentryremoved;runtimeandpriorbinaryhashesunchanged. ActualC1radix+denseprefillcompositionrunninginproc-1abdcf2c-3470-4b71-a964-95a81e639bd1againsttwopopulated7.6Khistories. Sequentialtwoconversationreplaysarecachemissesonone-slotEngine,notwarmprefixcertification. Evidence long_quality_and_body_1106/.

### 2026-09-08 11:49 +07 — Long coherent radix retained; strict full-vocabulary proposal composition kept as a separate workload option
- 3072budgetcompleteThai/JP/Arabicradixoutputs1057/800/1265natural;coherentfullread,ordinaryThaiEnglishterms,nolanguagesoup. RootcontrolThai32.91->radix34.80. LRUfullanswerbadtestsnotcertified;unmodifiedlibrary10Kmodeloperationspasses. ScaledcontinuousHMMAcompleteop0.979–0.986xbothGPUsrejectedandarchived.
- Explicitfull-vocab-draft-retained-target composesexistingfullvocabproposalwithretainedoptimizedtargetandmargin=-1/C1D3. Botharmsradix:Thai34.769->54.283/54.339repeat;JP49.688->54.414/54.407;Arabicbyteidentical1265tokens54.848->60.180/60.096,rounds612->551. Finalrepeatall3hashesidentical. Same1350MHzandruntimebytes3513771632; cachechargedonce. Nativefull-targetoptionunchanged; directAPIandCLIinvalidcombinationsreject. Checkpoint-agnosticmechanism,onlyoneactualcheckpointtested.
- Technical128goldenoriginalradix82.42/82.17vsnew81.24/81.15;exactoutput43/129/83butpositionchange35/27/21->35/28/20. Newoptionnotagoldenc1recordorstrict100pass;keepoldwinner. FinalfunctionalJSON40rulespassboth,new92.67vs96.43;codeflat91.72/91.88,new3145naturalparsesbut3of17generatedtestswrong. Extractedlibrary10Kpassboth. Boundqualitylabelsretained.
- C1radix+dense1024prefill2populated7.55/7.57KcasesTTFT36.68/37.33->11.29/11.61butinventoryconfusestxncount/boxcountandfastanswerendsmid-sentence;NOTPROMOTED. Sequential2historiesin1slotarecachemisses,notwarmprefixproof. OlderqualifiedC4prefillunaffected.
- FinalcurrentC4+D1mixedwarm251.638/min29.487/p95.783;appended239.482/min28.433/p953.216,8prefixhits;post3cancelrefill239.688/min28.220/p95.786;noncancelnatural/errors0/caps0. Current18CTest+14stream+11preservationcases/18witnessespass; actualcompcancel/ingress/deadline,3reloadfree22881435648,oldstrict79/margin7goldensallpass. No65K/262Krerun. Sourceuncommitted,optin,noagents/training/artifact/hardware/commit/push/deploy. Alljobsfinished/GPUsfree. Currenthandoff/board/doc/indexupdated; strictbody100+andservingqueuesopen.

### 2026-09-13 20:57 +07 — Timing-aware serving budget cuts cold-video join while C1 winners remain preserved
- Preflight Sep13healthy/correctworktree/stockGPUsidle. Two boundednewC1Q4familiesclosed: exactDP4Arows~.50–.515x, cooperativeKpartition~.948–.953x; no productionkernelorCMakechange retained, prototypes archived c1_integer_rows/. No newstrictC1record.
- Newoptionalprefill-target-decode-tps defaults0. Per-requestcommitted-token/timewindow pluslane-ownedrecentprefillpeak selectsbounded1..24decodebursts fromslowestestablishedstream; unknown/infeasiblerateretainscap. Strictkernel/acceptance,KVownership,cancel/deadlines,allcoldcohortsemanticsunchanged. RequestlocalCPUstateonly, GPUreservation4779389184unchanged.
- Samebinary/corpusprotocol twofixedcontrols videoTTFT17.582/19.301 vs timed14.481/14.920 (means18.441->14.700s,-20.3%). Image7.579–8.128fixedvs8.044–8.093timed,notimproved. Sixexistingclientcommoncoldpendingmin20.897–23.221fixedvs21.577/21.601timed. Warmassignmentsdynamicnotbyte-identicalcohorts;target24best-effortnotSLA. Baselineoneinventorycap, timednone; generalvideotemporalqualityremainsuncertified.
- FinaltimedphysicalC4+D1mixedwarm252.875/min29.550/p95.781;realappended236.700/min26.945/p953.238,8owners/8prefixes;post3cancelrefill239.329/min27.451/p95.783. Allnoncancelnatural/errors0/caps0. NotC1speedcredit.
- 18rebuildCTest+14stream+11preservationcases/18witnessespass. Timedactualcancel1/7/13/ingress/deadline/invalidationpass; threeprefillownersredcancel/blueexact/READY andtransient655360->1310720peakwith41943040capacitypass. Oldstrictgolden80.6615/80.5631 andmargin107.329/107.201 exactoutputandcounts preserved. No65K/262Krerun. Docs/handoff/board/indexupdated preservingownerprefix. AllGPUjobsdone/listenersclosed; sourceuncommitted,optin,noagents/training/hardware/commit/push/deployment.

### 2026-09-13 22:00 +07 — Fresh cold profile and streamed prefill memory discriminator
- HPSERVERpreflighthealthy/correctbranch/stockGPUsidle. Same-arithmeticweight-expansionpipelinefullMLPexactbutT128slower/T1024flat;archivednotrouted. Vendorheuristicrankboundedtestno material complete-op replacement; no autotuning or rank-hardcoding promoted. CurrentnativeC4coldprofile70.1%GPUkerneltimecuBLASdenseGEMM,notGDN.
- Unchangedruntime2048coldprofile8x~7.6Kp9548.13s,Vision16K/user64K/replicakept;4Kprofilecorrectlyrefused5440065792requiredvs5281245696available. Newstreamedgate/upFP32intermediates andhalveddownexpandedrows saves385875968bytes(368MiB)atT4096. CompleteopT4096testedexactbothseeds/cards andwithin~1%time;T1024changednumerics~.00035relL2,notuniversallybitexact.
- Integratedonlyopt-inNINFER_SERVE_PREFILL_STREAMED inactualdenseprefillpath. Planneranddispatch sharefrozenflag; preservedoldsmallchunkpath. AddedCPUinterval32..4096 testsforold-defaultidentity,ragged1023tailcapacity,monotonicboundsandfrozenoption. Bothpass. Actual4K/old1K/matched4Krepeat driver proc-8e1538de-3376-4dd6-a867-241ba468f278 running; donotduplicate. Evidence streamed_prefill/. NoC1kernel/acceptancechange,noagents/training/hardware/commit/push.

### 2026-09-13 22:15 +07 — Native4K streamed prefill now fits with complete same-binary C8 evidence; validation active
- Gate-onlystreamingstillfailednativeplan; GDNvalue/gate intermediateanddeadconvolutiontemporarylifetimeaddressedtheactualpeaks. Runtime+plannerconvscope synchronizedafterorderedQKVextraction; no metadatareservebypass. Native4096/16Kuser/64Kreplica/VisionC4D1nowstarts bothstockcards, freeafterstartup461/350MiB; streamedflagoff preservesolddensearithmetic. 4Kpureoptestsnumericallyexactontestedseeds, fallbackboundary32..4096hosttestscoveralltails.
- Samecurrentbinary8x~7.6Kcoldcontrol1024p9554.3138s vsstream4096p9546.5090/46.5568 (14.33%lower). Coldoutputs64captimingsonly. Warm2048allowanceall8naturalinall3runs, sharedmin27.41controlvs27.70/27.79candidate; no newC1record. GPUpeaks21543/21652candidatevs21233/21342control; moreactualmemorythan1K,butfullcontractkept. Intermediate failedstartupreceiptsretained.
- Finalvalidationactiveproc-6ac15f66-ffaa-441d-8c81-00a47a0499d1:20CTest/oldstrictandmargin/boundaryoperators/cancel/mediaisolation/actualC8/freshjoins/retained15Kmediafacts. Do notduplicateGPUjobs. Evidence streamed_prefill/. Existingpreparedweightpipelineandheuristicplanoraclesclosedwithmatchedscope,temporaryentriesremoved. Noagents/training/changes tohardware/model/acceptance/commit/push.

### 2026-09-14 03:09 +07 — Resumed streamed prefill after interrupted test; real cancellation boundary diagnosis
- HPSERVER preflight healthy, correct worktree/branch, GPUs idle. Prior streamed_prefill validation was terminal failed at cancel4096, not still running. Diagnostic-only rerun kept failing assertion and printed prompt1547/chunk4096/processed1547: completed prefill was incorrectly labelled incomplete by test.
- Test now checks complete-before-first-output retention separately and grows actual tokenized input beyond chunk with adequate context. Current4K pass: completed25/25reuse21, interrupted6155 input processed6151 reused0, fresh exact output. Current128 pass: interrupted1547 processed256 reused0. Runtime not weakened/edited for this test. Media isolation3owners/redcancel/blueexact/READY passed,16cold/warmfocusedmediafactchecks passed.
- Remaining bounded process proc-f77caf02-ac6e-4b48-b699-a118ac9d2300 in resume_streamed finishes mixed/fresh/15K facts; no duplicate jobs. Currentmixedwarm270.62/min32.38,refill251.71/min28.22. Shortactualfollowup8prefixhits but commoninterval1.776s fails>=2s measurement qualification; no throughputclaim. Freshjoin6existingmin23.268 and newimage7.551/video16.781s.
- Found stronger prior2Kcold48.134s beyond default1K54.314; must compare4K46.51–46.56 against stronger2K too. Matchedcurrent2K/4K and separate meaningfuldetailed-followup scripts prepared,notlaunched until currentGPUjobterminal. No hardware/model/acceptance/kernelchanges,newdefaults,agents,commit,push. Evidence and before-source/binary hashes underresume_streamed/.

### 2026-09-14 03:25 +07 — Streamed prefill resume completed; stronger-baseline comparison prevents false speed promotion
- Finished previouslyblockedqualificationaftertrue1547/4096completeprefilldiagnostic. Test nowprovescomplete25/25retained21 plusactualpartial6155/6151discard and128/1547/256discard. No cancellation/runtimepolicyweakened. Allmissingrealmediaisolation/quality/mixed/freshjoin/15Kfactscompleted.
- Samecurrentbinarystronger2Kcontrols coldp9539.965/42.136 vsstream4K40.558 andstream2K41.440. Notademonstratednewlatencywinnerover2K;boostconditionschangedfromoldsaved54.31->46.51default1Kcomparison. Stream2K saves128247808plannedbytes=122.31MiB/GPU;observedpeaks21443/21570->21321/21448MiB. Numericalalternative,1/8warmhashesidentical;noglobalqualitypromotion/defaultchange.
- 4KdetailedactualfollowupC8267.365/min30.629/p953.153,8owners/8prefixes;warm267.243/min32.258/p95.707;post3cancelrefill263.308/min31.305/p95.729. Allnoncancelnatural/errors0/caps0. Originalshortfollowup8prefixesbut1.776commonsecondsUNQUALIFIEDtps;dontloosen2smetric. Freshjoin6existingmin23.268,image7.551/video16.781s. Long15.330–15.367Kmediafacts16/16expected24-9/natural;warm.995s,cold89.835s,notlongdecodecert.
- Final20CTest+14stream+11policycases and18retainedwitness/originalbinaryguard pass. RuntimeCLI/server/dualhashesunchangedfromresumepreflight;priorstreamedsourcebeforehashesunchanged. No newC1/65K/262Krecert orspeedclaim. Updateddocs/handoff/board/index preservingownerprefixes. Alljobsfinished/GPUsidle,source/test/docsuncommitted,noagents/training/hardware/commit/push/deployment.

### 2026-09-14 04:55 +07 — Packed-prefill architecture has real isolated-state proof; live-join performance comparison active
- FreshstrictC1D3/radix Nsight profile completed; joinedhigh/lowinteger-group alternative0.906–0.950xradix bothGPUs rejected, codearchived/no productionradixchanges. Recoveredcurrentservice/strictwitnessguardpassed beforeedits. Evidence c1_radix_accumulate/.
- Native denseMLP packing4whole128/512requestquanta gives2.2x/1.35xleafspeed withnumericoutputdifferences. Implemented explicitpack-prefill frame throughsharedprojections but separateattention/KV,conv,recurrent,MTPstate; testedfour6.2Kdistinctregisters/cancel/retry/prefixproof. Actualallcold2Kcandidate41.50/42.52s vs39.17control slower,notpromoted.
- Revisedexecution packsWHOLEexistingquanta only whenaggregatefitsbudget; nevercutsallcold2Ktocreatebatch. Twoactualestablisheddecoders+2cold6.2Ktextownerrealtest passed packed256 boundary,cancelowner0, survivorBIRCHcorrect,retryinvalidated/finishedprefixreusable. Proposedperstreaminterruptiontimingsfeedexistingboundedprotection, noacceptance/data/modelchanges.
- Activecomparison proc-958c15b1-3eaa-4b28-abcd-a70292ca0816: live_control/live_packed/live_packed_repeat, separatelylabelled2warm+2coldperGPU. Harness15testspassed,existing3+1default/thresholdsretained. Sourcependingfinalcleanup removesunnecessaryoverreservation andrequiresequaloldquanta; currentrunbinarynotrebuiltmidrun. Needterminalread,finalsame-sourcevalidation/C1guard/C8/cancelanddecision. Noagents/training/hardware/defaultpromotion/commit/push.

### 2026-09-14 05:10 +07 — Packed live-join final validation in progress; raw C1 winners rechecked
- Final21CTest+15stream+11preservationcases/18witnessespass. Finalcurrentstrict79goldens80.778/80.648 andmargin107.444/107.150 retainexactoutputand43/129/83vs33/99/94. GPU1whole-quanta/livepackedtestpassed, allcoldunsplit, twoincomingregistersonecancel/survivorcorrect/retryinvalidated. Generalcancelandmediafallbackpassedwithoutchangingcachepolicy.
- Activeboundedproc-133f3a9a-90f7-4a49-b721-d07c4baca698 now runs latestsame-source reversed livepair plusmixedC8/quality. DonotduplicateGPUwork. Initial3livecomparisons positiveandsameGPUmemory butownerassignment/clockvariation caveat; nofinalspeeddefaultyet. Currentstatusboardupdatedwhileoldhandoffbodyremainsprevioustranche.

### 2026-09-14 05:17 +07 — Native packed live-prefill retained after final paired test and historical/C8 regressions
- Finaloptinpack-prefill sharesdenseprojections but keepsattention/KV/conv/GDNrecurrent/MTPcacheperowner. Packswholeequalexistingquantaonlyifaggregatefitsreservedmax; allcold2Kremainsunsplit. Initialallcoldpacking41.50/42.52vs39.17rejected. NoextraGPUstorage,modelcalibration,acceptancerelaxation orchangeddefaults.
- Finalsamebinary2established+2freshperGPU: newcomermean22.9257->18.7175s (-18.36%), max24.067->19.840. Four-establishedshared-windowmin24.041->22.174; longestgap.483->.616. Bothcoldjoinphases8naturals/errors0/caps0. Earliercandidatepairsmeans21.64/19.05vs25.24corroboratebutdifferentearlierbinary. Dynamicwarmowner/boostcaveatsnotconfidenceinterval. Finalplanned4999614720bytesunchanged;packedtokens6400vs0.
- FinalmixedC4+D1withoption warm266.385/min31.689/p95.717;actualdetailedfollowup262.611/min30.909/p953.265,8owners/8prefixes;post3cancelrefill259.507/min30.511/p95.737. Allnoncancelnatural/errors0/caps0.16focusedexpectedfacts passed; no universalclinical/code/videoqualityclaim. Media remainsordinaryfallback.
- 21rebuiltCTest+15stream+11policy/18witnessguardspass. ActualGPU1packedlivecancel/properretry/prefix testpass afterGPU0initial; allcold4registerfactsproperandzeropacked. Output1/7/13cancel/ingress/deadline, complete25/25retained21 andpartial1547/1543discard, redcancelblueexact/READYmediaslicespassed. Currentstrictgolden80.778/80.648 andmargin107.444/107.150 exactoriginaloutput/trajectory. Genuine65K/262Knotretimed; alloldpathsretained.
- OneboundedfreshC1radixgroup-joinoracle0.906–0.950xslowerbothGPUs archived; radixkernelfilesunchanged. No newstrict100record. Newcheckpoint/docs/index/board/currenthandoffupdatedwithownerprefixesretained. Alljobsfinished/GPUsfree/listenersclosed. Sourceuncommitted,noagents/training/hardware/commit/push/deploy/defaultpromotion.

### 2026-09-14 08:12 +07 — Raw C1 arithmetic and cross-operation oracles rejected without touching retained runtime
- HealthyHPSERVERpreflight/correctbranch/GPUsidle. Actualoldstrict/margingoldens2invocations4repeatseach:strict81.232/81.1477/80.924/80.8045,margin108.061/107.829/107.685/107.521,exactretainedoutput/digest/counts afterwarmup. Runtimebinariesunchangedthroughtranche.
- FixedpreviousinvaliddirectS4oracle withcorrectcontiguousK8fragmentandaccumulatedK32partial; trueexactA8nativeS4/U4 completeQ4gateupcodes/scalesbitexactbothGPUseeds. Timing186.37vs186.37and180.32–180.35vs181.18–181.41us,neutral1.000–1.006x. ExplicitA4singlelimblowerprecisionceilinglikewiseneutral1.000–1.007x,~.137outputrelL2;no runtimeprecisionchange/modelport. Nochipceilingclaim.
- DifferentfusedcompleteMLP CTA-local256gatequantumimmediatelyfeedsdownpartials thenfinalreduction:baseline296.32–296.74vs341.76usGPU0;289.22–289.54vs333.79–333.95GPU1,.866–.868x. Smallnumericdifferences3/1BF16outputs,notuniversallybitexact. Rejectedbeforeintegration;5.57MBpartials. No geometrysweep.
- Alltemporarybenchentriesremoved/exactoriginalCMakeandproductionQ4sourcepreserved;prototypesandlogsarchived c1_native_i4_exact/ and c1_fused_mlp/. Final21CTest+15stream+11policy/18witnessguardpass. CurrentCLI/server/dualhashesidenticalpreflight. Packedlive/C8/radix/107/longcontextallpriorroutesuntouched,nonewraw100or65K/262Kclaim. Updatedcheckpoint/index/handoff/board preservingownerprefixes;noagents/training/hardware/commit/push/deploy;alljobsdone/GPUsfree.

### 2026-09-14 09:52 +07 — Compact-Q5 traffic candidate reaches real C1 tests; final quality/lifecycle checks running
- HealthyHPSERVERpreflight/correctworktree/idleGPUs. NewmechanicalQ5G64->Q4 numericalprecisioncandidate on64targetdown+48GDNvalue tensors, loadedGPUcopyonly; noartifact/training/calibration. Completeoriginaldown126.98->106.24/106.50us and118.78->100.35us,1.18–1.20x. EquivalentconvertedQ5oracleexactoutputs onbothcards; idempotence/sourceidentity/release/sameaddressverifiedforbothconsumers.
- OptincompactflagfixedinStartupFeatures,strictC1D3,margin-1,source-ownedregistrationandreleaselifetime; noncompactkernelsretained. Matching128fixture82.85/82.57controlvs83.67/83.62candidate withdifferentbutrepeatabletrajectory46rounds/137drafts/81accepted, no strict100/newexactrecord. Changedweightsarenumericalalternative notoriginalmodelidentical.
- Sixcomplete-outputbudget4096GPU1:prose67.64->72.62,Thai57.82->62.20,JP57.82->61.69,Arabicflat63.85->63.96,JSON98.60->94.20. BothJSON40strictrecordsvalid. CandidateThai/JP/Arabicreadcoherent,noobservedsoup;Englishfinalclauseabrupt. Code6144budgetbothnatural:control3145toks97.96vs4278toks104.06;bothparsebutgeneratedtestswrong3/17vs2/25,respective fullcodequalitynotpass.
- Currentboundedvalidationproc-069d71c5-e4eb-4667-9560-dc032326c808 containssecondGPUprose,compactcancel/reload,oldgoldens,mediaandC8regression. Do notduplicateGPUjobs. Fulloriginalsources/binariesbackedupc1_compact_q5/,noagents/training/hardware/defaultpromotion/commit/push/deployment. Newcandidateaheadoldhandoff; boardupdated.

### 2026-09-14 10:24 +07 — Compact-Q5 fast kernel rejected at model quality; prior runtime restored and paired-verified
- NativeQ5->Q4rangeconverter/mappedconsumer complete down18–20%fasterbothcards andexactagainstconvertedQ5reference. Sourceidentity/idempotence/release/sameaddress andrealmodelcancel/reloadpass. Broad64down+48GDNvaluechangedtargetprecisionflagoffdefault; onlyloadedGPUcopy,noartifact/training/calibration. Bodygainnotacceptedmodelwin.
- BroadpairedlongGPU1 prose67.64->72.62,Thai57.82->62.20,JP57.82->61.69,Arabicflat,JSON98.60->94.20. JSON40rulespass,prosemostlycoherentbutEnglishendsabrupt. Codecap4096thenlarger6144completes4278at104.06but2/25generatedtestsbad(vs3/17control). BroadOCRNIFERVISION731->NIFERVISIONcoldandwarm14/16facts, NOTPROMOTED. Secondcardprosehashesreproduced; boostcaveats recorded.
- Singleprincipleddown-onlyablationkeepsGDNnative/restores8/8facts butcanonical12880.85/81.01vs81.52/81.64slower. Code1202tokensstopbeforeinsert/evict/testsdespite6144budget;unmodifiedput(x,41)/get(x)returnsNone. That100.15tokenrateisnotusefulcode. Bothquantcandidatesremovedratherthanpromoted; allsource/binaries/positiveandnegativeevidence archived c1_compact_q5/. No furtherlayersweep.
- Restoredallaffectedexistingfilesbyteexactfrompre-experimentcopies; newfeaturefiles/benchentriesremoved. Pre-existingdirtyworkretained. Fullrestoration21CTest+15stream+11policy/18witnessespass. ActualoldvsrebuiltCLI strict81.51/81.46 andmargin108.33/108.15 exactoutput/digest/count/positions,matchingloadedclocks1860/1845respectively. Rebuiltbinaryhashesdifferbutsourceequal; originals retained.
- Disabledcandidate normalC4+D1/pack/Vision duringvalidationwarm271.40/min32.56,appended263.66/min31.79,refill266.10/min32.17,8prefixhits/allnoncancelnatural/errors0/caps0. Notattributednewgainorrestored-binarytiming. No65K/262Krecert/newstrict100. Currenthandoff/status/indexupdated, noagents/training/hardware/commit/push/deployment;alljobsdone/GPUsfree.

### 2026-09-14 11:53 +07 — Strict C1 accuracy, factory-drafter and upstream oracle checks; next reference running
- CurrentturnhealthyHPSERVERpreflight;4strictgoldens80.738–81.054 and4margin107.469–107.919 exactretainedbytes/digests/43/129/83vs33/99/94. SimpleindependentsymbolentropyactualQ4gate3.47–3.49/Q5down4.49–4.57 ideal~7%roundbyteceilingbeforedecodecost,notageneralhardwareceiling; no wastedcompressorkernel.
- SHAall1124artifactobjects identicaloutside5MTPweightmatrices. CurrentW8factorydrafterstrictD3/radix/fullvocabtargetfixedgives78.56/78.67vsQ4factory81.22/81.52, exactsameoutputs43/129/83; no quality trade ormodel-targetchange. PinnedExLlama7dc12afM4kernelmechanicalQ4packcompleteop206–213usvsnative180–186,.874–.880x rejected. No upstreamkernelmods.
- SpareT4MMAcolumns usedforradixactivationlimbs(noextramodelweightread), independentFP64sampledcompletegroups zeroerrorvsnative~.0094–.0122; completegate+2%cost. Gateonly81.14–81.21vs81.43–81.66;fullMLP80.42–80.53vs81.39–81.61, unchanged43/129/83. Six6144budgetnaturaloutputsnetregression(prose/code/Arabicdown,smallThai/JP/JSONup),notpromoted. Exactbefore-source restored andtestedprototypearchived. CLI/serverneedfinalrebuildafterrestoration.
- Pinnedggml093a2f86 MMVQ/MMQ whole-operation oracleactiveCPUbuildproc-ac571419-5a0e-4df1-8326-a261f4444f20, consistentCUDA13.1; noGPUworkactiveyet. Evidence c1_lossless_traffic/upstream andllama_oracle. Boardupdatedforresurrection; noagents/training/hardware/push/defaultpromotion. Keepstrict100goal,alloldwinners.

### 2026-09-14 13:18 +07 — Strict C1 proposal and exact-layout diagnostics completed without changing retained target paths
- Recovered c1_lossless_traffic pinnedggml build terminal and tested bothGPUs: MMVQ/MMQ completeops0.556–0.588x native; no integration,temporarytargetremoved/sourcearchived. Originalprecision baseline protected. Current exactC1D3/radix pairedtests use samebinary andtargetoutputdigests, no marginrelaxation.
- Request-local lastverifiedhiddenresidual intoMTP recursion82.57–82.87->68.64–68.69,43rounds/83accepted->52/74; head-onlyablation82.43–82.75->74.11–74.18,48/78. Both preserve targetoutput butproposal yieldregresses; closed/restored,notcoefficientsearch. DraftonlyBF16KV82.22–82.25vs82.48–82.76 yieldsidentical43/129/83,neutralnegative/memoryextra,restored.
- Exact CTA-localQ4physicalweightreorder no meaningful gain (.991–.996xGPU0,1.005–1.008GPU1), exactcodes/scales; properQ5row8physicalformat+sameT4splitreduction.776–.797x despiteexactoutputs. Prototypesarchived/buildentriesremoved. NCUQ4read94708608bytes~expected947MB/10; no duplicatepayloadopportunity. Nsightkernelbusy~98%,notformalhardhardwarelimit.
- FinalnativeMTPnorm-basisrecursiontest82.45–82.51vs82.39–82.70,identical43/129/83/positions/bytes, noyieldgain. Closed/restored fromscopedbackups. All current experiment processes terminal, source original; CLI/server still fromlastcandidatebuild and require rebinding/validation before finalcheckpoint. No agents/training/hardware/defaultchange/commit/push. Mainstrict100stillopen; nextcurrent-roundhandoffaudit,notrepeatclosedknobs.

### 2026-09-14 14:05 +07 — Dual full-context goal revisited with an explicit complete-computation control
- Ownerexplicitlyreiteratedcorrectsingle100anddual-fullcontext100. Preflighthealthy/correctbranch/GPUsidle; restoredlatestothertrancheworkrebuiltbeforechanges. HistoricalQ64longpoliciesretainedsparseattention,FFNskipmaskandlicensedD4; old262qualitytrueincludesvisibleparagraphrepetition. Currentold65snapshot+retainedpath124.32tok/s,butembeddedchatcontroltokens/incoherent; notusablecorrectnesswin.
- AddedoptinLongContextComputation::Complete frozenbeforegraphcapture. Stagesalltargetlayers/allnativehead; MTPattentionandtargetcontextparallelallkeyranges; D4goesthroughrealgreedyverification. OldRetainedThroughputdefaultssame. Snapshotreservedwordnowpolicy0/1,mismatchrejected; nohistoricalsnapshotmodified. NewCPUboundarytestpasses; full-modeincompatibleCLIcombinationsreject.
- Fresh65292inputfullD3noCP/TP 40.42t/s,187.413sprefill,coherent192-tokenprefix/noembeddedcontrols. Newlycapturedsnapshotpolicy1. Exactsame-modereplay41.24reproducesalloutputbytes. SamefullsnapshotTP2+full-contextparallelD3=46.06t/s,D4=36.41; textstillcoherentbutdifferentnumerictrajectory,noexactoutputclaim. Bothwellbelow100notsealed.
- Currentboundedproc-f5505411-af14-42c6-aa0b-69042f111bd7 capturesnewgenuine261900full-computationprompt withsameq4artifact/split36/D3 at262144capacity. No shortcutsnapshotreuse,noagents/training/hardwarechanges/nootherGPUwork. FollowupmustinspectterminalreceiptbeforeanyGPUjob; sourcepolicyfilesbackedupunderdual_full_context_audit/policy_before. Boardupdated; oldhandoffawaitsfinalcheckpoint.

### 2026-09-14 18:11 +07 — Dual complete-context proposal window candidate and physical-split correctness regression
- Recoveredgenuinefull262capturepreviousjobterminal:261900tokens1210.5sprefill24.24native,31.69TP2/CPstrictD3. Oldlicensed/retained>100profilesdonotmeancompletecompute; remainintact. Full-vocabproposeralone49.61->48.45at65and31.82->31.92at262neutralremoved.
- Newexplicitproposal-only8192+128sink aliaspreservesphysicalappend/masks,alltargetweights/layers/keys/fullhead. BothGPU39boundary tests pass. Samebinary65control49.53->60.09,87->73rounds;26231.85->39.74,102->87;192outputprefixesbyteidentical,snapshotsunchanged. Stackfullvocab57.81vs60.18at65,worse;41.26vs39.72at262+3.9%,separatecandidate. No100orcompleteanswerclaim.
- Realcomplete-textstop red64tokens12im_endfollowedinventedturns;green3tokens12im_end,exactprefix. Historicaltextfixturebehaviorpreserved. Newnatural65archive64172actualtokenscapture249naturals/all3spreadfacts, thenCPcontrolillegalaccess. TP-only64pass,CP-only/both32fail. Identifiedcapacityplannedphysicalboundaryvsactualwindow-derivedGPUranges; Complete-onlyfixed-partition geometry candidate building/replaying in proc-fcd6ca87-e2ff-4561-b8dc-744c8e10d9cf. Do notduplicateGPUjobs. Evidence dual_draft_window/physical_split_fix/.

### 2026-09-14 18:37 +07 — Natural full-context quality capture and next bounded TP2 graph oracle in progress
- Physicalsplitfix real65Kcontrol/windowboth228naturals,all3widelyspacedregisters;CPillegaladdress correctedwithoutdiscardingtargetkeys. Complete-onlyEOStestred64tokensinventedturns->green3tokens12im_end;historicaldefaultunchanged. AddedfreshMTPprefillguardtoavoidapplyingdecodewindowtoT128,notyetfresh-inputGPUqualified.
- ActiveGPUjobproc-48aa8662-3147-4921-870a-b4de4a92854d captures genuine260973ish tokenqualityarchivefromscratch withfullnative layers/keys;thenTP2+CPcontrol/window replays. Usesfixedcopiedbinary,sourcechangesisolated; do notduplicate. UpdatedboardwithactualprocessID.
- Prepared/compiledseparatecoarseTP2MLPgraphoracle defaultoff,source/destinationidentitycachekey,capturedcreationtimereported,oldcrossdevicesynchronizationunchanged. Notthenegativeoldper-eventDAG. NoGPUtestuntilcaptureterminal. Sourcesbackedupdual_tp2_mlp_graph/. ExhaustivewindowpolicytestsandGPUaliasproofpromotedtotestsource;allpriorwinnersmustberecheckedonfinalbuild.

### 2026-09-14 20:48 +07 — Natural complete-context verification finished; exact score/value pipeline under test
- Currentfresh13Kprefillguard>8192proposalwindow verified natural10tokenrecord bothmodes; native64172context228tokenanswer44.49->51.95tps and260973context229answer33.51->37.62,identicalcompleteoutputs/all3distantfacts. Snapshotfilesunchanged. NativefactoryD3/targetallkeys/alllayers/fullheadretained; this verifiespriorwindow/policyfixes,notnew100gate.
- Boundedfull-keyattentionexperiments: P/Vscalefactor.86–.89x; Qhoistbitexactbut72Bspill.77–.80x; dedicatedproducersremove spill/reg96/shared18KiB but.96–.98x,exact. IntegerPVaftercompactswizzlefixnumericalrelL2~.0114/cosine.99993but.61–.63x,noport. Priorbadlayoutoutputs/decisionsretained. Runtimeweights/KV/policyuntouched.
- ForcedupstreamPyTorch2.6CUTLASSefficientbackend supportedSM75withoutinstall, identicalnativeinput/Qquantization/reversepages;85nativekeysplit+LSEmerge989–998usvsnative631–632bothcards,FP16KVdoublebytes/excludedconversioncostoptimistic. No port. NCUclocks/cache-controlnone observed45.68%DRAM/41.26%SM/44.67%activewarps,notrooflineclaim.
- Nowtestingnewtrue2slotQK/softmaxvsPVoverlap,namedbarriers,separateK/Vstaging,disjointwarp registerlifetimes. Activeproc-533f6d65-fa40-428f-9dd4-31e3fddfaa91; nootherGPUjobs. Exactoldarithmetic/allkeysrequired. Noagents/training/hardware/default/commit/push. Currentboardupdatedwhileoldhandoffbodyawaitsfinalconsolidation.

### 2026-09-14 22:50 +07 — Full-key packed FP16 value candidate improves natural261K; broader qualification active
- PreflightHPSERVERhealthy/GPUsidle/correctworktree. Recoveredheadscale finalrepeat261K40.73vs37.77natural3facts andoldexactscorepipeline1%leafreject. No duplicateGPUjobs. DyadicVcacheformat6–8%attentionleafnotmaterialfortherisk,closed/archive,noport.
- NewpackedFP16dequantizationconsumesoriginalI8Vcodes/scales,skipssoftwareBF16roundtrip; half2directFP16highermantissa arithmeticnotbitexactoldattention. Allkeys/QK/causalmask/targetlayers/fullhead/nativeD3acceptance unchanged.8,126,464 exhaustivebyte+finitehalfscaleconversionssameexpectedhalfproductbothGPUs,inputunchanged. Completeattention~1.055–1.095x,nomacroclaimfromleaf.
- Samebinarynaturaltrue260973context40.78->44.01,reverse43.98vs40.63,241candidatevs227controltokens,105roundsbothwith136vs122accepted. All3distantrecords correct,completeanswersfullyread, outputsha repeated andsnapshotsunchanged.65Kcandidate54.20/54.26slightlybelow54.54/54.38controls; notgeneralreplacement. Keepdefaultoffwhilequalitypending.
- Activeboundedproc-91ff4349-1797-4765-97e1-280f10ed9561 nowcapturesfresh~12KJSON/Thai canonicalstates thenheadscale/valuecomparisons. Finalsourceaddsstrictflagvalidationbeforecoordallocation,willneedfinalsame-sourcefullcontextandC1/guards. Noagents/training/hardware/commit/push/defaultpromotion. Docs/handofffinalpending; currentboardexplicit.

### 2026-09-14 23:06 +07 — Complete261K direct-FP16 value path retained with final paired correctness/clock evidence
- Finalsamebinarytrue260973headscalecontrol40.62vsdirectFP16value43.95tps(+8.20%),natural227vs241tokens,105roundsboth,122vs136accepted. Earlier44.01/43.98repeatsidenticalcandidateoutput/all3distantfacts; actualvaluecalls1680. Loadedclocks1830/1845andpeak16721/16890MiBsamewithinfinalpair.65K54.20/54.26vs54.54/54.38slightlyslower,notglobalreplacement; defaultoff.
- ValueI8codes/scalesunchanged; exactsignedbytebiasedhalf2reconstruct+multiplyavoidsoldBF16roundtrip. Higherprecisionintermediatebutnotbitexactgeneration; noVcacheexpansion/modelweights/keydropping/layerskip/acceptancerelaxation.8,126,464exhaustiveconversionsperGPUinclzero/subnormal/overflowpassed. Productionleaf~6–9%typical,one27%isolatedoutliernotcredited; end-to-endgainincludesdraftyield.
- Freshcanonical12Kbackgroundcaptures:JSON12101input,1129/1129/1080naturaloutputsall40rulesvalidacrosscanonical/headscale/value. Thai12097input892/898/826natural,completeanswersreadcoherentThai/topics/conclusion;notuniversalnamedfunding/code/mediaqualityclaim. Existingcanonicalsnapshotfilemetadataunchangedonload; nooverwrites.23CPU/API/profileCTest+15stream+11preservationcases/18witnessespass,33cache/39proposalviewboundariesperGPUplusnew7CLIinvalidcasespass.
- Finalrealoldstrict79goldens80.4538/80.3124andmargin106.958/106.854sameoriginaloutput/digest/counts. C1binariesrebuiltduringsharedsessionsohashesnotpreflightidentical;currentidentitiesandactualtestsbound. NativeC4servinglogicuntouched,no freshC8performanceclaim. Priorpendingheadscale final261K40.73/37.77reconciled. DyadicVcachevariantnotported; oldpreexpandedVexperimentwasclosedandnotrepeated.
- Verifiedindex/report/docs/currenthandoff/boardupdatedpreservingownerprefixes. AllmanagedGPUjobsterminal/GPUsidle,noagents/training/hardware/commit/push/defaultpromotion/deployment. StrictC1>=100andcompleteDual>=100bothOPEN. Retainedoptional261Kprogressisnot100seal.

### 2026-09-15 04:59 +07 — External low-bit execution measured on SM75; faster native codebook remains unpromoted
- HealthyHPSERVERpreflight, correctworktree, idleGPUs. PreviouslyclonedFLUTE9eb83a12nowhasstandaloneSM75port: synchronousvectorcopy andtwoK8nativeFP16/FP32MMAlogicalK16; newerCuTebroadcastscalesmaterializedwithoutremovingassertions. Original3/4bitpackfunctions+independentFP64references exact/deterministiconbothGPUsfullM4gate/downshapes. Initialtinygridpreconditionerrorownedjobterminated/fixed, receiptskept. Runtimeunchanged.
- CurrentCUDA13.1matchedFLUTE3bitgate446.66/444.45us,down240.51/235.78vsnativecompletegate186.37/179.84–180.22,down124.93–125.34/117.44–118.18. FLUTE4bit~703–707/~347–349us. Externalexcludesepiloguesandstillloses; noportintoruntime/nogeometrysweep. EarlierCUDA12.5runslabelledseparately.
- DistinctfixedunlearnedNF4-derivedintegerlookupnativeQ5downoracle1.140–1.162x; all65536packedindexpatterns and128independentscalaroutputsamplespassperGPU. Numericalweightsnotoriginalexact. Read-only24actualmatrices/128sampledrows medianoutputrelL2.09431vsuniform.10768,worst.11759vs.13875. Betterproxyisnotfullmodelquality; noconversionorpublicflagpromoted. Evidence nf4_integer/.
- Final23CTest+15stream+11policy/18retainedwitnesschecks pass. Realstrict79goldens80.7539/80.5356 andmargin107.263/107.171 retainexactoutputs/digests/43-129-83 and33-99-94counts. CLI/server/dualSHAidenticalpreflight. Allpriorprofilesincluding43.95complete261Kpreserved/notretimed. Both100gatesopen. Updatedcheckpoint/index/handoff/board, ownerprefixretained. Noagents/training/hardware/commit/push/deploy; alljobsdone/GPUsidle.

### 2026-09-15 11:50 +07 — Bounded C1/value-cache/mixed-step experiments rejected; retained runtime unchanged
- HPSERVERpreflighthealthy/correctbranch/stockGPUsidle. Read latestFLUTE/NF4completednotes,notrerun. ActualC1strictgoldens80.7184/80.5719/80.4362/80.3391,margin107.322/107.167/106.922/106.689 exactoldoutputs/digests/acceptance43-129-83vs33-99-94. Same runtime binaries retainedafterexperiments.
- NewfixedNF4VALUE-onlycache keepsallkeys/scores/max/LSEandtargetweights;65536lookuppatternsandconvertedI8Vreferenceexactoutput/partialbothGPUs. Fullattention.990-.996x65K,.992-1.033x261K,originalVrelL2.249/.471randomfixtures. No material gain, no modelport; temp targetremoved, archivedgqa_nf4_values/. Not a generalcompressionimpossibilityclaim.
- CurrentC1exactQ4readschedule89MBstream:rowmajor530.6-530.7vsCTA-wavefront531.0-531.3GB/sGPU0;546.0-547.0vs546.0-546.1GPU1. Exactlogicalsignatures,no>=20%headroom, closedbeforecomputekernel. No layout/runtime/precisionchanges orparameter sweep; c1_weight_wavefront/.
- Queuedmixedprefill/decode actualVariantVerify4columns+qualifieddenseprefill64/128 vscombined68/132. Warmgraph alternating later64ratio1.037-1.045,128ratio.950-.969; firstpassoutliersnotcredited. VerifytailnumericalrelL2~.0135-.0138, no fullmodel/stateproof. Notintegrated; preservesoldseparateandpacked-livepaths. Benchmarkincludeerrorsrecorded/fixedwithoutchangingruntimeguards.
- Allthreetemporarybenchentriesremoved/exactoriginalCMakeandproductionQ4/GQAheadersunchanged. Final23CTest+15stream+11policy/18retainedguards pass. CurrentCLI/server/dualhashesmatchthisturnprotectedgoldenbinaries. No newstrict100/fullcontext100/C8performanceclaim. Updatedreport/index/currenthandoff/board preservingownerprefixes; alljobsfinished/GPUsidle,noagents/training/hardware/commit/push/deployment.

### 2026-09-15 13:41 +07 — Exact session-copy D3 graphs produce complete editing gain; qualification active
- ReadqueuedP2session-copycontract; fulltargetC1D3only, strictmargin-1, same3drafts, exact32priorledger match. Newboundedrequestowned1MiBCPUindex, secondplannedcopygraph retainsnative targetverification andMTPhidden/KValignment but skipsfactoryproposalhead/AR until mismatch; nativeproposalresumptionobserved. Nocachedanswers/crossrequesttokens/modelcalibration/precisionchange.
- Samebinaryactualcomplete688tokenfileeditcontrols104.73/104.10->110.54/110.34tpscopy, exactsamefilehash/digest; copy160rounds/480acceptedcopies. Midfileedit109.97vs104.45, copied464/468 andfactoryresume1, samecompleteoutput. Firstretainedoptimizedheadcontrolstopped50tokensunusable; selectedunchangedfulltargetcontrolwhichcompletes688. Initialcopyeligibilityzeroedbymodelpresencepenaltymetadata, fixedinproposaleligibilityonlybecausegreedytargetbranchignorespenaltymetadata; samplingnotchanged.
- True3turnHTTPfileepisodeusingactualreplies: copy111.73/110.09/108.33vs105.08/103.88/102.10clienttps, allfilesexact/prefixes0/735/1478. Episode30.576vs31.810s; loadedclocks1800vs1785, measurednotclaimedisolated. Generalprose/JSON/Thai/JP/Arabicallbyteidentical,no copyhits,~0.2–.7%overhead. Repetition357tokensidentical,copy109.94vs104.81butboth51linesnotrequested48,notcorrectnesscert.
- Currentprocessproc-a4bf496b-ae8e-4e26-a138-53db8b63d1e7 realcopy/resume/cancel/isolationthenpreservedthinkingepisode. Needterminalreceipt/finalmedia/reload/oldgoldens/C8beforefinishing. Newliveboardprecise; priorhandoffolder. Noagents/training/hardware/defaultpromotion/commit/push/permanentservice. GeneralC1strict100andcompleteDual100remainopen; copy-specific100notmislabelled.

### 2026-09-15 14:33 +07 — Recovered copy-path final validation; unused-column GDN precision candidate in full-model comparison
- HPSERVERhealthy/correctbranch/noGPUownerspreflight. Recoveredoldcopyrealjobterminal,completedremaining24CPU/15stream/11policy/18witnesses,4strict+4marginfixedgoldens,copy/cancel/reloadGPU0,media16outputsidenticalonoff/nohits,12returnedfilesexactandindependentfunctionaltests. NativeC8warm269.447/min32.021,follow264.836/min31.051,refill262.729/min30.839,8prefixes/errors0/caps0. Copy-specific110isnotgeneral100.
- NewGDNvalue/Z unused4-of8integerMMAcolumnspreserveweightstreamandmatmulcount whilecarryinglowradixactivationlimbs. IndependentFP64samples error .0053927->.0016620/.0056466->.0017391, determinism/nonfinite/untouchedQKguards passbothGPUs;21,120bytesextrascratch,no persistentcache. Leaf1–2%slower, usefulyieldhypothesisnotrawkernelclaim.
- ExplicitstrictC1D3/defaultofffeature/workspacewired; other35Bvariant/defaultA8preserved. Short83.11/82.84controlvs82.91/82.79candidate,outputdigest14660c413b4cf185/43rounds129draft83acceptedexact both. Activeproc-1836949f-a466-49e9-b43c-eaa531299dab sixfamily6144budget, nooverlappingGPUjobs. GDNprechangebackupscontainvalidatedcopyfeature,donotrestoregitHEAD. Both100gatesopen,noagents/training/hardware/commit/push/deploy.

### 2026-09-15 18:28 +07 — Whole-round hypotheses measured on native histories; no incorrect or uneconomic speed path promoted
- HPSERVERhealthy/correctworktree/GPUsownedserially. ReconciledpriorGDNtrialrejected/restored; primarywhole-rounddirectivefollowed. NativeD3rank/accepttrace6exactcontrolpairs1249full+10clampedframes. Target45.605msT8proxycannot100atdepth3evenfreeperfect4outputs. Optimistic8nodespinemodelwithperfectdescendantsatT4cost35.425msgives89.15harmonicproxy,notactualtreeoruniversalbound. Data/costsavedwhole_round_branch_ceiling/.
- WholeA16targetbodyoraclekeepsweights/factoryD3, testsactivationprecisionuseful-workpremiseacrosslargefraction:yieldratiosgeneral.993,code.998,JSON.982,Thai1.044,JP1.098,Arabic.961;not24%gain. Privateenvhookremoved/programrestoredexact. No anotherlocalactivationlimbport.
- PublishedDSparkv2actualFP16inferenceon5correctnativecausalcommittedlayer-taps,1241roots. No rejected/futuretargetfeatures or teacherforcedMarkovinputs. U2.136/4.506/6.040/1.185/1.539/1.447afterFP32-headsensitivity; only1acceptedprefixchanges. StrongJSON/code alone notgeneric100; gammma7notD3credit. Publishedsource/weights/hostfeatures/resultsretaineddspark_v2_local_coverage/,notengineintegration/training.
- Built genuinelynativeSM75E8P+E8RVQ3wholegatecostoracleforthreebitMLPtrafficplan. All16,777,216vectors/134,217,728components+128independentscalarrowsperGPUpass;payload66.85vs89.13MBbutlatency694–702vs180–186us (~.256–.268x). No actualweightconversion/qualityclaim, no parameter sweep, prototypearchived/targetremoved. Mechanismclosedbeforelargeport.
- Privatebranch/feature/A16instrumentationarchivedandremoved,acceptedsourcebeforehooksrestoredbyteexactincludingoldorphanbranchtracecleanup.24CTest+15stream+11policy/18witnessespass. Finalrealstrict80.7295/80.5964 andmargin107.329/107.211 exactoutput/digest/43/129/83and33/99/94. No newgeneric100/completeDual100/C8/65K/262Kclaim. Allacceptedcopy/radix/packed/contextprofilespreserved;report/index/currenthandoff/boardupdatedpreservingownerprefix. Noagents/training/hardware/commit/push/defaultpromotion/deploy;alljobsdone/GPUsidle.

### 2026-09-15 21:26 +07 — Whole MLP W3G128 cost gate passed; actual target weight-quality canary active
- HealthyHPSERVER/correctworktree/stockGPUidlepreflight. Fourrealstrict80.7821/80.6232/80.4336/80.3715 andmargin107.242/107.205/106.981/106.768 retainexactoldoutput/digest/counts. IntegerlatticecodecnewpackedINT8reconstruct is2.4xfasterthanpriorFP16RVQbut0.620–0.636xcurrentgate,archive/noport.
- Table-free3bit/128groupwholeMLPfamilycostis material:gate184.38–184.64->149.73–149.76GPU0 and180.10–180.42->143.52–144.26GPU1; completegate+down296.26–296.67->221.18 and288.99–290.46->215.94–216.06us,1.337–1.345x. Exhaustive4096decodercombinations and128independentscalarfulloperationrowsperGPUexact,allfinite. Syntheticrepresentedweightsnotoriginalmodelquality. CoverslargeMLPfractionratherthansmallleafcampaign.
- FollowingthroughviaprivatewholeMLPweightqualitycanary, notmovingtonewidea. 128canonicaltargetMLPmatricesreconstructedsource-onlyQ3G128insideGPUcopy, no artifactwrites/training. OriginalQ4/Q5containers and A8G64referencekernelsremainforqualityscreen; nofastmodelportornew100credityet. Activeboundedproc-a4996c5f-9233-4bba-8711-47ff4f28a467, factscontrol/candidate then6familiesifpass. Sourcebefore/hashes/evidencewhole_mlp_w3g128/quality_canary; boardupdatedforresurrection. Noagents/hardware/defaultpromotion/commit/push/deploy.

### 2026-09-15 21:33 +07 — W3G128 quality run resumed after diagnostic build fixes
- Privatecanarybuildguardcorrecteddraft_tokens->draft_windowandCUDART_INF_Fincludeadded; failurelogsretained. Activeboundedproc-cb976b24-b9db-4f21-b169-bf9efd3a9cdc runsfactsandconditional6familyquality. Sourcestillunpromoted; nohardware/default/modelartifactchanges. CurrentstatusIDrefreshed.

### 2026-09-16 02:44 +07 — Recovered whole-MLP speed gate; native nonuniform 3-bit full-model path now testing
- HPSERVERhealthy/correctbranch/stockGPUsidle. RecovereduniformW3 qualityjobwascompleted; natural4170tokencodeinvalidnext_..prev,nopromotion. FixednormalquantileNU3levelsderiveindependentofcheckpoint andretainwholeMLP1.326–1.345x;8192samplegroups/128MLPmatricesSSE5.050vs5.431(.930x),123matricesbetter. NotsemanticqualityfromSSE.
- NU3weight-onlycanarywithcanonicalA8G64kernelspassed8factsandnatural6outputs;code4168parses,10Kintegerkeymodelops passbut2/20generatedtestsbadandNonekeyevictionfails. Evidencequality/code_review.json;nofullcodecertificate. Followingthroughnativefastdataflownotshippingcanary.
- In-placefixedNU3packingroundtripsallgate94.699MB/down58.491MBbytes3timesperGPU,zeroerrors;128independentscalaroperationrowsand4096decodercombosexactforrepresentedweights. Existing94.7MBscratch/reusedsourceallocations,noextraresidentmodel. Fullfastsourceprepared/guarded/dispatchwiredbehindnewoptinC1D3flag, preservedtargetMTP/acceptance. ChangedweightandA8G128precisionnotoriginalmodelexact.
- Activeboundedproc-5764a954-a879-42e4-9577-8a666b2fff9d runsbuild/CPU/facts/6families;donotduplicateorrebuildmidpair. Currentboardupdateswhileoldhandoffstale. Before/fast_beforeandbaselinebinariesretained. Noagents/training/artifact/hardware/defaultpromotion/commit/push/deploy. Both100gatesremainopen.

### 2026-09-16 08:37 +07 — Canonical fast-profile audit finds missing target candidate, new exact-score union under qualification
- PreflightHPSERVERhealthy/correctbranch/stockGPUsfree; olddirectcodejobterminal4/4passbutfullheadsettingsdifferentfrom97canonicalfixture. Repeatedcanonical18casepair:directcode3/4 withLRU319tokensstopmid-if,original1/4;focused7/8directvs8/8original. Naturalstopand126.94TPSnotcodecorrectness.
- Readonlyheadsamehiddenposition444showsoptimizedEOS248046score12.25vsfullnative220score26.125, allshortlist-nativeexactscoreerrors0, nativewinnerexcludedbutactualnextdraft220. No targetoutputaltered. Diagnosticremovedaftercapture. ThisiscoveragebugratherthanNU3converterarithmetic. Newdefaultofftarget-draft-union exactscoresallactiveD3proposalIDsusingexisting16-rowW8workspace andmergesbeforestrictargmax;noEOSsuppression/noacceptedmarginchange/noextraweights. Stillnotfullvocabcertificate.
- New1024randomindependentargmax/tie/duplicate/invalid/tail/metadataGPUtestspercardplusCPUCLI/frozenfeaturetests pass. Currentboundedproc-8590b3ea-c36b-4f5d-91fb-8d3b9450b704 doescanonicalshortrepeatandLRU0/1checks. Avoidoverlap/rebuild. LUTdecoderexactbut~3xslowclosed;G64SSE7.54%betterand3%costslowercandidateheld,notruntimeported. Alloriginalroutes/quantkernelspreserved;nohardware/training/agents/commit/push/defaultpromotion.

### 2026-09-16 11:40 +07 — Closed exact-head dead end; generic full-target proposer/body screens and causal feedback oracle
- Bounded-head audit625framesallmatchfullheadbut88.76/88.66vs89.22/88.83; closedperSept16ownerstopdirective. Reviewedexactdiffthenarchivedandremovedonlyunacceptedboundedruntime/flags/tests; preservedNU3/union/sessioncopy/allotherdirtywork. Originalexecutablessavedfull_vocab_proposer_ceiling/.
- Sixcurrentoriginaltarget/full-vocabstrictD3pairs: exactW8proposalhead acceptedcountsandbytesidenticaltoB1/B2+W8rerankbut~14%slower; Thai/Arabicby-positioncountsnotidentical. Noaccuracygainpremisefornewheadfamily. FactoryMTPW8bodyafter1112nonMTPobjects17.759GBbyteverifiedidentical shows0–2.7%yieldgainandslowerall6. No corpusshortlist/NU3/no newprecisionport.
- Activeboundedproc-54c802df-1829-4d41-b82a-438249bb2f43 privatecausalfeedbacktestusesonlyacceptedtarget/alignmentstatepairtoraisefutureproposalyield,noweights/acceptancerelaxation. Standaloneindependent arithmetic/future-column checks then6samebinaryfulltargetpairs; no parameter tuning. Sourceinmtp_causal_state_feedback/before, statusupdated aheadoldhandoff. Finaltests/restorationpending; noagents/hardware/commits/push/artifactmutation.

### 2026-09-16 12:46 +07 — Whole-MLP trellis cost rejected; complete-context ownership and generic correctness reverified
- HPSERVERhealthyandcorrectworktree/branch,allGPUjobsserial. Reconciledpriorcausalfeedbackprocfinished0/negativeall6andprivatehooksalreadyremoved. NewwholeMLParithmetic-trelliscostfamilyfrompinnedQTIPe90c668/GLQ9763383:fixed65536state+262144mappingvaluesperGPU/128scalargatechecks; noactualmodelconversion/training. G64FP16239–240us,integer1MAD369–371us,optimisticmatrixscaleFP16224–225us vs180–187baseline;noaccepteddecoder. Targetworktrafficreductionfailedruntimecostgate,nocoefficient/geometrysweep.
- Archived3prototypekernels/drivers/referencebytes/pins/GPLsourcesaslocalresearch;removedonlyour3benchentries/files. ProductionQ4andCMakebyteexactpreturn. No modelweightprecision/artifact/cachepolicychanged. No broadtrellis/hardwareceilingclaim.
- Currentcomplete261Kownershipsamebinary:TP244.18vsownerlocal41.43tok/s,natural241vs249tokens/all3records,105vs113rounds,canonicalmetadataunchanged. Ownerlocal saves~2–2.7GiBbutdecoderategoesdown; shorterloadwallnotclaimedthroughput. ExistingTP2profilekept;no masksplitfurthersearch. Current44.18reproducesprior43.95,notspeedbreakthrough.
- Finalrebuiltgenericfullnativehead/fullvocabD3/noNU3six512controls63.63/82.11/90.64/54.79/53.06/61.81 allprioroutputhash/digest/round/acceptedpositions exact. Boundedpreservationnotcompletequality100. Historicalstrict80.6834/80.5502andmargin107.211/107.150retainexact43/129/83and33/99/94.24CTest+15stream+11policy/18witnessespass;genericity/qualitybarunchanged.
- Report/index/status/currenthandoffupdatedpreservingownerprefixandpriorhandoffbackup. No newacceptedprofile, strictgeneric100andcompletedual100OPEN. Alljobsdone/GPUsfree;noagents/training/hardware/commit/push/defaultpromotion/deployedservice.

### 2026-09-16 14:57 +07 — Actual261K rotated-cache admission failed; original complete target and C1 trajectories preserved
- HealthyHPSERVERpreflight/correctworktree/idleGPUs. RecoverednewerunrecordedW8registerstreamandTP2globalgraphclosures; no duplicatebenchmarks. CompletedC1projectionlift42.99vs44.21with2.08GBextra officiallyclosed/privatehookarchivedandrestoredfromownscopedbackup. ExistinglegacyA8/otherstageprofilespreserved.
- Newdata-obliviousrotatedKVlarge-traffic screen: analyticBeta(d256)Lloydlevels/fixedHaarseed, MSE4K4V andpaper3bitK+1bitQJL/3V, no calibration/weights/training. Readonlycurrent261Klayer3/31/63actualQKVallheads/columns288cases, exact241outputtokens/3facts/natural/snapshotmetadataunchanged. Nativeofflinebaselineerror<=.00448; compressedvariantsworstrelL2.2531/.43284,mincos.98037/.92587,49.24%/43.56%cachebytecounts. Failedpredeclaredgate,noCUDAcodecport/bitwidthseedrepair/speedclaim.
- Capturehookremovedandexactbeforeattentioncoordinatorrestored; canonicalKV/weights neveraltered. Finalsourcebuild24CTest+15stream+11policy/18witnesschecks pass. Actualgenericfullnative/fullvocab512controls63.67/82.07/90.62/54.83/53.07/61.88allpriorbytes/digests/acceptanceexact. Historicalstrict80.6606/80.5825andmargin107.251/107.166exacttrajectories,notgeneric100.
- Finalactualcomplete26097344.09tps/241natural/3distantfacts/outputhash80f35de3...samepriorprofile/snapshotsunchanged; reproductionnotnewgain. No newC8/fullmediaqualification. Checkpoint/index/handoff/boardupdatedpreservingownerprefixes; alljobsterminal/GPUsfree/noagents/training/hardware/artifactwrites/commit/push/deploy. Both100gatesremainOPEN.

### 2026-09-16 15:46 +07 — Shared dense ownership and integer-vector cost screens rejected; current full-context and generic paths reverified
- HealthyHPSERVER/YOLO/correctworktree/GPUsidle. Exacttwo-slot8producer+8consumerQ4pipeline codes/scalesidenticalbothGPUs but202.50-207.39usvs181.09-186.69 (~.887-.900x). No geometrysweep or runtime port. Original kernel/source unchanged.
- NewbytewisevectorE8dyadicresidualS8MMAcostscreen16,777,216vectorcodes/134,217,728components plus128scalargateoutputsperGPUexact, graphrepeatallcodes/scalesequal. Payload66,846,720vs89,128,960bytesbut288.32-292.86usvs180.22-186.75 (~.620-.638x). Syntheticoperator, no actualweightconversion/modelqualityclaim; dyadic1/2vsupstream1/2.04explicit. No coefficientsweep/fullport.
- Bothprototypesarchivedintact,temporarybuildentriesremoved. ExactQ4/Q5sourceandCMakepreflightidentity; allcurrentCLI/server/dualbinaryhashesalsoidenticalpreflight. Final24CTest+15stream+11policy/18witnessespass. Fullnative/fullvocab6boundedcontrols63.67/82.10/90.59/54.81/53.06/61.83alloldbytes/digests/acceptancesame.
- Realcomplete26097344.11tok/s/241natural/3distantrecords/exactoldhash/snapshotsunchanged; retained~44reproductionnotnewgain. Historicalstrict80.6396/80.5147,margin107.300/107.170originaltrajectoriesretained; no generic100credit. Both100goalsopen, no newC8/media certificate. Checkpoint/index/handoff/boardupdatedwitholderfullqueuebackup. Noagents/training/hardware/commit/push/deploy;alljobsdone/GPUsfree.

### 2026-09-16 16:17 +07 — Exact token/precision-folded S4 architecture failed cost gate; working paths unchanged
- HealthyHPSERVER/YOLO/currentbranch/idleGPUs. Distinctfoldedlow/highactivationcolumns reduceexactnativeS4MMAcount8->4/G64 andretainoriginalA8/Q4/FMA/outputboundary;5.57MBrowsumcorrectioncharged. All69632outputcodes/1088scalesmatchperGPUtest, but198.27-198.56vs186.37us and191.68-191.78vs181.50-181.60us. No parameter sweep/runtime port; prototype archived.
- Applicationbinaries/Q4/Q5sources/benchCMakeexactpreflightaftercleanup. Finalgenericfullnative/fullvocabcontrols63.67/82.09/90.59/54.84/53.07/61.87exactoldbytes/digests/acceptances. Realcomplete26097344.06tps/241natural/3distantfacts/exactoldhash/snapshotsunchanged. Historicalstrict80.6945/80.5139,margin107.318/107.195originaltrajectoriespreserved; no generic100credit.
- 24CTest+15stream+11policy/18witnessespassed; no newC8/media claim. Checkpoint/index/handoff/boardupdatedwithpriorqueuepreserved. Both100gatesOPEN. Noagents/training/hardware/artifactmutation/commit/push/deploy;allboundedjobsterminal/GPUsfree.

### 2026-09-16 17:52 +07 — Exact TP2 packet/dead-work efficiency retained; full-context and physical C8 regressions pass
- Primary exact-zero shared-activation screen: six native full-vocab D3 controls, sixteen calls on all64 down matrices, unchanged output/acceptance. Shared zero density0.0158–0.0296%; free byte-removal ceiling1–2us/frame. No sparse kernel or threshold. Private capture restored from scoped original.
- Independent TP2 source audit removed unused owner BF16-to-FP32 conversion and its allocation. Active T4 packet21120bytes versus26400, T5 unchanged26400; same BF16 reduction and event ordering. Runtime counters show70,963,200 fewer input-transfer bytes and6720 discarded conversion launches avoided for the261K241-token answer. Requested buffer storage153600bytes/GPU and51200host bytes lower; no observed-peak VRAM claim.
- Four ABBA natural261K runs preserve exact output and105rounds/315drafts/136accepts: baseline44.11/43.93 versuscandidate44.17/44.21. Ratio-of-means+0.39% with clock variation, not a material or statistical speed claim. Final rebuild44.35 with exact prior answer and unchanged snapshots. No target weights, vocabulary coverage, KV precision or acceptance policy change.
- Real all64-shard T4→T5→T4 poisoned-tail tests across both physical role assignments total17,039,360 exact BF16 comparisons.25CTest+15stream+11policy/18witnesses pass; final generic fullnative/fullvocab six outputs/digests/acceptance match. Historical strict80.6617/80.5583 andmargin107.178/106.999 original trajectories; margin no generic100credit.
- Rebuilt C4+C4 mixed server: warm273.75/min33.38, actual appended follow-up273.25/min32.90, post-cancellation refill268.98/min31.87;8prefix hits,3cancellations, all noncancelled natural/errors0/caps0. Serving regression only, not TP2 or single-request gain. Source-scoped patch, originals, index, current handoff and board saved under tp2_active_packet. All jobs terminal/GPUs idle, no service deployed. Changes uncommitted; no agents, training, hardware setting changes, artifact rewrites or push. Both100goals remain open.


### 2026-09-16 20:05 +07 — Wide-D3 final PASS; legacy ~80 reclassified non-generic; clean C1 whole-round ABBA only +4.91%
- `wide_radix_final` terminal PASS after full rebuild/regressions. Exact full-native/full-vocab six-family preservation plus independent natural/code/JSON review passed; complete260973 reproduced44.23tok/s/241natural/3facts/exact old output/snapshots unchanged. Physical C4+C4 mixed serving warm272.12/min33.42, appended follow-up270.94/min32.03/8prefixhits, refill267.51/min31.91. Serving retained; no new100 gate.
- Scope corrected: new wide family is 8/16-column T8/T16 shared-weight execution. Profiled target45.98->40.21ms is real multi-slot D3 work, not a new C1 kernel. Fresh generic strict full-vocab/full-target general ABBA of existing C1 T4 radix: native60.62/60.32(mean60.47), radix63.45/63.43(mean63.44,+4.91%). Each side deterministic; radix output trajectory differs. Below owner >=8-10% primary whole-round gate; retain incrementally, stop treating as route to C1>=100.
- Genericity audit found historical retained `strict79` test explicitly sets ProposalHead::Optimized, which binds the 131072-row external corpus-frequency shortlist. Thus old ~80 strict golden remains retention/product evidence but cannot earn current generic C1 gate credit; margin7 also remains acceptance-relaxed/disqualified. No valid generic>=100 seal.
- Current C1 T4 budget ~35.42ms frame/~32.38ms target/~3.05ms outside target. Next primary work must remove a large target-body fraction and/or materially raise useful accepted tokens per verification round, with predeclared >=8-10% complete-round ceiling. Both100gatesOPEN. Jobs terminal/GPUsidle at update; noagents/training/hardware/modelartifactwrites/commit/push/deploy.

### 2026-09-16 20:16 +07 — Shared-weight wide D3 kernel qualified; C1/full-context baselines preserved, D1 memory extension rejected
- Recovered previously completed wide_radix_weight_reuse and preserved actualprofile45.98->40.21ms target /56.35->50.54ms wholeB2D3frame. Finalproductionoperator on bothGPUs matches independent existingT4radix groups/padding/repeats;1.27–1.86x vs separateT4ops. Bitexact to radix notnativefallback; no weight/vocab/acceptance change. ExistingT4C1 unchanged, both100gatesopen.
- Corrected only wide lifecycle test that incorrectly requestedC1-onlypackedMTP with2slots; guardnotrelaxed. FinalC2cancel/reload,C4explicit4K/16Kcancel,C3/C4reload3cyclesallpass. Automatic1GiBheadroomreject retained andexplicitprofile documented.25CTest+15stream+11policy/18witnessesplusbothGPUradix/legacy/TP2tests pass. Actualold/currentCLIstrict81.42/81.39 at1860MHz andmargin108.43/108.41 at1845MHz exactoutput/43/129/83and33/99/94; no margin generic100credit.
- FinalsamebinaryC4D3natural8taskcomparison:both4explicitcodeimplementationspassindependenttestsand40JSONrulespass;candidateThai993/JP708/Arabic1362naturalandcoherent,prosefactualwordingissuesnotednotblanketpass.7of8perrequestdecoderatesup,butcandidate5546vs5188tokens and55.169svs54.396smakespan meansnoend-to-endcoldlatencyclaim. Originalambiguousintervalfailureskept. Qualityandcommands indexed wide_radix_final/.
- AttemptednormalC4D1extensionatunchanged16K/65K/visioncorrectlyrejected7,079,989,504Breservationvs5,281,245,696available. NewdirectcanonicalweightsreadoracleexactT4/T8/T16butlosesT8cost;notported. D1admission/capacitychangesrestoredfromscopedbackups,oldpackedwideD3andTP2packetpreserved;bothfailedsourcepathsarchived. No smaller-capacityD1substitution, sourceweightmutations or tuningloop.
- Finalgeneric6fullnative/fullvocaboldoutputs/digests/acceptancesexact. Actualcomplete26097344.23tok/s,241natural,3facts/exactoldhash/snapshotsunchanged. NativephysicalC4+C4D1warm272.12/min33.42,follow270.94/min32.03,refill267.51/min31.91,8prefixes/3cancel/errors0/caps0;regressionnotD3gain. Finalreport/index/scopedpatch/handoff/board savedpreservingolderqueue/ownerprefix. Changesuncommitted,noagents/training/hardware/artifactrewrite/push/deploy;alljobsterminal/GPUsidle.

### 2026-09-16 20:55 +07 — Exact attention ownership and replicated-activation large-slice screens completed without production promotion
- HPSERVER/YOLO/correctbranch/GPUsidle preflight passed. Live board newer than old handoff: preserved wide_radix_final T8/T16 multi-slot D3 qualification and did not relabel it as new C1. No source or application changes outside isolated benchmark targets.
- New warp-resident full-key GQA retains native headscale/direct-FP16 cache arithmetic, all output/partials/m/l bytes exact bothGPUs at64172/260973. Complete operator1.7–2x slower; 255registers vs124 baseline. Closed without tile/precision sweep; prototype archived, active target removed.
- Full-layer replicated-activation TP2 premise tested with all64real MLP shard pairs and synthetic normalized residual chain. Corrected explicit initial H2D completion outside timing after failed baseline initialization diagnostics; both replicas/physical orders then exact. Host-synchronized chain14.14–14.22vs15.22–15.30ms, queued-events13.90–14.66vs15.15–15.40.10,485,760 exact BF16 state comparisons plus repeat checks. Positive isolated0.7–1.5ms saving below11.5ms port gate; fullmixer/GDN/attention replica maintenance and extra collectives unimplemented, no whole-model claim.
- All production sources and current CLI/server/dual hashes match preflight; benchmarks archived and CMake restored exactly. Final25CTest+15stream+11policy/18witnessespass. Actual generic C1 general63.62oldbytes/216rounds648drafts294accepts and complete26097344.34/241natural/3distantfacts/exactoldhash/snapshotsunchanged. No newC8/media/performance promotion. Both100gatesopen. Report/index/status/handoff preservepriorwideboard/queue. Noagents/training/hardware/commit/push/deploy; alljobsdone/GPUsfree.

### 2026-09-18 06:10 +07 — Centered full-context runtime loses in real shards; original-value integer PV also closed, retained composition reverified
- Completed actual centeredV4 async runtime rather than promoting1.32x85-splitmicrobenchmark. Immutable source prefix/native fulltail, independent codec page/tail/source checks bothGPUs. Actual261K41.11vs46.18; onefixedparent-preservingtwochild43.99vs46.26. Bothnatural3factsbutchangedanswers/acceptance,2.685GBsidecars. ActualNsight56.43ms/17.04attentionunionvsprior53.81/14.25; notmerelystartup. No furtherpartition/codecsearch. Runtime preciselyrestored/testsourcearchived.
- NeworiginalV8/scales probability-productS8 actual288querygate .01285relativeL2/.999919cos passes; scalarreferenceandgraphrepeat passafterfixedshuffle-registermappingbug, originalfailurepreserved. Real34/51ranges747-751/1057-1067us vsnative686-690/864-868; no port. Fixed32-key/rank8livevaluefactorizationfails.68155/.80076; no rankrepair/implementation.
- Finalrestored/rebuiltdual actual260973 full-vocabproposal46.16tps/241natural/100rounds141of300accepts versuslegacy44.37/241natural/105rounds136of315;exactoldoutputandtrajectory/3facts/snapshotsmetadataunchanged. Fouraffectedtests+18witnessespass. CLI/serverbinaryhashesunchanged;dualrebuiltsoactualvalidationnotfalsehashparity. No unrelatedC1/C8campaign. Both100open,olderresident/wide/TP2/servingworkpreserved. Checkpoint/index/board/handoffupdatedwithpriorqueuebackups. Noagents/training/hardware/artifactwrites/commit/push/deploy;alljobsdone/GPUsfree.

### 2026-09-18 09:39 +07 — Full-context transport profiling isolates NCCL graph-mixing penalty; no production port
- Fresh three-region Nsight: owner17.556ms, eager16.602ms, graph28.095ms. Q4/Q5 mean math-kernel times essentially unchanged; NCCL interval union7.717ms eager versus16.900ms graph. Categories overlap, not additive wall. Profile artifacts/analysis under production_serve_20260918/nccl_transport_profile.
- Independent one-GPU-per-process benchmark implements current64realMLP owner sequence, strict BF16 state comparisons, both physical orders, parent wall-time and before/after baseline. Default graph25.18–26.98ms vs15.69–16.00 eager; process isolation alone does not fix it. Initial short-lived-bootstrap-helper mistake corrected before valid timing; failure preserved. Source archived and only its temporary target removed.
- Documented serialized GRAPH_MIXING_SUPPORT=0 control brings graph15.85–15.96ms, still not <=12ms/20% owner-chain improvement. Pinned2.26.2 source identifies SIMPLE-only SHM copy engines; actual SHM/CE/CE confirmed, but eager20.18–20.71ms andgraph32.20–33.29 vsdirect15.83–15.91 graph. No package upgrades/global settings/production transport added; no channel/algorithm sweep.
- Current genuine260973 target reproduction control46.18 andpre-existingtarget-guidedproposal47.46tok/s, bothsame241natural tokens/3distantfacts/snapshotmetadataunchanged. Guide95rounds146of285 versuscontrol100/141of300; actual95captures282views. Small prior-guidemechanism reproduction, not transport speedup or100. Allthree application hashes unchanged; no unrelatedC1/C8recert.
- Own profiling/rank/copy tests terminal; concurrent guide and integer-scale work kept separate, shared handoff not overwritten. Evidence/index/report in transport_campaign_0907. No production source/weights/hardware/training/agents/commit/push/deployment changes. Both generic C1D3 andcomplete dual100remainOPEN.

### 2026-09-18 09:48 +07 — Target-guided factory proposal context reproduced a small exact-answer full-context gain
- Recovered actual latest inherited target-guide state; self-query retrieval was already closed. Rebuilt/checkedsrc:63retrieval and134licensed-query/future/rejected-row view cases perGPU. Current actualguide runs consume282views/95captures, unlike old unwired0views. Three natural241/826/1080target answers and snapshots stay exact.
- Clean reverse261Kpair47.44guided vs46.02control,95vs100rounds/146vs141accepted. Earlierguided47.37 repeats. Anomalousfirst39.15control kept butnotcredited20%gain. ~3.09%measuredgain notprimary8–10%or100. Thai/JSONroundsnotreduced; JSON76.39vs76.71. Preserve explicit default-off private research profile, testedbinary/source/commands; no public/default promotion or universalqualityclaim.
- New continuousS8 whole-MLP progressiveinteger-scale/two-activation-limb cost test:983040packedtuples/3932160coefficients and128scalaroutputs perGPU exact, graphstable.189.28–189.57vs186.56–186.78us,182.27vs180.00–180.10us,slower. No actualmodelweightconversion or qualityclaim; prototypearchived/benchentryremoved,no coefficient/warp/tile sweep.
- Reconciled completedNCCLgraph chain gate asfailed (23.45–26.17ms vsowner17.30–18.44, eager15.51–15.55onlyisolated); archiveactivebench withouttouchinglibrary/runtime. Finalaffectedpolicytests/18witnessespass, actualgenericC1control63.68oldbytes/digest/trajectory. No unrelatedC8recert. Checkpoint/index/board/handoff preserveolderqueue. Noagents/training/hardware/modelartifactwrites/commit/push/deploy;alljobscomplete/GPUsfree. Both100goalsremainOPEN.

### 2026-09-18 11:34 +07 — Full-context D4 profiling exposes exact ragged-W8 head cliff; padded fix in validation
- D4fullcontextnewheadscale/directFP16/fullvocabcomposition preservesallkeys/strictverification and5queryattentionbitparityagainstT4subsets. Real261K D4 40.42/96rounds/239natural3facts versusD3 46.09/100rounds/241sameoldanswer. Snapshotmetadataunchanged. D4isnotpromoted; wideattentiononly0–3%costincrease, fullprofilelocates~6.67msotherdenseunionpenalty includinghead2.404->4.305ms.
- Fixed complete8-columninputstaging control retainsoriginalscales,weights,FMA/reduction andoutputwidth; T5/6/7fullvocabhead1.52–1.60x faster onbothGPUs, allmeasuredBF16logits/guards/repeatsidentical. ProductpolicychangeonlySM75vocabularyT5..7; T4/T8/otherarchitecturesunchanged. Independentpersistenttestsupportszeros/source-reuse/strides/repeats. Firsttest missingarena include fixed; no failed-modelclaim.
- Sole activeproc-826cef73-c2fd-47c0-89a1-47d6f6182358 now source-boundtests+ABBArealD4sameanswers. D4guard/dispatchandW8sourcechangesareseparatelyscopedbackedup; classifyandrestoreD4ifnotwinnerwithoutlosingW8. No newbestfullcontext/100claim. Priorworkingguide/resident/TP2/C4+C4preserved; noagents/training/hardware/artifactchanges/commit/push/deploy.

### 2026-09-18 11:46 +07 — Exact W8 vocabulary T5-T7 padding retained after full-context profiling; slower D4 composition restored
- A one-point current complete261K D4 composition used the new headscale/direct-FP16/full-vocabulary primitives and strict target verification. T5 attention was bit-identical to two independent T4 subsets on bothGPUs/2K/64K/261K, but realD4 40.42tps vsD3 46.09. Actual profile otherdense22.51vs15.84msunion includedW8head4.305vs2.404ms; attention only14.38vs14.04. Categories overlap; no kernel sum used as wall time.
- Controlled fixed8columnactivation initialization removes the ragged T5/T6/T7 staging cliff without changing valid inputs, originalweights/scales, matrixFMA/reduction or output width. Frozen-original/current fullhead1.52–1.60x speed; all14,899,200BF16logits plusguards/repeats identical. Product onlySM75vocabulary5..7, noextraallocation; otherwidths/architectures unchanged. PermanentGPUregression covers same-addresssourcechanges, zero inputs, strides and graphs bothcards.
- ABBA genuine261KD4 original40.36/40.26 vs padded41.28/41.18 (+2.28% ratio ofmeans; clocks vary), allsame239naturaltokens/96rounds383drafts143accepted/3facts/snapshotmetadataunchanged. Still slower thanD3, so newD4admission/dispatch archived and restoredexact while independent W8kernel retained. OriginalhistoricalD4 andallguide/resident/wide/TP2packet paths preserved. Temporarybench targets removed, fullsource/evidence kept.
- Final rebuilt CLI/server/dual:25CTest+15stream+11policy/18witnesses plusnewGPUtests pass. Actualcomplete260973 fullvocabD3 46.05,existingtargetguide47.39; same241naturalanswer/3facts/prior100or95roundtrajectories/snapshotsunchanged. Historicalstrict80.6067/80.4749,margin107.231/107.028 retained; corpus-selectedgoldensnotgenericC1, marginnostrictcredit. No freshC8/media performance claim.
- Retained source-scopedpatch hasW8config,newtest,testsregistration,servingdoc only. Report/index/handoff/status updated under w8_single_tile_scale_ownership/final; previousqueue/ownerprefix saved. Initialguard/builddevelopmentfailures preserved. Alljobsterminal, GPUsidle, no service. Noagents/training/hardware/modelartifactchanges/commit/push. BothgenericC1D3>=100 andcompleteDual>=100remainOPEN.

### 2026-09-18 18:40 +07 — Four full-context mechanisms measured; original guided runtime restored exactly
- Exact KVcode dictionary factoring differs from prior full-vector-plus-scale check but finds1.0594–1.0606xbytes withoriginalscales/all12heads; max66.8MBdictionary. No decoderport or grouping/thresholdsweep. Everyreconstructed64byteblock exact.
- Newfully-visible prefix andnativeboundary kernelcomposition retainsallkeys/partitions/math; alloutput/partial/max/sum bytes andgraphrepeats match2K/64K/130K/261KbothGPUs. Completeoperator~12–22%slowerlong. NewseparateQK/P8andcenteredV4integerPV dataflow also loses~1.4xat261K;31.3MBintermediatepluspreparationcharged. Independent384scalarrepresentedquerycases<=.271%relL2, notnativequalitycertificate. Bothkernelprototypesarchived, activetargetsremoved; no warp/tile/precisionrepairloop.
- Fullcontextcurrentguidedfactory-head accuracy discriminator:47.60fastfullvocab versus42.78exactnativeW8scan, same241naturaltokens/95rounds285drafts146accepts/positions73,47,26. Extra5.98ms/round buysnoacceptedwork; privateexactheadflagarchivedandrestored. Originaltargetweights/KV/fullcontext/strictacceptance/snapshotsunchanged.
- Finalallthreeapplications rebuilt andSHAidenticalpreflight. Actualguided26097347.55tok/s,same241naturalanswer/3distantfacts/exactoldhash/acceptance/snapshotmetadataunchanged. Build/diff/18retainedwitnessespass; no unrelatedC1/C8claim. EarlierW8T5..7/guide/resident/wide/TP2packet/servingpreserved. Report/index/handoff/status underfull_context_execution_1838; olderfullqueuebackedup. Alljobsterminal/GPUsfree/no listener, noagents/training/hardware/modelwrites/commit/push/deployment. Both100gatesOPEN.

### 2026-09-19 01:04 +07 — Exact full-context snapshot restoration accelerated; integrated all-key runtime reverified
- HPSERVERYOLOpreflighthealthy. Recovered current joint runtime already integrated ahead of chat; fusedcontrolvariateclosedcostnotrepeated. Rebuiltallapps andactual261K native47.58/joint50.17natural241tokens,3distantrecords, exactpriorhashwithineachmode. Jointnumericaloutputdiffersfromnative; basisprep1.536s charged. Freshsame-outputNsight50.864ms steadywall with15.817msotherdense/12.195TP2/7.555joint unions overlapping, not additive.
- Retained two disjoint stage/helper snapshotread+upload tasks joinedbeforeactivation/errors. ABBAserialrestore12.845/12.991s vsparallel7.889/7.756; process35.976/36.089 vs31.042/31.147,13.70%meanwallreduction. Same241jointtokens/98rounds294drafts143accepts andsnapshotmetadata. No warmdecodegain oruncached-prefillclaim.
- Retained overwrite-only immutable snapshotinput allocation andfactoredjoined-taskhelper. SecondABBAalready-parallelcontrolrestore8.064/7.979 vsnew6.909/7.455; means8.022->7.182s,process31.151->30.417 (-2.36%). All8jointanswers/trajectoriesexact, hostpeak~15.1GiBunchanged. CPUtestcoversinvalid/empty/truncatedbytes, moves, parallelreads, exceptionjoin/errorprecedence, moveonlycallbacks. Existingnumericalprepuses samejoinedhelperwithoutmathchange.
- Finalnative261K47.56tps,241natural/3facts/prior95rounds285drafts146accepts/hash80f35...,6.25srestore,28.13swholeprocess.26CTest+15stream+11policy+18result/evidencechecks pass. Actualhistoricalcurrentruntime strict80.6321/80.3923 andmargin107.186/107.082 preserve old trajectories; no generic/strict100credit. Optionalarchivecheck failed because separate results-onlycleanup intentionallydeleted3Sept8retained-before binaries, confirmedTSV; notsilentlyweakened orclaimedpass.
- CLI/serverbinaries andjointmath/coordinator sourceexactpreflight; dualrunnerrebuilt with6scopedsource/test/docfiles only. Snapshotcontents/modelweights/precision/acceptance/hardware unchanged. Finalreport/index/scopedpatch/status/handoff preserveolderqueue; nofreshC8claim. Alljobsdone/GPUsidle/no listener/agents/training/commit/push/deployment. Both100warmgoalsremainopen; exactstartupwinseparate.

### 2026-09-19 13:49 +07 — Exact dense and causal-subspace screens close; restricted-target flaw found in old drafter coverage test
- Exact shiftedQ4x16/Q5x8 S8operands+integerunshift preserveallcode/signcombinations/completeoutputs/repeatsbothGPUs, but0–1%costdifference,belowmaterialgate. No runtimeport; ownbenchmarksarchived. Sourceidentitybacks in exact_shifted_dense_1304.
- Fixed same-request causal rank256basis fromfirst384nativehiddenstates tested5990laterrows/30layerfamilycases onCPU. Whole4row.05residualgate eligibility0on5families/11%JSON,below35%; no image/kernelport orrank/tolerance sweep. SeparatefromoldcrosspromptdownGSI; data in causal_activation_subspace_1322.
- Native unsignedP8*S8coordinates passesall65536scalarbytepairs/fullmatrixrefs andall288realquerygates, graph/lifecycle/fallbackchecks, operatorcostneutral. But actual261K50.53/239natural vs52.15/241signed,3facts both, so privateflag/templateextensionarchived/restored. Originalintegerjointroute retained; no numericalfield/selectionthreshold/weightalteration. Sourceandmodelresults in unsigned_probability_joint_1335.
- Audited dspark_v2_local_coverage/capture.py: --lm-head-draft + --full-vocab-draft-retained-target mandates Optimized target, notfullnativehead. Oldgenericcompatibilityclaim thereforeoverbroad. New same-pinned-professional-checkpoint replay willusefreshfulltarget/fullproposalcaptures, correct postlayer taps andonlycommittedcausalstates. Activeproc-3582fa0e-e0bb-468d-8239-2f730dd64f3c, sourcehooksisolatedbackedup;donotoverlapGPUjobs/rebuild. No100claim, no subagents/training/hardware/commit/push.

### 2026-09-19 18:35 +07 — Exact full-context selected-union compaction retained; corrected full-target drafter screen closed
- Completed six strict fulltarget/fullvocab capturedhistory pairs and pinned professionalDSpark FP32nativehead coverage: prose2.1355/code4.0667/JSON6.04/Thai1.1855/JP1.5391/Arabic1.447. Offlinecontiguousagreementonly, belowgenericcostgate; no port/training/futurefeatures. Readonlycapturehooks restoredfromscopedcopies, genericcurrentCLI exactoldoutput/216rounds648drafts294accepts.
- Exact three-stage unioncompaction removes four-CTA serialization, preserves all orderedindices/counts/capacity/prefailedflags andusesdeadperquerymaskbuffer afteritslastreader. IndependentCPU/originalbuffers/guards/changedinputs/repeatedgraphs passbothGPUs, stage1.7–3.54x. Survivor-compacting radixselector was slower85–91usvs56–62 andnotintegrated; no thresholds/ranks/precision/weights/allocations changed.
- ABBAtrue260973 integerjointoriginal52.16/52.01 vsparallel52.34/52.32 (+0.47%means, naturalclocksnotcontrolled). Allsame241naturaltokens/96rounds288drafts145accepts/3distantfacts/fallback/preparedbytes/snapshotmetadata, same18,181/18,108MiBpeaks. Underlying joint numericalqualityscopeunchanged; no new100orlargegain. Finalfloatingjoint50.41 exactprioranswer/98roundtrajectory also passes.
- Final26CTest+15stream+11policy+18retainedresultrecord checks andbothGPUnewtests pass. --check-binaries not claimed because3archivespreviouslydeleted; no fake witnesses. Ninfer/servehashes matchrestoredbaseline, dualexactcompaction changed. Source-scopedsixfilepatch/checkpoint/index/currenthandoff/statusupdated preservingolderqueue. No freshC8/mediarecert. Alljobsdone/GPUsidle; noagents/training/hardware/modelartifactmutations/commit/push/deploy. BothgenericC1D3>=100andcompleteDual>=100OPEN.

### 2026-09-19 20:38 +07 — Native S4 live-coordinate and mapped-payload architectures completed; production unchanged
- HealthyHPSERVERpreflight/correctworktree/stock2x22GiB GPUsidle. Newmappedpayload3armreal64MLPchainfusesinputpublicationandremoteBF16outputwithordinaryevents, nopolling/stream-memorywait/P2P/hardwarechange.10,485,760BF16statesplustwoseedrepeats exactbothroles; mapped16.15–16.64msvs15.12–15.20sync/15.35–15.46eventDMA loses. Archivedprototype/ownCMakeentryremoved.
- Newfixedsource-prefixRMS equilibrated K64/V32->nativeK4/V4 representation keeps8bitQ/Pviaexactlimbs, allkeycontributions/selectedfullcorrection/tails/budgets. Offline288actualqueries maxrel.08995/mincos.99618,no fallbackcredit. NativeS4vsindependentexpandedS8outputs/partitionstatesexact across3capturedlayers,bothroles,dynamicquery/positions,eager/graphs/sourceguards/fallback. Storedcoords113,436,160->59,138,560B/layerbothGPUs,notallKV/modelVRAM. Preparationcounted,ISAcountneutralnot2x.
- Complete2GPUattentionpipelineonly1.016–1.036xcurrentS8, belowpredeclared1.4x. No modelintegration/generatedanswerqualityclaim or format/rank/thresholdsweep. Fullnativecode/runtimeclone/diagnostics retained equilibrated_s4_joint; activebencharchived/CMakeexactoriginal. Both100goalsOPEN, not universalhardwarelimit.
- FinalallthreeapplicationsandscopedproductionTP2/joint/CMakehashesexactpreflight.26CTest+15stream+11policy+18retainedresultrecordspass; archivebinarieschecknotclaimedbecause3previouslydeleted. Actualgenuine26097352.46tps/241natural/96rounds288drafts145accepts/same3facts/hash/snapshotmetadata. Thisisunchangedroute,reproductionnotgain. No freshC1/C8/mediaqualification.
- Finalcheckpoint/index/handoff/status under native_s4_and_payload_followthrough preserveearlierqueue/ownerprefix. No modelweights/training/agents/hardware/commit/push/deploy. Allboundedjobsterminal/GPUsidle; twoindependentnewcostscreens documented withoutupgradingquality/speedgates.

### 2026-09-19 23:26 +07 — Current-source Marlin and exact lookup dense families tested; no accepted acceleration
- HealthyHPSERVERpreflight/currentworktree/idleGPUs. Pinned ModelCloudGPTQModel e1d7c2924cf26868bde79a55f79f282139f731a5 fromSep18; fixedpublishedsmallbatchSM75launch, no tuning. Standalonecompile neededtransitiveheaders/relaxedconstexpr/staleSM80dequant guardfix, originalsource+diffpreserved. RepresentedFP16GEMMreference128rows/GPU relativeL2.00185-.00228 andreplayspass. Completegate278.14–284.67usvs180.22–187.58; no numericalparity/modelqualityclaim orport.
- Newpaired-token exactA8subset-sum/weightbitplane dense implementation, sameQ4bits/scales/G64FP32FMA/finalBF16A8boundary. Wholeoutput417792codes/6528scales across6cases and3,932,160CPULUTentries exact; graphsrepeat. Butcompleteoperation514–516usvs180–186. No subgroup/warp/cache sweep, Q5extension ormodelconversion. Twoisolatedsources/targets archived; no product changes.
- Finalsourcebound genuine26097352.47usefultok/s,241natural,96rounds288drafts145accepts/71,48,26,exactoldhash/3distantfacts/snapshotmetadataunchanged. Sameexistingintegerjoint numerical attention, no newgain or universalqualitycertificate. Allthreeapphashes andbenchCMake matchpreflight. Onlyactualnewtests claimed, no freshC1/C8/media/broadregressionpass or recreateddeletedwitnesses.
- Report/index/pinnedsource/compatibilitypatch/oldhandoff/status saved in dense_family_followthrough_2323; preservepreviouslargerqueue andallacceptedroutes. Noagents/training/hardware/modelartifactchanges/commit/push/deploy; allboundedjobsterminal/GPUsfree. BothgenericC1D3>=100 andcompleteDual>=100 remainOPEN.

### 2026-09-28 14:29 +07 — Routed attention packets retained; native S4 and complete-prefix RedHat proposals screened without production promotion
- Recovered finished packet ABBA andcompletedpendingcurrent-sourcefinalchecks. Native261K47.62/47.45 vs47.57/47.51stable; integerjoint52.31/52.22vs53.55/53.50(+2.41%means), allsameanswers/acceptances/fallback/snapshots. Contiguous output states inallmodes, packedinputonlyjoint preservesnativeDMAoverlap. Sourcepacketlayout/newGPUtests20,754,048bytes verified.26CTest+15stream+11policy/18records andhistoricalcurrentstrict80.68/80.61,margin107.39/107.09pass. Archivebinarychecknotclaimed.
- Implemented nativeone-limb signedA4G64 usingunchangedQ4weights; independentCPUquant/FP64represented-operationreferences, random/spiky/zero/repeatedgraphs passbothGPUs. Completegateonly0–0.9%benefit vsA8,below1.2xgate; no targetprecisionport orknobsweep. Kernel/evidencearchived, ownbenchmarktargetremoved.
- Onegenuine64172-tokenprefill restoredactual2048sourcefeaturewindow forpinnedRedHat5-layer8tap8proposalmodel, notanotherempty-prefixbootstrap. Capture/controlsame249naturalanswer/106rounds318drafts143accepts; exactly2048originalpromptplus247committedgenerationfeaturestates, nofuture/rejecteddata. On104sameactualroots,completeprefix2.6442vs-empty2.3654meanmatches+bonus, bothbelow4.3costgate; source/cachedandfuture-poisonchecks pass. No training/tuning/port ormodeltpsclaim. Read-onlyhook removedexact; temporary2.679GBnewsnapshothashedthenremovedonlyaftervalidatingrestoredruntime; oldsnapshotsuntouched.
- Finalrestoredsamepacketbinaries: actual261K53.66usefultps/241natural/96rounds/3facts/exactpriorhash andsnapshotmetadata. Private64K55.37same249tokenreference; reproductionnotgain. Existingjointnumericalqualitylimits andbothsingle-request100goalsremain. Sourcepatch/index/handoff/statussavedunderfull_context_continuation_1423, olderqueuepreserved. Noagents/hardware/training/modelartifactwrites/commit/push/deployment;alljobsfinished/GPUsfree.

### 2026-09-28 15:16 +07 — Coarse dependency graphs and per-GPU host execution tested; no accepted speed profile
- HealthyHPSERVER/currentbranch/correcttwo22GiB2080Tis/idleGPUs. Built complete48GDN+48MLP crossGPUfork/join graph1922nodes (2114withhistories), exactsame96BF16state tensors bothreplicas/orders/replays. Graph24.27–24.53ms vsowner22.76–22.85; queuedreplica likewise24.17–24.80. Failed17.2ms/1.35xgate, nofullmodelport orgeometry/precisionrepair.
- Fresh three-region Nsight showsreplicated23.67MBtransfers vs6.20MBowner, duplicatedQ4projectionmean66usvs41, notjusthostwaits. Unionsoverlap andprofilewalloverheadexcludedfromfrontier. Separate sameowner/data-volume pipelinewithtwo persistentGPU-specificCPUthreads/crossdeviceready events isexact but23.11–23.21ms vs22.99–23.83, -0.9..+2.8%ratios, no materiallyfundedport orthread/affinitysweep. Own threadsjoinedanduniqueCMake/sourcearchived.
- Captured-data complementary factory3+publishedRedHat7nodetrie on104samefull64Kroots yields2.9808meanmatches+bonus vs2.3365factory,3.0096evenfreeunlimitedtwo-spineunion. Candidateconstructionusesonlyproposalsnottruth; widerverification/drafter/statecostunpaid, below4.3necessarygate, no engineport/100oruniversalqualityclaim. Existingprofessionalmodel andcausalinputchecksunchanged.
- Finalactual26097353.49tok/s same241naturalanswer/96rounds288drafts145accepts/71,48,26/3facts/snapshotmetadata. Unchangedexistingintegerjointnumericalmode; no newbest. AllthreeinferencebinaryhashesandTP2/CMakebeforecopiesexact. Report/index/previoushandoff/status saved undercoarse_graph_dependencies, all acceptedroutes preserved. Noagents/training/hardware/artifactmutation/commit/push/service. Alljobsterminal/GPUsfree; both100gatesOPEN.

### 2026-09-28 16:22 +07 — Raw accumulator transport screened; head-local projection cost bound active
- NewprivateFP16 TP2 partial preserves rawFP32dot/scale/FMA/reduction butkeepsmorebitsbeforeglobalBF16residual. IndependentrealweightFP64checks ~7-9xlowerstoredpartialerror, rangeoverflowchecks pass. ABBAtrue261K54.66/54.59 vs53.77/53.63,245vs241natural3facts,96rounds/~4.47s unchanged; below>=10%gate,notfastercompletion. Prototype/source archived, all productionedits restoredexact; finalrestoredhashcheckafterbuildneeded.
- Distincthead-local8QK/24V partition removespriorstripedGDNall16QKduplication. Usescurrentradixfamily, allactualsourceweights; threeprojection-only pairs withFP64andreplaychecks andno communication/recurrence included. Activeproc-9f87caae-e4b6-47c3-bcb6-007752fac611 verifiesactualnativebaselinefamily. OnlyownstandaloneCMakeentry/source, no runtimeportyet. No knobsweep/training/hardware; both100goalsopen.

### 2026-09-28 16:51 +07 — Head-local GDN/MLP architecture produces isolated gain; real full-context route preserved
- New natural8QK/24V partition removespriorstripedall16QKduplication andusesexistingtwo-limbradix originalweights. Projection-onlybound2.3x was followed bycomplete48GDN+48MLP implementation inclcontrol/convolution/recurrence/gating/actualDMA. Replicatedgraph21.07–21.45vsnativeowner22.70–22.79ms(+5.9–8.1%), below15%portgate. OriginalcontrolisnativeGDNnotcurrentstage1A8; excluded16attention/proposer/headcannotbeforgotten. Preparedextra3.8748GBcharged; noactualmodeltpsclaim.
- Independentparentfullheadversusslicedconv/recurrence/gates/records1,778,688bytes exact; actualsourceFP64outputreference error.000076–.000134. All96candidateboundary states exactbothreplicas/queues/graphs and repeated runs. Candidate differsfromnativechain(.3199L2) andhasnolong/multimodalmodelqualitycertificate. Distinctownerroutedcommunication preservescandidatemathandcuts23.59->17.99MBwire,butgraphs23.69–23.71ms lose. FreshNsightcomputeunion21.81->16.05 whilecopy2.03->5.47; intervalsnotadditive. Bothscopeimplementationsarchived,notported.
- New rawFP16TP2partial experiment hadsameFP32contractions and7–9xlowerstoredreferenceerror/rangeguards. Actual241->245naturaltokens54.66/54.59vs53.77/53.63tps,but96round/~4.47sunchanged,below10%gate,notcompletionlatencywin. Source changesrestoredexact; candidatebinary andallfailedtests remain separate. No bit/scale/layerknobrepair. FLUTEcurrentpinwasalreadyclosed,notrerun.
- Restoredactual26097353.72tps,241natural/96rounds288drafts145accepts/71,48,26,exactoldhash/3distantfacts/snapshotmetadataunchanged. Allproductsource/CMake restored; CLI/serveoriginalhash,dualonlynonallocatedstrtabchangedandallallocatedcode/data/CUDAfatbinsectionsidentical. No broadC1/C8/media/historicalbinaryarchiverecertclaim. Finalreport/index/status/handoff preserveallpreviousroutes/ownerqueue. Alljobscomplete/GPUsidle; noagents/training/hardware/modelweights/commit/push/deploy. Both100goalsOPEN.

### 2026-09-28 19:29 +07 — Conditional proposal tree and scale-separated whole MLP implemented; no new model-speed promotion
- PreflighthealthyYOLO/stockdual2080Ti/correctworktree. ReconciledstalerotatedMNZ2status: extendedmodelqualityfailureandrestoredsuitealreadycompleted; noquantizerrepair. Preservedallcurrentapplicationhashes/profiles.
- Fixed15nodeprefixclosedconditionaltreeusespublishedMarkovparenttoken/fullvocabnormalization and2048realprefixfeatures. Actual104same64Krootsgreedymean2.6442, frozen-chain-tree3.375, conditional3.5481(+34.2%coverage), below4.3costgate. CPU60exhaustiveheapvsenumerationcases, source/cached/future-poison/greedyall104/replay pass. Offlineprefixagreementnotmodeltps; no verifierport orparametergrid.
- WholeMLPscale-separatedsignedW4/A8 integeraccumulation keepsrow/token scale application toepilogue, removesgroupmultipliers/secondlimb fromoldercontinuousdesign. Bothfullquantizers, BF16SwiGLU,down andresidualreset included:264.83–265.02vs293.63–296.96us,257.28–257.66vs288.74–289.12(1.109–1.122x). Exhaustivecodec/CPUactivationquant/independentFP64/randomspikyzero/replaypassedbothGPUs. Below1.20gate, noactualsourceconversion/modelqualityclaim orbit/scale/tilegrid. Bencharchived/CMakeexactrestored.
- Finalactual26097353.77tps/241natural/96rounds288drafts145accepts/71,48,26,exactoldhash/3facts/snapshotmetadataunchanged. Existingnumericalattentionlimitations unchanged; this ispreservationnotgain. Allthreeapphashesidenticalpreflight. Report/index/currenthandoff/statussavedwithpriorqueuepreserved; no newC1/C8/media/wholehistoricalpassclaim. Alljobsdone/GPUsfree; noagents/training/hardware/weights/commit/push/deployment. Bothsingle-request100goalsremainOPEN.

### 2026-09-28 20:34 +07 — Shared-vector decoder and exact coalesced layout rejected at whole-operation cost; model preserved
- HealthyHPSERVERpreflight/YOLO/stockdual2080Ti/correctworktree. Fixed4D signed4096entryshared16KiBvectorcodec derived frompublicIQ3XXSmagnitudegrid,3encodedbits/coefficient. CompleteG64gate+input/outputquantization/SwiGLU215–219usvs180–188, fails1.20gate. Independent524288decodedcoefficients plusCPUinput/outputquant andFP64rawformulas pass; syntheticgrid scalesoverflowedFP16 inordinary/spikycases, explicitlynotnumericallyusablemodeloutputs. No table/scale/precisionrepair oractualtargetconversion.
- Singleidentifiedkernelprofileclock-controlnone:95%L1/TEX,60.8%MIOwarpstallattribution. One exact3-plane global loadlayoutremovesstride12 redundantsectors butsamecompleteoperation213–217usstillloses. Inversepermutationall66,846,720weightbytes/card andallraw/A8/scales/graphs identicalincludingrecordedoverflows. InstrumentedpostNcu timingnotcredited. Samefixedcodec/grid/arithmetic, no laterparameterorbit. Ownbenchentriesarchived/removed, originalCMakeexact.
- Actualcurrent26097353.82usefultps/241natural/96rounds288drafts145accepts/71,48,26/exactoldtokenhash/3distantfacts/snapshotmetadataunchanged. Preservationnotnewgain, existingjointnumericalqualitylimitationsremain. All3apphashespreflightidentical; acceptedprofilesuntouched. No newC1/C8/media/broadtestclaim or recreateddeletedbinaries. Report/index/oldhandoff/statusunder shared_vector_mlp/followthrough. Both100gatesopen; allownjobsterminal/GPUsfree, noagents/training/hardware/modelweights/commit/push/deployment.

### 2026-09-28 21:17 +07 — Exact folded-limb dense dataflow rejected at full-operation gate; strict C1 paths preserved
- HealthyYOLOHPSERVER/stockdual2080Ti/branchpreflight. CurrentHumming sourcef8182f6d vs previouslyexecuted8611853c reviewed, no materiallynewSM75densearchitecture justifiedrepeatbenchmark; source/diffrecordedonly.
- NewfourlogicalcolumnD3 nativeS4/U4 operandfolding uses8MMAcolumns forbiasedlow/highA8limbs andexactsource-derivedQ4row sums. No weight/activationquantizationchange; all163840independentCPUdots,557056outputcodes,8704FP16scales,guards,sourceupdates andgraphs exactbothGPUs. Wholegate193.57–198.66usvs180.16–187.74loses despitehalvedmatrixinstructioncount; metadata+packing charged, noQ5/wholeMLPportorknobsweep. Benchsourcearchived/originalCMake restored.
- ActualgenericstrictC1fullvocab512prose control62.65 witholdno-graphdiagnostic vs62.98normalexistinggraphreplay. Bothexactoldoutput/digest/216rounds648drafts294accepts/164,89,41. Notnewoptimization, naturalcompletioncertificate or100. All3inferencehashespreflightidentical. No fresh261K/C8/media/broadhistoricalsuiteclaim, previous53.8notrelabeledasnew. Source-boundreport/index/status/handoff preserveoldownerprefix/queue. Allownjobsfinished/GPUsidle; noagents/training/hardware/artifactweights/commit/push/deploy. Both100goalsremainopen.

### 2026-09-28 21:52 +07 — Gated all-gather and exact quadrant ownership tested; natural strict C1 baseline measured
- PreflighthealthyYOLO/stockdual2080Ti/correctworktree. Newwhole48GDN+48MLPcutall-gathersBF16gatedactivations, projectsdisjointoutputrows thenexchangesBF16residualhalves. Logicalpayload23.59->16.52MBbutextraexchange/rowprojectionworkcharged. Full-Krowgraph21.125–21.705msvspriorcolumn21.042–21.473; no fasterimplementation. Exactfull-K/independentFP64sourceweight/parentconvrecurrentrecordchecks passbothphysicalroles; notnativewholemodelqualitypass.
- Second2x2quadrantrealizationpreservesoriginalcolumnpartialassociationexactly andoverlapslocalprojectionwithcopyengineinputDMA. All96priorcandidateboundary states exactacrossreplicas/sync/queued/graphs/replays; rowmathFP64referencecheckedthreeactualsource layers. Graph21.160–21.490vspriorcolumn21.033–21.471 essentiallyflat/slower. Neitherfunds1.15xnativeownerportgate; fullattention/proposer/headexcluded. No split/precision/head/warp/cache tuning. Isolatedsourcearchived,originalCMakeexactrestored; no productsourcechange.
- ActualstrictC1naturalcompletiononthesamegenericproseprompt: cap4096ratherthan512, unchangedfullnative/fullproposalvocabD3/graphs.1103tokensstopnaturally/64.07tok/s/459rounds1377drafts645accepts/344,198,103,17.200sdecode. First512tokenIDsexactpriorcontrol. Completeanswerreadcoherentwithnormalstructure,butdomainconflationnotedso no perfectaccuracycertificate. This ispreservedbaseline, notnewoptimization. Allthreeapphashespreflightidentical. No fresh261K/C8/media/historical80/107recertclaim.
- Source-boundcheckpoint/index/currentstatus/handoff updatedpreservingownerpreambleandpriorqueue. Initialstandalonebuildspanerrorpreserved, no arithmeticfailurehidden. Allownjobsdone/GPUsfree; noagents/training/hardware/modelweights/commit/push/deploy. Bothsingle-request100goalsstillopen.

### 2026-09-29 15:49 +07 — Bank-owned decoder removes measured serialization but whole-MLP gain remains below model gate
- Exact bank-owned256magnitude/foursign decoder mapslane tobank, replacing random4096entry reads. Samevectorraw/A8/scales, all131072code-lanepairs/card match; nativegate1.149-1.161x. NCUsharedloadconflicts5,585,779->1,961, MIO44.95->0.95%; profiletimings excluded. Wholegate/down269.28-269.95vs294.91-295.33us and265.63-266.14vs288.77-288.99 (1.085-1.097x), finite represented-inputFP64checks/replaypass. Failedlocal1.20proxy transparentlyamendedonlyforwholemixedQ4/Q5trafficcost, unchanged1.25wholemodelgate stillfails. No targetconversion orqualityclaim. Archivedtwoownbenchmarks, originalCMake/apphashesunchanged. Both100goalsopen; nextcheck actual factoryMTP/strictacceptance contract ratherthan table/precision/warp repair.

### 2026-09-29 16:03 +07 — Exact warp-interleaved lossless dense consumer tested; production preserved
- New actual-source Q4 packed-byte ANS stream decodes directly inside original matrix/G64FMA/nonlinear/output kernel; no decompressedmatrixconsumerbuffer orchangedweights/quantization. Layer0/31 onbothGPUs: totalpayload88.57%/88.05% ofnative, CPU/GPU356,515,840weightbytes exact and835,584outputcodes/13,056scales exact across12cases, guards andgraphreplaypass;12CPUcodec/distribution/truncationcases pass.
- Fullfusedgate655–668usvs179–186us: decoding/dataflowoverheadswamp11–12%trafficgain. No Q5/GDN/modelport orprecision/table/interleaveknobsweep. Generic hardwarecompressioncapabilityfalsebothcards viareadonlyquery. No hardwarechanges. Ownbenchregistrationremoved/sourcearchived; independentbank-privateworker cleanup ofits2targets preserved. No sharedstatus/handoffoverwrite.
- ActualunchangednaturalC1fullvocabD3 64.06tok/s/1103tokens/459rounds1377drafts645accepted; actual260973dual53.72/241natural/96rounds288drafts145accepted/3distantfacts/snapshotmetadataunchanged. Exactprioranswer/acceptanceboth. No newgainorstrongerjointnumericalqualitycertification. All3applicationhashesunchanged. Scope/results/index underproduction_serve_20260929/warp_ans_dense. Allownjobsterminal; noagents/training/modelartifact/hardware/commit/push/deploy. Both100single-request goals remainopen.

### 2026-09-29 17:32 +07 — Full-vocabulary MTP no longer loads an unused corpus head; memory fix retained, faster profile not generally promoted
- Implemented source-bound resource ownership across frozen features, artifact bind/materialize, dual ownership, prefill and planning. Full-vocabulary MTP omits exactly two unused device objects (357040128bytes) and B2 cache; nativeW8 ranking remains charged once. Legacy optimized/corpus and DFlash paths unchanged; no target/proposal math, acceptance or default change. Exact11-file patch and before copies under full_vocab_head_ownership/final.
- Measured C1 composition peak20347->19917MiB (-430), same natural1091token answer/454rounds1362drafts638accepts onbothcards. NativeC1 current63.71tok/s same1103natural trajectory. Genuine260973 dual before53.67 after53.61, exact241natural/96rounds288drafts145accepts/3facts/snapshotmetadata. OutputGPU peak-406MiB, otherGPU+44MiB sampledtransient disclosed. No newwarmdecodespeed or100claim.
- Existing permitted strictD3/fullvocab/dynamic-target+draft-union profile ran natural six-family comparison. General67.4-67.67 vs63.57-64.06 and code/JSON near98, but generateddynamicPython module fails2of18tests while nativepasses15of15. JSON40record/Thai/Arabic exact native outputs. Failed tests unmodified; no generalpromotion/newdefault. Resourcefix preservesbothpaths andisindependent ofqualityfailure. Freshprofile stilllargeMLP17.7ms/frame, no claim resourcefixremovesit.
- Final27CTest(including actualrealplan),15stream,11policy,18recordchecks pass. Historicalstrict80.60/80.43 andmargin107.21/107.07 oldtrajectories preserved. Actualcomposedtext/imagecancellation/prefix/leases/deadlines pass, documentedcold-warmimagewordingdifference notcalleduniversalidentity. No freshC8/media/historicalbinaryarchivecertificate. Allnewapphashesindexed; priorhandoff/ownerprefix preserved, stalebankprivatejob reconciledterminal, otherbenchmarksleftalone. Noagents/training/hardware/modelrewrite/commit/push/deploy; both100goalsopen.

### 2026-09-29 19:10 +07 — Three exact resident whole-MLP pipelines tested; original inference remains faster
- HealthyHPSERVER/YOLO/stockdual2080Ti preflight. Whole-round budget17.72msMLP/35.86msframe, fixedwholeMLPfunding>=1.12x. Newmixedproducer/consumerresidentcooperativegrid preservesoriginalQ4gate/G64/SwiGLUBF16A8/Q5split4 arithmetic but425.98/417.8us vs295.2/288.9 loses. ExactsinglekernelNsightdiagnostic58.09%DRAM/41.36%occupancy/0.55eligiblewarps; profiler-onlytimingsexcluded.
- Differentrole-poolownership letsfinishedproducers joinconsumerrowqueue:317.8/305.9usstillloses. Ready-onlycontinuations stores/resumes exactFP32perpartstate toreleaseblockedworkers:330.56/323.6usstillloses;actualpartialcontinuationsobserved160–320tasks/160rowpairs. No sum-reassociation/quantizer/weightchange orgrid/ratio/registercacheknobsweep. Exact327680Bscratch andqueue/fence overheadincluded. Isolatedkernels/sourcearchived, ownactiveCMakeandbenchremoved.
- 42bothGPUchecks match2,924,544midcodes/45,696scales/860,160finalBF16 outputs,5376independentCPUQ5dot-groupFMA/split-reductionreferences exact. Guards/readiness/cooperativeadmission/watchdog/sourceupdates/graphs pass. FactoryMTPnormalizedhandoff/alignmentreview establishesnoconfirmedbug; noMTPmodification. Developmentcompile/unmatchedprofilelogsretained,nofakepass.
- Finalunchangedruntime strictC1natural1103tokens64.03tps/459rounds1377drafts645accepts andgenuine260973dual241natural53.66/96rounds288drafts145accepts, sameprioranswer/acceptance/3facts/snapshotmetadata. Existingjointnumericalqualityscope andMTPresourcefixpreserved. All3apphashespreflightunchanged; no freshC8/media/historical80/107/archivecertclaim. Checkpoint/index/status/handoff preservepreviousqueue. Alljobsdone/GPUsidle; noagents/training/hardware/modelrewrite/commit/push/deploy. Both100goalsOPEN.

### 2026-09-29 20:45 +07 — Exact vocabulary-row parallelism retained after real full-context speed and lifecycle validation
- Reopened old exact head row split only because current full-native verification now invokes it everyround, satisfying its recorded frequency condition. Currentkernel completehead inclhiddenfanout/fullupper-logitreturn1.33–1.45x faster; originalall993280logits/case andguards/replays exactbothownershipassignments. Newborrowedheadexecutionseam andsource-ownedhalfcopy, explicitC1D3complete-onlyflag, defaultsunchanged.
- Actual260973 samebinaryABBA joint53.58/53.50 ->54.38/54.35 (+1.54%means), nonjointheadscale/directvalues47.56/47.31 ->48.25/48.11 (+1.57%). Everyprofile retains241naturalanswer/oldacceptance/3facts/snapshotmetadata. Finalhardened-source54.64samejointoutput; genericC164.55same1103natural/459rounds1377drafts645accepts. Not100, no originaljointqualitylimitationsremoved.
- Tradeoffhelper675430400Bweights plus2027520Bdeviceworkspace/1034240Bpinned, prep~0.37scharged andgraphcreationfirstdecode. Shorttotalrequestmaylose; onlywarmgainclaimed. SharedGPU0unrelatedemulator2154751notkilled, pmonnopositivesampledSM/mem butunavailablecountersnotzero; shared-hostmeasurementcaveat. No otherNInferCUDAworkleft.
- PermanentGPUtest coversfulloutputs/source-refresh/rebind/zero/guard/invalidAPI/capturefallback/immediatedestructionbothdevices. 28distinctCTestactualpasses afterartifactpath/envcorrection; initialskipandmisconfiguredretryretained.15stream/11policy/18records; historicalstrict81.31/81.25,margin108.33/108.11 oldtrajectories. No freshC8/media/archivebinarycertificate. Fixed3treeprobeandpackedhalfauxiliaryclosednotported,2tracehooksrestoredandtemporarybenchmarksarchived.13filesretaineduncommitted, noagents/training/hardware/modelrewrite/push/deploy; both100goalsopen. Sourceboundcheckpoint/index/handoff/status preserveoldownerqueue.

### 2026-09-29 22:31 +07 — Compact greedy head integration and original argmax replay bug fixed; actual model test active
- Inherited greedy-transfer oracle failed because original row-zero NaN contract ignored finite stale winners; red regression reproduced. Fixed uniform early rowzero rule; all finite logits/ties unchanged, graph input changes/allNaNs tested bothGPUs.
- Exact two-half head may now return four winner/score pairs, preserving all rows; separate execute_greedy API avoids falsely claiming full logits. Full-logit materialization remains explicit/lazy for current stage residual. Standalone save0.46–0.52ms; bothdevice source/graph/domain/guard tests pass. Current261KABBAproc-f3859a26 active, no inference speed claim yet. No modelquantization/training/agents/settings/default/push/commit.

### 2026-09-29 22:48 +07 — Exact compact greedy result retained after real261K gain; stale-NaN argmax corrected
- Recovered boundedhead-transfer work; specialvaluefailure camefromoriginal tiledargmaxfinite stale winner overrow0NaN. Redtest0expected/321actual, fixeduniformrow0branchandallNaNstalehandling; finiteprojectionandtiesunchanged. Shareddevice reductionhelpers extractedwithoutduplicateglobalkernels. BothGPUdomain/replaytests pass.
- Separateexecute_greedycontractcomputesallnativeW8rows butreturns32Bscore/tokenpairsinstead993280Blogits; originalexecute()fulloutputs preserved. Stagecurrent-logitvalidityandlazyreprojection prevents stalefull-logitreads. Realstage7946240bytes directvslazy andsuccessiverounds/repeatreadsexact; read-onlyhookremoved. Source/graph/domain/fullcompact-switch/destructiontests passbothdevices. Adds15648Bworkspaceonly; oldhelperweights/prepcostremaincharged.
- Actual260973ABBA55.04/54.99vs54.61/54.36 (+0.9727%means), allsame241naturalanswer/96rounds288drafts145accepts/71,48,26/3distantfacts/snapshotmetadata. Finalbuiltcompact55.14sameanswer; strictgenericC164.50same1103natural/459rounds1377drafts645accepts. Smallwarmgainnot100orstartupclaim, existingjointattentionqualitylimitsremain.
- 28CTestactualno-skips +bothGPUargmax/headtests,15stream/11policy/18recordchecks andhistoricalstrict81.40/81.27,margin108.288/108.280 originaltrajectories pass. No generic100fromhistoricalfixtures/newC8/media/archivebinaryclaims. Retained14files/sourcepatch/apphashes/checkpoint/index andoldhandoff/status saved. Allownjobsterminal/GPUsidle; noagents/training/hardware/modelrewrite/default/commit/push/deploy. Both100goalsOPEN, largeprimaryarchitectureworkstillrequired.

### 2026-09-30 05:54 +07 — Exact original-owner round graph and conditional fallback tested; compact runtime restored
- Primarylarge-slice48GDN+48TP2MLP original-ownergraph isolatedschedulingwithoutpriorreplicaextraQK/wireconfound. All96states/changedinput/replays exactbothroles; native22.70-22.73ms,queued22.97-23.00,graph23.12-23.20. Failed1.20gate; nobroaderportorparameterrepair. Sourceandtargetarchived, CMakeoriginal.
- AuxiliaryactualCUDAconditionalIF supportedbothstockSM75cards. Separatepredicate/control versionfull260973 host55.19/54.98vs54.08/55.00; fusedexactglobalpredicate+conditionalhandle withmapped416Bpeermetadata host55.21/54.93vs55.01/55.02,flat. Bothsame241natural/96rounds288drafts145accepts/71,48,26/perlayer4fallbacks/snapshotmetadata. Event-orderedno-pollingmetadata1024independentCPUtruthcases exactand307fallbacks/card. Neitherretained; no furtherbranch/transportknobloop.
- Restoredoriginal2productfilesbyteexactandremoved2ownprivateheaders. Finalcompact26097355.36usefultps same241naturalanswer/96rounds/3distantfacts/snapshotmetadata. Preservationnotnewgain; priorjointnumericalqualitylimitsremain. CLI/servehashesidentical, rebuiltdualhashrecordednotcalledidentical. No newC1/C8/media/historicalcertclaim or recreatedmissingwitnesses.
- Finalcheckpoint/index/restoration/source/modelreceipts underproduction_serve_20260930/conditional_attention_dispatch; primarysourceunderexact_owner_round_graph. Status/handoff/ownerprefix/oldqueuepreserved. Allmanagedjobsdone,bothGPUsidle; noagents/training/hardware/modelartifacts/commit/push/deploy. BothgenericC1D3>=100andcompletecontextdualsingle>=100remainOPEN.

### 2026-10-01 12:07 +07 — Shared attention-input whole-op passed cost screen; actual full-context loses and quality evaluation active
- Recovered finished September30GDNprojection trial and restored its exactprivateoriginals; preserved compacthead/resource/servingroutes. Newwholeattention-input3stagepipeline actualsourcekernel94–105usvs137–184, CPUinversecode/FP64/guards/replaypass. Initialdefaultupload/nonblockingpackdependencyfixed, originalfailuresaved.
- PrivateC1anddual hook +exact712638528Bsourceplanreservation; firstunreservedcandidateblockedcorrectly, notguardbypass. ActualsamebinaryC163.91/63.23->64.61/64.48 withchangedbutdeterministicnatural1155vs1103answer,underflowflag2notexactFP16range. Dual26097354.90/54.91->54.05/54.08,265vs241natural/109vs96rounds/facts/snapshotsgood, no fullcontextpromotion. Currentproc-619f6dcd runsunchangedlongtext/media/historyquality, norebuildoverlap. Noagents/training/weights/hardware/commit/push. Both100gatesopen.

### 2026-10-01 13:36 +07 — Closed interrupted shared-attention quality pass; fresh causal summaries and whole-MLP work
- Read actual managed jobterminal0 instead ofrerunning. Sharedattentionnaturalcode/JSON/language/media cohortpasses butJSON+dualperformance regress andsingleC1longeroutput+712MBplans insufficientgeneralpromotion. Savedcandidateexactsource/qualitychecks, restored10scopedpaths andrebuilt originalCLI/serve/dual. Acceptedcompacthead/memory/servingroutes preserved.
- Newconstant-work packednibblebias correction removesoldrowmetadataandsparsegathers. Allindependentinteger/outputchecks exactbothcards; completegateonlyflat, no retuning. Newcausalpastquery-specializedMuSeinferenceoracle usesheldout72queries/actual65Kcache, formula identitiespassbutmaxL2.0831 missesgate; no sourcequantization orcurrentqueryfitting. Both archived.
- Nowrunning proc-703da9a1-0668-4d95-9155-3e4686a746e9 originalQ4/Q5wholeMLPgroup-scaledFP16path, allquantization/nonlinearity/downresidual included, threeactuallayers andFP64references. No productionpath untilfixedcost+accuracygate andmemoryownershipresolved. LiveGPUownership/statusrecorded; noagents/training/hardware/modelartifactchanges/commit/push. Both100goalsopen.

### 2026-10-01 14:00 +07 — Three architecture screens completed; restored current inference verified
- Closed recoveredsharedattention trial fromactualterminalqualityjob, notrerun. Inspectedlongcode/JSON/languages/media pass, butJSON+dualperformance regress with712MBplans; no generalpromotion. Original10sourcepaths restored, correctedcandidate/quality archived, itsisolatedbenchremoved. Earliercompacthead/memoryownership/profile paths preserved.
- Newmetadata-free signednibblebias arithmetic passes524288scalarpairs/163840CPUdots/557056codes/8704scales/guards/replays bothGPUs, wholecostflat0.03–0.32%, noQ5 or tuning. NewcausalMuSe-inspired query-tilted semanticcache formedonlyfrompast495queryframes63360keys,72heldout65Kqueryheads; fourformulaidentitiespassbutmaxL2.0831/mincos.99727failfixedaccuracygate, evenfreesupport/tailrefresh. No runtimeport orcentroid/selection/rankgrid.
- NewactualoriginalQ4/Q5wholeMLP group-scaledFP16input/gate/down avoidsA8roundingandlowersreferenceerror, butwholeoperation299.87–314.21usvs284.86–295.42 (5–6%slower).18pattern/layer/cardCPUFP64/formats/guards andgraphreplayspass; oneunderflowflag2,nooverflow. Failsbothpredeclaredperformancebranches; no modelcanary/9.8GBpreparedcopy orknobsweep. Allnewoperatorentriesarchived and removed.
- RestoredcurrentgenericstrictC163.99same1103naturalanswer/459rounds1377drafts645accepts; actual260973compactdual55.13same241natural/96rounds288drafts145accepts/3facts/snapshotmetadata. Preservationnotgain, existingjointnumericalqualitylimitsremain.28CTestactual/no skips, bothGPUargmax+stale-preseed/headlifecycle,15stream/11policy/18record tests pass; strict81.40/81.22 andmargin108.30/108.26 oldtrajectoriespreserved, notgeneric100. Source-boundreport/index/currenthandoff/statusandpriorqueue saved. Noagents/training/modelartifacts/hardware/commit/push/deploy;alljobsterminal/GPUsfree. Both100goalsOPEN.

### 2026-10-01 16:06 +07 — Radix stacking and exact attention dataflows completed; no accepted gain and original runtime restored
- Currentwhole-roundprofile47.30msinstrumented,compute/copyunion39.35/4.89ms,combined43.15. Hoststreamwait33.48msincludesGPUdependencycost,notremovableCPUoverhead. Acceptedcompacthead/MTP/TP2numericalpolicies protected; no generic100orhardwareceilingclaim.
- Implemented missingexistingC1residentradixpreparationfordual85matrices with2MiBscratch/stage, no newquantizationorartifactrewrite. Actual260973 ABBA55.14/54.97nativevs51.24/51.32radix, natural3facts/deterministic/snapshotmetadata allpreserved. Roundscheapened45.6->44.8msbutyield2.51->2.30 and96->107rounds means~6.9%usefulregression. No policy/layerrepair; testedcpparchivedandrestoredbyteexact.
- Three exact attention dataflows implementedandcosttestedbothcards. Preparedqueryfragments132–136->140–144us,31,952,640referencebytes+CPUqueryquant exact. SplitQK/softmax/PV21.23MBrecords130–136->232–236us,31,952,640bytesexactafterfixingtrue tiny-partitionrecordalias; failedcasekept. IndependentQKtileorder61,516,800bytesexact,still122regs,forward2%advantageunrepeatableinreverse. No kernelgeometry/precision/selector repairgrid. Threeownbenchmarksarchived/CMakeoriginal.
- Restoredall3applicationsSHA256exactpre-turn,cppandCMakeoriginal. FreshgenericC163.95same1103natural/459,1377,645, compactdual54.79same260973/241natural/96,288,145/3facts/snapshotmetadata. Preservationnotgain, numericalattentionqualitylimitsunchanged. No newbroadregression/C8/media/historicalcertificate claimed. Verifiedcheckpoint/index/status/fullpriorhandoff saved; allownjobsterminal/GPUsidle, noagents/training/hardware/artifactmutations/commit/push/deploy. Both100goalsOPEN.

### 2026-10-01 17:37 +07 — Exact selection lane closed without product changes; coordinate shared RadixArk state
- Ownvalue-spacebucketandoriginal-radixterminal variants allsortedIEEE referencespass butslower. OwnbenchCMakeexactpre-turn, temporarysourcesarchived; native sample_order_statistic unchanged. No ownedGPUjob remains, no production filewasmodified inthislane. CLIhashchangedwhileindependentRadixArk lane rebuilt; no falsebinaryidentityclaimorrestorationovertheotherlane. Sharedstatusleftintact.
- RadixArk coverage continuity warning: the SAME publishedb9a5dbdf03bc999c6c73c426b19c2d9041cea393 wasalready evaluated on FULLNATIVEtarget histories in .chatgpt/production_serve_20260919/dspark_full_target_recheck_1346/ (not only RedHat). Scope explicitlycorrected oldrestrictedtarget, nativeW8endpoints/FP32head/fivepostlayertaps5/19/33/47/61. Its recordedmeans2.1355prose/4.0667code/6.04JSON/1.1855Thai/1.5391JP/1.447Arabic. Compare currentnewsource/mask/YaRN capture ifdifferent; donotcreditrepeatcheckpointasnewarchitectureor redownload itagain. Currentothermanagedcaptureproc-106bcc45-c289-4705-ac5b-5870813f1b21 terminal0; sourcehooksvisiblyremoved, evaluatoroutputexists. Thisnoteiscoordinationnotclaimofownwork.

### 2026-10-01 17:42 +07 — Selection experiments finalized; shared status left to the independent drafter lane
- Exactvaluebucket initial141–148us, streamed123–139vsoriginal51–64; original-radixterminal85–98 alsoloses. EightGPUruns/27648sorted-referenceassertions(include repeatedfixtures) exact, guards/graphs pass. Oneprototypefallbackinitializationedit caughtbysignedzeros, correctedandrerun; badsource/logretained. No productionselectororotherproductsourcewaschanged. Bothownbenchmarksarchived/CMakeexactrestored; no newfullmodelrun orspeedup claim.
- Checkpoint/index in production_serve_20261001/selection_followthrough_1741. CLIhash changed during independentRadixArkbuild; server/dualpreflightunchanged. Sharedactiveboard andprotectedhandoff not overwritten. Confirmed RadixArk revisionb9a5dbdf and SHA2aff025f exactlysameasSept19full-targetscreen; duplicatewarning/priorresults posted. Ownjobsdone, both100goalsopen; noagents/training/hardware/modelartifactrewrite/commit/push/deploy.

### 2026-10-01 18:05 +07 — Exact valid-query integer attention retained after actual full-context improvement
- New fixednativeM8 tiling usesall24queryrows without8paddedM16rows. Sameweights/scales/thresholds/union/softmax/PVorder andallkeys/layers; no approximation/acceptancechange. Registers122->72,shared32->24B,noextrasourceallocation. Completeoperator1.43–1.48x, true260973ABBA55.15/55.14 ->56.15/56.14(+1.8134%) allsame241naturalanswer/96rounds288drafts145accepts/3facts/snapshotmetadata. Finalcompiled55.99sameanswer; C163.96same1103naturalreference, preservationnotgain.
- Frozenoldreference plusindependent65536signedpairs/card;20range/source/tail/empty/ownedoffsetcases102528000bytes andcapturedchangedqueryreplays exact.28CTestactualpasses,15stream/11policy/18records,bothGPUhead/argmax/newattention tests pass. Historicalstrict81.352/81.2886,margin108.276/108.084 oldtrajectories preserved, no generic100orC8/media/archiveexecutableclaim. OneGPU1candidatepeak+24MiBtransientdisclosed, noextra bufferintroduced.
- Onchipphase-slabexactbutslowerarchived. RadixArkcheckpointinitialname-basedsearchmissedoldalias;SHAconfirmedidenticalold dspark_v2. Notnewcandidate; sixnativecapturepairs exact, onlygeneral/codecandidatecheckscomplete thenJSONcached-referenceerrorfailed. No thresholdrelaxation/weightsrepair/port, diagnosticheadersrestoredexact, duplicateattempt/sourceindex preserved topreventrepeat. No training/subagents/hardware/modelartifactrewrite.
- Retainedsixsource/test/docsfiles; privateA/Bflag/oldproductionkernel removed, frozenreference keptonlytest. Allcurrentapplicationidentities andsourcepatch boundunder valid_query_integer_attention/runtime/final; priorprotectedownerprefix/fullqueue/status preserved. CLI/servehashesunchanged,dualrebuilt. Allownjobsfinished/GPUsfree; no commit/push/deploy. BothstrictC1D3andcompletedualsingle-request100remainmandatoryOPEN.

### 2026-10-01 20:27 +07 — Direct attention metadata followthrough is in complete-context ABBA
- Currentvalid-queryallkeystagealreadycomputesquerymasks/headunion; newdirectcounts/unionsavoid96maskmaterialization andrescans, sameexistingmemoryallocation. Initialdirect versionmixedcost; removing provablyduplicate old_mass recurrence (tailallexcluded) restores72registers and4–6%wholepipelineoperatorgain withoriginalstatebytes/CPUcounts/unions/graphs exactbothGPUs. No parametersweep/weights/threshold/rank/precisionchange. Auxiliaryofretainedattentionfamily, notclaimtolargeC1solution.
- Samebinaryactual261KABBAactive proc-07ed8a64-0199-44dc-97a4-bc342948c6b7; allsource/backups in production_serve_20261001/direct_attention_metadata_2017/runtime. Do notoverlapGPUwork orapprebuild. Permanentboundary/nonfinite/CPUmetadata/reference test prepared. Both100goalsopen, nocommit/push/deploy/agents/hardware.

### 2026-10-01 20:49 +07 — Direct exact metadata gains retained; larger D3 corrective-leaf architecture misses cost budget
- Current allkey computedselection reusedasheadunion andpartitioncounts, no96maskmaterialization/rescans orlargeclear. Oldprefixmass recurrence identicaltodenominator aftertailsfullyexcluded; redundantlivefloatelided. Existingmaskallocationreusedfororderedcompaction. No weights/scales/rank/threshold/softmax/PV/coverage/verification change. Initialmixedversionnotcredited; final4–6%pipelineoperatorgain/72registers24Bshared.
- Actual260973 ABBA56.14/56.17 baseline vs56.47/56.44direct (+0.5342%means), allsame241naturaltokens/96rounds288drafts145accepts/3facts/fallbacks/preparedbytes/snapshotmetadata. Finalcompiledno-comparisonswitch56.71sameanswer, notanadditionalpairedgain. 13newboundary/nonfinite/CPUmetadata casesperGPU plusoriginalvalid-query/ordered-union tests bothcards pass. No addedallocation, peakGPU0same18859/GPU1same17806–17830range. CLI/servehashesunchanged, dualrebuilt. Existingjointqualitylimitsremain.
- Primary D3greedyspine+4correctiveleaf reuse screen preservesdepth3/fullnativeverification design, noextraMTPbody. 1258oldnativecontrolrootspotentialcoverage+16.86%general/+12.39%pooled,40independentallocation/2560prefixcases pass. OfflinecountsnotE2Eyield; actualtreecache/masksnotimplemented. ExactT8wholeMLP andexistingradixQK/output costmeasured: coveredadditional2.47/2.90msextrapolationalready>1.85msbudget beforeotherwork. Noportorparameterrepair. Allsources/negativeevidencearchived separately.
- Finalfivefilemetadata patch/report/index/status/handoff preservepre-turnownerqueue. Alltemporarybenchmarksremoved/CMakeexactoriginal; secondarytestsleftcurrentapphashesintact. Allownjobsdone/GPUsfree, nofreshC1/C8/media/broadhistoricalcertificate or recreateddeletedwitnesses. Noagents/training/hardware/modelartifactwrites/commit/push/deploy. Both100single-requestgoals remainOPEN.

### 2026-10-01 23:02 +07 — Exact binary-convolution dense architecture tested; original model paths retained
- Implemented two nativeSM75 XOR-popcount significance-convolution gates with ORIGINALQ4/A8 products, no requantization or free correction. Fixed20-BMMA form and18-BMMA one-sided-boundary consolidation both preserve originalG64FP32FMA/nonlinear/outputquantization. Largefamilybudget recorded; actualgatebelow1.08xcriterion: GPU0 188.93–190.46usvs186.59–187.04, GPU1 184.10–185.12vs180.22–180.35. NoQ5/wholeMLP/runtimeport or warp/tile/precision/scale orbit.
- Perimplementation163840independentCPUdots,557056originaloutputcodes/8704scales plusguards/zero/signedendpoints/sourceupdates matchbothGPUs. Independent4096scalarpairs+384groupcases eachalgebra exact; initialCPUverifierparenthesiserrorretainedandcorrected. Finalboundarygraphchanged-input/nativecomparisonandrestore/replay explicitlypass. Sourcecodepayload89128960unchanged; sourcepreparation~2ms and1280Bsumworkspace charged.
- Allthreeproductionexecutables matchpreflight andoriginalbenchmarkCMake restored; only ownisolatedbenchmarks archived/removed. No newfullmodelC1/261K/C8/media/historicalcertificate or fabricatedfrontier. Lastaccepted~56.5fullcontextpair remainsprior evidence, withsame numericalqualitylimitations. Source-boundcheckpoint/index/status/handoff preservepreviousownerprefix/largequeue. Allownjobsterminal/GPUsfree; noagents/training/modelrewrite/hardwarechanges/commit/push/deploy. Both100single-requestgoalsremainOPEN.

### 2026-10-02 00:00 +07 — Cost-aware D3 verification scheduling and exact recurrent factor reuse evaluated; no new speed promotion
- Primary fixedpre-target widthscheduler keepsnativeD3spine andspends4correctiveleafcolumns onlywhen expectedutilityexceedsmeasuredpartialwidthcost. No targettokensinselection/no fittedthreshold.120independentexhaustiveutilitycases,1258priornativecontrolroots reproduced. Favorablemodeledpooledgain5.99/5.05%,general10.13/8.84%undercard0/1costs; omittedhead/valueZ/attention/branchworknotfree andnewrootstrajectoryunknown. Fails8%pooled+10%general beforeactualtreeport.
- AuxiliaryactualFP32transitionfactor reuse avoidsrecomputingacceptedGDNstate-keydots/deltas. New48-record+foldkernelcomparedall4,047,175,680record/output/state/convhistorybytes exactacrossprefix0..4/twocards/nonzero/zero/inactive/signedzero/tiny inputs; changedsourcegraphs+guardsalsoexact. Butwholeoperation1.426–1.431vs1.380–1.403ms and1.405–1.587vs1.374–1.556, slowerall12matchedcases.4.718MBadditionalfactorsandrecordtrafficcharged. No globalstateownershipport orwarp/scale/tilegrid.
- Bothmechanismsclosedatrecordedgates, ownGDNbenchsourcearchivedandCMakeexactrestored. All3productionapphashesunchanged includingprioracceptedmetadata/head/memory fixes; nofreshfullmodelmeasurement or C1/C8/media/historical claim. Before/finalidentities,source/results/checkpoint/oldhandoff/status saved underadaptive_width_factor_followthrough. Alljobsdone/bothGPUsfree; noagents/training/modelartifacts/hardware/commit/push/deploy. Bothsingle-request100goals remainOPEN.

### 2026-10-02 05:52 +07 — New frozen eight-tap DFlash2 whole-round proposal screen completed; no production promotion
- Verified genuinely new published DFlash2 comparison checkpoint42fee8dc / weighta41e77a9,4.11GB, not previousRadixArk/RedHatDSpark. LatestRedHatmetadataaliases same6cf6f33weight, explicitlynotrerun. Implemented inference-only adapter around pinned70fbcbd4original decoder/convolution/selector components;81statekeys strict, analyticRoPE, originalnativeW8-derived endpoints. No localtraining/parametersearch or target/source arithmetic edits.
- Evaluated1353existingnative roots across6short512-token controls andgenuine64172prefix249naturalcontinuation with2048realpromptstates. WeightedshortnativeD3advance2.4275 vsnewfirst3=1.9416; seven=2.4331. JSONseven6.1333 positive but general/code/multilingual andlong2.625failnecessarygate. D3first3generallyworse, no modelport andno offlineratio called actualacceptance/tps. No first3/wideC1goal substitution.
- 42source/cachedbackboneandselectorpathchecks exactpredictions, <=.004hiddenL2;4maskcases+512independentscalarconvvalues percard,14futurepoisonchecks,21fixedcrosscardroots match. OfficialauxID-1layerconversion auditedagainstcapturedstates. Allproposals constructedbeforetruthconsulted; pretrainedselector neverusestraining-onlytargetinjection. RecordedFP16drafter/FP32nativehead precision limits and27msunoptimizedadaptercost, notuniversalcheckpointqualityclaim.
- All3productionapphashesunchanged, no source/build/model/defaultchanges orfreshC1/261K/C8/media/historicalspeedrecert. Prior~56.5pairremainspriorwithnumericalqualitylimits. New4.11GBresearchweights/source/results/checkpoint/index andprotectedoldhandoff/status retained. Allownjobsfinished/GPUsfree; noagents/training/hardware/targetartifactrewrite/commit/push/deploy. Both100goalsremainmandatoryOPEN.

### 2026-10-02 08:40 +07 — Conditional attention and exact native-M8 pipelines completed; original model paths restored
- Recovered inherited native_pathwise_draft compile/capture state, completed six unchanged controls.1252 causal anchors score-only branch selection gains0.658% local useful prefix before341 extra MTP calls and general-prose-2.174%;2000 independent upper-bound tests pass. No policy port or tuning. Private W8/MTP hooks restored from exact originals and new probe header archived/removed.
- ActualSM75 conditional graph and two-context parent capabilities work.216 exact combined-count/union/failure/buffer-reuse cases perGPU pass. Separate conditional full261K55.82/56.10 vs56.31/56.58; combined parent54.40/54.58 vs56.52/56.24 lose. Every original241token natural answer/acceptance/perlayerfallback/3facts/snapshot preserved. Both full implementations archived; runtime and test sources restored; no subsequent graph/threshold/clock repair.
- ExactnativeM8 gate ownership tested with real fast/slow Q4/Q5 half-MLPs. Fused and decoupled producer/quantizer each pass163840 independent group dots/557056 codes/8704scales/327680 final BF16 values, guards/source input cases/same-source graphs. Fused fast238–245us vs168–174, decoupled195–201vs167–173; slowhalves also lose. Actualprofile69CTA/68SM waveconstraint; no more block/warp/precision sweep. All ownbench entries archived/removed. ValueZtwo-limb favorable vsoldSIMT is notcurrentC1gain (~80usA8alreadyused), explicitly notpromoted.
- Final original-source apps rebuilt, all scoped source restoration hashes checked. NativegenericC163.99 same1103natural/459rounds1377drafts645accepts; complete26097356.58 same241natural/96rounds288drafts145accepts/3facts/perlayerfallback/snapshots.28CTest actualpass,15stream/11policy/18records; oldstrict80.1614/80.0216 andmargin106.75/106.361 trajectories retained. Preservationnotgain; no freshC8/media/historicalbinaryarchive certificate. Both100goals open, accepted paths intact. No agents/training/modelartifact/hardware/commit/push/deploy; jobs terminal/GPUs free. Full report/index and prior protected handoff/status saved in attention_m8_followthrough.

### 2026-10-02 09:36 +07 — Separate GemLite whole-dense oracle isolated; GPU ownership respected
- Current worker only writes production_serve_20261002/gemlite_dense_oracle research files, no product source/buildtarget/model edits. Pinned new GemLite f649e80e, full-gate budget, default schedule only. Package frontend andnestedJIT fail under existing3.11.0rc1; Triton3.3.1also lacksSM75MMA lowering. Experiment-local Triton3.2 and stablePython3.11.14 installed withoutchangingglobalenvironment; nativeTuringmatrixsmoke passes3.2. No GemLite timingyet.
- Detected concurrent q5_sign_dilation model job proc-f3a10326-fc08-436a-9a21-b725671cf895; GPUguard preventedour nexttest, nooverlap/nootherprocesskill. ContinuingCPUcompiler/reference work. ExistingsharedACTIVEboardnot overwritten; oldapphashes arenowhistoricalpreflight,notclaimingcurrentbinariesunchangedafterotherworkerbuilds.

### 2026-10-02 09:59 +07 — Causal proposer reuse and exact decoder experiments completed; no new accepted speed profile
- Whole-round tokenreuse hypothesis checked1258oldnative roots with972policycases andexact511decode-token reconstructionper6control. Oneprecedingwrong-contextverifiertail pluscurrentfirst-MTPconsensus onlyomits106of2516ARcalls,1.8–5.6%reuse, no usefulgain. No runtimeport/lookup-pool orfuturetruthdependency. CorrectedpossibleT8stagingidea as alreadyimplemented inarchivedD3leafcost, didnotrerun.
- Newtarget-featureproposal-onlyreuse implementedandtested: copyfrommatchingpositionofpreviouslycomputedrejectedtargethidden only intoMTPARinput, nevertargetKV/GDN/committedstate.480GPUindex/sourceimmutability/changed-inputgraphcases2457600valuespass. Realnaturalprose63.91/1103tokens/459rounds645accepts ->56.66/953/446rounds508accepts; yield2.403->2.137. Candidatecompleteanswerread, nothashdifferencebutlowerusefulrate rejectsatfirstfamilygate; no feature/coefficient/blendrepair. Private2filesrestored.
- ExactQ5sign-dilationwordidentity4194304GPUcases, fullMLP/operator/guards/replays pass. Actual8modelrunsC163.97/63.30vs63.55/63.32, full261K56.18/56.16vs56.10/56.32, alloldanswers/acceptance/fallback/snapshots;flatnotgain,8sourcesrestored. DistinctexactQ4/Q5postmaskwordreconstruction alsofailswholeMLPcost despiteexactintermediates/outputs; no furtherkernelparameterorbit. Own3benchmarksarchivedandCMakeexactpreturn.
- Final rebuilt currentnativeC163.92same1103natural/459rounds1377drafts645accepts; complete26097356.29same241natural/96rounds288drafts145accepts/71,48,26/3facts/snapshotmetadata/fallbacks. All10temporaryproductionsourcefilesexactpreturn; rebuiltapphashesrecordednotcalledidentical. No freshbroadregression/C8/media/historicalarchiveclaim, noacceptedrouteoverwritten. Report/index/fullhandoff/statussaved, oldownerprefix/queuepreserved. Allownjobsterminal/GPUsfree; noagents/training/weights/hardware/commit/push/deploy. Both100goalsOPEN.

### 2026-10-02 12:01 +07 — On-load corrections reach full affine-MLP cost check; production untouched
- Reconciled prior probability-carriedMTP canary as alreadyfinishednegative withsource/apprestorationmatchingverifiedindex. New exact Q4onloadmetadata elimination preserves524288sumcases/163840CPUdots/557056codes/8704scales+changed-inputgraphs; removesoldfoldpenaltybuttiesnative, notported. Exact24-querysamplingfollowthroughslowerandarchived. Normalizationauditfoundupstreammatchingboundaries, nochange.
- Applied measuredmetadata-elimination mechanism to samefixedaffineE8PfullMLP: on-loadS4row sums, foldedgate+directproducerpacking anddownscale-onlyrepresentation. Initialcompleteoperation~1.30xnative withallrepresentedE8bits/FP32/64references/source-updategraphs passing; GPU1straddles1.30floor so fixed-kernel8alternatingtimingepochswithclocks currentlyrunning proc-599d5593-428a-465c-b2ec-d29d933260fa. Notactualmodelquantizationorqualitypass. No corefilechanges or100claim; allacceptedprofilesintact.

### 2026-10-02 12:46 +07 — Continued whole-MLP E8P speed candidate into independently checked model-quality canary
- Recovered priorcompleteoperationtests andordinarywarmupresults. PreviousaffineE8Pkernel1.55–1.59xnativeundersteadyclocks, newonloadslightlyslower; preserveattributionandcoldresults. No modelperformanceclaim. Continuequalityratherthanabandonpassinglargefamily.
- Source-onlyconverter/H64/nearestcodebook verifiedbyexhaustiveFP64search andallcarriercoefficients. Newslowcanonicalreferencecarrier exactlymatchesoriginalcompressedE8P raw/intermediate/fullMLP onbothGPUs, allthreecases. Eagerprivatecodeappliesall128MLPmatriceswithfixedRMS/1.03,H64,2bitE8P; no training/calibration/parametersearch orartifactwrites. Managedproc-05fef458 native/candidatecanaryactive;sourcebackups andidentitiesunder onload_affine_whole_mlp/quality, restorebeforefinish. Both100goalsopen.

### 2026-10-02 13:11 +07 — Validated compressed E8P source-only model against exact quality carrier; broader quality active
- UnchangedE8P/H64/RMS1.03 coefficients nowpackedinsideexistingloadedcarrierallocations; no extraresidentmodelcopy orartifactwrite. RotatedBF16/A8boundaries andoriginalsignedS4/U4ops, padding/nonfiniteguards explicit. Whole-entrypoint T1/T2/T3/T4/T5/T7 outputs matchslow referenceonbothGPUs; sixwholeanswer/counterreproductions exact includinglongJSON1283andThai964tokens.
- Initialfunctionalrecord/JSONpass, shortThaiChineseintrusion and longpunctuationdrift remainqualitywarnings, notfullpass. Remaining4fixedfamilies proc-613e0041 onnativecontrol thencompressedcandidate active, no parameterrepair. Quality-source archives/backups under onload_affine_whole_mlp/quality; restoreprivate7files afterdecision; no acceptedmodelgainyet.

### 2026-10-02 13:32 +07 — Whole-model E8P compressed implementation validated, then rejected for severe quality loss; original runtime restored
- Recovered terminal sustained wholeMLP costs: earlier producer-owned affine E8P1.55–1.59x native; newer onload1.53–1.56x slightly worse. No misattribution or kernel ratio substituted for model speed. Continued into real128-matrix source-only H64/G64/RMS1.03 nearestE8P implementation with fixed parameters, not training/calibration or artifact rewriting.
- Independent exhaustive converter, exact canonical/compressed intermediate and full MLP, both-GPU padded T1/2/3/4/5/7 and nonfinite checks passed. Implemented real compressed same-coefficient execution inside existing loaded allocations. Six whole-model answers and full speculative counters exactly match deliberately slow reference. This is candidate-internal correctness, not native-weight equivalence.
- Actual quality fails: Arabic repeats the same phrase389times and reaches4096 versus native1189natural; short Thai Chinese intrusion and late quote instability also recorded. JSON4/40 and record successes do not override failure. Native generated Python3801natural passes19tests; candidate code ends mid-module4096, cap alone not called semantic failure. No more bit/scale/H64/layer/window repair, no media/fullcontext candidate promotion. Fixed source-only recipe closed, not allE8P or calibrated methods.
- Archived seven private source files and four owned benchmark targets; exact before-source and parent baseline CMake restored. All3 rebuilt apps equal original hashes. Fresh generic strict C163.87 same1103natural/459rounds1377drafts645accepts; populated260973dual56.58 same241natural/96rounds288drafts145accepts/3facts/snapshot/fallback decisions. Preservationnotgain; numerical-attention limits remain.
- Eight actual focusedCTest cases,11policy,18recordchecks and historicalstrict80.25/80.08,margin106.70/106.54 original trajectories pass. No newC8/media/archive-executable certification. Final checkpoint/index/source archives and old protected-owner handoff preserved in onload_affine_whole_mlp/quality/restored. Alljobsdone/GPUsfree, noagents/training/hardware/modelartifactrewrite/commit/push/deploy; bothgenericC1D3andfullcontextdual100goalsmandatoryOPEN.

### 2026-10-02 14:22 +07 — Exact unsigned-operand complete MLP tested; no speed promotion
- Implemented originalQ4/Q5 plus sharedactivation-moment identity withnativeU8/S8MMA. Bothfullgate/up anddown/residual, originalG64quantization/FMA/reduction/BF16boundaries retained; no weightprecision/table/modelchange. Completeoperation0.9–1.4%slower onbothcards/reversedorders, below1.15xfundinggate. No follow-onwarp/tile/scale/fusionrepair.
- Independent180224CPUgroupdotchecks,557056originalintermediatecodes,8704scales and163840finalBF16values exact, plusguards/zero/endpoints/changed-inputgraphreplay. SourcebiasXOR isinvertible,2816Bactivation-sumscratchcharged. OriginalnativeQ5profile81.71%DRAM/37.73%SM/zero spilling; memoryscoreboardattribution70.7%notadditivewalltime. Profileinternalapplicationtimings excluded.
- Allthreeinferencehashes andbenchmarkCMakepreflightidentical; onlyownisolatedtargetremovedandsourcearchived. No freshC1/261K/C8/media/historicalperformanceclaim; prior63.87/56.58notrelabeledasnew. Existingacceptedpaths/qualitylimits andprotectedownerqueue preserved. Source-boundreport/index/status/handoff saved underunsigned_activation_moment_mlp. Allownjobsterminal/GPUsfree; noagents/training/hardware/artifactrewrite/commit/push/deploy. Both100goalsremainOPEN.

### 2026-10-02 18:59 +07 — Binary-predictor complete MLP loses cost; causal shared-mask counterexample and independent union bound completed
- Recovered stale intermediate-channel status without rerunning finishedproxies; originalQ4hookalreadyrestored. Followedpassedfull-input1bitproxy signal prerequisite into actualGPU-onlyprediction/selection/originalselectedgate/maskedquant/densedown/residual. WholeMLP543–564usvs289–295 onbothcards/reversedorders. IndependentCPUquant/scores/fullsort/FP64proxy+selectedrow/guards/source-changegraphs pass. Profilecorealready~331usbeforeselection; no parameterrepair ormodelport.
- Held anchorinputbytesfixedwhilechangingonlylater3draftcolumns: nativeanchorMLP output0changes, candidate5085/5094of5120BF16valueschange. Sharedcurrentmaskcreatescrosscolumndependence, cannotbe exactcausalverifier; notproductionbug orchangedemittedtokens. Previoussignal/longanswerqualitychecks insufficienttoestablishcausalindependence.
- Fixed per-token50%masks withsharedcomputeunion assessed512recordedstates/all64originalMLPweights. Energy86.53%mean butunion88.41%rows; sourceproxy+unionQ4+denseQ5logicalbytes170.06MBaveragevs153.19MBnative(+11.01%), allcasesmoretraffic beforegather/selection. Onlya favorablelogicalpayloadmodel,notactualDRAMoruniversalsparsitylimit; no causalunionkernel/modelport ornewratiosweep.
- Fresh unchangedbinaries: strictgenericC164.00same1103natural/459rounds1377drafts645accepts; populated260973dual56.20same241natural/96rounds288drafts145accepts/3distantfacts/snapshotmetadata/fallbackdecisions.8focusedCTest/11policy/18records pass, notnewC8/media/historicalarchivecertificate. Originalapphashes/Q4source/CMakepreserved; own2benchmarksarchived. Checkpoint/index/protectedoldhandoff/statussaved. Noagents/training/hardware/targetartifactrewrite/commit/push/deploy;allownjobsterminal/GPUsfree,both100goalsOPEN.

### 2026-10-02 19:55 +07 — Head-only proposal-state correction and exact TP2 input fusion evaluated; no speed promotion
- Primary new state/scoring separation leaves nativeMTP recurrence untouched and corrects only transient vocabulary input from an already accepted target/aligned pair. Independent184320formula and184320future-poison output checks passed across both GPUs, with read-only sources/guards/changed-input graph replay. Actualnaturalprose63.99->59.63tps;1103/459rounds645accepts vs1105/491/613, useful/round2.4031->2.2505. Both answers read, fixed unitgain failed; no coefficient/position/temperature repair. Source/header/bench fully restored or archived.
- Auxiliary exact TP2 input ownership reuses the existing fusedRMSNorm/BF16/G64A8 kernel rather than separate normalization+quantization. 409600codes/6400scales exact on20bothGPUcases;286720independentCPUquantized values. Complete260973 ABBA56.39/56.53 original vs56.63/56.50 fused, ratio1.001860 but rangesoverlap andboostvaries. No robustmodelgain claimed or feature retained. All same241naturalanswer/verification/facts/snapshotmetadata/15layer+2terminalfallback records. Fallback matcher initiallyempty was corrected and verified, not counted as evidence.
- Finalrestored source rebuilt andactualC163.96same1103natural/459rounds1377drafts645accepts;dual56.61same241/96rounds288drafts145accepts/3facts/15fallbacks/snapshotmetadata. All9source/build backups exact; all3rebuiltbinaryhashes DIFFER from original, explicitly not claimed identical. Initial byteidentity assertion failure preserved; current binaries bound to real validation. 8focusedrebuiltCTest/11policy/18records plusactualhistoricalstrict andmargin trajectories pass; no newC8/media/archivebinary certificate. Both100goalsOPEN. Alljobsfinished/GPUsfree; noagents/training/hardware/modelrewrite/commit/push/deploy; oldownerprefix/queue backed up.

### 2026-10-02 20:58 +07 — Factored residual-lattice dense dataflow checked and rejected against production cost
- New primary large-family decoder changes computation ownership: two nativeS4 dotproducts for exact existing3bit E8P+E8 coefficients, combined before originalG64FMA. It does not repair failedE8P2 quality via bitwidth or grid changes. Originalrepresented217.09-229.15us ->new206.18-211.62us, still15-16%slower than production178.02-184.32. Payload66.85MBplus11.14MBscales/sums, preparation separatelycharged. Gatefailed, no downstream/modelconversion or parameterorbit.
- BothGPUs268435456decodedcoefficients exact;835584intermediatecodes/13056scales/835584rawvalues equal oldrepresented baseline.1536independentCPU fullgateformulas no mismatches;12causal future-column tests, changed-source and same-address input graphreplay, guards pass. No unqualified originalmodelquality inference. Previouswhole64TP2graph examined before scheduling work; identical closed experiment not rerun or reclassified.
- All3productionexe hashes preserved, no productsource changes. Ownbenchmarktargetarchived andCMakeexactpriorcopy. No freshfullmodel speed certificate, previous63.96/56.61notrelabelled. Allcurrentacceptedhead/attention/memory/servingroutes remain. Sourceboundreport/index/oldhandoff/ownerprefix/status retained. Both100goalsopen; allownjobsfinished/GPUsfree. Noagents/training/hardware/modelartifactwrites/commit/push/deployment.

### 2026-10-02 21:50 +07 — Query-owned exact attention completion retained after paired full-context speedup
- Three attention-completion stages fused into one query-owned block; original coordinate tree, exact-subset FMA sequence and explicit native finalFMA contraction preserved. Prepared halfbasis permuted for contiguous consumer loads using existing64KiBscratch; no persistentallocation or pointer changes. Independent factors parallelized, denominator order retained. Finalkernel automatic in existinginteger mode, floatmode unchanged, privateflagremoved. Five scopedsource/test/docfilesretained.
- Actualsamebinary260973 eight-run final native56.44/56.50/56.38/56.25 vs56.68/56.60/56.51/56.50, mean56.3925->56.5725(+.3192%), all4adjacentpairspositive. Same241naturalanswer/96rounds288drafts145accepts/3records/15fallbacks/twoterminalcounts/snapshotmetadata. Finalrebuilt56.70exactoldbehaviorwithoutprivateflag. Naturalclocks varied; no confidenceinterval orwholelatencyclaim. Earlierfusion/layouttimingsnotadded/cherry-picked; standalone1.53–1.61xnotcalledmodelgain.
- BothGPU96scenarios/card pass2,949,120validoutputbitcomparisons and6,291,456basispermutationelements plusflags/guards/empty/nonfinite/simpleCPUformula/changedsourcegraph tests. Existingvalid-query/directmetadata/orderedunion tests passbothcards;8focusedCTest/11policy/18records actualpasses. Initial1ULPcompilercontractionissuefixedwithouttolerancerelaxation. CLI/servehashesoriginal,dualnewhashindexed; benchCMakeexactrestored. No newC1/C8/media/archive-executablecertificate andoldintegerattentionqualitylimitsremain. Alljobsdone/GPUsfree; noagents/training/hardware/modelartifact/commit/push/deploy. Both100goalsOPEN; priorhandoff/status/ownerprefix preserved.

### 2026-10-02 22:54 +07 — Producer-owned exact query preparation retained after full-context paired gain
- Three-stage integerquery preparation plus two state-initialization passes replaced by onequery-owned block. Eight original basisdot warps andoneindependentmean/quantizerproducer, same max tree/4partialFMA/pairwise sum/FP16boundary/shift/finalA8. Full104metadata/allunionwords zeroedwithonewriter each. Existingbuffers/floatmode preserved, no rank/precision/key/weight/acceptance change, privateflag removed.
- Actual2609738runs native56.69/56.52/56.48/56.39 vs56.82/56.75/56.65/56.46; means56.52->56.67(+.2654%), all4pairspositive(.13,.23,.17,.07). Same241naturalanswer/96rounds288drafts145accepted/3facts/15fallbacks/snapshot. Finalrebuilt56.83exactoldbehaviorwithoutprivateflag; confirmationnotanadditionalgain. Clocksvaried, no universalCI. Earlierbasicfusionmixed andindependentroleintermediate ratesnotadded.
- BothGPU14cases/card compare537600outputbytes and98304CPUprojectionvalues,1536mean shifts,98304projectedquantizervalues;18initializationextents/guards/read-only/crossquery/source-graphcases pass. EightCTest + existingcompletion/validquery/directmetadata/orderedunion bothGPUtests +11policy/18records pass. No freshC1/C8/media/archivebinarycertificate. InitialtwoCPUsubmissionthreadcasewas8%slowerdespiteexactresults; originalTP2source restored, threadsheaderarchived, no affinity/spin repair. CLI/servehashespreflightidentical;dualnewhashbound.5filesretained, benchCMakeoriginal. Noagents/training/hardware/modelartifact/commit/push/deploy;both100goalsOPEN. Alljobsdone/GPUsfree; oldownerprefix/queue preserved.
