# nInfer RTX 2080 Ti / SM75 progress ledger

This file is an **append-only project memory** for the Turing port. Record major architecture decisions, performance checkpoints, accepted/rejected experiments, and anything a future agent needs to avoid repeating work.

Use Bangkok time (`Asia/Bangkok`, UTC+07:00). Prefer `python3 scripts/append_progress.py "Title" "Details..."`. Do not rewrite old entries to make the story cleaner; append a correction entry if an earlier statement becomes wrong.

## Current reference

- Project root: `<local-path>`
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
- Moved Git root from `<local-path>/Documents/ninfer-3090` to `<local-path>`.
- Recreated CMake 3.31.10 environment inside `.tools/cmake-venv`; old `<local-path>/Documents/ninfer-build-tools` removed.
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
