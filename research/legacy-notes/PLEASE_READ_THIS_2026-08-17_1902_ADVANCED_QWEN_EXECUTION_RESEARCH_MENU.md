# PLEASE READ — advanced Qwen-specific execution research menu after sharded MTP baseline

Owner/adviser research note, 2026-08-17 19:02 +07. This is **not** an instruction to interrupt the current sharded-MTP implementation. Finish and benchmark that coherent path first. This note records the next higher-leverage design space so the project does not prematurely stop at ordinary two-stage pipeline parallelism.

## Immediate priority stays the same
1. Finish correct sharded MTP3 target verification with GDN replay/fold semantics.
2. Pipeline MTP verification rounds across 2–3 independent sessions.
3. Add packed/in-flight ready-session batching only where it improves real useful tok/s.
4. Benchmark populated context frontiers and preserve the upstream-oriented numerical correctness gate.

## A. Intermediate-hidden speculative predictor at the GPU boundary
The stage0->stage1 boundary hidden is already computed and transferred. Treat it as a free early-exit feature source. A cheap checkpoint-specific predictor can propose future tokens while GPU1 finishes the exact late layers. This is conceptually aligned with speculative pipeline decoding / layer-parallel speculative decoding.

Audit first: at candidate split layers (e.g. 28/32/36/40), record how predictive the boundary hidden is of the final target token / existing proposal-head winner on a diverse corpus. Report top-1/top-k agreement, target-rank distribution, confidence/margin, and incremental predictor latency. Do not build a heavy head until the oracle/predictability audit says it is worthwhile.

If useful, prefer a tiny predictor or projection into the existing proposal-token space rather than a second full vocabulary head. The predictor is speculative only; wrong predictions must affect acceptance/throughput, never target semantics.

## B. Extremely low-bit speculative sidecar is fair game
Native Turing B1/BMMA, ternary/BitNet-like weights, or S4/A4 can be used for a **draft/ranking predictor** even if they are too approximate for the target model. This is a safer use of aggressive low-bit math because exact target verification remains authoritative.

Do NOT infer that arbitrary Qwen target weights can simply be rounded to ternary because BitNet-style models exist. If exploring B1/ternary/S4 here, measure candidate recall/rank and end-to-end accepted useful tokens per second.

## C. Proper source-weight PTQ may unlock a different dual architecture
The previous Q5->Q4 / Q4->Q3 work requantized an already-quantized NInfer artifact. That does not close the BF16-source calibrated PTQ space. Preserve a future research lane for GPTQ/AWQ/OmniQuant/rotation/outlier-aware or vector/codebook-style compression, adapted carefully to the hybrid GDN architecture.

The payoff is not merely faster kernels. If model + per-session state can be made small enough to fit a whole 262K-capable session on each 22-GB card, switch to full-model replication/data parallel serving:
- GPU0 runs complete NInfer for user A
- GPU1 runs complete NInfer for user B
This is the cleanest route to near 2x aggregate bandwidth / ~72+72 checkpoint-lane throughput, with no PP bubbles or distributed GDN rollback. Treat "replicate whole model per GPU" as an explicit memory-compression milestone.

## D. KV compression can have nonlinear architectural value
INT4 or otherwise compressed KV is worth auditing not only for attention bandwidth but because it may be the difference between "must pipeline-shard" and "full replica fits per GPU" at long context. Quantize carefully and test long-context quality; do not modify persistent GDN recurrent state aggressively until weight/KV avenues are exhausted.

## E. Dynamic/precompiled split selection
Do not assume one fixed split layer is optimal for every workload. The output head, attention cost, GDN mix, context frontier, verify width, and concurrency change stage balance. Profile known per-layer/per-phase costs and allow a small set of prequalified split points selected by mode (e.g. short MTP decode, long-context MTP decode, 2-session, 3-session). Avoid runtime migration of huge weights; select among preloaded/precompiled layouts where memory permits.

## F. Hybrid context parallelism only if long-context attention becomes dominant
At populated 128K/262K, if full-attention KV scanning dominates, consider splitting only those long-context attention reductions across devices while retaining depth PP for the rest. This is lower priority on the present non-P2P topology; measure first. Do not cargo-cult TP/CP into a PCIe/NODE topology where communication costs more than the work saved.

## G. Scheduler should become a policy layer, not a second inference runtime
Keep canonical model semantics in TextContext/Variant/ops. `TextShardStage` should remain a thin state/transport adapter. `apps/dual/main.cpp` is an experiment/CLI, not the permanent serving runtime. Long term, expose reusable primitives for sharded verify, replay/fold, packed ready-session execution, and optional predictor hooks; keep scheduling above them.

## H. CPU/GPU overlap and graphing are late-stage polish, not the first breakthrough
Once MTP/packed execution works, overlap acceptance/bookkeeping/batch formation with GPU execution and use CUDA graphs where stable. Do not optimize host scheduling before the target-round architecture is right.

## Decision framework after first dual-MTP benchmark
- >=120 useful tok/s aggregate: excellent; optimize balance/bubbles and long-context behavior.
- 100-120: acceptable but substantial headroom likely remains; profile stage imbalance, draft overhead, head cost, and packed-session opportunities.
- 80-100: architecture still leaving important bandwidth/reuse on the table; investigate predictor/packing/split and checkpoint-qualified ~72 machinery.
- ~45-60: likely still scalar-like or failing to amortize target sweeps; do not polish around it.

Keep hard gates: single-GPU INT8 >=80 remains open research; dual performance mode >=100+ minimum. The ~72 checkpoint-qualified lane remains important evidence and should be requalified, not forgotten.
