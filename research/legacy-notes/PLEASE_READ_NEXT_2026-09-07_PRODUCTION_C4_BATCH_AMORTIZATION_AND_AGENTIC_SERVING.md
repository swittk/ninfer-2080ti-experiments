# NInfer next research direction — production C4 batch amortization and agentic serving

**Date:** 2026-09-07

This note is an owner/supervisor research update for `feat/sm75-dual-gpu-full-context`. Read `AGENTS.md` first. All existing stock-hardware, no-reset, no-training, correctness, multimodal, and no-subagent rules remain absolute. This note does **not** lower or replace any existing hard gate.

## End state

The target is not another benchmark-only `apps/dual` frontier. The target is one production `ninfer-serve` OpenAI-compatible endpoint backed by two independent stock 22 GiB RTX 2080 Ti Engines, with stable per-request ownership, exact prefix validation, dynamic membership, cancellation, multimodal support, and the existing long-context route preserved.

Keep the already-qualified frontiers intact:

- stock single short: ~107.27 useful tok/s;
- dual short: ~104.94 useful tok/s;
- genuine populated 261,900-token context: Q64 median 104.53 useful tok/s, representative quality pass;
- physical C4+C4 text: Q109 166.85 aggregate, minimum 20.32/user.

The existing production-serving owner gate remains physical text/image/video/mixed C8 >=160 aggregate and >=20 useful tok/s for every user, with acceptable TTFT/p95/fairness/churn/prefix behavior.

**But do not treat 166.85 aggregate as the architectural ceiling.** Once the production gate is genuinely met, continue aggregate optimization while a bounded complete-system C4 ceiling remains materially above measurement. A practical research progression is: first make sustained production C4 on each card clearly outperform that card's C1 aggregate; then clear 200+ physical C4+C4 aggregate; then pursue roughly 250-300 if measured full-body ceilings still support it. Those are optimization targets, not replacement hard gates and not promises.

Do not game this by choosing only code/JSON or repeated-copy workloads. Report at least three separately labeled classes: representative mixed serving, long-context, and agentic/repeated-copy. Never merge them into one headline number.

## What the archaeology says is already exhausted

Do **not** reopen these merely under a new paper/project name unless a new complete-system mechanism is genuinely different:

- generic QServe/OmniServe porting: SM80 `cp.async`/m16n8k32 core does not transfer; the Turing residual loader/compute family was already measured;
- generic Humming, BitBLAS/Ladder, TEAL, VPTQ, QuaRot, W2/W3/W4 framework ports at the already-tested shapes;
- static proposal-prefix truncation and request-local active vocabulary. The production 131072-row draft head is already a **frequency-ranked non-contiguous shortlist** produced by `tools/convert/qwen3_6/common/draft_head.py`; P0-R tested shrinking that ranked list through 98K/65K/32K/16K and P0-AP tested request-local active vocabulary;
- ordinary sparse-proposal certification/indexed rerank families already bounded by P0-AB/P0-BF and related artifacts;
- KV4/K4V2/K4V4/tiered/lossless/low-rank/sparse-KV families already closed by the numerical or complete-system gates;
- FAFO/VeriCache/PipeInfer/PARD/ASD/lookahead/token-recycling/consensus/periodic-anchor families already bounded;
- TP2 mailbox/mapped-memory/stream-memop/transport permutations already closed;
- generic long-context GQA split/fraction/warp/cache/rebalance sweeps;
- naive native-T16 grid-y token tiling from Q90/Q91. The live opportunity is the **Q95/Q96 weight-fragment-reuse family**, which is mechanically different;
- more shared-C8 `apps/dual` microvariants. The winning ordinary eight-user topology is physical C4+C4.

The deleted/prunable old worktrees do not appear to hide a later distinct mechanism. Their surviving branch refs mostly contain early Aug-18 A8/C4/TP2 correctness work that is already represented in current source, `progress.md`, or the surviving `.chatgpt` evidence. Treat the current worktree plus artifacts as the richer authority.

## P0 — finish the missing production T16 weight-amortization bridge

This is the strongest current lead.

The real production steady-decode profile already holds `avg_decode_batch=4.00` on each replica, so the current problem is **not lack of continuous batching**. The Nsight short C8 profile shows approximately 85% of GPU kernel time in the dense Q4/Q5 body:

- Q5 row-split split2 ~32.2%;
- Q4 small-T MMA ~22.7%;
- other Q5 row-split/MMA paths ~29%;
- Q4 row-split ~3.9%;
- proposal head ~3%; target W8 head ~2%; GDN/attention are much smaller.

The current production source still handles important T9-T16 dense cases by composing qualified <=T8 calls. In particular, `Variant::post_mixer()` slices T9-T16 into T8 chunks, and `mtp_post_mixer()` similarly composes prepared Marlin <=T8 slices. The new `NINFER_SERVE_C4_A8` candidate extends the A8 layout to C1..C4, but that alone is not the same as the already-proven fragment-reuse kernels.

The repo already has exact operator evidence:

- Q95 T16 Q4 SwiGLU: 1.15269x versus two qualified T8 calls;
- Q95 T16 Q5 residual-add: 1.50535x;
- Q96 T16 GDN value/z projection: 1.18560x;
- Q98 T32 Q4: 1.19826x;
- Q98 T32 Q5: 1.59663x.

These are the clearest unharvested measurements in the repo. Integrate the **true Q95/Q96 T16 fragment-reuse path into the real Program/Variant C4 MTP execution**, not another harness wrapper. Preserve C1 exactly and keep B1/B2/B3/ragged fallbacks conservative until separately qualified.

Qualification order:

1. Run a sustained one-replica C4 test with 512-1024 output tokens per request, not a 3-token `READY` smoke. Hold B4 residency long enough for real throughput measurement.
2. Same prompts/seeds/sampling and same card for baseline/candidate. Require target output contract and MTP trajectory classification to be explicit.
3. Confirm profiler evidence that the T16 Q4/Q5 kernels actually replace the two-T8 path. If the top dense percentages do not fall, the intended route was not reached.
4. Only then run physical C4+C4 through `ninfer-serve`.
5. Re-check cold/staggered/refill because layout transitions around prefill must remain correct.

The 15:07 `c4_a8_smoke` is a correctness smoke only: each request generated only three tokens and `avg_decode_batch` was mostly ~1. It must not be used to classify C4 throughput.

## P0 ceiling oracle — single-card short-context C8 / T32

After T16 production integration, answer the real hardware question directly: how far does one TU102 scale when enough independent columns share a weight stream?

Use a **bounded ceiling experiment**, not a product change: if memory permits with a deliberately small max-context/KV budget, run one card at C8 long enough to sustain the T32 MTP geometry. Q98 already proves exact T32 Q4/Q5 fragment reuse with material gains. Measure the full model and identify whether aggregate continues materially above C4.

If C8 cannot fit, quantify which allocations block it (per-session recurrent state, KV, graph family, transformed head/cache, workspace) rather than guessing. This tells us whether future dynamic short-context higher concurrency is worth engineering. Do not sacrifice the production 8K/C4 or 262K contracts merely to make this oracle fit.

## P1 — explicitly measure the MTP-vs-plain-batch crossover

Modern Qwen3.8 serving stacks do not assume speculation wins at every concurrency. A current RTX 3090 vLLM deployment reports MTP as the low-concurrency mode and plain large batching as the high-concurrency mode, with the crossover around C8.

Our current sustained production C8 profile shows only about 52-56% MTP draft acceptance on the short mixed requests. Q109's physical C4 timing also shows the proposal body dominating each replica: roughly 4.64-4.67 s proposal versus ~1.02-1.04 s target stage0 and ~0.09 s target stage1/head.

Q77 tested batched ordinary decode earlier, but it was an older shared-dual route and is not a decisive production-replica crossover test. After the real T16 dense paths are in place, run a bounded **one-replica sustained C4 MTP3 vs ordinary exact decode** A/B through the current Engine. If MTP still wins, close the crossover at C4. If ordinary wins, use a simple concurrency/profile dispatch only if it preserves CUDA-graph and correctness contracts. Do not introduce unstable dynamic per-round switching.

## P1 — multi-lane packed prefill / decode-maximal scheduling

The production scheduler has chunked staged prefill, but `ConcurrentExecutor` still has one `prefill_lane_` owner per Engine. Multiple cold prompts therefore remain substantially serialized even though warm prefix TTFT is now excellent.

Research direction: Sarathi/Sarathi-Serve's useful idea is not merely "use chunks"; it is **decode-maximal batching**, where a prefill chunk fills otherwise underutilized compute while ongoing decodes piggyback, and chunking makes those mixed iterations regular. FlashInfer now exposes both unified mixed BatchAttention and Prefill-On-Decode as an executable design reference.

NInfer-specific progression:

1. First support multiple staged prefill owners so cold requests are not semantically tied to one prefill lane until completion.
2. Then investigate packing compatible prompt chunks from multiple cold requests into one high-M model pass. Dense projections should naturally amortize weights across the packed token matrix; attention/KV writes remain request-indexed/ragged.
3. Preserve decode-maximal service: an admitted cold request must not stall all live decode lanes for a whole prompt.
4. Only if the full model can carry mixed prefill/decode safely should attention be unified in one mixed kernel; use FlashInfer BatchAttention/POD as an oracle, not as permission to import an SM80-specific backend blindly.

Success is cold/staggered TTFT p95 plus E2E throughput improvement **without** degrading warm steady decode or fairness.

The current chunk512 evidence is useful but not sufficient: it improved 2K cold E2E from ~42.9 to ~48.1 aggregate and warm to ~90.6, while cold TTFT remains tens of seconds. That says chunk size alone is not the final scheduler.

## P1 — prefix- and shape-aware replica admission

Current `ReplicaRouter` is intentionally simple: stable locality hint + active-count balancing. Keep exact prefix validation inside the Engine, but the router can become more informative without moving KV between replicas.

SGLang's current routing/scheduling work provides two transferable ideas:

- cache-aware routing: prefer the worker likely to own a substantial reusable prefix while load is reasonably balanced, otherwise fall back toward shortest queue;
- shape-aware routing: separately account for projected input/prefill load and output/decode load rather than treating all active requests equally.

For NInfer, expose only **side-effect-free admission hints** from each Engine: reusable-prefix estimate after tokenization, running decode count, staged-prefill tokens, and free admission capacity. Route by a bounded score rather than a raw full-text fingerprint. Long-lived agent sessions should also retain their reusable leaves preferentially under pressure; SGLang's session-aware radix cache is a useful semantic reference.

This is primarily a TTFT/cache-hit/fairness improvement, not the first saturated-C4 throughput lever. Do it after the dense T16 path unless the server is otherwise blocked.

## P2 — production stacked session-history copy drafter for agentic workloads

This is **not** a general-model headline metric. It is a separate agentic accelerator that matches the owner's eventual YOLO/software-worker use case.

NInfer already ran the `p0_s_long_lookup` oracle. It showed high exact lookup potential from prior prompt/output tokens, including LZ77-style overlap, but no production source n-gram/session-copy drafter exists.

A current Qwen3.8 RTX 3090 llama.cpp deployment stacks an exact session n-gram copy drafter ahead of MTP: after an exact 32-token history match it drafts up to 64 verbatim historical tokens, target verification remains exact, and MTP handles all non-copy rounds. Their 8-turn file-editing workload reports 108.8 -> 260.8 tok/s overall, with warm turns 237-288; prose improved much less and a pathological self-repeat case regressed ~1.7%.

A clean NInfer version would:

- maintain a bounded host/session token-chain table;
- fire only on a sufficiently long exact suffix match;
- draft verbatim copy; target verifier remains the authority;
- preempt MTP only on those copy rounds;
- keep zero model-specific prompt heuristics;
- benchmark an actual multi-turn file-edit/tool-agent episode, representative prose, and a pathological repetition control separately.

Do **not** use this to claim the general C1/C4 frontier. Report it as `agentic/session-reuse` throughput. It is valuable because the intended product is an OpenAI-compatible worker endpoint, not because it makes benchmark prose magically faster.

## P2 — capacity and high-concurrency mode only if the C8 oracle proves it

If the one-card C8/T32 ceiling is strong but production memory blocks C8, do a memory ownership audit before inventing another scheduler. Quantify per-request recurrent state, KV pages, graph allocations, proposal/target head caches, transformed layouts, and optional vision residency. Only then consider a short-context high-throughput profile with higher per-card concurrency.

External RTX 3090 stacks can reach very high aggregate rates at large concurrency largely because they trade context/cache capacity and mode selection for batch width. That is evidence that batch width can matter enormously; it is **not** permission to weaken NInfer's 8K/C4 or 262K capability.

Optional multimodal capacity lever: if enabling vision materially prevents the desired text concurrency, measure whether the vision tower can be lazily resident/offloaded while preserving the same API. Treat this as a memory/product experiment, not a decode optimization.

## External research sanity checks

Use these as scaling/design references, never as apples-to-apples targets:

1. **syv-ai/qwen38-27b-rtx3090** — same Qwen3.8-27B family on one RTX 3090. A realistic MTP serving table reports roughly C1 111 tok/s, C2 192, C4 269, C8 407 at default sampling (greedy about 120/199/281/414). This demonstrates that C4 aggregate can be far above C1 when the dense path actually amortizes weights.
2. **0x7067/rtx3090-llm-lab** — another same-model profile reports C1 ~104 and C4 ~358 aggregate under its retained vLLM arm. More importantly, its session n-gram work and small-batch MMQ analysis identify the same two themes seen in our artifacts: repeated-copy speculation can be enormous for agents, and quantized dense projection dominates the normal verify step.
3. **sergiuszm/ninfer-5090** — NInfer itself, Qwen3.6-27B groupwise-int, reports C1 185.8, C2 247.0, C4 309.5, C8 535 aggregate. This is strong evidence that NInfer's architecture is not intrinsically unable to scale batches; the SM75 groupwise-int path is the missing optimization surface.
4. **Sarathi / Sarathi-Serve** — chunked prefill + decode-maximal batching; useful scheduling model for mixed prefill/decode service.
5. **FlashInfer BatchAttention / POD** — current executable reference for a single mixed prefill/decode attention launch with paged KV. Use as a design/perf oracle where SM75 support is relevant, not a drop-in assumption.
6. **SGLang RadixAttention/router/session cache** — cache-aware and shape-aware admission, longest-prefix scheduling, session-aware cache retention.
7. **TensorRT-LLM CUDA graph batch-size work** — exact graph-shape coverage matters because padding/fallback can waste throughput. NInfer already captures MTP graphs for each B1..B4; preserve that advantage. Do not add dynamic speculative-depth behavior that silently exits the captured path.

## Reporting discipline

For every candidate, keep the result categories separate:

- **general production serving:** sustained 512-1024 token outputs, mixed realistic prompts, physical `ninfer-serve`, C1/C2/C4 per replica and C4+C4 physical aggregate;
- **long-context:** qualified genuine 65K/261.9K contracts, never extrapolated from 8K;
- **agentic session reuse:** multi-turn code/file/tool workloads, explicitly labeled;
- **multimodal:** image/video/mixed measured physically, not inferred from text.

Always record actual batch residency (`avg_decode_batch` or equivalent), useful committed tokens, MTP acceptance/yield, TTFT p50/p95/max, inter-token/inter-chunk p95, fairness, loaded clocks, and per-replica memory. A short request that terminates after three tokens is a smoke test, not a throughput sample.

## Immediate next action

Finish the current C4 A8 correctness work, then **do a sustained C4 measurement that proves whether the real Q95/Q96 T16 fragment-reuse family is actually executing in production**. If the current candidate is still two-T8 composition, do not spend cycles microtuning it: wire the true T16 reuse path first. Re-profile. From that result, decide whether the next bounded experiment is the single-card C8/T32 ceiling or the C4 MTP-vs-ordinary crossover.

Production correctness, prefix reuse, churn, TTFT, and physical multimodal certification still come before any final seal. No subagents.


## Live update — 15:26 +07: sustained C4 A8 is a real major advance

The sustained measurement requested above landed before this note was discovered by the supervisor:

- `.chatgpt/production_serve_20260907/c4_a8_text2k/result.json`
- warm physical HTTP C4+C4: **152.6686 aggregate E2E tok/s**; TTFT p50 **0.724 s**, p95 **1.564 s**; minimum visible decode **22.41 tok/s/user**; fairness **0.9624**; 8/8 complete with no errors;
- when both replicas are fully decode-ready, server intervals reached roughly **181-199 committed tok/s** at `avg_decode_batch=4.00`;
- churn: 127.22 aggregate E2E with the intended 3 cancellations and minimum visible decode 31.09/user;
- cold remains 47.13 aggregate / TTFT p95 60.05 s, and refill remains 96.63 / TTFT p95 28.76 s.

Therefore **preserve and qualify the C4 A8 mechanism; it is not a dead smoke path.** First identify exactly which dense route/layout transition produced the ~1.8x warm E2E gain versus the prior ~85.6 baseline and prove its output/trajectory/product contracts. Then profile the sustained A8 path itself. The Q95/Q96 true T16 fragment-reuse recommendation remains open only where the A8 path still composes two T8 calls; do not replace a faster A8 route with an older primitive blindly. The immediate research question is now: can the A8 path plus true T16 reuse (where structurally compatible) move sustained physical C4+C4 from ~190-200 decode toward the 250+ region while cold/prefill scheduling is fixed separately?
