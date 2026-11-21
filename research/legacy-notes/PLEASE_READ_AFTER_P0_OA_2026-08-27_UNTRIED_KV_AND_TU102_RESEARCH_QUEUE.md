# READ AFTER CURRENT P0-OA — UNTRIED KV/TU102 RESEARCH QUEUE

Do **not** interrupt P0-OA native affine K4/V2. Finish it, restore source, and record quality + real E2E first. This is the next queue only if P0-OA does not solve the long gate.

Current situation: single80, dual-short100+, populated65K100+ are reached. Representative populated262K >=100 remains. At 262K D4 yield can already be near fully licensed, so the remaining gap is primarily target/KV bytes and target-pass cost, not another generic acceptance sweep.

## Do not reopen
Already covered/upper-bounded/closed here: global KIVI/TurboQuant/KVarN; Quest/page top-k; exact-score GQA-group top-k upper bound; StreamingLLM sink+tail; static prefix summaries; semantic retrieval; generic SnapKV/PyramidKV/LServe-like eviction at comparable retained budgets; broad short dense Q4/Q5 microgeometry; ordinary D4/D5/tree/Bole/PEARL/fifth-proposal speculation. P0-OA already covers native global affine K4/V2.

## P1 — highest priority after P0-OA: layer-wise K/V mixed precision
Research: KVTuner (arXiv:2502.04420), ChanMix (ICLR 2026), KVmix (AAAI 2026), KV Pareto (EACL 2026). These all support the idea that KV sensitivity is heterogeneous; KVTuner includes Qwen-family evidence.

I do not find an actual layer-wise K/V precision search in our progress/artifacts. Qwen3.8-27B has only 16 full-attention layers, so brute-force a hardware-restricted per-layer palette.

Offline first, no CUDA:
- use actual 65K + representative262K captured queries/K/V;
- perturb one attention layer at a time, all others canonical INT8;
- test `INT8`, `K4/V4`, `K4/V2`; also cheaply test `K8/V2` and `K8/V4` because keys are usually more sensitive than values;
- measure attention-output cosine/rel-L2, target-token/logit stability and downstream representative behavior;
- search the 16-layer byte/quality Pareto frontier and report weighted average bits + projected 262K bytes/round.

First runtime implementation should use one **fixed format per layer** so cache layout stays contiguous and dispatch is cheap. No token-dynamic precision initially.

## P2 — fixed key-channel high-precision sidecar
Research: MixKVQ (ACL 2026), ChanMix (ICLR 2026), KVQuant (arXiv:2401.18079). MixKVQ argues key precision should depend on channel quantization difficulty + query relevance while V remains aggressively low-bit.

Bounded oracle from the best P0-OA/P1 format:
- for each of 16 layers x 4 KV heads, rank key channels by actual query-weighted error contribution;
- keep only top 1/2/4/8% channels INT8, or add a compact 2/4-bit correction plane; rest stay K4; V stays V2/V4;
- pack fixed sidecar channels contiguously/offline; avoid random per-token gathers;
- fixed masks first. Query-aware dynamic masks only if fixed masks clearly win.

This is distinct from page/token eviction and may rescue K4 without rereading full INT8 K.

## P3 — attention-aware bit allocation, not raw KV reconstruction
Very recent: *KV Cache Compression Through the Lens of Transform Coding* (arXiv:2608.14191, AATC) optimizes attention-aware rate-distortion and reports near-lossless compression on Llama/Qwen families.

Use this as an offline allocator, not a wholesale port: optimize against attention-output / target-logit distortion using our real query captures, restrict choices to our finite palette (`INT8`, `K4/V4`, `K4/V2`, optional key sidecar), and reuse only fixed Hadamard/variance transforms already justified by KVarN. No learned transform/training. Feed the resulting layer/head/channel bit allocation into P1/P2.

## P4 — tiny data-dependent outlier-token sidecar
*Accurate KV Cache Quantization with Outlier Tokens Tracing* (arXiv:2505.10938) finds unusual tokens that break normal low-bit KV assumptions. This differs from our fixed sink/recent windows.

Cheap oracle only: keep <=0.5/1/2/4% content-dependent old tokens canonical INT8, everything else low-bit. Stop if more than a few percent is needed or layout becomes random-gather dominated. Prefer page-local compact exception lists.

## P5 — TU102 native S4 QK only after K4 quality passes
Repo search currently finds no native `s4.s4` GQA path. SM75 supports S4 MMA, but QK needs an A4 query.

Only after a K4 cache is worth serving: microbench per-group A4 query + packed K4 native S4 MMA versus current unpack/dequant route; compare attention error, SASS/registers, latency and E2E relevance. Reject immediately if A4 query hurts target behavior or unpack/compute is not material after K-byte reduction.

## P6 — later product win: reopen single ~100
Single80 was frozen before later discoveries. After the 262K branch reaches a natural checkpoint, do a bounded one-GPU compatibility matrix for P0-M proposer, indexed rerank, qualified margin policy, K16 final-head recall and any semantically justified fast licensed-accept shortcut. Complete E2E + six-family quality required.

## Paper ideas not to port literally
UltraQuant 2026 uses AMD CDNA4 FP4/FP8 MFMA; methodology is relevant, kernel path is not SM75-transferable. DuoAttention/CompressKV/SnapKV/PyramidKV/LServe/Quest should not reopen generic eviction: our exact-score selector and sink/tail upper bounds already close comparable retained-page economics. Retrieval-head/channel sensitivity can still be used as a **precision-allocation signal** for P1/P2. No training/fine-tuning.

## Order
1. Finish P0-OA.
2. If insufficient: P1 layer-wise mixed-precision oracle.
3. If close: P2 key-channel sidecar and/or P3 attention-aware allocation.
4. P4 only as bounded rescue.
5. P5 only after K4 quality passes.
6. Do not write new P1-P4 kernels until offline byte/quality accounting shows a plausible path toward representative262 >=100.

Always report real/projected bytes per round, target-stage time, useful tok/round, E2E tok/s, representative262 quality, and survival under the actual two-card physical-KV/helper topology.
