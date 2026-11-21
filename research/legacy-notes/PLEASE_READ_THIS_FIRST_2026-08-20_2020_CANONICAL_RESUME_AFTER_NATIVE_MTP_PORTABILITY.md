# CANONICAL RESUME — 2026-08-20 20:20 +07

Live state when written:
- branch `feat/sm75-dual-gpu-full-context`
- HEAD/remote `c99f5bd7df3c8410322df79a6475d817205a30a4`
- tracked source clean
- no producer running; GPUs idle
- last work ~19:36: factory↔trained native-MTP alpha0.25 general canary rejected

Hard gates remain:
- single Core-X >=80 useful t/s
- dual one-request >=100
- populated ~262K dual one-request >=100

Read current `progress.md` plus:
- `PLEASE_READ_THIS_FIRST_2026-08-18_MANDATORY_PREFLIGHT_AND_CANONICAL_LEDGER.md`
- `PLEASE_READ_THIS_FIRST_2026-08-18_NO_WATCHER_THEATRE_KEEP_WORKING.md`
- `PLEASE_READ_AFTER_CURRENT_E2E_2026-08-20_1912_LONG_CONTEXT_SM75_DFLASH2_FP16KV_AND_TARGET_COST.md`

## Current trusted frontiers

Single exact proposal-B2 historical anchor: ~71.8–72.1 t/s.
Dual populated-65K fixed-depth4 reference: ~59.05 t/s, 3.90625 useful/round, ~67.219 ms/round.
262K: capacity PASS with physical INT8 KVP + dedup T5 TP; no >=100 throughput result.

## What is solved / accepted

- physical sequence-sharded KV and exact online-softmax merge;
- deduplicated two-GPU TP2 storage;
- true single-pass T5 Q4/Q5 TP (~1.53x isolated MLP discriminator);
- fixed depth4 + corrected snapshot bootstrap semantics;
- parent-indexed tree convolution/GDN recurrence;
- compact all-node ReplaySSM factors + selected-path fold;
- exact ancestor-only tree attention; tree tail cost negligible (<0.04 ms at N22);
- native Qwen3.8 MTP maps to Speculators/FastMTP and trains full-rank offline;
- N17-N64 grouped R64xC32 input projection route: N22 48-layer input 64.674->29.568 ms (2.19x), complete GDN tree 94.2->59.5 ms (1.58x).

## Why we are still stuck

1. Proposal quality is not portable. Full-rank native-MTP training improves code/JSON/Thai/Japanese/Arabic/medium and one independent65K holdout, but materially regresses general technical prose.
2. Wide tree target cost is still dominated by Q4/Q5 projections. Tree recurrence/attention are no longer the wall.
3. ~65K committed-prefix KV/attention is a real hardware/runtime wall; public dual2080Ti SM75 vLLM is ~55.3 t/s at64K even though it can exceed100 at4K.
4. Our pair has no CUDA P2P/NVLink and the internal GPU is ~9% slower.
5. 262K must keep capacity while also reducing target round cost; external drafters can consume too much VRAM.

## Native-MTP training status

Factory2500 / fresh4000 / continuation work proves the head learns. Offline K5 yield improved roughly 2.16->2.40 on the large independent holdout. But online portability is mixed.

Selected candidate online:
- general ~-8.3%
- code/json/multilingual/medium mostly positive
- independent65K holdout +5.6%
- independent65K train prefix -1.2%
Classification: LONG_CONTEXT_QUALIFIED_SHORT_REGRESSION; do not promote.

Fresh4K online general ~-9%, medium strongly positive; rejected.

Cross-weight prefix snapshots are ILLEGAL because snapshots contain MTP KV/block-table state. Candidate snapshots must be model-specific with artifact+prompt provenance.

Alpha0.25 factory↔trained blend general canary also regressed (~43.28->39.06 t/s), so do not turn interpolation into a giant sweep.

## HIGHEST-PRIORITY NEXT TEST — matched65K BF16/F16-vs-INT8 physical-KV A/B

Public dual22GB RTX2080Ti vLLM Qwen3.6 GPTQ-INT4+MTP3 reports at PP65536/TG512:
- FP16 KV 85.5 t/s
- INT8 KV 50.1
- TQ K8V4 40.2
- TQ4NC 36.9
while short-context INT8 can be faster.
Source: `weicj/vLLM-2080Ti-Definitive`, qwen36 KV throughput sweep.

Run strict matched NInfer A/B: same artifact, prompt, model-specific snapshot, MTP depth, split/TP, clocks, generation. Compare complete round, attention/KVP ms, useful/round, TPS, VRAM, exact digest.

If BF16 fits and is materially faster at65K, allow context-specific policy:
- 65K speed route = BF16/F16 KVP
- 262K capacity route = INT8 KVP

Do not assume one KV dtype must win all contexts.

## SECOND PRIORITY — exact Qwen3.8 DFlash2 sidecar

Exact-target Qwen3.8 DFlash2 now exists publicly (`z-lab`/`incoai` variants including GGUF). Old Qwen3.6->3.8 zero-shot transfer remains CLOSED but is irrelevant to exact-target trained weights.

Run sidecar before custom external-drafter training:
- healthy Core-X
- n-max4 and7
- general/code/JSON/Thai/Japanese/Arabic
- 8K and, if feasible,65K
- greedy and canonical sampled separately
- record draft ms, AL, sustained TPS, VRAM/context ceiling.

Do not abandon native MTP. DFlash2 must win complete economics and may be unsuitable for262K due drafter memory.

## THIRD PRIORITY — native-MTP objective/data v2

Do NOT just run more identical CE steps. Current problem is accepted-prefix portability.
Use research-backed alternatives:
- D-PACE dynamic accepted-length position weighting (arXiv:2605.18810);
- AngelSpec MTP TTT / on-policy multi-depth rollout;
- target-generated data;
- long-context roots;
- family-balanced sampling;
- online actual acceptance for selection.

## Target-side followups only after the above data

- audit/test FP16 recurrent state if production state is FP32; require >=5-8% E2E ceiling;
- model SM75 W4A8/INT8 activation + IMMA.8816 only if >=8-10% target-round ceiling;
- dedicated K+1/multi-query committed-prefix attention if BF16-vs-INT8 shows current KVP dataflow is pathological;
- source-audit public `2080Ti-LLM-Toolbox`, `vLLM-2080Ti-Definitive`, `FlashQLA-SM70-SM75` as executable oracles, not automatic replacements.

## Do not reopen

- simple Q4/Q5 cache/geometry microvariants;
- factory-MTP tree as previously tested;
- Qwen3.6->3.8 DFlash transfer;
- prompt/history lookup;
- tiny LoRA/horizon adapters;
- SpecMQuant hierarchy at current target cost;
- GDN host-staged TP;
- event-DAG/coarse scheduler routes;
- cross-weight snapshot reuse.

After every material proposal/KV/target result, recompute single/65K/262K economics and move immediately to the next bounded discriminator. No watcher/wait state.
