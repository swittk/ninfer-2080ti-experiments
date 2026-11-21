# OWNER REVIEW / NEXT EXECUTION — 2026-08-20 21:00 +07

Resume ONLY in `../ninfer-2080ti-dual262k`, branch `feat/sm75-dual-gpu-full-context`, reviewed HEAD `c99f5bd7df3c8410322df79a6475d817205a30a4` unless it has legitimately advanced. Do not work in the older dirty root `feat/rtx2080ti-sm75` worktree.

At review: both 2080 Ti GPUs idle (P8, 0%), tracked source here clean, newest material artifact = 19:36 alpha0.25 MTP general-canary rejection. No watcher mode.

Hard gates unchanged: single healthy Core-X >=80 useful decode t/s; dual one-request >=100; genuinely populated ~262K dual one-request >=100. No OC/power/memory-clock tricks. Keep correctness/determinism and general model usefulness; proposal yield != engine speed.

## Already solved / do not redo

Accepted: physical sequence-sharded KV + exact online-softmax merge; dedup TP2 storage; true single-pass T5 TP; fixed depth4 + snapshot-bootstrap fix; parent-indexed tree conv/GDN; compact all-node ReplaySSM + selected-path fold; exact ancestor-only tree attention; native Qwen3.8 MTP full-rank training; N17-N64 grouped R64xC32 Q4/Q5 route (N22 input 64.674->29.568 ms, complete GDN tree 94.2->59.5 ms).

Closed unless materially new evidence: Q4/Q5 cache/geometry microvariants, Q4-down, Q5 4-bit-core+sparse exceptions, block64 sparsity, host-staged GDN TP, event DAG/coarse scheduler, 2+3/3+2 microbatch verifier, old factory-MTP tree, prompt/history lookup, tiny hidden/logit adapters, Qwen3.6->3.8 DFlash transfer, hierarchy at current target cost, cross-weight prefix-snapshot reuse.

Trusted anchors: single exact ~71.8-72.1; populated65K depth4 ~59.05 useful t/s / ~3.906 useful per round / ~67.2 ms round; 262K capacity PASS but no >=100 throughput.

## Current MTP work: classify correctly

Target correctness is not the failure. Trained candidates are deterministic and target-isolated; proposal portability is the failure.

- selected full-rank: general -8.31%; code/JSON/multilingual/medium mostly positive; independent65K holdout +5.60%, train prefix -1.23%; `LONG_CONTEXT_QUALIFIED_SHORT_REGRESSION`.
- fresh4K: general -9.00%; medium +23.1%; multilingual positive; code/JSON near neutral; `REJECT_MATERIAL_SHORT_REGRESSION`.
- alpha0.25 blend: 43.275->39.055 t/s; useful/round 2.3774->2.1453; `REJECT_GENERAL_CANARY`.

STOP turning interpolation into a sweep. Alpha0.10/0.15 may get a cheap general-only canary only if useful for immediate closure; do not burn hours. Next serious MTP attempt changes objective/data, not alpha or CE step count.

Snapshots include proposal MTP KV + block-table mutable state and are weight-specific. Artifact SHA + prompt SHA provenance remains mandatory.

# PRIORITY 1 — implement missing BF16 physical-KV route, then matched65K A/B

Important repo audit: the accepted physical context-parallel GQA/KVP route is currently hard-coded INT8, so this is NOT just flipping `--kv-dtype bf16`.

- `apps/dual/main.cpp` rejects `--context-parallel-gqa` unless `KvCacheStorage::Int8Group64`.
- `apps/dual/context_parallel_gqa.cu` plans helper K/V as I8+FP16 scales, computes split inventory as I8, requires I8 local cache, and uses an I8-only partial-range launch.
- `src/ops/launcher/gqa_attention_decode.cu` already has `launch_tc_partial_bf16`, while `gqa_attention_split_capacity(...)` already accepts BF16. Thus this should be a bounded dtype-generalization, not a new attention architecture.

Implement the smallest clean extension:
1. Thread KV dtype/storage through `plan_sequence_sharded_kv`, coordinator, StageEngine, DeviceScratch and helper-cache planning instead of baking I8 into geometry.
2. BF16 helper cache = BF16 K/V planes, no quant-scale planes. Preserve current I8 layout exactly.
3. Extend cached small-T partial-range dispatch to BF16 using the existing BF16 producer/warp geometry; do not invent another attention kernel.
4. Preserve same two-range `(acc,m,l)` reduction + deterministic owner merge and physical page ownership; no duplicate full shadow cache.
5. Make snapshot/import/storage dtype-correct. Never reinterpret INT8 snapshot bytes as BF16 state.
6. Requalify current INT8 route unchanged.

Correctness: qualify each dtype against its own canonical same-dtype range/merge reference. Do NOT require BF16 and INT8 output bytes to match each other; they are different numerical cache representations. Require deterministic repeat digests within each route and no state/page/ownership corruption.

Then run strict populated65K A/B with the canonical factory-MTP control first, avoiding trained-MTP portability as a confounder. Same logical prompt, generation length, depth4 control, split/TP policy, clocks and artifact; use legal dtype-specific snapshots. Record attention/KVP ms, complete target/round ms, useful/round, useful TPS, transfer/merge costs, VRAM on both GPUs, and deterministic trajectory per dtype.

External evidence is strong but not proof: public dual22GB SM75 vLLM Qwen3.6 GPTQ-INT4+MTP3 reports PP65536/TG512 FP16 KV 85.5 t/s vs INT8 50.1, while at4K INT8 is faster 95.5 vs81.7:
https://github.com/weicj/vLLM-2080Ti-Definitive/blob/vllm-2080ti-deifinitive/docs/qwen36-kv-throughput-sweep.zh-CN.md

If BF16 materially wins65K, accept context-regime-specific policy — **DO NOT replace, de-prioritize, or remove INT8**. INT8 remains a first-class production route because its lower KV footprint is strategically important for very long contexts and for preserving VRAM headroom. The goal is not to crown one global KV dtype; it is to support both cleanly and choose by context/capacity/performance regime.

Concretely, preserve this architecture unless local evidence later proves otherwise:
- short/moderate context: whichever route wins matched E2E;
- ~65K: BF16/F16 may become the speed route *if* locally proven;
- 131K/262K and memory-constrained regimes: INT8 remains the expected capacity route and must continue receiving correctness/performance support.

Do not contort the INT8 implementation to make the BF16 experiment easier, and do not regress its existing physical sequence-sharded path, storage efficiency, or 262K capacity. The BF16 work should generalize shared abstractions so both paths stay ergonomic and independently optimizable.

Also do a bounded 262K BF16 allocation/capacity probe before assuming it cannot fit: the public SM75 fork now ships FP16-KV 256K profiles on dual22GB. That is not proof for our no-P2P layout; it only justifies a cheap capacity test. Even if BF16 happens to fit, compare its VRAM margin and performance against INT8 rather than automatically promoting it for long-context service.

# PRIORITY 2 — exact Qwen3.8 DFlash2 sidecar, no NInfer integration yet

After BF16 A/B is classified, test exact-target Qwen3.8 DFlash2 on healthy Core-X as an executable proposer reality check. Old Qwen3.6->3.8 transfer remains CLOSED and is irrelevant to exact-target weights.

Use a current supporting sidecar runtime (llama.cpp DFlash2 PR #27342 or equivalent) built for SM75 plus exact Qwen3.8 DFlash2 GGUF. Start n-max4, then7. Current community evidence says n-max4 can beat7 on sustained long generations and reduces verifier/draft burden; local result decides.

Cohort: general/code/JSON/Thai/Japanese/Arabic; ~8K first, then65K if memory allows; greedy and canonical sampled separately. Record draft block ms, accepted/emitted length, sustained TPS, VRAM/context ceiling, and task variance. Sidecar TPS is NOT an NInfer frontier.

Before any NInfer integration, combine measured DFlash2 draft cost + selected-path width/AL with the CURRENT NInfer BF16/INT8 target-round model. Integrate only if projected complete economics can materially move 80/100/262K. Public exact-Qwen3.8 reports prove DFlash2 is real, but also workload-sensitive and memory-hungry; 3090 numbers do not transfer automatically.

Current references:
- 2x3090 report: narrative ~120 t/s, code ~218, 7 drafts, AL~3.35, ~47.8% acceptance, ~131K context ceiling due drafter memory: https://www.reddit.com/r/LocalLLaMA/comments/1vsccit/
- llama.cpp Qwen3.8 DFlash2 usage is associated with PR #27342.

# PRIORITY 3 — native MTP v2, but only after P1/P2 discriminators

Keep native MTP alive: it is VRAM-cheap and is probably the most attractive 262K proposer if portability is fixed. Change the training problem:
- target-generated/on-policy roots;
- family-balanced sampling including general technical prose;
- long-context roots;
- multi-depth rollout;
- acceptance/prefix-survival-aware objective (D-PACE style dynamic position weighting);
- online actual accepted length/E2E as selection metric, not CE alone.

AngelSpec now directly supports MTP TTT/on-policy multi-depth rollout, D-PACE weighting, long-context training up to128K, sequence packing and online real acceptance eval:
https://github.com/Tencent/AngelSpec
D-PACE: https://arxiv.org/abs/2605.18810

Do not run another long identical CE continuation just because loss goes down. The observed failure is cross-family accepted-prefix portability.

# Only after P1/P2 data — target-side lanes

1. FP16 recurrent-state discriminator if ordinary production GDN state is FP32; keep FP32 accumulation where required; only pursue with modeled >=5-8% E2E ceiling.
2. SM75 W4A8/INT8-activation route using Turing IMMA.8816 only if largest target buckets imply >=8-10% complete-round ceiling. QQQ is a mechanism reference, not a drop-in kernel: https://arxiv.org/abs/2406.09904
3. Dedicated K+1/multi-query committed-prefix attention if BF16-vs-INT8 shows current long-context dataflow is pathological.
4. Use public SM75 stacks as executable oracles, not replacements: `vLLM-2080Ti-Definitive`, `2080Ti-LLM-Toolbox`, `FlashQLA-SM70-SM75`.

Research nuance: `FlashQLA-SM70-SM75` reports ~2.08x GDN-stage improvement on 2080 Ti but only ~0.61% decode gain in its measured whole request. So do not make another giant GDN-local optimization lane unless a local complete-round ceiling says it can matter:
https://github.com/weicj/FlashQLA-SM70-SM75

SpecMQuant still explains why wide tree verification on a 4-bit target can destroy the bandwidth advantage; our measured N22 projection wall independently confirms it. Do not reopen hierarchy/tree-width merely because the tree semantics now work:
https://arxiv.org/abs/2505.22179

## Execution discipline

After every material KV/proposer/target result:
1. exact correctness/determinism gate;
2. real E2E useful TPS, not verifier/draft throughput;
3. classify ACCEPTED/REJECTED/CLOSED;
4. update artifact conclusion + `progress.md`;
5. recompute single / populated65K / populated262K envelopes;
6. immediately continue to next bounded discriminator.

Do not cherry-pick code prompts. Do not conflate proposal acceptance with engine speed. Do not revive closed routes without a new quantified ceiling. No watcher/wait state and no stopping after merely writing a conclusion.


# OWNER OVERRIDE — NO DELEGATED AGENTS / NO CODEX

This NInfer workstream is too delicate and stateful for delegated implementation agents. **Do not invoke Codex, codex exec, Claude, OpenCode, sub-agents, agent loops, or any autonomous coding wrapper for implementation/research/benchmark decisions.** Do not create scripts whose purpose is to launch or supervise such agents.

All source inspection, edits, builds, benchmarks, profiling, correctness qualification, and scientific classification must be performed directly by the primary ChatGPT session through the trusted repository/host tools. External tools may be used only as normal compilers, profilers, benchmark binaries, scripts, or reference runtimes—not as coding/reasoning agents.

If prior local orchestration artifacts mention `codex exec`, `--full-auto`, `claude -p`, or similar, treat them as stale/invalid process artifacts. They are not permission to delegate.


# OWNER OVERRIDE — NO HOMEGROWN MODEL TRAINING / WEIGHT TUNING

Do **not** train, finetune, distill, interpolate, blend, mutate, or otherwise create custom MTP/draft/model weights for this NInfer project. Do not launch more local native-MTP CE/on-policy/TTT/D-PACE training, homemade DFlash/DFlare/AngelSpec training, LoRA/adapters, or checkpoint interpolation. Prior local training experiments remain evidence only; they are not the forward trajectory.

Allowed model-side inputs are: (1) the model's original factory/native MTP weights, (2) upstream/published professionally trained speculative checkpoints such as exact-target DFlash2, and (3) mechanical format conversion/quantization/export that does not retrain model behavior.

The optimization trajectory is inference/system-side: exploit native MTP more effectively, scheduling/depth/topology, KV/cache/state reuse, verification/target kernels, training-free draft-KV/windowing or vocabulary methods, multi-GPU execution, and integration of upstream trained drafters. If a paper's main gain requires training new weights, treat it as research context only unless an upstream ready-made checkpoint matching Qwen3.8 exists.
