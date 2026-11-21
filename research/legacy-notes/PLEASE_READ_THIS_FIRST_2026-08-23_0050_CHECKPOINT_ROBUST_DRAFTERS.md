# 00:50 adviser correction — checkpoint robustness first; DFlash2 demoted, PARD/TLI/SLEM next

Read repo-root `AGENTS.md` first. Stock GPUs only. No hardware tuning/reset/reboot. No coding agents. No local behavioral training. Remaining hard gates: dual-short >=100, populated65K >=100, genuine populated262K >=100 useful complete-path tok/s.

## Owner correction: DFlash2 is checkpoint-specific

The product must not depend on pristine `Qwen/Qwen3.8-27B` behavior. Abliterated/fine-tuned/merged Qwen3.8 variants are legitimate target artifacts. A fast path that becomes useless when target weights are modified is not a general dual/long solution.

DFlash2 is target-checkpoint dependent:
- its published card names `Qwen/Qwen3.8-27B` as base/target;
- it consumes target hidden states at layers 5/19/33/47/61;
- DFlash training learns against those target representations;
- embedding/lm_head may come from the live target, but that does NOT realign the five learned draft layers to a modified hidden-state distribution.

Abliteration changes exactly those residual-stream representations: common Qwen3.8 ablations subtract a refusal direction from many residual-stream writer matrices. Correctness remains target-verified, but DFlash2 acceptance can fall until the drafter is pure overhead. There is currently NO published DFlash2 acceptance result for an abliterated Qwen3.8-27B.

Nuance: collapse is not guaranteed. One community abliterated BF16 Qwen3.8 report still measured native-MTP draft acceptance around 83.8%, showing low-rank target edits need not destroy every learned auxiliary predictor. But MTP != DFlash2 and this does not justify a DFlash2 port.

### Revised DFlash2 status
DFlash2 is OPTIONAL / CHECKPOINT-SPECIFIC ONLY. Do not implement it as the primary NInfer trajectory.

Only if trivial, do a compatibility probe on the actual modified target we care about: capture its five taps, run the official DFlash2 reference offline, measure six-family acceptance, and stop immediately if it lacks dual100 margin. Do not optimize DFlash2 kernels before this compatibility proof.

## First close P0-BO
Current D4 probe is already negative: single 68.16 t/s, dual 49.01 t/s. Write the conclusion and close. No D4 sweep.

## Do NOT reopen plain Qwen3.5-0.8B
This repo already tested the ordinary standalone Qwen3.5-0.8B Q4 learned drafter and CLOSED it. Best optimistic economics were roughly 41.56 single / 29.62 dual65K / 22.03 dual262K useful t/s. Its AR draft cost is too high. Do not resurrect it.

## P0-BQ — checkpoint-robust published PARALLEL drafter: AMD PARD
This is materially different from the closed 0.8B AR drafter.

AMD PARD is explicitly target-independent: one published PARD model is intended to accelerate a family of targets without consuming target hidden states or retraining per target. AMD publishes `amd/PARD-Qwen3-0.6B` (~0.8B parameters). Current vLLM supports it as a parallel drafter with `num_speculative_tokens=12` and `parallel_drafting=true`.

Sources:
- https://github.com/AMD-AGI/PARD
- https://huggingface.co/amd/PARD-Qwen3-0.6B
- current vendored vLLM docs: `.chatgpt/p0_prefill_20260821/upstream/vllm/docs/features/speculative_decoding/parallel_draft_model.md`

PARD proposes the block in parallel rather than paying one small-model forward per drafted token, so the old Qwen3.5-0.8B cost closure does NOT classify it.

### Vocabulary mismatch: solve losslessly, no training
`PARD-Qwen3-0.6B` uses Qwen3 vocab 151,936. Qwen3.8 uses 248,320. Do not modify/train PARD to Qwen3.8.

Use published heterogeneous-vocabulary speculative decoding:

**TLI (Token-Level Intersection)**
- token-string intersection + masked draft logits + ID map;
- target distribution remains exact/lossless;
- current vLLM supports `use_heterogeneous_vocab=true` for generic draft models.

**SLEM (String-Level Exact Match)**
- draft uses its full tokenizer;
- draft output string is retokenized with target tokenizer and verified as target tokens;
- also lossless;
- use when intersection is too small for TLI.

Sources: ICML-2025 heterogeneous-vocab SD paper https://arxiv.org/abs/2502.05202 ; vLLM TLI support; merged HF Transformers universal speculative-decoding reference; active vLLM SLEM integration.

### P0-BQ discriminator — oracle before NInfer code
1. Use only published `amd/PARD-Qwen3-0.6B`; no adaptation/training.
2. Compute exact tokenizer-string overlap with the actual Qwen3.8 tokenizer. TLI if overlap is high; otherwise SLEM.
3. Run published PARD on committed target prefixes for general/code/JSON/Thai/Japanese/Arabic and on the actual checkpoint variant class we intend to support (base + an abliterated/modified checkpoint if available).
4. Measure accepted length and full-survival for K=4/8/12. Target decisions remain NInfer-authoritative.
5. Benchmark one PARD parallel block pass on stock TU102; record latency and loaded clock. Use vLLM as the executable discriminator rather than porting PARD first.
6. Compute serial complete-system economics with current canonical-indexed target T widths. If serial misses 100, do not automatically close: PARD's main value is independence for pipeline overlap.

## P0-BR — reopen PipeInfer ONLY with measured PARD
Earlier PipeInfer work found a materially useful hypothetical:
- free independent correct proposer -> target pipeline oracle ~143.5 t/s;
- factory MTP could not realize it because target-dependent chaining/corrections caused bubbles;
- the old conclusion explicitly says to reopen for an owner-allowed independent proposer.

PARD is a newly published independent **parallel** proposer and does not consume the current target block's hidden-state taps. If P0-BQ shows nontrivial acceptance and low enough TU102 cost, feed the measured PARD numbers into the existing PipeInfer simulator before writing runtime code:
- allow PARD to use the stage0-GPU bubble while stage1 target work is in flight where dependency permits;
- model correction/cancellation from measured PARD full-survival, not a perfect oracle;
- charge real no-P2P boundary traffic;
- charge real PARD draft time on its actual GPU placement;
- use the TLI/SLEM-retokenized target width for verifier cost;
- compute dual 128/256 and then 65K/262K ceilings.

If measured PARD+PipeInfer <100, close cleanly. Do not locally tune PARD weights.

## Why this is more robust to abliterated/fine-tuned targets
PARD does not consume target hidden-state representations. A target edit can still reduce token agreement because behavior changes, but it does not make the drafter's learned layers interpret a shifted internal feature manifold. Target verification preserves correctness. This is the desired general-engine failure mode: lower acceptance on divergent checkpoints, not architectural incompatibility.

Target-independent does NOT mean universally high acceptance. Six-family + checkpoint-variant measurement is mandatory.

## Other target-robust research / don't recycle
- Plain Qwen3.5-0.8B external draft: already CLOSED on cost/quality in this repo.
- Token Recycling, LogitSpec, Lookahead, suffix lookup, active vocab, candidate cache: CLOSED by measured economics.
- DFlash/EAGLE/DSpark-style target-feature drafters: checkpoint-specific unless a matching published speculator exists for that exact target.
- FLy / loose acceptance is checkpoint-agnostic and training-free but lossy. We already observed aggressive accept-all/periodic gates wreck Thai/Arabic/code. Only reconsider a conservative FLy-style margin/deferred gate if a real drafter gets very close to 100 and the six-family quality gate survives; not the primary lane.
- PARD-2 target-dependent mode is not useful here unless running a published target-independent mode; no local adaptation/training.

## Priority / autonomy
1. Close P0-BO D4.
2. DFlash2: optional cheap compatibility probe on the actual modified checkpoint only; NO mainline port unless it unexpectedly proves portable.
3. Prioritize P0-BQ: published PARD-Qwen3-0.6B + TLI/SLEM acceptance and stock-TU102 block-cost oracle.
4. If viable, P0-BR: measured PARD-backed PipeInfer ceiling using existing machinery.
5. Port only if complete-system dual100 survives with margin and representative/checkpoint-variant acceptance is portable.
6. Then immediately attack 65K/262K. Continue autonomously; single80 is frozen.
