# RESUME NOW — native MTP is NOT exhausted; fix the quantized verifier wall too

Read current `progress.md` first. Live HEAD before this note: `5184d78`. Current hard-goal frontier remains 59.05 useful tok/s populated-65K; single exact anchor ~72 tok/s.

Keep the newly accepted tree substrate: parent-indexed conv/GDN, compact ReplaySSM selected-path fold, exact ancestor-only tree attention, and the AngelSpec systems smoke. But the real 48-layer sweep changed the bottleneck: Q4/Q5 target projections dominate width scaling, not recurrence/tree attention. N8/N12/N16/N22 complete GDN+fold = 29.4/49.3/58.7/94.2 ms; input projection alone = 18.5/32.8/35.8/64.7 ms.

## 1. Open a REAL native-Qwen MTP fine-tune lane now

vLLM Speculators now supports FastMTP-style fine-tuning for models that already ship MTP. This is NOT a separate DFlash model:
- native pretrained MTP weights are extracted from the verifier;
- only MTP layer(s) train;
- target embed_tokens/lm_head stay frozen/shared;
- recursive multi-step FastMTP loss is used;
- improved MTP is stitched back into the original verifier checkpoint for ordinary native-MTP serving.

Primary refs:
- https://github.com/vllm-project/speculators/releases
- https://github.com/vllm-project/speculators/blob/main/docs/user_guide/tutorials/train.md
- https://github.com/vllm-project/speculators/issues/267

Published evidence: Qwen3.5-9B 3-step acceptance 89.36/72.34/61.70%, AL 3.23; Qwen3-Next-80B mean accepted tokens 2.01->2.46 (+22.5%) after one epoch. Factory MTP is not necessarily the endpoint.

### Bounded Qwen3.8 native-MTP discriminator

1. Sidecar-audit Speculators `qwen3_5` MTP support against our Qwen3.8-27B config/native one-layer MTP. If only config/name plumbing is missing, patch the SIDE CAR, not NInfer.
2. Use offline hidden-state training; start with existing diverse traces.
3. Train actual native MTP with recursive 3/4/5-step loss, full verifier vocab, no token curation.
4. Stitch/export candidate weights and run an OFFLINE NInfer proposal oracle on general/code/JSON/Thai/Japanese/Arabic + long-context samples before runtime integration.
5. Report per-position acceptance, useful tokens/round and tree candidate mass versus factory MTP.

Gate: materially shift held-out proposal distribution, not merely fit train rows. If it does, the old current-MTP tree closure does NOT apply to the new distribution.

For single >=80, prioritize this at least as highly as DFlash: no separate ~12.6-ms block drafter and only ~11% E2E gain is needed from ~72.

## 2. Primary consumer evidence: native MTP + optimized target can exceed 100 on this dense hybrid family

Closest reproducible primary data is Qwen3.6-27B on 2x3090:
- W4A8 + built-in MTP n=5, full 262K: 73.3 narrative / **106.1 code tok/s**, warm accept ~5.0 / 80%; n=6/8 regress.
- matched AutoRound INT4 + INT8-PTH KV + native MTP n=4: **88.05 narrative / 120.17 code decode**, avg accept 75.3%, per-pos 0.94/0.83/0.69/0.55.
Refs: https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md and model README.

This proves DFlash is NOT required for >100 on the same dense Qwen3_5/DeltaNet family. A secondary site attributes Qwen3.8 2x3090 AutoRound 262K ~94-104 t/s to @Tech2Wild, but no primary artifact was located; do not use it as proof yet.

## 3. Our W4 tree wall matches SpecMQuant almost exactly

Paper/code: arXiv:2505.22179, https://github.com/AI9Stars/SpecMQuant

Their result: on 4-bit weight targets, larger speculative trees become compute-heavy; verification/decode ratio rises and destroys W4's memory-bandwidth advantage. Their solution is hierarchical speculation: a cheap intermediate verifier turns a rich tree into a short SEQUENCE, then the expensive quantized target verifies that sequence. W4A16 Llama-3-70B: 2.78x, 1.31x over EAGLE-2.

This is materially different from our closed factory-MTP tree and directly matches our projection-dominated N8/N12/N16/N22 curve.

Before implementation, build an ECONOMICS oracle from existing tree traces:
- assume a cheap intermediate scorer selects one likely root-to-leaf path from a richer DFlash/AngelSpec/native-MTP tree;
- determine final target T4/T5/T6 width, required intermediate accuracy and allowable cost for 80/100/120;
- explicitly test whether a FastMTP-tuned native head could be the intermediate scorer/filter.

Final target stays lossless; the intermediate only chooses proposals. If the oracle predicts >=100 at realistic accuracy/cost, prefer hierarchy over directly forcing N16/N22 through the W4 target.

## 4. If target still verifies many nodes: dedicated SM75 multi-column projection path

Profile N8/N12/N16/N22 first: exact Q4/Q5 kernel route, full weight scans, bytes read, occupancy, and whether nodes are one packed pass or effectively re-streamed.

Research principles, not drop-in code:
- Marlin keeps near-ideal W4 performance to M~16-32 with striped weight reuse (`IST-DASLab/marlin`), but it is SM80+, so do NOT port it literally to TU102.
- FlashDecoding++ (arXiv:2311.01282) identifies flat-GEMM dataflow and double buffering as major small-M inference issues.
- Genesis P67 is an ATTENTION K+1 multi-query kernel, not our projection fix, but proves spec-verify needs dedicated K+1 paths rather than generic fallbacks.

NInfer already proved this principle at T5: a shared single-pass T5 TP weight stream gave 1.53x isolated MLP speedup. Build a benchmark-only packed T8/T12/T16/T22 Q4/Q5 input/output projection route that reuses each resident weight tile across node columns. Avoid repeated full weight streams. Start on the real 48-layer GDN tree benchmark, then full verifier.

Gate: large movement only — target >=1.5x projection-bucket improvement at N8+ OR enough complete-round projection to put realistic AL~7-10 on a >=100 trajectory. Do not integrate another 3% local result.

## 5. vLLM/3090 lesson: copy mechanisms, not headlines

club-3090's working Qwen3.6 dense paths combine native MTP with target-engine details, not magic MTP alone. Their Genesis P67 is a proper K+1 multi-query verify kernel for compressed KV; their docs explicitly show broken/incorrect results without the correct multi-query path. Also note Qwen3.6 native MTP has a measured n-sweep knee: n=5 wins in one W4A8 config; n=6/8 regress. Do not assume wider is always better.

There is a secondary public attribution of Qwen3.8 2x3090 AutoRound 262K at ~94-104 tok/s, but I did not locate the original primary report. The primary same-family Qwen3.6 27B numbers above are enough to establish the mechanism is real.

## 6. DFlash/AngelSpec is now lane C, not the entire plan

AngelSpec systems feasibility is proven; keep it alive, but do not spend the next long block only generating a DFlash corpus while the target W4 verifier cannot exploit wide trees efficiently.

Compare three measured paths:
A. factory vs FastMTP-tuned native MTP distribution;
B. SpecMQuant-style rich-tree -> cheap-intermediate -> short-sequence target economics;
C. packed arbitrary-width target projection ceiling.
Only after these decide whether large-scale DFlash/DFlare training is worth it.

## Immediate execution order

1. Native FastMTP/Speculators Qwen3.8 compatibility + one-step/offline smoke NOW.
2. In parallel profile N8/N12/N16/N22 projection routes/bytes; implement packed multi-column discriminator if its ceiling is material.
3. Build SpecMQuant hierarchy economics oracle from existing tree traces; no runtime implementation until it predicts a real win.
4. Recompute single/65K/262K hard-goal envelopes from measured A/B/C results.
5. Only then scale AngelSpec training.

Hard gates remain single Core-X >=80, dual one-request >=100, populated 262K dual one-request >=100. No watcher/wait state; after each closure move to the next bounded discriminator.
