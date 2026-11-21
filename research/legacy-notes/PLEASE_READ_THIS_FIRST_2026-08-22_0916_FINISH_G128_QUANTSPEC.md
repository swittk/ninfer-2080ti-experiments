# 09:16 adviser — finish P0-AH first, then QuantSpec-style long-context ceiling

Read repo-root `AGENTS.md` first. Stock GPUs only. No card settings/reset/reboot. No agents. No behavioral training. Hard gates remain 80 / 100 / 100 / 100 useful complete-path tok/s.

## Current stop state

At ~09:14 both GPUs are idle and no NInfer/build/research producer is live. Latest substantive work ended ~08:02 in `p0_ah_g128_marlin_runtime` after source restoration.

Late closures before P0-AH:
- P0-AB certified sparse proposal: **0 / 1008** hidden states certify at segment 128/64/32/16. CLOSED.
- P0-AG target-B1 K256 previously got CLOSED only because Arabic changed digest. **CORRECTION:** digest inequality alone is not a failure under the NInfer fast-MTP/turbo contract. The Arabic baseline and candidate are both coherent, technically competent answers; they share the first 66/256 generated tokens and then take different reasonable outlines. Treat B1/K256 as a quality-qualified turbo candidate, not scientifically closed.

## P0-AH is UNFINISHED, not scientifically closed

This is the immediate priority.

The worker already produced a mechanical factory-MTP Q4G128 artifact and isolated SM75 Marlin timings for all five MTP projections:
- input 63.23 us
- attention 86.14 us
- attention-out 42.15 us
- gate/up 195.32 us
- down 100.43 us
- total one MTP pass ~0.4873 ms; three passes ~1.4618 ms
- isolated numeric max rel-L2 ~0.00237, min cosine ~0.9999972

Artifact quantizer: per-group128 six-factor MSE clipping with one FP16 scale duplicated into adjacent G64 scale slots. This is a proposal/draft approximation only; canonical target verification still owns final output.

The normal runtime integration failed for a trivial CUDA compile reason, not for performance or model quality:
`kScalePermutation` is a namespace `constexpr int[64]` referenced from device code and NVCC reports it undefined in device code.

Do NOT continue the prebuilt-object/CMake detour. The permutation is exactly:
`perm(lane) = (lane & 7) * 8 + (lane >> 3)`
for lane 0..63. Replace the array lookup in the temporary runtime source with that arithmetic expression (or equivalent device constexpr function), compile the `.cu` normally as part of `ninfer_ops`, and run the real E2E A/B.

Completion contract for P0-AH:
1. real stock fixed-prompt A/B, repeated;
2. preserve final canonical target output; proposal trajectory may differ;
3. if fixed result approaches/crosses 80, immediately run the six-family representative qualification before promotion;
4. if it loses acceptance enough to miss the gate, close with measured complete-path result;
5. restore experimental source on closure.

This lane is especially important because P0-M Q4G64 factory-MTP already proved a real exact-final-output improvement to 77.97 t/s. G128 has a credible chance to reduce draft-body cost further; do not abandon it because of a compile/link plumbing mistake.

## Next materially different long-context research: QuantSpec-style quantized self-drafter

Paper/code:
- ICML 2025 QuantSpec: https://proceedings.mlr.press/v267/tiwari25b.html
- code: https://github.com/SqueezeAILab/QuantSpec

Why this is NOT the already-closed VeriCache lane:
- VeriCache oracle made the same-model draft attention/KV free/cheap but left the same-model dense/GDN draft path intact; its asymptote was only 95.63 t/s at 262K.
- QuantSpec uses a same-architecture draft with **hierarchical low-bit KV AND optional 4-bit Marlin/GPTQ draft weights**, reporting >90% acceptance and up to ~2.5x E2E on its tested long-context systems.
- Therefore it can attack the dense draft floor that made the compressed-KV-only oracle fail.

Do NOT port QuantSpec first. Build a Qwen3.8 complete-system ceiling using our canonical INT8 target:
1. target verifier remains current canonical INT8/full-KV NInfer path;
2. draft gets mechanically low-bit weights only (no training) plus a 4-bit/hierarchical KV approximation;
3. use measured/isolated SM75 Q4/Q4G128/Humming/Marlin costs, not paper GPU numbers;
4. use actual six-family and genuine262K acceptance/agreement traces;
5. charge helper-GPU role at its measured loaded clock and real no-P2P transfer unless overlapped;
6. compute complete-path 65K and 262K ceilings before any runtime port.

If even an optimistic QuantSpec-style weight+KV draft cannot clear 100 at 262K, close. If it can, then implement a minimal drafter. This is allowed because the draft approximation does not define final target output and requires no behavioral training.

CAS-Spec (NeurIPS 2025) is only a secondary scheduler after useful draft tiers exist; do not start there. Its value would be dynamically choosing between draft fidelities/lengths, not rescuing intrinsically slow draft models.

## Do not recycle

Do not reopen **certified sparse proposal**, direct W4 vocab, Jacobi/Lookahead, FAFO, SubSpec, KVarN/TurboQuant/KV4, lossless KV compression, same-model compressed-KV-only VeriCache, or the failed Q4/Q5 shared-memory producer-consumer schedules. **B1/K256 is explicitly reopened for behavioral-quality qualification** under the new `AGENTS.md` fast-MTP/turbo output contract. Do not require byte-identical digest; require determinism, no implementation/state bug, no pathological output, representative quality preservation, and material performance.

## Autonomy

Finish P0-AH, then QuantSpec ceiling if needed. Continue autonomously after either classification; do not stop at another sub-lane report. Recompute stock 80 / 100 / 100 / 100 after every accepted mechanism.
