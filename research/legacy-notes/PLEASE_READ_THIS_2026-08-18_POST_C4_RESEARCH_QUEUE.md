# PLEASE READ — POST-C4 RESEARCH QUEUE: ADAPTIVE SPECULATION + TU102 CHECKPOINT/KV CO-DESIGN

Owner/adviser research note, 2026-08-18. Do NOT interrupt a coherent C4 qualification/commit in flight. Finish that first. This note is a prioritized queue for the still-open single-GPU >=80 and dual single-user >=100 + populated long-context goals, plus robust multi-user throughput on low-acceptance workloads.

## Current evidence to preserve
- C4 cross-round preload demonstrated ~120.86/120.88 aggregate tok/s at identical 49.33% acceptance/output versus 92.80 before preload: a real scheduling gain.
- Scalar 2-session fast path remains ~107.3 aggregate at 55.32% acceptance.
- Distinct C4 prompts now expose extreme acceptance spread (roughly 65%, 37%, 27%, 9%); fixed MTP3 therefore cannot be the universal optimum.
- `acc9b4b` / `9138570` fast A8 path remains the production baseline. Do not regress it.

## 1. Highest-priority serving idea: adaptive speculative width / bypass
Primary evidence:
- Su et al., “The Synergy of Speculative Decoding and Batching in Serving Large Language Models”, arXiv:2310.18813. Optimal speculation length changes with batch size.
- Kong et al., “An Interpretable Latency Model for Speculative Decoding in LLM Serving”, arXiv:2605.15051. Serving speed depends jointly on effective batch, acceptance, draft length and verifier/drafter costs; acceptance alone is not enough.
- Zhang et al., “Learning to Draft”, arXiv:2603.01639. Directly optimize draft/verify cycle throughput rather than static draft budget/proxy acceptance.

Actionable bounded experiment with the EXISTING MTP checkpoint: maintain a per-session cheap online estimate (recent accepted-by-position / confidence if already available) and choose among no-spec / width1 / width2 / width3 at round boundaries. Do not cherry-pick prompts. Compare fixed representative corpus and C1/C2/C4. A very low-acceptance Thai-like session should not be forced to pay MTP3 forever. Report engine cost and algorithmic acceptance separately.

## 2. Dual single-user >=100: asynchronous/pipelined speculation rather than ordinary PP
Primary evidence:
- Butler et al., “PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation”, arXiv:2407.11798. Continuous asynchronous speculation + early cancellation is specifically intended to improve single-request pipeline utilization, low acceptance tolerance, and low-bandwidth interconnects.
- EAGLE, arXiv:2401.15077: second-to-top-layer feature prediction can be a much better speculation representation than token-only prediction.
- EAGLE-2, arXiv:2406.16858: context-aware dynamic draft trees improve lossless speculative throughput.

Do not immediately train a new model. First map PipeInfer-style overlap onto our existing sharded MTP state machine: can GPU0 start useful next-round/proposal work while GPU1 verifies the current round, with cancellation/rollback using the replay contracts we now trust? The existing C4 cross-round preload win is strong local evidence that this family of scheduling ideas maps to our hardware.

## 3. Long-context dual/single-user: reduce KV bytes, not only weight stalls
Primary evidence:
- KIVI, arXiv:2402.02750: asymmetric low-bit KV; keys and values prefer different quantization axes.
- KVQuant, arXiv:2401.18079: per-channel keys, pre-RoPE keys, non-uniform quantization and outlier handling; reports long-context speed/memory gains.
- KVTuner, arXiv:2502.04420: layer-wise mixed-precision KV, with Qwen-family sensitivity evidence and throughput gains over KV8.
- QServe, arXiv:2405.04532: W4A8KV4 system/quantization co-design and compute-aware reordering/register parallelism.
- NOVA-KV, arXiv:2608.04074: very recent attention-preserving transform/vector-quantization work; treat as research evidence, not yet mature production precedent.

Before writing kernels, run BF16/reference long-context quality calibration and offline KV sensitivity by attention layer. GDN recurrent state is NOT ordinary attention KV and must remain separate. A hardware-friendly KV4/mixed 4/8-bit plan could materially lower populated-128K/262K bandwidth and capacity pressure.

## 4. Single-GPU >=80: BF16-source checkpoint recipe optimization
Primary evidence:
- GPTQ, arXiv:2210.17323: second-order reconstruction-aware PTQ.
- AWQ, arXiv:2306.00978: activation-aware saliency; not all weights/channels are equally important.
- OmniQuant, arXiv:2308.13137: learned clipping/equivalent transforms for PTQ.
- APTQ, arXiv:2402.14866: attention-aware mixed-precision bit allocation using sensitivity.
- QuaRot, arXiv:2404.00456 and SpinQuant, arXiv:2405.16406: equivalent rotations/outlier suppression can make low-bit inference much easier.
- SqueezeLLM, arXiv:2306.07629: sensitivity-aware non-uniform quantization/outlier decomposition; sparse component may be undesirable on TU102, but sensitivity evidence is relevant.

Our current Q4G64/Q5G64 encoder is simple MAXABS+FP16 scale. Do a real BF16-source calibration campaign. First priority is SAME runtime formats: better clipping/code selection and layer/role-specific Q4/Q5/W8 assignment under a fixed quality/bpw budget. Only after quality evidence supports it should we add G32/new packing/rotation kernels. Never infer quality from greedy-output equality alone.

A particularly cheap offline candidate: Q4G32 costs 4.5 bpw (4 + 16/32), still below Q5G64 at 5.25 bpw. If Q4G32 approaches Q5G64 quality on selected roles, it may buy a useful byte reduction with simpler kernels than exotic codebooks.

## 5. Multi-user >100 should become robust, not just one acceptance regime
The C4 result already proves the scheduler can exceed 100 when target economics/acceptance cooperate. The next product question is not “can C4 ever reach 100?” but “can adaptive width/no-spec + exact-B microbatch scheduling keep representative aggregate throughput high across mixed acceptance?” Preserve isolated per-session outputs and compare against fixed corpus references.

## Qualification rules
- `PLEASE_READ_THIS_2026-08-17_2236_MTP_ACCEPTANCE_IS_NOT_ENGINE_SPEED_QUALIFICATION_RULE.md` remains authoritative.
- Separate target engine round-cost improvements from speculation/model-quality improvements.
- Checkpoint/KV experiments require BF16/reference quality evaluation, not output-digest coincidence.
- Long-context performance must be populated-context performance, not reservation-only.
- A paper result on A100/H100/L40S is methodological evidence, not a transferable speed claim for TU102.

## Suggested order after C4 commit
1. Adaptive no-spec/MTP1/MTP2/MTP3 policy on fixed corpus at C1/C2/C4.
2. Rebase/port proven A8 shared runtime work into the single-GPU >=80 lane and requalify the preserved ~72 checkpoint workload.
3. Build BF16-source sensitivity/quantization harness and search layer/role bit allocation using current Q4/Q5/W8 formats.
4. Profile populated 32K/64K/128K/near-262K; then test mixed/lower-bit attention KV only where it is actually dominant.
5. Bounded PipeInfer-style asynchronous proposal/verification overlap for dual single-user.

Do not treat any of these papers as permission to copy architecture-specific constants. Use them to generate TU102-specific experiments with exact correctness/quality gates.
