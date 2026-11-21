# PLEASE READ — single-GPU 80 is NOT exhausted; preserve/restore the qualified ~72 lane

Owner/adviser correction, 2026-08-17 18:18 +07. This is a **gentle status correction**, not a request to abandon the coherent dual/multi-session work currently in flight. Finish the current bounded dual change cleanly, but do not let `SINGLE_GPU_INT8_BYTE_REDUCTION_CONCLUSION.md` become a claim that single-GPU >=80 is physically or SOTA-methodologically exhausted.

## 1. Keep the real q45g correctness fix

Do **not** revert `38369da`. The stale/unwritten B2 partial-tail / CUDA-graph replay issue is a real state/data-validity bug and is a valid hard failure under the upstream-oriented correctness contract. This correction is not asking for the old broken replay behavior back.

Also continue to obey:
- `PLEASE_READ_THIS_2026-08-17_0830_STOP_CHASING_MTP_T1_BITWISE_PARITY.md`
- `PLEASE_READ_THIS_2026-08-17_0845_UPSTREAM_CORRECTNESS_HEURISTICS_NOT_BITWISE_RELIGION.md`

Do not reintroduce ordinary-T1/byte-identical output as a blanket gate. Use frozen operator numerical tolerances, state/indexing/rollback correctness, and first-fork margin/perturbation evidence.

## 2. Do not erase the known ~72 checkpoint-qualified lane

`perf/exact72-proposal-b2` / `9ca3a4b` is historical evidence, not something to silently replace with “65.3 is the only frontier.” Post-fix experiments also reached roughly 71.6-72.0 tok/s:
- B2 top31 + calibrated outlier coverage: ~71.60-71.77 tok/s on canonical 1024.
- W1/top56 + observed outlier coverage: ~71.86-72.01 tok/s on canonical 1024.

The rank audits did useful additional work: generic target-head B2/W1 shortlists are **not universally safe**; rare exact winners can have very poor proxy rank. That means the generic library path cannot simply hard-code corpus-derived token IDs and call itself universal. It does **not** mean the ~72 performance evidence should disappear.

Please keep two labels separate:
1. **portable/universal-safe baseline** (currently ~65.3), and
2. **this-exact-checkpoint qualified/turbo lane** (historically ~72, pending one clean post-`38369da` requalification under the upstream-oriented gate).

For this exact Qwen3.8 checkpoint, it is valid to have a calibrated/checkpoint-qualified mode if clearly identified as such. Requalify the fastest ~72 construction after the replay fix using a bounded diverse corpus plus exact/strict target-head diagnostics and first-fork/operator evidence. Do not reject it merely because a draft proposal differs: the proposal head is speculative and may be approximate; target verification/state correctness is the semantic authority. If the target-head shortlist itself causes an unexplained healthy-margin wrong winner, that remains a real failure.

## 3. “Single-GPU byte-reduction queue closed” is narrower than “single-GPU >=80 exhausted”

The recent work genuinely ruled out several **low-risk/current-format** ideas: block64 post-SwiGLU pruning, simple existing-artifact Q5->Q4/Q4->Q3, exact sparse residual representations, and exact S4 decomposition of A8. Preserve those negative results.

But important research classes were not completed and should remain on the single-GPU backlog rather than being declared exhausted:
- block32/block16 sparsity audit (explicit fallback in the 15:03/15:07 notes; no artifact found);
- **BF16-source calibrated PTQ**, rather than requantizing the already-quantized NInfer artifact (GPTQ/AWQ/OmniQuant-style second-order/activation-aware/clipping treatment);
- true calibrated **W4A4** / rotation or outlier-aware activation transformation. The exact S4-digits A8 experiment only disproves exact A8 emulation via S4; the existing naive A4 microbench is not a SOTA W4A4 quality test;
- lower-bit/vector/codebook or mixed-precision methods where actual stored bytes fall materially below Q4/Q5;
- architecture-aware treatment for the hybrid GDN layers instead of assuming Transformer-only transforms transfer unchanged;
- improved speculative-decoding economics (dynamic width/tree/draft policy or better draft-head calibration), since useful tokens per expensive target sweep remain a major lever.

No need to launch all of this immediately while the dual milestone is coherent. Just correct the project status: **single-GPU >=80 remains an open research target, not an exhausted/physically impossible one.**

## 4. Current dual work

The current heterogeneous-session / multi-session dual work is reasonable to continue. Keep the hard gates intact: one-GPU INT8 >=80 remains required eventually; dual performance mode >=100+ remains required. Do not let capacity/functionality success substitute for those performance gates.
