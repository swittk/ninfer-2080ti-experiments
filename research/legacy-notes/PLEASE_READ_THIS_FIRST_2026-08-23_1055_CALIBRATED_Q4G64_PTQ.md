# 10:55 adviser — external/proposal branches exhausted; attack Q5 target roles with calibrated Q4G64 PTQ

Read `AGENTS.md` first. Stock GPUs only. No hardware tuning/reset/reboot. No coding agents. No local behavioral training. Mechanical PTQ/quantization/export is allowed; do not finetune model behavior, train adapters, learn a drafter, or optimize full-precision target weights.

## STOP STATE / scoreboard
At 10:47 Aug 23 both GPUs are P8/300MHz idle, no NInfer/build/research producer. Last substantive result is P0-CY at 09:40 and progress at 09:42. Hard authority remains: single **80.185 certified/frozen**, dual short **65.910**, populated65K **58.75**, genuine262K **32.76**.

Recent closures: VPTQ M=1 gate/up alone 2.724ms (slower than entire P0-M proposal body 1.619ms); q-distribution/rejection sampling, qdist-PipeInfer, ASD-regret/suffix reuse, sensitivity-pruned self-draft, verifier-keyed/Oilbird semantic pool all below dual100 even under favorable/free costs. Do not reopen them.

## Critical distinction: old BF16 Q5->Q4 did NOT use calibrated PTQ
The Aug-18 BF16-source q4down canary is real and must not be ignored. But inspect its actual encoder: `.chatgpt/int8_tu102/bf16_recipe/audit_quant_sensitivity.py` uses canonical `quantize_matrix(... Q4G64_F16S)` plus a fixed 12-point per-group maxabs clipping sweep (1.0..0.70). It does NOT use GPTQ/Hessian reconstruction, AWQ activation-aware scaling, AutoRound optimized rounding, OmniQuant-like transforms, or QuaRot R4. The conclusion only closes plain/clipped BF16->Q4G64 subset enumeration.

Current public Qwen3.8 evidence is materially stronger: multiple fresh W4A16 AutoRound/GPTQ checkpoints quantize most text linears successfully, including uncensored/abliterated Qwen3.8 variants. Common sensitive exceptions are `linear_attn.in_proj_a/b`, embeddings/lm_head, and sometimes MTP; that is consistent with leaving the hybrid-GDN-sensitive pieces high precision rather than proving all Q5 roles require Q5. Sources to read:
- `Avuja/Qwen3.8-27B-int4-AutoRound` (AutoRound 0.14.2 W4A16; most text linears W4, `linear_attn.in_proj_a/b` + MTP kept BF16);
- `johannrplaster/Qwen3.8-27B-Uncensored-int4-AutoRound` (abliterated/uncensored target, W4A16, sensitive GDN projections preserved);
- `kernelogic/Qwen3.8-27B-Uncensored-GPTQ-Int4-sym-G128-MTP-BF16`;
- Intel AutoRound rotation docs; QuaRot fixed-Hadamard reference.

## P0-CZ — calibrated Q4G64 target PTQ, SAME runtime format first
Goal: move selected current Q5 target roles to existing NInfer Q4G64 without changing the hot kernel or target architecture. This is higher leverage than another schedule sweep because Q5->Q4 removes ~19% persistent/streamed bits for converted matrices.

### Phase A: offline reconstruction discriminator — NO artifact/runtime yet
1. Reuse the complete local BF16 18-shard source and captured real activations from the existing bf16_recipe / teacher traces. Do not redownload or use qualification prompts for calibration. Use a disjoint generic corpus if more activation rows are needed.
2. Start with **MLP down only**. Preserve all GDN/attention Q5 roles at current precision. Compare on exactly the old q4down candidate layers so the delta is encoder quality, not layer selection.
3. Produce Q4G64 codes + FP16 G64 scales with three bounded encoders:
   A. old best clipped RTN authority;
   B. GPTQ-style second-order/error-compensated Q4G64, `desc_act=false`, no asymmetric zero point;
   C. AWQ/activation-weighted scale search or AutoRound-style rounding constrained to the same symmetric Q4G64 storage.
   These are quantization algorithms only. Do not update BF16 model weights, train rotations, finetune, or use task labels.
4. Score actual layer-output reconstruction on captured post-SwiGLU activations, not only weight L2. Report relative output L2/cosine and target-head/logit effect for the same held-out rows used by the prior canary.
5. Admission: calibrated Q4 must reduce the old clipped-Q4 excess output error by at least **2x** on the candidate layers and approach current Q5 output error closely enough that a one-layer teacher-forced canary is plausible. If not, close calibrated Q4G64 before building an artifact.

### Phase B: one/two-layer canary, then bounded subset
6. If Phase A passes, build exactly one best and one second-best calibrated Q4 down layer into the existing NInfer Q4G64 artifact format. No new CUDA kernel. Run the same multilingual canary set used by Aug-18.
7. Only if both canaries are behaviorally sound, greedily form at most two bounded subsets (e.g. best 4 and best 8 layers by held-out output error). Do NOT repeat the old 2/4/8 enumeration using the same encoder; layer ranking must come from calibrated reconstruction.
8. Measure real stock complete-path target/MTP behavior. Changed deterministic target wording is allowed under the current behavioral contract, but representative quality, script/language integrity, state correctness and run-to-run determinism are mandatory.
9. If a subset survives, measure target-stage wall directly. Current populated65K MLP Q5-down bucket is ~10ms/round across both roles; even ~15-20% reduction here is material and combines with other long-context work.

## P0-DA — fixed QuaRot R4 only if calibrated Q4G64 still misses quality but is close
Do NOT use SpinQuant learned rotations; that requires optimization/training of rotations. QuaRot's fixed Hadamard is function-preserving and needs no behavioral training.

For MLP down, apply the local R4 transform only: offline rotate `down_proj` input dimension with an exact block Hadamard and apply the corresponding online Hadamard to the post-SwiGLU activation before quantization/down projection. SpinQuant's public code uses exactly this `apply_exact_had_to_linear(... down_proj, output=False)` pattern and an online full Hadamard for `down_proj`.

First do a pure numerical oracle: rotate BF16 weights/held-out activations, confirm BF16 output invariance, then quantize the rotated down weights to Q4G64 and compare output error against Phase-A GPTQ/AWQ. Qwen intermediate 17408 is non-power-of-two; use only a mathematically valid factored/block Hadamard (QuaRot `get_hadK` style) and verify exact invertibility before quantization.

Only if rotated Q4 materially improves held-out output error should you benchmark an SM75 online-Hadamard kernel. Total transform cost across 64 layers must fit comfortably inside the Q5->Q4 byte saving; if the transform consumes the saving, close. Prefer fusing the transform into the existing SwiGLU/A8 epilogue only after the isolated cost oracle passes.

## What NOT to do
- Do not reopen plain q4down2/4/8, global W2/W3, BitBLAS, VPTQ, CodeGEMM, TEAL/shared-G64, external drafters, or old vLLM-2080Ti archaeology.
- Do not run AutoRound/SpinQuant as an unconstrained full-model training-like campaign. Quantization-only code/scale/rounding optimization is allowed; no learned rotation parameters or model-weight finetuning.
- Do not change NInfer storage/kernel format in Phase A/B. The point is to test whether better PTQ can exploit the already-fast Q4G64 operator.
- Do not reset the dirty promoted tree.

## Priority
1. Record/close any unfinished CM/shared-G64 bookkeeping if needed.
2. P0-CZ Phase A: old clipped RTN vs calibrated GPTQ/AWQ/AutoRound-style **Q4G64** on existing BF16 MLP-down candidates, scored on real held-out activations.
3. If the reconstruction gate passes, one/two-layer artifact canary then at most two calibrated subsets, stock E2E.
4. Only if calibrated Q4 is close but not enough: P0-DA fixed QuaRot R4 numerical oracle, then online-transform cost.
5. Recompute dual/65K/262K ceilings after any accepted target byte reduction. Continue autonomously.
## Mandatory multimodal/vision preservation while doing PTQ

The optimized Qwen3.8 artifact still contains the full Vision path: **333 `vision/*` objects**, about
**0.275 GiB persistent weights**. The Vision backbone is 27 layers (hidden1152/intermediate4304) and
merges to the text hidden width 5120. Current formats already quantize Vision linears separately
(Q4/Q5 plus BF16 norms/biases); P0-CZ must not touch those weights.

The current single-device runtime already supports `--vision` and has a dedicated
`mtp_bridge_multimodal`; MTP+Vision is supported. The current **dual text-shard loader explicitly
rejects `features.vision`**, so the present 65.910/58.75/32.76 dual path is operationally text-only.
That is a missing dual integration, not the final product contract.

### PTQ calibration/qualification rule
Because visual embeddings then traverse the same 64 text layers, any Q5->Q4 target-backbone change
must be scored on both text and multimodal hidden distributions. For P0-CZ Phase A, augment the
disjoint calibration/reconstruction capture with a small multimodal corpus (images + at least one
video) and report MLP-down output error separately for text-origin and vision-origin/vision-conditioned
columns. A candidate that wins text reconstruction but materially regresses multimodal reconstruction
does not advance. Do not use the held-out multimodal qualification prompts for code/scale selection.

Before any target PTQ subset is promoted, add a bounded multimodal regression set covering at least:
OCR/document/screenshot reading; chart or diagram reasoning; natural-image spatial/counting/reasoning;
multi-image comparison; short video/temporal reasoning. Also run a long-context multimodal retention
case with media near the beginning and a query near the end. Standard public sets such as DocVQA,
ChartQA, OCRBench, MMMU and Video-MME are suitable sources, but a small fixed local subset is enough
for the gate if it is representative and not used for quantizer fitting.

### Dual/full-context architecture direction
Do not shard or duplicate the Vision tower just because the text model is layer-sharded. The clean
first implementation is:
1. Materialize the ~0.275-GiB Vision tower/merger on **logical stage0** (the stage that owns text
   layer0), not on a permanently named physical GPU. This avoids a backwards cross-device transfer
   of 5120-wide merged visual embeddings. Keep normal runtime role calibration.
2. Reuse the existing `VisionPrefillSession`/scatter semantics. Vision encode is prefill-only; inject
   the merged BF16 [5120,V] embeddings into the same chunked stage0 text-prefill path, then pass the
   ordinary hidden boundary to stage1 exactly as for text. Do not replace vision placeholders with
   token embeddings.
3. Reuse `mtp_bridge_multimodal` semantics at reusable-prefix/suffix boundaries. Prefix reuse must be
   keyed by the actual media/preprocessed-vision identity/scatter metadata; never reuse a text/vision
   frontier across different media.
4. Keep only one persistent Vision copy. The current frontend permits up to 32,768 merged vision
   tokens; their BF16 merged output would be ~320 MiB if fully materialized, so preserve streaming/
   release behavior rather than pinning all media outputs forever. The existing runtime already
   releases consumed media payloads. Account Vision workspace/transients explicitly in the dual
   planner and CUDA-graph reservation.
5. At 262K the existing physical run is roughly stage0 weights/state/KV 8.80/2.97/2.80 GiB and
   stage1 9.13/2.08/1.95 GiB before other reservations. The Vision weights are small enough that a
   single stage0 copy is plausible, but prove startup headroom with the planner; do not assume or
   duplicate allocations.
6. The 65K/262K performance gate is on **total populated context including vision tokens**. Vision
   encode/prefill latency is a separate TTFT metric. After prefill, useful decode must still target
   >=100 tok/s; do not waive the hard gate for multimodal requests.

Do not interrupt P0-CZ merely to implement dual Vision now. But design every promoted target format
and memory-plan change so this integration remains possible, and do not call the final full-context
dual engine complete while `features.vision` is rejected.

