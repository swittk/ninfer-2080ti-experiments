# SUPERSEDED AS PRIMARY TRAJECTORY

**Do not treat DFlash2 as the default next architecture.** It is a learned drafter trained for the pristine `Qwen/Qwen3.8-27B` target and has no published acceptance evidence on abliterated/fine-tuned Qwen3.8 variants. Read `PLEASE_READ_THIS_FIRST_2026-08-23_0050_CHECKPOINT_ROBUST_DRAFTERS.md` first. DFlash2 is now only an optional cheap compatibility probe on an explicitly supported target checkpoint.

# 00:40 adviser — single80 is certified; close D4 and evaluate newly released Qwen3.8 DFlash2

Read repo-root `AGENTS.md` first. Stock GPUs only. No hardware tuning/reset/reboot. No coding agents. No local behavioral training. Hard remaining gates are dual-short >=100, populated65K >=100, genuine populated262K >=100 useful complete-path tok/s.

## Current authority / stop point
At ~00:35 Aug 23 both GPUs are idle and no NInfer/build/research producer is live. Latest substantive work ended ~22:37 in `p0_bo_current_d4` with one raw single and one raw dual run and no conclusion.

**Single80 is DONE and frozen.** P0-M + qualified combined-B1 + direct indexed exact rerank passed 12 stock complete-path reps: median **80.185**, mean 80.153, min 79.84, max 80.48, deterministic same digest/trajectory. Do not polish single further.

Current dual-short authority is canonical factory MTP + promoted indexed target: **65.910 t/s** median (65.80-65.99), 47 rounds / 141 drafted / 78 accepted / 34,26,18, deterministic. The fully promoted P0-M+B1 package was worse on dual (63.095), so dual should use the canonical indexed frontier unless a new mechanism supersedes it. Populated65K remains **58.75** and genuine262K **32.76** until recertified otherwise.

## First close P0-BO current D4
Raw promoted-code D4 discriminator is bad:
- single D4: **68.16 t/s**, 39 rounds, 154 drafted, 87 accepted, 33/25/17/12;
- dual D4: **49.01 t/s**, 49 rounds, 196 drafted, 77 accepted, 30/20/16/11.
Deeper width raises useful/round but verifier/draft economics lose badly. Write the conclusion and close; no D4 tuning sweep. P0-BN already proved splitting T4 into T2 microbatches restreams weights and is worse.

## NEW P0-BP — official Qwen3.8-27B DFlash2 checkpoint (released Aug 2026)
This is materially new relative to the earlier DSpark / old-DFlash / PipeInfer closures. Inco AI + z-lab have now published an exact-target **Qwen3.8-27B-DFlash2** draft model for `Qwen/Qwen3.8-27B`, plus published GGUF quantizations. It is professionally published upstream and therefore allowed under the owner's checkpoint rule; do NOT train/finetune anything locally.

Authorities:
- https://huggingface.co/incoai/Qwen3.8-27B-DFlash2
- mirror: https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
- published quantized drafts: https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF
- implementation: https://github.com/z-lab/dflash
- llama.cpp DFlash2 support is PR #27342.

Published architecture/config facts:
- block-diffusion, one-pass block drafting; verify block size 8 / 7 speculative tokens;
- 5 local sliding-attention layers, hidden 5120, intermediate 17408;
- target auxiliary feature taps at layers **5, 19, 33, 47, 61**;
- sliding window 2048; selector top-k16, rank256;
- BF16 checkpoint ~2B params; published GGUF sizes: BF16 ~3.86GB, Q8_0 ~2.06GB, **Q4_K_M ~1.14GB**.

Published Qwen3.8 acceptance is materially higher than MTP at the same 7-draft width: GSM8K MTP 5.02 vs DFlash2 5.46 useful/verification step; MATH-500 4.72 vs 5.28; HumanEval 3.91 vs 4.39; MBPP 3.99 vs 4.79; MT-Bench 3.74 vs 4.10. Published single-stream H200 throughput also materially beats MTP. The published Q4_K_M draft reports **5.39 acceptance length** on the GGUF GSM8K evaluation, so mechanical low-bit deployment does not obviously destroy draft quality.

### IMPORTANT: oracle before port
Do not blindly enable the existing DFlash flag. Qwen27 currently has `DFlashConfig::supported=false`, `maximum_dflash_draft_tokens=0`; the family runtime's existing DFlash path targets the older contract. DFlash2 has a different block drafter + selector.

1. Download/use the published upstream checkpoint only (prefer Q4_K_M for first cost screen; BF16 only as correctness reference). This is allowed published weight use, not local training.
2. Add a diagnostic-only NInfer feature tap for the **current promoted target** at layers 5/19/33/47/61 on six representative families. Capture enough chronological features/positions for the official DFlash2 reference to draft the next 7-token block. Source restore afterward.
3. Run the official DFlash2 reference offline on those captured NInfer features and measure actual draft acceptance against NInfer target decisions. Use chronological state only. This directly answers whether published alignment survives our target quant/arithmetic path.
4. Separately benchmark one published Q4_K_M DFlash2 draft pass on a stock TU102. Do not infer cost from H200. Record loaded clock and block latency. If easiest, use llama.cpp PR #27342 as the executable cost oracle before any NInfer port.
5. Placement oracle: the 1.14GB Q4 draft fits comfortably on either 22.5GB card. Prefer the dynamically faster role. With split32, target taps 5/19 are stage0 and 33/47/61 stage1. If draft lives on stage0/faster role, only the three late 5120-wide BF16 feature rows need cross-host transport per new position (~30KiB/position); if on stage1, only two early rows (~20KiB/position). Measure, do not assume. Keep DFlash2's 2048-window draft KV/features resident on the draft GPU.
6. Measure/derive the **actual NInfer T8 target verification cost** on the current canonical indexed dual-short path. Then compute complete round cost = target stage0 + stage1/head + DFlash2 block draft + feature transport + exact verify/fold/control. Use measured NInfer acceptance from step 3.
7. Advance to a minimal runtime only if the costed six-family dual-short projection can clear 100 with margin. If it cannot, close before porting.
8. If dual-short passes, immediately cost it at populated65K and genuine262K. DFlash2 draft attention is sliding-window 2048, so draft cost stays bounded, but the exact target still pays long-context GQA. Do not assume DFlash2 solves the long gates; measure T8 long target cost first.

### DFlash2 and PipeInfer
Do NOT automatically reopen old PipeInfer. DFlash2 consumes late target taps including layer61, so it is not a free independent proposer available while the same target block is still in stage1. First test ordinary serial DFlash2 economics. Only analyze overlap later if a concrete schedule exists that uses already-available previous-round features without future leakage.

## Research lanes NOT to chase now
- Speculative Pipeline Decoding / PPSD (May 2026) is structurally interesting, but the published speculation heads are trained checkpoints for Qwen3.5-4B/9B, not Qwen3.8-27B. Local training is forbidden, so it is not an actionable exact-target lane now.
- PipeSpec requires a hierarchy of aligned draft models; no new exact Qwen3.8 hierarchy is published.
- Existing PipeInfer with factory MTP and multi-branch successor PipeInfer were already costed below dual100. Reopen only if DFlash2 or another newly published proposer changes the measured dependency/yield economics.
- P0-BI accept-all/periodic anchors produced 98-110 t/s but failed representative quality catastrophically; do not use favorable-prompt speed as evidence. The later top-k acceptance gates at ~92-98 t/s are speed ceilings, not a quality-qualified frontier, and still miss 100.

## Priority / autonomy
1. Write P0-BO D4 conclusion (closed).
2. Start **P0-BP DFlash2** with capture/alignment + stock draft-cost oracle.
3. If costed dual-short ceiling >=100 with representative acceptance, implement minimal clean Qwen3.8 DFlash2 path and certify.
4. If DFlash2 cannot clear dual100, close it and research another materially distinct dual dependency architecture; do not reopen closed accept-all/tree/lookup families.
5. Once dual100 is real, continue immediately to 65K100 and 262K100.
