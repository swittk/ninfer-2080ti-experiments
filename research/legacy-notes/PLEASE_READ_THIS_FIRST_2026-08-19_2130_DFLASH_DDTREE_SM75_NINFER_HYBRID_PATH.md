# RESUME NOW — benchmark DFlash/DDTree on SM75, then graft it into NInfer

Read current `progress.md` first. Last local result: 21:07 low-rank proposal-logit corrections CLOSED. Current trusted populated-65K frontier remains ~59.05 useful tok/s; single Core-X exact anchor ~71.8-72.1 tok/s.

## Research correction

There is already a consumer CUDA DFlash+DDTree implementation that explicitly supports RTX 2080 Ti / sm_75: `Luce-Org/lucebox` (`server/`). Primary refs: https://github.com/Luce-Org/lucebox ; DFlash arXiv:2602.06036 ; DDTree arXiv:2604.12989.

Published Luce same-hardware result, single RTX 2080 Ti 22GB, CUDA 12.4, Qwen3.5-27B Q4_K_M + matching DFlash draft:
- AR HumanEval mean 19.88 tok/s;
- DFlash+DDTree HumanEval 53.42 tok/s, acceptance length 8.14;
- Math500 49.01 tok/s, AL 7.30;
- GSM8K 43.55 tok/s, AL 6.53;
- one HumanEval prompt 70.70 tok/s at AL 10.67;
- build uses `-DCMAKE_CUDA_ARCHITECTURES=75`;
- draft BF16 is converted to FP16 on Turing for tensor-core speed.

Single RTX3090 Luce reports ~129.5 HumanEval mean at AL 8.31; community/demo peaks can exceed 200 on easier prompts. Do NOT compare headline peaks to our hard gate.

**Do not abandon NInfer for llama/vLLM by assumption.** Luce's published SM75 DFlash mean (~53) is below our NInfer ~72 exact anchor (different model/workload, so not an exact A/B). NInfer has real SM75 target-engine strengths. Luce proves the missing algorithmic piece — block drafting + tree verify with AL ~6-10 — is feasible on Turing.

## Do this first: Luce SM75 sidecar benchmark

Use a separate scratch checkout; do not contaminate NInfer source.
1. Clone/pin current Lucebox with submodules.
2. Build only DFlash server/test path for sm_75.
3. Use its documented Qwen3.5/3.6-27B target + matching DFlash draft.
4. Run target-only and DFlash+DDTree on healthy Core-X.
5. Capture AR TPS, DFlash TPS, AL, draft ms/block, tree-verify ms, memory, clocks; sweep DDTree budget 12/16/22 if feasible.
6. This is a hardware/runtime calibration, not our final Qwen3.8 result.

Bound effort. If Luce runs, keep it as executable oracle/reference. If it doesn't, mine its implementation and return to NInfer; do not spend a day fixing their server.

## NInfer should keep the target engine; port the algorithm

NInfer already owns tuned SM75 Q4/Q5/GDN paths, exact B2 machinery, physical sequence-sharded INT8 KV, deduplicated T5 TP, 262K memory architecture, deterministic replay/fold, and the ~72 exact single anchor.

Luce has complementary machinery we should study/port semantically:
- target hidden-feature taps -> DFlash KV injection;
- block-16 parallel drafting;
- DDTree best-first tree;
- `ggml_ssm_conv_tree`;
- `ggml_gated_delta_net_tree` / `_tree_persist`;
- fast rollback and target-feature compaction;
- FP16 tree intermediates;
- GPU draft top-K and verify argmax;
- Turing BF16-draft -> FP16 conversion.

Treat Luce as an implementation mine, not the replacement runtime.

## Qwen3.8 drafter compatibility gate

No public Qwen3.8-27B DFlash was found in current Z-Lab model list. Public `z-lab/Qwen3.6-27B-DFlash` exists.

Our Qwen3.8 target: hidden 5120, 64 layers, vocab 248320. Public Qwen3.6-27B DFlash is dimensionally aligned: hidden 5120, 64 target layers, vocab 248320, block 16, target taps `[1,16,31,46,61]`, 5 draft layers.

Cheap discriminator before training:
1. Capture Qwen3.8 hidden states at 1/16/31/46/61 on general/code/JSON/Thai/Japanese/Arabic.
2. Feed them through public Qwen3.6 DFlash without runtime integration.
3. Measure block token accuracy, longest accepted prefix, and DDTree/oracle useful yield.
4. If useful, fine-tune from it. If weak, do NOT close DFlash; representation drift between 3.6 and 3.8 is expected.

## Training scale: stop treating the 1.7K adapter failures as evidence against a real drafter

The 1,719-record hidden/LoRA experiments validly close those tiny adapters only. DFlare (arXiv:2606.02091) scales DFlash training from ~800K to ~2.4M samples and adds layer-specific target-feature fusion. Spec-AUF (arXiv:2607.01893) optimizes the accept-until-first-failure prefix directly.

If Qwen3.8 needs training, build a real multilingual/code/JSON/reasoning target-trace dataset at hundreds-of-thousands+ anchors, preferably initialized from the closest public DFlash checkpoint.

## Tree verification is mandatory for 262K >=100

Our own envelope says ~8.85 useful tokens/round are needed at T5-like 262K cost with ~4ms draft. Fixed T4/T5/T6 cannot reach this even with perfect drafts.

The current-MTP tree oracle being CLOSED does NOT close a new proposal distribution. DFlash+DDTree changes the proposals. Also inspect:
- JetSpec arXiv:2606.18394: one causal-parallel head emits a conditioned tree in one pass;
- DSpark arXiv:2607.05147: lightweight sequential correction + confidence-scheduled verify;
- Bole/SpecLA-style compact recurrent update factors for hybrid/GDN tree verification.

Do not build another tree from current MTP logits. For a new DFlash/JetSpec distribution, require exact target selection, compact GDN state handling, and >=6 useful/round at 65K / roughly >=9 at 262K before hard-goal claims.

## `DFlash2` terminology

No single canonical primary project named `DFlash2` was found. Recent references usually mean DFlash + SGLang Spec V2, DFlash+DDTree, or later extensions (DFlare/DSpark/D^2SD), plus consumer forks like Lucebox. Spec V2 mostly attacks host overhead; our scheduler ceiling already says host cleanup alone cannot bridge 59->100. The block drafter + packed/tree verifier is the major change.

## Secondary single-GPU lane

Audit DeltaLog (arXiv:2608.15533): deferred dense recurrent-state materialization via compact update logs. ReplaySSM is not automatically equivalent. Only build a K=2/4/8 discriminator if current normal GDN decode really writes dense state each token and the E2E projection is >=5%.

## Order
1. Lucebox SM75 sidecar benchmark NOW.
2. Inspect Luce DFlash/DDTree/tree-GDN seam; map to NInfer.
3. Qwen3.6-DFlash -> Qwen3.8 hidden-feature transfer oracle.
4. If promising, prototype NInfer DFlash feature/KV-injection interface; if weak, plan real fine-tune, not another tiny LoRA.
5. Packed DDTree/JetSpec-class verification + compact GDN state handling.
6. DeltaLog audit for single >=80.
7. Keep six-family held-outs and long-context gates; never promote one easy code prompt.

Hard goals remain single >=80, dual one-request >=100, populated ~262K dual one-request >=100. Do not stop after another closure; run the Luce SM75 discriminator first.
