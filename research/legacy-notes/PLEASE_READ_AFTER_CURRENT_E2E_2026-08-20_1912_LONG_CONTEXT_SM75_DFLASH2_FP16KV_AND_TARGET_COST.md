# READ AFTER CURRENT E2E — long-context SM75 evidence changes the next queue

DO NOT INTERRUPT the currently running corrected selected-native-MTP E2E. Finish and classify it first. Hard gates remain single >=80, dual one-request >=100, populated ~262K dual one-request >=100.

## External SM75 reality check

`weicj/2080Ti-LLM-Toolbox` / `vLLM-2080Ti-Definitive` runs Qwen3.6-27B-AWQ + native MTP K=3 on dual modified 22GB RTX2080Ti **with NVLink**:
- 4K/TG128 peak-repeat median decode **101.3 t/s**;
- 64K/TG512 decode **55.3 t/s**;
- sequential serving MTP acceptance 80.2%, positions .927/.854/.782.
Refs: https://github.com/weicj/2080Ti-LLM-Toolbox/blob/main/reports/summaries/qwen36-27b-awq-vllm-mtp-k3.md and repo root.

Thus TU102 can cross100 at short context, while ~64K is a real wall even in a purpose-built SM75 vLLM stack. Our 65K59.05 is not uniquely broken.

Our topology is harder: `nvidia-smi topo -m` says GPU0<->GPU1 NODE, NVLink inactive; torch reports CUDA P2P false both directions. Do not assume NVLink TP behavior transfers.

## PRIORITY A — matched 65K BF16-vs-INT8 physical-KV A/B

The SAME public dual2080Ti stack shows Qwen3.6 GPTQ-INT4+MTP3 at PP65536/TG512:
- FP16/default KV **85.5 t/s**
- INT8 per-token-head **50.1**
- TQ K8V4 40.2
- TQ4NC 36.9
At4K, INT8 is faster (95.5 vs81.7), so this is context-dependent.
Ref: https://github.com/weicj/vLLM-2080Ti-Definitive/blob/vllm-2080ti-deifinitive/docs/qwen36-kv-throughput-sweep.zh-CN.md

We currently qualify long C1 with physical INT8 KVP. After current E2E do a strict matched NInfer discriminator:
- same artifact/prompt/snapshot provenance/depth/TP2/split/clocks;
- `--kv-dtype int8` vs `bf16`;
- attention/KVP ms, complete round ms, useful/round, TPS, VRAM, exact output/digest.

Do not assume vLLM transfers; local evidence decides. If BF16 fits65K and is materially faster, allow bifurcated policy: BF1665K speed route, INT8 262K capacity route. Inspect INT8 dequant/scales, page gather, K+1 verify shape and whether BF16 gets a better multi-query dataflow.

## PRIORITY B — exact Qwen3.8 DFlash2 now exists

Old Qwen3.6->3.8 zero-shot transfer remains CLOSED, but that closure is irrelevant to a separately trained exact-target drafter. Current exact-target identifiers include `incoai/Qwen3.8-27B-DFlash2` / GGUF.

A current 2x3090 vLLM report: 120.1 narrative /218.3 code t/s, 7 draft tokens, AL3.35, acceptance47.8%, patched P2P; context ceiling131K because of drafter memory. Ref: https://www.reddit.com/r/LocalLLaMA/comments/1vsccit/qwen3827b_on_2x_3090_vllm_dflash2_218_toks_single/

Current llama.cpp testers use `incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M`; n-max4 can beat7 on sustained long generation in some reports.

Run a SIDE-CAR exact-target discriminator on healthy Core-X before more custom external-drafter training:
- Q4_K_M drafter; no NInfer integration;
- n-max4 and7;
- general/code/JSON/Thai/Japanese/Arabic;
- 8K and, if feasible,65K;
- greedy and canonical sampled config separately;
- draft ms/block, emitted/accepted length, VRAM, sustained TPS.

Do NOT abandon native MTP. DFlash2 must beat trained native MTP in complete economics.

## PRIORITY C — target engine lessons from fast Qwen3.8 3090 stacks

Current ~138 t/s single3090 Qwen3.8 reports combine DFlash2 with FP16 recurrent state, INT8 activations, quantized head/MTP and split-KV verify attention. The lesson is full-stack target-cost reduction, not magic drafting.

### C1. FP16 recurrent state
Audit production ordinary GDN state dtype/traffic. If FP32, make opt-in FP16-state discriminator with FP32 accumulation where required. Qualify six corpora + long continuation. Pursue only if projected E2E >=5-8%; single72->80 needs ~11.1% total.

### C2. SM75 W4A8 / INT8 activations
QQQ arXiv:2406.09904 demonstrates high-ceiling W4A8 GEMM. Its kernels are not SM75 drop-ins; the transferable idea is Q4->INT8 unpack in registers + per-token activation quantization + TU102 `IMMA.8816` + fused scale handling.
Profile largest current Q4/Q5 target buckets and calculate ceiling first. Prototype only if complete target round could move >=8-10%; no more tiny geometry variants.

### C3. K+1 committed-prefix attention
Tree-tail attention is already negligible. The long-context committed-prefix scan is what matters. Use the BF16-vs-INT8 A/B to decide whether to reopen its dataflow and compare against dedicated multi-query verify approaches rather than generic decode/prefill paths.

## PRIORITY D — native MTP training: change objective/data if portability still fails

Current training really improves independent offline yield, but online short E2E still showed a material general-prose regression and long-context portability is under corrected qualification.

If selected candidate fails:
1. Do NOT just run far more identical CE steps.
2. Cheap factory<->trained weight interpolation alpha sweep offline first: find a zero-inference-cost Pareto point that recovers general prose while preserving code/JSON/multilingual gains.
3. Try acceptance/prefix-survival-aware position weighting inspired by D-PACE (arXiv:2605.18810), which dynamically emphasizes positions limiting expected accepted length rather than fixed positional decay.
4. Keep target-generated/on-policy data, disjoint roots, and model-specific proposal snapshots. Never reuse MTP KV/snapshots across different weights.

Speculators training docs explicitly require target-generated data for native MTP and recommend much more data for production quality. Optimize actual accepted prefix and E2E, not just loss.

## PRIORITY E — use public SM75 vLLM as an oracle, not a replacement

If practical after the above A/Bs, source-audit or sidecar-run `vLLM-2080Ti-Definitive`. Diff mechanisms that matter:
- AWQ Marlin flat GEMM;
- FlashQLA SM75 GDN;
- FlashInfer/FA2 attention;
- CUDA graph/async scheduling;
- native MTP K=3.

Their short101.3 proves our hardware class can cross100. Their long64K55.3 proves context is the larger wall. Their rig has NVLink; ours has no CUDA P2P, so communication conclusions need local validation.

## 262K reality

Do not lower the goal, but a drafter alone is not credible. Public dual2080Ti evidence establishes 262K capacity, not >=100 decode. Current exact-target DFlash2 dual3090 example reports a131K ceiling. Club-3090 native-MTP full262K can exceed100 on some code paths but not universally on narrative.

For our no-P2P Turing pair, populated262K >=100 likely requires BOTH a much stronger proposal source and a major committed-prefix/target-round reduction.

## Order AFTER current corrected E2E

1. Classify current trained-MTP candidate honestly; leaked/shared-snapshot wins stay invalid.
2. Matched65K BF16-vs-INT8 KVP A/B.
3. Exact Qwen3.8 DFlash2 SM75 sidecar.
4. FP16 recurrent-state discriminator if state is FP32.
5. Ceiling-model then SM75 W4A8/IMMA only if target linears still dominate enough for a large win.
6. If native MTP remains nonportable: interpolation + acceptance-aware objective before more identical CE scaling.
7. Recompute single/65K/262K hard-goal envelopes after each material target-side change.

No watcher/wait. Finish the current run before touching this queue.
