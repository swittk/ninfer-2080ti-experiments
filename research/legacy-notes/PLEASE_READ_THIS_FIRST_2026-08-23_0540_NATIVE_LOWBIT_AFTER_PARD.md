# 05:40 adviser — external drafters exhausted; return to native low-bit target/MTP with bounded SM75 oracles

Read `AGENTS.md` first. Stock GPUs only. No hardware tuning/reset/reboot. No coding agents. No local behavioral training. Remaining gates: dual-short >=100, populated65K >=100, genuine262K >=100 useful complete-path tok/s.

## STOP STATE / DO NOT RESET THE TREE
At 05:30 Aug 23 both GPUs are stock-idle and no NInfer/build/research producer is live. Last substantive artifact is P0-CE at 04:22:57.

The worktree is intentionally DIRTY with the promoted single80/indexed-rerank package plus accepted dual/prefill work. **Do not reset/checkout HEAD over tracked changes.** HEAD is still `1697ed3b...`; the working tree carries the product frontier.

Current scoreboard: single **80.185 certified** (freeze); dual **65.910**; populated65K **58.75**; genuine262K **32.76**.

## External drafter campaign is exhausted
Do not continue checkpoint-robust drafter fishing without materially new evidence. PARD+TLI/SLEM+free PipeInfer best impossible short was **93.74**, 65K **57.34**; PARD trees **65.55** free-proposer; PARD2 target-independent **91.43 short / 55.93 65K** free-proposer; sequential Qwen3-0.6B best supported SLEM K3 **97.13** with proposer deleted; SWIFT/KnapSpec **61.93** real; Vegas/SparseSpec **67.84** free-verification; ASD accept-all D3 **98.095**; sticky exact-anchor best **96.68**. These are closed.

## Long-context fact: native target optimization is mandatory
P0-BZ gave sparse verification + Windowed-MTP impossible advantages. 65K: delete ALL GQA -> 78.81; + short-window proposal -> 91.22; + proposal entirely free -> **97.04**. 262K: delete ALL GQA -> 52.39; + short proposal -> 66.51; + proposal free -> **70.55**. Thus 65K100 cannot be solved only by KV/attention/proposal; the target dense/state floor must improve. 262K needs much more.

## Do NOT reopen closed quant families
Selective BF16 Q5-down->Q4 (`q4down2/4/8`) is closed. Global W3 target is closed (Humming T4 W3/W4 ~0.875, only ~0.775 ms/round optimistic saving). Global W2 target is closed on quality (even favorable four-centroid G64 has ~0.28-0.43 rel-L2 on real blocks). Exact W3+residual, Q5-as-Q4-base+residual, S4A8, ordinary geometry/prefetch/cache sweeps, Humming W4, QServe transfer, and producer/consumer pipeline are closed.

## P0-CF — factory-MTP-only W2/W4 oracle
This exact contract appears untested. Factory MTP is only a proposer; canonical target verification remains authoritative, so a mechanically W2 MTP may lose acceptance but cannot directly corrupt correctness.

Use BitBLAS/Ladder only as an operator oracle first. BitBLAS supports INT8 activations x INT4/INT2/INT1 weights on V100-class Tensor Cores; SM75 has the relevant INT8 Tensor Core family. Sources: https://github.com/microsoft/BitBLAS and Ladder OSDI24.

1. Build one stock-SM75 BitBLAS INT8xINT4/INT2 discriminator on exact factory-MTP shapes: K10240/N5120, K5120/N14336, K6144/N5120, K5120/N34816, K17408/N5120; T4 first.
2. Compare W4 against current P0-M Q4G64/Marlin isolated MTP body and production proposal wall. If W4 is not competitive, still record W2/W4 ratio; no framework port.
3. Mechanically requantize ONLY factory-MTP projections to symmetric G64 W2 from the actual current native MTP weights. No gradients/distillation/finetune/adapters or qualification-prompt calibration.
4. Use captured six-family MTP hidden states to measure projection error and proposal top1 agreement. Target-W2 failure does not automatically close proposer-W2.
5. If full W2 is too destructive, test at most ONE mixed W2/W4 MTP recipe: keep the most error-sensitive projection(s) W4, rest W2.
6. Real E2E only if isolated MTP body saves >=~25% and proposal agreement suggests useful acceptance. Report yield/acceptance, not just kernel speed.
7. Recompute dual/65K/262K. Current proposal is only ~4 ms/round short-dual, so even deleting it cannot solve dual100 alone; this is additive, not a gate claim.

## P0-CG — one-shot BitBLAS/Ladder W4A8 target-Q4 dataflow oracle
Humming W4 projection-only was slower than current fused NInfer; that closes Humming, not every independent INT8xINT4 tensorization.

Benchmark BitBLAS INT8xINT4 on the exact hot Q4 gate/up T4 geometry. Current NInfer `q4_a8_swiglu_t4_interleaved_chunk40_to_a8` is ~167-168 us and already includes SwiGLU+A8 epilogue. BitBLAS projection-only must be decisively below that (target <=~140-145 us) to leave room for equivalent epilogue. If >=150 us, close immediately with no port. If it wins, port only the minimal Ladder tensor-transform/unpack schedule into NInfer while preserving G64 scales and fused epilogue. Query/key only if gate/up wins. Do not propagate to Q5; W5/decomposition families are already closed.

## P0-CH — small selective W2 target subset only if W2 hardware economics are exceptional
Global W2 target quality failure does not prove that a small subset of low-sensitivity Q4 matrices cannot be W2. Old repo authority explicitly says serious per-layer/role bit allocation was never completed. But do this ONLY after P0-CG/P0-CF prove W2 execution is materially faster.

Use original BF16 source and checkpoint-generic mechanical sensitivity ranking. Prefer weight-only reconstruction error; if activation sensitivity is needed, use a disjoint public calibration corpus only. No training/optimization on qualification prompts. Start with Q4 MLP gate/up only; leave Q5, GDN-sensitive value/output, attention output, embeddings and vocab at current precision. Move only the least-sensitive layers to W2 under a fixed average-bpw budget; every candidate must pass six-family behavioral-quality + deterministic checks. Stop at first quality cliff. This is EXL2-style mixed-precision as a design precedent, not an ExLlama runtime dependency.

## QSpec / ML-SpecQD — reference only
QSpec (EMNLP'25) confirms the no-retraining idea of lower joint-quantized SAME-target drafting with higher-quality quantized verification. Its released kernels require SM80+, and our QuantSpec/SubSpec same-model draft ceilings are already poor. Do not port QSpec now. Reconsider only if P0-CF shows an unexpectedly cheap/high-acceptance W2 same-checkpoint draft.

## Priority / autonomy
1. Record PARD/PARD2/Qwen0.6 closures in handoff/progress if needed; do not reopen them.
2. P0-CF: BitBLAS W4/W2 exact-MTP-shape operator discriminator + factory-MTP-only W2 acceptance oracle.
3. P0-CG: one-shot BitBLAS W4A8 target gate/up oracle; close immediately if projection-only cannot beat ~145 us.
4. Only if W2 execution is exceptional, P0-CH selective W2 subset of low-sensitivity target Q4 layers; no global W2, no Q5->Q4 replay.
5. Recompute dual/65K/262K after every accepted native win. Keep single80 frozen. If bounded low-bit oracles fail, research a materially new target projection/dataflow architecture rather than another external drafter.
