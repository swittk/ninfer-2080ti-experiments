# After current CM/shared-G64 lane — VPTQ/codebook low-bit oracle

Do NOT interrupt an in-flight P0-CM/shared-G64 qualification for this. Finish/classify current work first. Read `AGENTS.md`; stock GPUs, no hardware tuning/reset, no coding agents, no local model training.

## Why this is genuinely new
Our scalar/mechanical W2 work failed for two independent reasons: BitBLAS W2 is slower than promoted P0-M Q4G64/Marlin, and scalar W2 destroys proposal/target fidelity. That does **not** classify vector-codebook PTQ.

**VPTQ** (Microsoft, EMNLP 2024) is a vector post-training quantization method designed for ~1-2 bpw. The current project explicitly advertises 1-2-bit compression **without retraining** and has dedicated CUDA inference kernels. Current Aphrodite hardware support lists VPTQ on Volta/Turing/Ampere/Ada/Hopper. This is owner-legal only in pure PTQ/`woft` form: no post-quant fine-tuning, no behavioral training.

Sources:
- https://github.com/microsoft/VPTQ
- https://arxiv.org/abs/2409.17066
- https://huggingface.co/docs/transformers/quantization/vptq
- https://aphrodite.pygmalion.chat/quantization/support-matrix/
- optional generic 2026 quantization adapter: https://github.com/wejoncy/QLLM (VPTQ + mixed 2-8 bit across HF models).

QuIP# is NOT the first choice: its published fast E8P kernel was designed for Ampere+ and the full method uses fidelity fine-tuning. CodeGEMM is also not a direct runtime candidate: official build targets SM90 and its public quant script uses tuning.

## P0-CP — VPTQ factory-MTP-only operator oracle FIRST
The strongest first fit is the factory MTP, not the target verifier. MTP proposals are three causal/small-M passes and canonical target verification protects correctness. Existing VPTQ inference is GEMV-oriented, so this is where codebook quantization has the best chance.

1. Clone/pin current Microsoft VPTQ under `.chatgpt/`; do not modify production source. Build only the inference extension for **sm_75**. If upstream build needs a trivial arch flag, that is allowed; if kernels use an Ampere-only instruction, record and close.
2. Benchmark the VPTQ kernel on the exact five factory-MTP matrix shapes already used in P0-CF: K10240/N5120, K5120/N14336, K6144/N5120, K5120/N34816, K17408/N5120. Measure M=1 first (and exact real MTP input dtype/layout as closely as the kernel allows).
3. Current promoted P0-M body authority is **1.619 ms for all three dependent proposal passes**. Admission gate: a VPTQ 2-3bpw five-matrix pass must project to <=~0.40 ms/pass (<=~1.20 ms/three passes) before scale/codebook/epilogue/state integration. If it cannot beat this with >=25% margin, close before quantizing weights.
4. If timing passes, mechanically VPTQ-quantize ONLY the actual factory-MTP matrices from the current checkpoint in **without-finetuning / woft** mode. Calibration/Hessian collection is allowed as PTQ, but use a disjoint generic corpus; do not optimize on the six qualification prompts. No gradient updates to model behavior, no adapters, no post-quant tuning.
5. Measure six-family MTP proposal top1 agreement, accepted-by-position, useful/round, and E2E. Because target verification is canonical, changed draft tokens are allowed if quality remains target-correct.
6. Test at most two codebook budgets: ~2bpw and ~3bpw (or nearest practical VPTQ formats). No giant codebook sweep.

## P0-CQ — VPTQ/CodeGEMM T4 target oracle ONLY if MTP result justifies it
Target verification is T4 and current NInfer gets weight reuse extremely well. Ordinary codebook GEMV often scales badly with M>1. Therefore do not full-model VPTQ first.

A) Run the VPTQ operator at M=4 on exact hot gate/up shape. Compare against NInfer fused Q4 gate/up ~167-168 us including SwiGLU+A8. A codebook projection-only path needs to be roughly <=130-140 us to leave room for epilogue and be interesting. If it is >=150 us, close VPTQ target use.

B) If VPTQ M=4 is slow but M=1 is excellent, read **CodeGEMM** (NeurIPS 2025) before inventing a kernel. CodeGEMM replaces repeated codebook dequantization with a small on-chip `Psumbook` of activation-centroid inner products, then gathers by code index. It explicitly targets the M=1..16 regime and reports much better M=4 scaling than AQLM/QuIP# in its paper.
Sources:
- https://github.com/naver-aics/codegemm
- https://proceedings.neurips.cc/paper_files/paper/2025/hash/3151e460c41ba67dc55412861184ef35-Abstract-Conference.html

Do NOT port CodeGEMM directly: its released setup hardcodes sm_90 and its public PTQ recipe includes local fine-tuning. Use only the **Psumbook/dataflow idea** as a bounded SM75 source-level oracle if VPTQ proves codebook quality + M=1 economics are strong enough to justify custom work.

## Why this could matter beyond short MTP
P0-BZ showed 65K stays at ~97 t/s even after deleting all long-GQA and all proposal cost, so target weight/state traffic needs an additional several-percent reduction for any 65K100 composite. A genuinely fast ~2-3bpw target representation would attack the correct physical floor. But target work is admitted only after M=4 operator economics and held-out quality are proven.

## Priority
1. Finish current CM/shared-G64 work.
2. P0-CP VPTQ sm75 M=1 exact-MTP-shape timing oracle.
3. Only if <=~1.20 ms/three-pass projected body: pure-PTQ/woft MTP quantization + six-family acceptance/E2E.
4. Only if MTP succeeds: VPTQ M=4 target gate/up discriminator.
5. Only if codebook quality is strong but M=4 execution is the blocker: inspect CodeGEMM Psumbook dataflow for a small SM75 oracle.
6. Recompute dual/65K/262K after any accepted native win. Do not reopen scalar W2, Q5->Q4, BitBLAS, old vLLM-2080Ti archaeology, or local training.
