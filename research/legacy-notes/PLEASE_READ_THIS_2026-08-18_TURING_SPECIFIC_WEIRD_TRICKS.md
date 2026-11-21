# PLEASE READ — Turing-specific weird tricks worth bounded experiments

Do not interrupt coherent work. These are ranked follow-ups after current qualification / single80 / dual-C1 priorities.

## 1. Tree/branch packing is also a HARDWARE optimization on SM75
Turing integer Tensor Cores want matrix-shaped work; single-user decode is naturally GEMV-ish. A shallow <=8-node speculative tree/pseudo-tree creates T4/T6/T8 verifier geometry, lets weights be reused across candidate columns, and stays inside our optimized T1..T8 A8 kernels. Evaluate tree-MTP not only by acceptance but by target ms/round versus linear MTP at identical useful-token accounting.

## 2. Re-open exact certified coarse-to-fine target head, but NOT the old loose global Cauchy bound
Old B2->W8 global Cauchy certificate failed badly (bounds ~100-200 vs winners ~20-30). A materially different candidate is GROUPED residual certification, ideally Q4->W8 rather than B2->W8:
  exact_i = approx_i + residual_i
  |residual_i dot h| <= sum_g ||r_i,g||_2 ||h_g||_2
Use the same G64 (or coarser bounded) groups. 248320 rows * 80 FP16 residual norms is ~38 MiB metadata, tiny versus the ~1.35 GiB W8 head scan. Coarse Q4/INT1/INT4 Tensor-Core pass + grouped upper bounds + exact W8 rerank/fallback could preserve exact greedy semantics while avoiding most full-head bandwidth. Only pursue if offline bound tightness looks credible; fallback must remain exact.

## 3. Drafter-only full W4A4 on native Turing INT4 Tensor Cores
SM75 has native s4/u4 MMA. Old work tried pieces (MTP gate/up B2, MTP down W4/A8, exact A8 digit decomposition), not a clean BF16-source full MTP-body W4A4 calibration. Because MTP is only a drafter, target correctness is preserved; bad quantization only lowers acceptance. Build/offline-calibrate first, then kernel only if acceptance/cost model says worthwhile.

## 4. INT8 P*V for MTP attention
Current INT8-KV QK already correctly uses native m16n8k32.s8 Tensor Cores. Do NOT redo that. PV is still BF16: V codes are dequantized to BF16, then BF16 MMA. For drafter-only attention, test folding each V per-key/per-group scale into softmax probability, quantizing scaled probabilities per tile/group, and using u8/s8 or s8/s8 IMMA for PV. This is approximate but target-safe because MTP is verified. If compelling, only then consider quality-qualified target use.

## 5. Verification-guided sparse MTP KV at long context
At 64K-262K, make the DRAFTER read a sparse KV subset while target verification remains full/exact. Strongest version: reuse target verification attention/block importance from round N to choose MTP KV blocks for round N+1, avoiding a separate selector. Turing-specific optional selector: B1/INT4 key signatures + BMMA/IMMA to rank candidate KV blocks cheaply. MagicDec / recent SpecAttn-style work supports this family. Product metric is useful tok/s, not draft perplexity.

## 6. Exploit Turing's separate INT32 datapath deliberately
Turing can execute FP32 and INT32 work concurrently. For Q4/Q5 SIMT/dequant hot loops, inspect SASS/NCU pipe utilization and explicitly interleave next-fragment integer unpack/address (`LOP3`/`PRMT`/pointer math) with current FP32 scale/FMA accumulation, using multiple independent accumulator chains. Earlier warp/prefetch sweeps do not automatically answer a pipe-balance experiment. Reject if absolute kernel time does not improve.

## 7. Per-kernel L1/shared carveout A/B
Turing has a unified 96-KiB L1/shared resource with 64/32 or 32/64-style configurations. Current source search found no explicit `cudaFuncSetCacheConfig` / preferred carveout. Low-cost A/B: shared-heavy attention kernels prefer shared; low-shared streaming linear/proposal kernels test prefer-L1. Exact workload/digest. Expect small gains only.

## Already tried / don't blindly repeat
- INT8 QK Tensor-Core attention is already implemented.
- Exact Q4-weight/A8-activation decomposition through native S4 MMA was tested and slightly slower (~1.1%) in the earlier regime; only reopen if a materially new T8/tree geometry changes economics.
- Simple B2 global Cauchy certificate was far too loose; only grouped/tighter residual certification is a new hypothesis.
- 2:4 hardware sparsity is an Ampere-era feature, not a TU102 strength.
