# First SM75-to-NInfer-all port screen — rejected (2026-11-21)

This records the first attempted transfer from the archived RTX 2080 Ti kernels into the maintained [NInfer-all](https://github.com/iamwavecut/ninfer-all) source. **Not an upstream speedup claim; do not open a PR for this candidate.**

Source basis: NInfer-all `cc15a072962ef162c166b0200b82dbaba1772749` and our historical Q5 RowSplit small-token kernel (`src/ops/linear/q5/q5_rowsplit_gemm_simt.cuh`). An isolated local experiment added an opt-in `NINFER_SM75_Q5_R4_RESEARCH=1` route for exact 5120×6144 Q5G64 BF16 projection, T=4–6. The standard NInfer-all route remained the default.

### Direct GPU measurements

Hardware: original modified 22 GB RTX 2080 Ti on GPU 0, stock 250 W limit, driver 580.126.09, CUDA 12.9.86. Deterministic pseudorandom Q5 codes/high bits/scales and BF16 activations, 5120 rows × 6144 columns. Reference uses the maintained two-warp K-split kernel; candidate computes four rows per CTA.

| T | Maintained K-split kernel | Ported SM75 four-row kernel | Port result |
|---|---:|---:|---|
| 4 | 70.7 µs | 100.0 µs | 41% slower |
| 5 | 84.6 µs | 112.7 µs | 33% slower |
| 6 | 95.1 µs | 121.6 µs | 28% slower |

These figures average two opposed-order timing rounds, each comprising three synchronized batches of 80 GPU invocations. T=6 showed substantial baseline clock/timing variation (81.6–108.6 µs), so these are screening measurements, not a confidence interval. The slower result was consistent across T=4–6.

All BF16 output values matched **bit-for-bit** against the existing K-split implementation: 20,480, 25,600, and 30,720 results for T=4,5,6. Independent sampled CPU FP64 oracles passed under BF16-appropriate tolerance on 14 rows per T. Full-engine correctness or end-to-end speedup was **not** claimed. The main NInfer-all source build was stopped after this negative direct result.

The Q5 plain-linear Op may not be exercised by the model's principal MTP execution, which also invokes fused linear-add paths. Consequently, even a winning kernel could not by itself explain our older observed ~37 ms MTP round versus ~63 ms on the earlier maintained release benchmark.

**Conclusion:** reject this kernel port. Newer NInfer-all implementations have overtaken this specific historical Q5 route. The next worthwhile experiment is to profile the complete MTP verification/proposal/target-head path under a same-source, same-weights comparison and target only demonstrated bottlenecks. Keep original prompt/draft policies strict; never substitute relaxed quality for throughput.

Original public archive: [swittk/ninfer-2080ti-experiments](https://github.com/swittk/ninfer-2080ti-experiments). Earlier end-to-end comparison: [November 21 live benchmarks](2026-11-21-live-sm75-comparison.md). No upstream issue or PR was opened.
