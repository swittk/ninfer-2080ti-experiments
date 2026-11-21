# 04:45 adviser — unstick from closed speculation/KV lanes

Read `AGENTS.md` first. Stock GPUs only. No card settings/reset/reboot. No agents. No behavioral training. Gates remain 80 / 100 / 100 / 100 useful complete-path tok/s.

The fresh stock profile points back to target projection dataflow. Do not return to vocab-head, direct-KV4, Quest/page sparsity, semantic retrieval, Jacobi/lookahead, FAFO, VeriCache, lossless-KV compression, or static proposal-prefix lanes; those are closed.

## P0-V — matched SM75 W4A8 external oracle, then TRUE producer/consumer pipeline

This is the highest-priority short-single lane. Historical NCU already measured roughly:
- Q4 A8 SwiGLU: ~540-554 GB/s, ~67.5% long-scoreboard, near-full occupancy, low L2 hit;
- Q5 down A8: ~521-535 GB/s, ~55.8% long-scoreboard;
- GDN Q5 A8: ~462-480 GB/s, ~42% long-scoreboard;
- Q5 6144 SIMT is a different class (~304-340 GB/s, high L2, only ~13.9% long-scoreboard).

The old owner audit explicitly identified a true Turing producer/consumer load-use pipeline as the next structural gap, but progress only records ordinary geometry/prefetch variants and says the true pipeline was deferred. Do not mistake `prefetch2`, chunk20/80, warp-count, cache-hint, or dual-row experiments for this.

First, use upstream Humming as an executable schedule oracle on the exact hot Qwen shapes. Humming officially supports SM75 INT8 activations and symmetric INT1-8 weights. Benchmark W4A8 T4/T5 with the closest representable group/scale semantics. If exact Q4G64/F16-scale semantics are not directly representable, use Humming only as a performance/dataflow oracle, never as correctness proof.

If the external/operator ceiling is materially positive, implement one true SM75/Turing pipeline on the hottest Q4 path:
- loader-ish warp/group or explicit staged producer/consumer work;
- double-buffer shared/register fragments;
- load next weight/scale fragment while current MMA executes;
- materially increase load-to-use distance;
- no cp.async/TMA (not available on Turing);
- preserve current stored codes/scales and qualified arithmetic semantics.

The current profile has ~11.2 GPU-ms/round in Q4 gate/up and ~7.1 in Q5 add/down-family work. Single80 needs only ~1.6 ms/round complete-path saving, so a real ~10% structural gain across those dominant paths is enough in principle. Start Q4; only propagate to Q5-down if the pattern wins. Treat Q5-6144 separately.

QServe/OmniServe is a design oracle here, not a quantization contract: study its compute-aware offline reordering and register-level INT4->INT8 dequantization to reduce CUDA-core unpack/scale overhead. Do NOT adopt QoQ/KV4/approximate quantization semantics. Mechanical exact reorder of existing Q4 codes is allowed.

## P0-W — SubSpec ceiling for dual-short >=100

SubSpec (NeurIPS 2025) is lossless and training-free: it constructs a highly aligned draft model from mechanically low-bit substitute layers of the target, sharing remaining layers/KV. This is materially different from the closed 48/64-layer pruned self-drafter and from trained draft models.

Our existing free-independent-proposer PipeInfer oracle exceeds ~140 useful t/s, so the missing object is an allowed cheap independent aligned proposer. SubSpec may fill that gap. Do NOT port the framework first. Build an optimistic but real Qwen3.8 ceiling:
1. mechanically derive candidate substitute layers only (no training);
2. benchmark low-bit substitute projection cost on the 1350-MHz helper using Humming W2/W3/W4 or another exact executable low-bit oracle;
3. measure actual next-token / multi-token agreement on the existing six-family traces;
4. charge real no-P2P host transfer for the small draft boundary/state unless proving overlap removes it;
5. model independent-drafter pipelining against the measured target round.

If optimistic complete-path ceiling cannot clear 100, close. If it can, implement minimal independent proposer and then consider EasySpec-style layer-parallel drafting to use both GPUs more efficiently. EasySpec itself assumes a smaller draft model, so it is secondary to proving a SubSpec-like drafter.

## Lower priority — exact finite-precision attention certification

BLASST/Skip-Softmax style block skipping is approximate as published and cannot be promoted. A new exact variant would only skip a KV block when its denominator and every value-channel contribution are provably below the current floating-point/BF16 rounding threshold, so exact output bits cannot change. This is distinct from Quest's q·k top-key/page selection. Cheap oracle only; long262K needs ~3x, so do not spend kernel time unless certified skip rate is unexpectedly huge.

## Do not stop after one closure

Run P0-V first. In parallel/afterward bound P0-W. Recompute all four stock-card gates after each accepted mechanism and continue. Do not report final merely because one candidate closes.
