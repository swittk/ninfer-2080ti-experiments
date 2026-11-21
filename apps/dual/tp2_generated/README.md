# TP2 block69 kernels

These generated SM75 kernels implement the calibrated 69/67 group64 row split used by the opt-in dual C1 MLP coordinator:

- fast/Core-X shard: 8,832 intermediate rows;
- slow/internal shard: 8,576 intermediate rows;
- Q4 gate/up and Q5 down are both resident on their assigned device;
- T4 outputs are partial BF16 MLP projections converted to FP32 before the owner-device reduction.

The split and generated code are frozen production inputs, not generated at build time. Provenance and microbenchmark evidence remain under `.chatgpt/dual_c1_20260818/tp2_asym/` and `tp2_chain64/` in the development worktree.
