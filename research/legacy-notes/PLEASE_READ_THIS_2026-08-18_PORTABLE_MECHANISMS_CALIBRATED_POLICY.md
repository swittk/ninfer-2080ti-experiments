# IMPORTANT — portable mechanisms, calibrated policy

The active 2x TU102 machine is a valid development/optimization platform, but the cards are not performance-equivalent: the Core-X card boosts normally (~1.7-1.86 GHz under current controls) while the direct internal card remains at 1350 MHz under load despite the same VBIOS, power limit, memory clock and supported clock table.

This does **not** invalidate the portable mechanisms. It DOES mean hardware-dependent placement numbers must not become product/model constants.

## Evidence that remains portable / useful
- correctness and replay/fold contracts;
- INT8 KV and its kernels;
- SM75 Q4/Q5/A8 kernels when candidate/reference are compared on a fixed device;
- packed verifier / rolling C4 scheduling mechanisms;
- cross-round overlap as a mechanism;
- MTP acceptance / shortlist / multilingual findings;
- tree/pseudo-tree, TP, context-parallel and sparse-drafter designs;
- same-device or same-fixed-assignment candidate-vs-reference A/B wins.

## Treat these as calibrated policy, NOT architecture truths
- split layer / number of layers per device;
- which GPU owns target head;
- which GPU owns proposal/MTP work;
- whether a particular helper/offload is worthwhile;
- TP row/head fractions;
- PP/TP/context-parallel threshold choices;
- absolute dual TPS and normal-vs-swapped placement deltas.

Current values are valid for this physical pair, not universal `RTX 2080 Ti` constants. Do not write logic equivalent to `if qwen27b -> split=32` based on this machine.

## Desired end-state
Portable mechanism + measured device profile. A lightweight calibration should eventually benchmark representative hot operations and communication on the actual devices, then choose/caches a machine profile keyed by relevant device/runtime identity. Useful inputs include:
- representative MLP/GDN/attention/head latency at T1/T4/T8;
- H2D/D2H / host-staged boundary latency for relevant byte sizes;
- sustained loaded clocks / bandwidth class;
- available VRAM / context requirement;
- concurrency target.

From that, select split, ownership, asymmetric TP fractions where geometry permits, and backend thresholds. Manual overrides should remain available.

## Qualification discipline now
Until autotuning exists:
1. record physical GPU UUIDs / role assignment and preferably sustained loaded clocks with benchmark artifacts;
2. compare candidate/reference on the same fixed assignment for software claims;
3. explicitly label placement/split/offload numbers as machine-specific;
4. requalify portability-sensitive placement policy on matched healthy/floating-clock devices before making broad claims;
5. do not pause core optimization work merely because this pair is asymmetric.

The 1350-MHz internal behavior should not shape production architecture. It is a calibration input / current-machine condition, not a model property.
