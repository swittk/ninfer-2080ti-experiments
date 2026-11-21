# PLEASE READ — reliability evidence is sufficient; package it, then move on

Do **not** interrupt an in-flight validation campaign. If one is running, let it finish. After that, please stop launching additional q45g determinism torture campaigns **unless a new mismatch/error actually appears**.

The reliability question now has enough evidence to justify closure and packaging:

- hardened CUDA-Graph campaign: 2 normal + 30 pressured runs, all identical
- hardened graph campaign without extra trajectory-digest instrumentation: 2 normal + 25 pressured runs, all identical
- reset-free B2 CUDA-Graph candidate: 2 normal + 30 pressured runs, all identical
- no-CUDA-Graph round-trace campaign: 1 normal + 30 pressured runs, all 372 MTP round records identical
- normal exact72 performance remains ~71–72 tok/s with the reference SHA/trajectory

The historical flake was real and worth investigating, but repeated clean campaigns across graph/no-graph, pressure, reduced instrumentation, and round-level tracing are now strong enough that more repetitions have sharply diminishing value.

## Next natural checkpoint: package the reliability work

Before starting another research direction:

1. Run the relevant unit/stress tests once on the final intended source state.
2. Review the reliability diff and remove diagnostics/debug-only code that is not meant to ship, while keeping useful permanent regression hooks/tests.
3. Commit the q45g cross-context determinism/reliability fix as a **separate local Git commit** on the exact72 lineage. Do not mix unrelated performance experiments into that commit.
4. Append a concise progress report via the repository's append-only progress workflow. Record:
   - the historical failure signature;
   - the likely/root cause if established;
   - the source changes that harden it;
   - the stress matrices that passed;
   - any residual uncertainty honestly.
5. Refresh the handoff/work-session state so the next session does not have to rediscover this investigation.

If the exact root cause cannot be proven beyond doubt, that is okay: document the strongest evidence and the hardening changes. Do **not** keep running endless 30-run campaigns solely to obtain philosophical certainty.

## After the reliability commit

It is fine to return to performance research, but start from the committed exact/reliable ~72 tok/s baseline and keep experimental work on a separate branch/checkpoint.

Do not restart already-exhausted ordinary Q4/Q5 geometry/cache/prefetch sweeps without a genuinely new hypothesis. The interesting remaining performance work is higher-level/architectural: e.g. more useful tokens per expensive model sweep (adaptive/deeper speculation, tree/suffix-style drafting where appropriate), or a genuinely evidence-backed dataflow/fusion idea. Preserve exact/reference behavior as the production lane and keep trajectory-changing turbo research explicitly separate.

**Short version:** finish the current run if one is active -> tests -> clean up -> commit reliability fix -> progress/handoff -> then brr performance again if desired. No more reliability campaign #37 unless something actually fails.
