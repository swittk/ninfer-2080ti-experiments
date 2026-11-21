# STOP THE WATCHER THEATRE — THIS IS AN ACTIVE PERFORMANCE JOB

The owner is explicitly calling out a recurring bad behavior: **doing one piece of work, hitting a mundane failure or missing artifact, launching a watcher loop, and then effectively stopping.** This is not acceptable for this task.

## Watchers are NOT work
A shell that only does `for ... sleep ... test -f ...` is bookkeeping. It does not count as progress. If the GPU/CPU is idle and the only live jobs are watcher/polling loops while performance work remains, **you are idle and you must resume real work immediately.**

### Hard rules
1. **Every watcher must have a live producer.** Before waiting for `foo/classification.json`, `decision.json`, `suite_summary.json`, `STATUS`, etc., verify that some real process is actively capable of producing that exact file. If no producer is running, DO NOT WAIT. Run/fix the producer yourself.
2. **A failed setup/build/materialization command is not a stopping point.** Inspect the error, repair the command/source/build state, rerun, then continue to the next experiment. Examples already seen and considered unacceptable stopping behavior:
   - redirecting into a directory that did not exist;
   - TP2 benchmark `main()` accidentally placed inside an anonymous namespace;
   - rejected attention-A8/cache diagnostic edits left behind and breaking CMake;
   - a generated benchmark source left as 0 bytes;
   - waiting for `classification.json`/`decision.json` when no process exists to create it.
3. **Do not create sentinel files or watcher loops as a substitute for finishing the analysis.** If the measurements already exist, write the classification/conclusion now and move on.
4. **No idle gap while the hard gates remain open.** When a bounded experiment closes neutral/rejected, immediately take the next highest-value item from the profile-driven queue. Do not stop to report that one knob failed.
5. **Use failures as branch points, not exits.** `compile failed` -> fix compile. `benchmark CLI wrong` -> inspect `--help`/source and run correctly. `profiler regex missed kernel` -> use exact symbol/listed kernel and rerun. `artifact missing` -> determine why and produce it. Only classify a path blocked when there is a real external dependency you cannot resolve locally.
6. **Keep real compute/research moving while long independent work runs.** A long compile, context-prefill, NCU capture, or artifact scan may justify parallel work on another independent item. A sleeping watcher does not.
7. **Do not confuse process-count with progress.** Three sleeping watcher shells and zero NVCC/NInfer/NCU/analysis producers = zero active work.

## Expected working loop
`measure -> inspect counters/SASS -> implement bounded hypothesis -> build -> operator bench -> exact E2E qualification -> classify -> revert/promote -> immediately next hypothesis`

Repeat this loop. The owner explicitly wants sustained iteration, not one-shot experiments followed by passive waiting.

## Current priority reminder
Single-GPU >=80 remains a hard gate. The profile says Q4/Q5/GDN hotspots are not exhausted. Keep iterating launch geometry, waves/SM, prefetch/dependency chains, INT32/FP/IMMA overlap, T6/T8/pseudo-tree amortization, and other counter-justified mechanisms. Dual TP2 is promising but does not replace single-GPU work.

---

# PROFILE SAYS NOT DONE — iterate, do not stop after one neutral knob

Owner is explicitly unhappy with the pattern "try one small thing -> neutral -> stop/report sadness". The exact-engine profile shows a concentrated optimization surface. Treat neutral experiments as one eliminated hypothesis, NOT evidence the engine is near optimal.

## Measured exact-target hotspots on healthy Core-X TU102
Nsight Systems 256-token exact-target run:
- Q4 A8 SwiGLU T4: **22.2% GPU time**, median ~168.0 us;
- Q5-down A8 T4: **14.1%**, ~106.7 us;
- GDN Q5 value/z A8: **7.8%**;
- another hot Q5 path: **7.1%**;
- full exact W8 target head T4: **5.1%**, ~2.46 ms/round.

NCU:
- Q4 A8 T4: DRAM 85.17% peak, L2 hit 5.23%, active warps 98.16%, **long scoreboard 77.04%**, math throttle 4.53%, MIO 0.27%, 59 regs/thread (64 allocated), 10 KiB shared, ~1 wave/SM. It reads ~94.75 MB: essentially the whole encoded gate/up weight once.
- Q5-down A8 T4: DRAM 82.64%, L2 hit 10.05%, active warps only **58.26%**, **long scoreboard 72.33%**, 59 regs/thread (64 allocated), 2 KiB shared, only **0.59 waves/SM**. It reads ~58.63 MB: essentially the whole encoded down weight once.

Default L1/shared carveout was neutral; that closes ONLY cache preference, not these kernels.

## Required iteration style
Run a matrix of bounded kernel hypotheses. For each: operator median + exact e2e same digest/trajectory where applicable. Revert losers and immediately test the next. Do not stop merely because a build/setup command fails; fix the command and continue. Keep a scratch table of attempted variants so we don't repeat them.

### Q5 first: obvious latency-hiding problem
Try multiple launch/tiling variants while preserving one-pass weight traffic:
1. Increase grid parallelism / smaller row tiles so waves/SM rises materially above 0.59. Sweep CTA/output-row partitioning and threads/CTA. Do not duplicate weight reads unnecessarily; measure DRAM bytes.
2. Sweep K-chunk/pipeline depth and prefetch distance; maintain 2+ independent outstanding global-load groups before consuming previous codes/scales.
3. Separate/overlap code and scale load/decode chains; vectorize aligned loads where legal. Inspect SASS for LDG width and dependency chains.
4. Deliberately interleave integer unpack/address (`LOP3`/`PRMT`/shifts) with FP scale/FMA and IMMA work on independent values to exploit Turing's separate INT32+FP32 issue capability.
5. Sweep warp count / CTA geometry against register allocation. 59->64 allocated regs plus only 58% active warps is worth attacking. Test variants that trade modest redundant arithmetic for more resident/issued warps.
6. Apply winning scheduling ideas to GDN Q5 value/z and the other Q5 hotspot; they collectively matter more than the head.

### Q4: bandwidth + very high scoreboard
1. Sweep load/prefetch ordering and independent K chunks to reduce 77% long-scoreboard stalls; use the existing one-wave/high-occupancy shape as baseline.
2. Try alternate warp-to-row/K mappings that preserve single weight read but create more independent outstanding memory operations per warp/CTA.
3. SASS-guided integer decode + FP/IMMA overlap; do not merely add generic prefetch that reproduces old experiments.
4. Sweep chunk geometry around current chunk40, threads/CTA, and scale/code vector load shape. Use NCU after a promising microbench to confirm stall reason actually moved.
5. Test whether a persistent/streaming CTA organization can reduce setup/barrier/address overhead without sacrificing DRAM saturation. Reject if bytes reread.

### Amortize weight scans, not just shave microseconds
The Q4+Q5 matrices are streaming their full weights each T4 verification. A major route to >=80 is more **useful verified candidates per weight scan**.
- Build a bounded **two-branch pseudo-tree** using existing packed verifier lanes/state semantics, targeting T6/T8 matrix geometry. Do NOT use curated token lists; branches come from generic proposal alternatives.
- First version may duplicate common-prefix compute if that keeps GDN state handling simple. The point is to test whether T8 weight reuse + higher useful acceptance beats T4/depth-only MTP.
- Report useful accepted tokens/round, target kernel timings, total t/s, and exact target output. This is both an algorithmic and SM75 hardware-utilization experiment.

### W8 exact target head
Only ~5% of exact profile, so don't let it monopolize effort. Still, get a valid NCU capture (previous name-filter got no kernel) and use it for a bounded schedule/top-k exactness experiment if counters justify it. Proposal B2 is target-safe in current controls; target B2 is NOT universally exact and must not be counted as exact-engine speed.

## Dual goal: TP2 bench MUST actually happen
The attempted `tp2_mlp_real/tp2_mlp_bench.cu` materialization failed and left a 0-byte file. Fix/recreate it normally; do not stop there.
Build the real T4 MLP TP2 microbenchmark:
- split gate/up rows across both GPUs;
- matching local down partials;
- concurrent device execution;
- real host-staged partial-hidden reduction/transfer included in wall time;
- compare against production one-GPU layer;
- because current cards are asymmetric, sweep asymmetric work fractions rather than assuming 50/50.
This is a bounded discriminator for the dual single-request >=100 architecture.

## Priority
Keep both fronts moving, but single >=80 remains hard gate. The current profile does NOT justify pessimism: >50% of GPU time is concentrated in a few kernels with measured memory-stall/occupancy issues, and T6/T8/tree-style amortization remains largely untested. Iterate until a class is genuinely exhausted, not until the first knob is neutral.
