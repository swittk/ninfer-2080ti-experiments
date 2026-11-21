# STOP: DO NOT SPEND THE DAY TRYING TO MAKE FAST BATCHED MTP BITWISE-IDENTICAL TO ORDINARY T1

**Owner/adviser priority correction — 2026-08-17 08:30 +07**

Read this before doing any more MTP-width tracing.

## THE IMPORTANT CORRECTION

**A difference between fast batched MTP output and ordinary one-token-at-a-time T1 greedy output is NOT, by itself, proof that our SM75 MTP implementation is broken.**

We re-read the original NINFER history. The upstream/original implementation explicitly knew about this distinction.

Historical evidence in this repository:

- `a40e17a fix(parity): verify mtp drafts with sequential target decode`
- `43f6544 fix(parity): align mtp verification loop`
- `920bf4e test(mtp): add round correctness gates`
- the old runtime had `mtp_strict_sequential`, whose purpose was to replay target verification with ordinary T1 kernels and prove equality with MTP-off greedy;
- the old MTP design docs explicitly discuss **batched-kernel numerical differences**;
- the historical Python/reference verification recorded a real old batched K5 mismatch at token index 37: the batched target path accepted a token that differed from greedy one-token target decode;
- later `b374d41 refactor(mtp): remove parity-only validation paths` deliberately removed `mtp_strict_sequential` and other parity-only machinery and relaxed the fast batched-MTP acceptance contract.

So the high-performance NINFER MTP path we ported to SM75 was **not historically defined as bitwise identical to ordinary T1 forever**. Width-specific batched Q4/Q5/etc. kernels may perform mathematically equivalent operations with different BF16/quantized accumulation/rounding. A near-tie can therefore flip an argmax without implying bad indexing, corrupt state, or an invalid speculative state machine.

Our previous SM75 correctness work was still meaningful: it preserved and stabilized the **existing NINFER fast-MTP semantics**, particularly MTP3, and made them deterministic/reliable. `38369da` remains the reliability baseline.

## DO NOT MISCLASSIFY THE CURRENT OBSERVATION

Current long-run comparison showed ordinary T1 vs diagnostic batched MTP first-token differences at approximately:

- d1/d2: token 170
- d3: token 89
- d4/d5: token 261

and eager/graph matched those divergence locations.

That proves **numerical/semantic non-bitwise-equivalence exists**. It does NOT yet prove a new SM75 state-machine bug.

The old upstream correctness gate also had a blind spot: the real-file strict/batched comparison generated only a few tokens in the short gate, while upstream documentation itself had previously observed a later K5 divergence. Do not interpret old green short tests as a promise of indefinite T1 bitwise identity.

## HARD STOP RULE

**Stop chasing ordinary-T1 bitwise parity unless you can demonstrate evidence of an ACTUAL algorithm/state correctness bug.**

You may continue MTP debugging only if you find concrete evidence such as:

- wrong causal mask / a column seeing future tokens;
- wrong RoPE/cache position;
- wrong GDN conv or recurrent-state slot selected/committed;
- incorrect KV commit/rewind/rollback;
- cross-column contamination or bad stride/indexing;
- workspace aliasing/stale data;
- a verifier target for a shared position changing because unrelated column DATA leaks into it rather than because arithmetic ordering differs;
- nondeterminism for the same binary/input/path;
- memory-safety/sanitizer evidence;
- disagreement with the intended original NINFER batched-MTP algorithm/state transition, not merely with T1 arithmetic.

If the evidence instead reduces to:

> same valid inputs/state/masks -> width-specific batched kernels -> small BF16/quantized numerical differences -> near-tie argmax eventually changes

then **DOCUMENT IT AND STOP. THAT IS NOT THE PERFORMANCE PROJECT'S CURRENT BLOCKER.**

Do not attempt to force every T4/T5/T6 kernel through T1 arithmetic just to obtain bitwise equality. That destroys the point of batched verification and can erase the performance we are trying to build.

## WHAT TO DO WITH THE CURRENT DIAGNOSTIC WORK

The current exact72 worktree contains a large amount of tracing/diagnostic code (`NINFER_MTP_EXACT_TARGET`, layer/substage traces, exact-GDN experiments, etc.).

1. Preserve the useful traces/artifacts under `.chatgpt/mtp_width_correctness/`.
2. Write a short conclusion classifying what was actually demonstrated.
3. If you found a **real state/indexing/mask bug**, make the smallest permanent fix and add a regression that tests that specific invariant.
4. If you did **not** find such a bug, restore tracked source to the `38369da` production/reliability baseline. Do not commit diagnostic slow paths or tracing scaffolding as production behavior.
5. A strict sequential/T1 comparison may remain a **debug/reference tool**, but it is not the performance-path contract unless the owner explicitly changes the product requirement later.

## IMPORTANT: DO NOT THROW AWAY THE MTP WIDTH RESULTS

The width campaign is still useful for performance economics even if widths are not bitwise-T1-identical.

Observed fast-path examples before the deep parity detour:

- strict JSON: MTP3 ~79.75 tok/s, MTP5 ~84.58 tok/s;
- long code: MTP3 ~94.13 tok/s, MTP5 ~99.03 tok/s;
- ordinary/canonical prose: fixed MTP5 was substantially worse than MTP3.

This still says adaptive/wider speculation may have real value on highly predictable workloads. But do NOT make bitwise ordinary-T1 equality a prerequisite for returning to performance work unless we explicitly decide to impose a stronger contract than original NINFER.

If adaptive width is revisited, document that changing batched width may also change finite-precision target numerics. The policy must be deterministic and model/checkpoint agnostic; do not use hardcoded prompt types or token IDs.

## NEXT PRIORITY: RETURN TO THE NCU-BACKED KERNEL OPPORTUNITY

Once the above classification is written and source is clean/re-grounded, resume the strongest measured single-GPU performance lead.

NCU already established:

- Q4 A8 SwiGLU: ~540-554 GB/s, ~67.5% long-scoreboard, ~99% occupancy, low L2 hit, zero spills;
- Q5 down A8: ~521 GB/s, ~55.8% long-scoreboard;
- GDN Q5 A8: ~462 GB/s, ~42% long-scoreboard plus fixed-latency wait;
- Q5 6144-output SIMT: ~304 GB/s effective DRAM read, high ~58% L2 hit, only ~13.9% long-scoreboard — a separate bottleneck class.

This is the first hard hardware-counter evidence supporting a **true SM75 load/use software-pipeline / double-buffering experiment** rather than another blind geometry sweep.

### Start with Q4 A8 SwiGLU

Before changing it, use narrow NCU/SASS/source correlation to identify the specific global-weight load(s) responsible for long-scoreboard stalls.

Then attempt ONE principled SM75/Turing-appropriate pipeline change, e.g. increasing load-to-use distance / register double-buffering / overlapping the next weight fragment with current compute.

Do not blindly copy Ampere/Hopper producer-warp designs: Turing has no `cp.async`/TMA.

Success must be measured by:

- reduced long-scoreboard / no-eligible-warp pressure;
- increased useful sustained DRAM throughput if possible;
- kernel-time improvement;
- **end-to-end tok/s improvement** on the production MTP3 lane;
- unchanged existing NINFER MTP semantics/determinism for the same configured path.

If the pipeline is neutral/regressive after a sensible implementation, record it and stop. Then test the next NCU-backed hypothesis rather than tuning forever.

Do not apply the Q4 solution blindly to the Q5-6144 SIMT kernel; NCU says it is not primarily DRAM/long-scoreboard bound.

## THEN DUAL GPU — FULL CONTEXT FIRST

Do not let this MTP parity rabbit hole delay the owner's larger system goal indefinitely.

After the strongest NCU-backed single-GPU experiment(s) are accepted/rejected:

1. move to two-GPU layer/pipeline sharding;
2. first milestone = maximum usable Qwen3.8 context, ideally full native **262144**;
3. each GPU permanently owns its layers, weights, layer-local KV, and GDN/recurrent state;
4. CUDA P2P is unavailable, so transfer only stage-boundary hidden activations via pinned host staging;
5. do not build remote-KV or chatty tensor parallelism;
6. after full context works, optimize combined/pipelined multi-user throughput.

Healthy GPUs:

- internal: `GPU-020906b4-0904-3471-e4ee-e41ec2350be3`
- eGPU: `GPU-2fffc240-5d58-fbc8-27d6-6b349448fda9`

Two independent replicas have already demonstrated ~136.6 tok/s aggregate decode, so replica mode remains the immediate ordinary-context multi-user baseline.

## SUMMARY IN ONE SENTENCE

**Unless you can show a real mask/state/indexing/rollback/cross-column bug, STOP trying to make batched MTP bitwise-identical to ordinary T1, clean up the diagnostics, and get back to the NCU-backed Q4/Q5 performance work; after that, go build full 262K dual-GPU context.**
