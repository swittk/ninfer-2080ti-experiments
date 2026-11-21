# RESUME NOW — current frontier is 59.05 TPS; attack coordinator serialization, then verifier geometry

This note supersedes stale chat/handoff state. Read current `progress.md` first.

## Current canonical truth

Branch: `feat/sm75-dual-gpu-full-context`

Current HEAD observed before this note:
`10d1427 perf(dual): add true single-pass T5 TP`

The worktree also contains a validated but not-yet-committed fixed-depth bootstrap correction in `apps/dual/main.cpp`.

Accepted architecture/results:

- native physical 60/40 sequence-sharded INT8-G64 KV is integrated;
- 262K physical KVP capacity passes;
- TP local weight storage is deduplicated by aliasing canonical local Q4/Q5 weights;
- 262K physical KVP + deduplicated T5 TP fits both 22-GiB TU102 cards;
- true single-pass T5 Q4/Q5 TP is accepted: 25.606 ms full Core-X -> 16.759 ms sharded chain, 1.5279x;
- T5 BF16 remote partial transport is accepted;
- fixed verifier depth4 is the accepted long-context budget;
- the snapshot-depth bug is fixed: restored depth4 now actually reaches four drafts after bootstrap.

Latest trusted populated-65K C1 qualification:

- 59.11 / 58.98 / 59.05 tok/s, median **59.05 useful tok/s**;
- digest `09ff33b553ac8a05` every run;
- 32 rounds;
- 127 drafted;
- 93 accepted (73.23%);
- accepted-by-position 29/24/21/19;
- 3.90625 useful tokens/round;
- 67.21875 ms/round;
- zero fallback.

Do not regress to the older broken-snapshot ~49 TPS figures.

## Closed work — do not rediscover

1. GDN head/state TP using the measured ~60/40 head split is CLOSED for this design:
   - 19.873 ms Core-X chain -> 19.212 ms dual = only 1.0344x;
   - it also failed the 48-layer state/numerical gate;
   - this is not worth runtime integration.

2. The tiny fixed two-branch last-draft depth2/3 tree oracle is CLOSED:
   - its zero-overhead upper bound did not beat accepted depth4.
   - This closes that narrow topology only, NOT all hardware-aware tree/verifier search.

3. Mixed T4 TP + canonical fifth-column tail is CLOSED because it streams MLP weights twice.

4. Depth5 fixed linear width is CLOSED for the measured 65K trajectory: same useful yield as depth4, higher round cost.

5. Old TP2 output-event-only async experiment is CLOSED. Do not repeat it unchanged.

## Largest current measured implementation gap: coordinator serialization

The post-T5 profile shows:

- T5 shard kernels themselves: ~10.8–11.1 ms/round per rank;
- TP coordinator: ~32.05 ms/round;
- 6,619 `cudaStreamSynchronize` calls: ~1.725 s over decode window;
- 19,034 `cudaMemcpyAsync` calls: ~0.610 s;
- only ~0.608 / 2.634 s of the verifier window has both GPUs busy.

The live source still has hard host fences, notably:

- `apps/dual/tp2_mlp.cu`: per-layer `owner->synchronize(); remote->synchronize();`
- physical KVP: owner D2H `cudaStreamSynchronize(owner_copy_stream_)` and helper completion synchronization before state return.

### NEXT EXPERIMENT: full event-DAG coordinator discriminator

This must be materially different from the previously rejected output-event-only TP2 test.

Implement the complete dependency graph for one fixed depth4 verifier round, not merely async output submission:

1. owner quant/input-ready event;
2. owner local engine waits on quant-ready;
3. remote input H2D waits on owner D2H-complete event without host synchronization;
4. local and remote engines run concurrently;
5. remote partial D2H is followed by a remote-complete event;
6. owner H2D/merge stream waits on that remote event rather than synchronizing the remote device on host;
7. owner-local engine completion is also event-driven;
8. merge waits only on the two precise dependencies;
9. preserve pinned host staging because topology has no useful P2P;
10. apply the same dependency-DAG idea to KVP Q/K/V/position staging and `(acc,m,l)` state return where possible.

Use cross-device `cudaStreamWaitEvent` where supported by the current CUDA runtime. Do not add busy polling or CPU watcher loops.

Start benchmark-only / opt-in if necessary. Measure:

- exact fixed depth4 65K snapshot workload;
- identical digest / 32 rounds / 127 drafts / 93 accepted if execution numerics are unchanged;
- wall ms/round;
- TP coordinator ms/round;
- number and total wall time of host synchronizations;
- two-GPU busy-overlap fraction;
- useful TPS.

Success gate: **material** reduction in normalized round cost, preferably >=8-10%, with no correctness regression. A 1% micro win is not the target. If the full event DAG cannot recover substantial wall time, classify it CLOSED and move on immediately.

Before coding, read the old 17:48 async-output conclusion and state explicitly why this full input+engine+return+merge DAG is different.

## Then: hardware-cost-aware verifier geometry, not another fixed tiny tree

If coordinator work alone does not put C1 on a credible >=100 trajectory, use the existing proposal logs and measured T1..T6 verifier costs to do an OFFLINE oracle first.

Do not implement another arbitrary fixed tree.

For each real proposal position / round:

- calibrate proposal top-k probability/margin to observed target acceptance;
- allow branching at whichever uncertain position has the best expected marginal gain;
- score candidate tree expansion by expected additional licensed tokens divided by measured additional verifier cost;
- include real T4/T5/T6 target cost and long-context KVP scan amortization;
- compare predicted useful TPS against accepted fixed depth4 59.05 TPS;
- evaluate general/code/JSON/Thai/Japanese/Arabic, not one curated prompt;
- no token membership lists, rescue vocabulary, benchmark-shaped thresholds, or learned prompt-specific hacks.

Only implement a Bole/Sequoia/SMART-like tree engine if this offline hardware-aware oracle predicts a MATERIAL gain (e.g. >=15-20%) on held-outs. If the oracle cannot beat depth4, close the direction cheaply.

A materially different long-context speculative lane that remains open is a same-model sparse/retrieved-KV drafter: canonical recurrent/GDN state plus a cheap recent/retrieved subset for full-attention draft passes, with the full physical-KVP target doing exact verification. Do not implement until an offline acceptance/cost model says it can beat current MTP depth4 economics.

## Single GPU >=80 remains open and neglected

`perf/sm75-int8-tu102` has not received fresh measured work in this interval. Current representative frontier remains about ~70 TPS.

Do not reopen Q4-down, fixed T6, generic cache-carveout, or already-closed Q4/Q5 geometry sweeps. Once the dual event-DAG discriminator is launched or waiting on a long run, use spare engineering time for a genuinely new single-GPU mechanism: hardware-aware verifier budget/tree oracle, sparse-KV same-model draft, or a major measured byte/fusion elimination.

## Execution discipline

1. Preserve/commit the validated fixed-depth bootstrap correction after its existing regression gate.
2. Full event-DAG coordinator discriminator.
3. Matched 65K depth4 qualification from reusable snapshot.
4. Reprofile and classify.
5. If positive, promote and then measure populated 262K C1.
6. If insufficient, offline hardware-aware verifier/tree oracle before implementation.
7. Keep single >=80 lane alive.
8. Append every result to `progress.md` and immediately choose the next largest measured blocker.

Hard goals remain:

- single Core-X one-user >=80 useful tok/s;
- dual one-request >=100 useful tok/s;
- populated ~262K dual one-request >=100 useful tok/s.

Do not stop after one result. Do not wait for another agent. Implement the next bounded experiment yourself.
