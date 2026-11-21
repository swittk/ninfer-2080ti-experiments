# OWNER INTENT AFTER THE 262K MILESTONE — NOW MAKE IT FAST, KEEP EGPU OPTIONAL, KEEP THE ~72 TURBO LANE

**2026-08-17 13:22 +07 — read before continuing dual-GPU work.**

The native-capacity milestone is now achieved. `8268050` proves the important architectural point: two healthy 22 GB RTX 2080 Ti cards can split Qwen3.8-27B, each hold about 7.96 GiB of weights plus about 8.14 GiB of local state, and run at `--max-context 262144` with capacity-invariant output on the short gate.

That means **capacity/correctness was phase 1 and is now sufficiently proven to move on.** Do not spend the rest of the project repeatedly re-proving that 262K allocates unless a later change creates a regression.

## The owner's actual serving modes / purpose

There are THREE useful modes. Do not collapse them into one.

### 1. One GPU available: fast normal serving

The machine must continue to work perfectly when the Thunderbolt/eGPU is unplugged.

The owner's preferred Qwen3.8 checkpoint-qualified serving lane is the separately validated **~72 tok/s MTP3+B2/turbo lane**. The clean/general `a15dbc4` / ~65.7 tok/s path is useful as a portable/reference kernel baseline, but it is NOT permission to forget or replace the ~72 tok/s qualified serving lane.

Single-GPU operation must never require the presence of the second GPU or the dual runtime.

### 2. Two GPUs available, ordinary contexts: replica mode

For ordinary/moderate contexts, two complete independent model replicas remain extremely useful. We already measured roughly **136.6 tok/s aggregate** with the two healthy GPUs simultaneously. Eventually the qualified turbo/B2 path can be used independently on each replica where appropriate.

Do not replace this mode with sharding merely because two GPUs exist.

### 3. Two GPUs available, huge context: sharded full-context mode

This is the purpose of `ninfer-dual`: one logical request can use both cards to reach the complete native 262,144 context.

This mode should be **opt-in / explicitly selected**, not an assumption that every boot has an eGPU attached.

The current `--device0`, `--device1`, and `--split-layer` parameterization is directionally correct. Do not bake in the current CUDA ordinal ordering, PCI address, or eGPU presence. CUDA ordinals may move across boots. Stable UUID selection/mapping is preferable for the finished user-facing interface.

If the second GPU is absent, normal single-GPU `ninfer` must remain unaffected.

## NOW: performance phase for sharded mode

The initial dual T1 path is deliberately synchronous and only about ~23.4 tok/s on the short gate. That was acceptable for proving the architecture. It is **not the performance destination**.

The current feed path serializes:

1. stage 0 compute;
2. device->pinned-host residual copy;
3. synchronization;
4. pinned-host->device copy;
5. stage 1 compute;
6. synchronization.

Therefore it is completely expected that `nvidia-smi` may show mainly one GPU busy at a time. That is a property of the correctness prototype, not the desired final behavior.

Now use measurements and engineering judgment to make sharded mode fast. Good directions include, but are not limited to:

- remove unnecessary per-token host synchronizations;
- async D2H/H2D on dedicated streams/events;
- double-buffer the 10,240-byte residual boundary;
- chunk/pipeline prefill instead of T1-replaying every prompt token forever;
- overlap stage work across independent requests/sequences so both GPUs stay occupied;
- rebalance the split based on measured stage time, weights, KV/state and head asymmetry rather than worshiping 32/32;
- CUDA graphs where useful;
- reintroduce MTP/speculation once the two-stage state path is stable;
- reintroduce the qualified B2/turbo ideas where they make architectural sense;
- benchmark 1/2/4/8 concurrent users and report per-user + aggregate throughput.

Do not demand ordinary-T1 bitwise identity from optimized batched MTP if upstream NINFER's own numerical/state acceptance contract would accept the difference. Preserve hard state/mask/KV/GDN/commit/rollback correctness and explain numerical forks rather than forcing all fast kernels through T1 arithmetic.

## Performance interpretation

- **~72 tok/s** remains the protected/current-checkpoint normal single-GPU serving reference.
- Experiments may be slower; slower experiments do not replace it.
- The ~65.7 clean/general branch is a useful engineering baseline, not the owner's desired speed ceiling/floor for the qualified Qwen checkpoint.
- Full-context sharding was allowed to start slow because it unlocks capacity a single GPU cannot provide.
- Now that 262K works, performance is the priority for that mode.
- Later, pipeline/multi-user scheduling should aim to have both GPUs doing useful work concurrently rather than alternating idle periods.

## Do not get trapped

Do not turn this into another all-day exact-parity archaeology project. Do not spend weeks polishing a synchronous 23 tok/s T1 implementation if MTP, overlap, chunked prefill, or scheduling can move the real serving path much faster. Likewise, do not hardcode prompt/token-specific rescue behavior just to win one benchmark.

Follow the trajectory autonomously:

**262K capacity is proven -> optimize the sharded path -> bring back safe fast-path ideas -> benchmark real serving -> preserve optional single/replica/sharded modes.**
