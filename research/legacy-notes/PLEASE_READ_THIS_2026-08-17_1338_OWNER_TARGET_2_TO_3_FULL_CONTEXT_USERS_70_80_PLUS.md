# OWNER TARGET UPDATE — 2–3 CONCURRENT NATIVE-CONTEXT USERS, >=70–80 TOK/S EACH; 100+ IS THE STRETCH TARGET

**2026-08-17 13:38 +07 — this refines the dual-GPU performance target.**

The owner does not merely want one 262,144-capacity request to run. The serving target is:

- **minimum useful target:** 2 concurrent users/sessions, each native-full-context capable, around **70–80 tok/s each** where practical;
- **desired target:** support **2–3 concurrent users/sessions** at that class of throughput;
- **stretch target:** **100+ tok/s per active user** if the hardware/algorithmic ceiling allows it;
- retain ordinary-context replica mode and the ~72 tok/s single-GPU checkpoint-qualified turbo lane;
- do not require the eGPU for normal one-GPU operation.

Distinguish carefully between **262K capacity** and **262K actually occupied history** when reporting throughput. The latter has much larger attention/KV bandwidth cost. Benchmark both useful short/moderate histories and genuinely long populated KV lengths.

## Memory is now part of the performance problem

The current BF16 dual milestone allocates about 8.14 GiB sequence state per GPU for one 262,144-capacity session, plus about 7.96 GiB weights. Duplicating that layout cannot fit 2–3 fully allocated sessions.

NINFER already supports **INT8-G64 KV** and upstream docs explicitly use `--kv-dtype int8` for 262,144 context. The dual runtime currently hardcodes BF16; change the dual/session design so INT8 KV is a first-class option and measure it.

A rough layout estimate from the existing format (I8 codes + FP16 scale per group64) puts one 262K shard state near ~4.3 GiB rather than 8.14 GiB before MTP-specific extras. That makes 2 sessions comfortable and 3 potentially feasible but tight on a 22 GiB card. Measure actual allocations; do not rely only on this estimate.

Do not create a complete duplicate model/stage per user. Use shared stage weights and a multi-session paged-KV/state pool/table-row design where possible, so extra users mainly cost per-session state rather than another ~8 GiB of weights per GPU.

## NCU is NOT globally finished

The short-context leaf-kernel campaign did enough to close broad blind Q4/Q5 geometry sweeps:

- Q4 A8 streaming kernels are already close to the SM75 DRAM ceiling; several prefetch/shared/prod-cons ideas failed;
- Q5-down likewise showed diminishing returns;
- GDN Q5 PC4 produced a real bounded win;
- Q5-6144 needs different scheduler/cache thinking and one high-register expansion already failed.

However, **do not interpret `SINGLE_GPU_FRONTIER.md` as “no more NCU work anywhere.”** The owner's target changes the relevant workload.

Before declaring the dual/full-context performance frontier, profile the kernels that dominate at realistic long populated KV lengths, especially:

- GQA decode at representative live windows such as ~8K, 32K, 64K, 128K, and near 262K;
- **INT8-G64 KV** attention separately from BF16;
- small-T/MTP verification attention at the widths actually used in production;
- multi-sequence/batched attention where 2–3 independent sessions are active;
- output-head/B2 cost on the second stage;
- boundary copy/synchronization/TB3 overhead after async overlap is implemented;
- stage utilization and memory-bandwidth counters during real two-/three-user pipeline runs.

At long context, attention/KV traffic can become a different bottleneck from the short canonical MLP-heavy profile. Use NCU to discover that bottleneck rather than continuing to polish already-saturated Q4 leaves.

## Throughput architecture matters more than tiny leaf wins now

For 2–3 users, prioritize mechanisms that improve useful tokens per weight/KV read:

- restore MTP3/B2/turbo semantics into the sharded path after state correctness is stable;
- batch/microbatch independent sessions through a stage so model weights are reused/amortized across users rather than reread independently for every user;
- pipeline stage0 of one request against stage1 of another;
- async/double-buffer the 10,240-byte hidden boundary and remove unnecessary host synchronization;
- chunk/pipeline prefill;
- evaluate adaptive MTP depth/width if it raises useful tokens per round on suitable workloads;
- consider wider B2 shortlist/adaptive shortlist or B3-like screening only if measured shortlist recall or head cost shows that is the limiting factor, not as a distraction from the dual pipeline.

The current split sweep shows the stage timings can be balanced; a split around the mid/high 30s should be selected by measured end-to-end stage time and memory headroom, not fixed dogmatically at 32/32.

## Acceptance / reporting

Do not claim success from aggregate tok/s alone. For 1/2/3/4 active sessions report:

- per-user decode tok/s;
- aggregate useful tok/s;
- actual populated context length per user;
- allocated context capacity per user;
- KV dtype and per-session state bytes;
- MTP rounds/acceptance/useful tokens per round;
- GPU utilization and DRAM bandwidth per stage;
- stage0/stage1/head/boundary time;
- memory headroom on both GPUs.

The project target is now explicitly **multi-user native-context serving**, not just a one-session capacity demo.
