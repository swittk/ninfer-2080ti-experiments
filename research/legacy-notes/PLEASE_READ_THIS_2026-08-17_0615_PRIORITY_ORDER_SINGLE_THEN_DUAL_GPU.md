# NINFER SM75 priority order — finish single-GPU exact work, then full-context dual-GPU

Date: 2026-08-17 06:15 Asia/Bangkok

This note is for the active coding/performance worker. The detailed continuity handoff is also at `.chatgpt/handoffs/current.local.md`.

## Do not reorder the goals just because a second GPU is now healthy

The owner wants the work done in this order:

1. Finish/reclassify the current model-agnostic B2 correctness-certificate experiment and establish what the actually portable single-GPU production lane is.
2. Use newly unlocked Nsight Compute hardware counters to look for clear remaining **exact/model-agnostic** single-GPU wins and take only evidence-backed opportunities.
3. Measure MTP draft widths 4/5 and keep adaptive/deeper MTP on the shortlist of larger algorithmic wins.
4. Only after the current single-GPU state is understood/preserved, move hard onto dual GPU.
5. Dual-GPU milestone #1: make the largest possible context work, ideally Qwen3.8 native **262144 tokens**, correctly.
6. Dual-GPU milestone #2: while preserving that full context, make both GPUs contribute useful inference work and then pipeline multiple users for high aggregate throughput.
7. Keep two-independent-replica mode as the default/sanity baseline for ordinary contexts/multiple users; it already measures ~136.6 tok/s aggregate with virtually no contention.

## Current single-GPU baseline / correctness situation

Reliability commit in sibling worktree `../ninfer-2080ti-exact72`:

`38369da — fix(sm75): harden q45g graph replay determinism`

Parent exact72 checkpoint:

`9ca3a4b — perf(sm75): checkpoint exact 72 tok/s proposal-B2 lane`

Cleaned portable/reference canonical 1024 run before certificate diagnostic was freshly reconfirmed at **72.07 tok/s**, canonical SHA and canonical MTP trajectory/stats.

Important subtlety: that 72-ish checkpoint still contains the older B2/top-128 approximate shortlist. It is heavily empirically validated on current Qwen3.8, but a fixed top-128 cutoff is not mathematically guaranteed for arbitrary fine-tunes/checkpoints.

B1/W1 token/row pin work is RESEARCH ONLY. It reached 73.38/73.11 on canonical but failed broader corpus/generalization and even dropped a true target winner on long-code. Do not reintroduce hardcoded token IDs or rescue rows into production.

## Current interrupted B2 certificate experiment

The only tracked dirty file in `../ninfer-2080ti-exact72` is:

`src/ops/linear/w8/w8_vocab_w4_a8_sm75.cu`

It implements a diagnostic Cauchy-Schwarz style certificate using load-time weight error metadata and runtime hidden-vector error norms. The intended production rule is correct: exact-rerank a B2 shortlist, accept only if every excluded row is provably unable to win; otherwise widen/fallback.

However the FIRST FORMULATION IS FAR TOO LOOSE.

Artifact:

`.chatgpt/portable_cert/cert256.*`

Observed:

- 256-token decode ~68.92 tok/s
- model/MTP behavior remained the expected one
- **392 B2_CERT_FAIL reports**
- real exact winners were usually ~20–30 score units
- excluded-row conservative upper bounds were commonly ~100–200

This means the current bound essentially cannot certify anything useful. Do not spend an open-ended session tuning the same formula. Either:

A. explain exactly why it is loose and try ONE clearly motivated tighter, still checkpoint-agnostic and cheap formulation with a credible path to useful pass rate; or

B. declare this certificate formulation impractical, remove/restore diagnostic code, and benchmark the genuinely trusted/full-exact alternative so we know the real performance price of universal portability.

Do not replace mathematical failure with heuristics.

## Nsight Compute is finally available

The machine was rebooted with:

`options nvidia NVreg_RestrictProfilingToAdminUsers=0`

Verified:

`RmProfilingAdminOnly: 0`

A normal-user NCU smoke test successfully read real SM75 counters including `dram__bytes.sum` and `sm__cycles_elapsed.avg`.

Once certificate work is classified, profile the clean 72-ish lane on the INTERNAL GPU and produce a ranked evidence-based optimization list. Measure dominant kernels, DRAM traffic/throughput, long-scoreboard/dependency stalls, LD/ST pressure, occupancy/register effects and compute utilization. Do not go back to blind geometry sweeps if counters do not support them.

Internal GPU UUID for all normal single-GPU work:

`GPU-020906b4-0904-3471-e4ee-e41ec2350be3`

Do NOT rely on CUDA device indices; eGPU hotplug changes enumeration.

## Healthy dual-GPU hardware now validated

Internal:

- UUID `GPU-020906b4-0904-3471-e4ee-e41ec2350be3`
- RTX 2080 Ti, 22528 MiB
- PCIe x16

Replacement Core X eGPU:

- UUID `GPU-2fffc240-5d58-fbc8-27d6-6b349448fda9`
- RTX 2080 Ti, 22528 MiB
- Razer Core X, TB3 40 Gb/s
- PCIe Gen3 x4 under load
- healthy boost ~1875 MHz, memory 7000 MHz
- no bogus SW-power-cap issue, no new Xid

Standalone same 256-token test:

- internal ~68.89 tok/s
- eGPU ~68.56 tok/s
- byte-identical output / matching MTP behavior

Simultaneous independent full-model workers:

- eGPU 68.47 tok/s
- internal 68.15 tok/s
- aggregate ~136.6 tok/s
- effectively no contention

CUDA peer access is unavailable BOTH directions. `nvidia-smi topo -m` reports NODE, no NVLink. Therefore do not build chatty tensor parallelism or remote-KV reads.

## Dual-GPU owner goals and architecture

Owner priorities:

1. Maximum usable model context, ideally native Qwen3.8 **262144**.
2. Serve multiple people with extremely high aggregate throughput.
3. Preserve high single-user speed.
4. Exact/model-agnostic correctness across fine-tunes/checkpoints.

Qwen3.8-27B implementation facts:

- 64 layers
- 16 full-attention layers
- 48 GDN layers
- hidden width 5120
- native context 262144
- maximum MTP draft tokens 5

For maximum-context mode, prefer coarse **layer/pipeline ownership**:

- GPU A permanently owns one layer subset + those weights + those layers' KV + GDN/recurrent state.
- GPU B permanently owns the other subset + its weights + its KV/GDN state.
- Transfer only stage-boundary hidden activations using pinned host staging because P2P is unavailable.
- Choose split using actual weight bytes, attention KV bytes, GDN state, runtime reservations, CUDA Graph memory/workspace and measured stage time; do not blindly split 32/32.

FIRST make a single session run the largest feasible context, ideally full 262144, correctly. Functionality/capacity first.

THEN optimize the full-context path: stage balance, host-transfer overlap, and multi-user pipeline scheduling so one request can be on GPU B while another is on GPU A. Compare against replica baseline at 1/2/4/8 users.

For ordinary contexts/multi-user service, independent replicas are already an excellent default because each model/KV stays completely local and TB3 disappears from decode hot path.

## Other required portability/perf checks

- Add Thai and Japanese (plus another non-Latin/multilingual case) to portability/certificate/fallback testing. Language may affect certificate/fallback frequency, never correctness.
- Measure draft-token widths 4 and 5 before assuming T3 is optimal. Coding/JSON historically have near-100% acceptance and may benefit dramatically from wider/adaptive MTP.
- Adaptive MTP and later tree/suffix speculation are higher-upside algorithmic directions after basic width economics are known.
- Do not use board OC, memory OC/forced clocks, or raised power limits as the primary path.

## Preserve

Preserve `38369da` as the reliability baseline. Checkpoint any validated model-agnostic performance improvement separately before opening another experiment. Do not restart endless q45g torture runs unless a new mismatch appears.
