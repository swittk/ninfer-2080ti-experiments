# OWNER HARD GATE OVERRIDE — STRICT C1 >100 REQUIRED; RELAXED 107 DOES NOT COUNT

Effective immediately, the owner's single-GPU/single-user gate is **STRICT C1 D3 >=100 useful tok/s** on one stock RTX 2080 Ti with acceptable output quality. Strict means `greedy_margin_delta=-1`; C1 remains D3. The current strict matching-fixture floor is ~79/80.5 tok/s, so the gate is OPEN and performance work must push upward from that strict lane.

The ~107 margin7 route is preserved as historical/experimental evidence only. It has **ZERO owner C1 gate credit** because same-prompt long Thai shows severe language soup versus the clean strict control. Language soup, comparable semantic drift, pathological repetition, or quality degradation is a **HARD FAIL regardless of tok/s**. Do not call 107 a passed single-user gate, do not use it as the C1 north-star floor, and do not substitute policy-assisted throughput for strict-path kernel/runtime progress.

Immediate priority: **strict C1 D3 architecture/runtime efficiency**. Target >=100 first, then continue toward 120/150/200+ if physically real. Preserve margin7 and all prior experiment paths, but do not spend the primary campaign optimizing a relaxed lane while strict remains below100. C4=D1 and physical C8=C4+D1 per GPU x2 remain unchanged. Finish only already-bounded validation from another lane; next architectural campaign is strict C1.

Machine authority: `tests/retained_sm75_profiles.json` + `scripts/check_retained_sm75_profiles.py`. Primary prose authority: `PLEASE_READ_FIRST_2026-09-08_NINFER_HEADLINE_FRONTIERS.md`.

# READ THIS FIRST — HEADLINE FRONTIERS ARE CUMULATIVE

**Primary authority:** `PLEASE_READ_FIRST_2026-09-08_NINFER_HEADLINE_FRONTIERS.md`. Machine guard: `tests/retained_sm75_profiles.json` + `scripts/check_retained_sm75_profiles.py`.

Non-negotiable C1 distinction: owner strict/current P0M **79.01**, older distinct certified SINGLE80 **80.185**, accepted D3 margin7 **107.27** (independently re-hit 107.255). Do not collapse them. New full-target `fast-full-vocab-draft` work has **no strict79 headline credit yet** because it is a different proposal/target route; matching-fixture A/B is required. Preserve dual-short/65K/262K, multimodal, physical-C8, long-media and prefill/media-cache headlines from the ledger as cumulative evidence.

# READ FIRST — AUTHORITATIVE C1/D3 TOPOLOGY + BOUNDED SM75 KERNEL CAMPAIGN

Owner correction, 2026-09-08. **This note supersedes any ambiguous wording in older Sep-8 notes that treats C1/C2/C4/C8 as peer topology choices or implies D1 is the C1 serving mode.** Do not re-derive settled topology unless the owner explicitly changes it.

## Settled canonical serving topology — treat as facts

1. **C1 = one stock RTX 2080 Ti + one real user/request = native factory MTP D3 (`draft_tokens=3`) canonical. ALWAYS.**
   - D1 at C1 is only a diagnostic/control. It is not the production baseline and must not be used to judge whether C1 performance has regressed.
   - Do not run another C1 D1 sweep to choose the width. The width choice is already settled.
2. **C4 per GPU = D1 canonical.** At C4, deeper speculative width competes with real cross-request batching; this fact does NOT propagate down to C1.
3. **Physical C8 production = C4+D1 on GPU0 plus C4+D1 on GPU1.** One-card C8/T16/T32 experiments are research/oracle work only, not the production topology.
4. **Raw one-GPU C1 useful decode tok/s is the owner's primary kernel-performance north star.** C4/C8 aggregate wins are valuable production-serving gains but cannot substitute for C1.

## Historical C1 authority to reproduce, not reinterpret

Authoritative receipt:
`.chatgpt/post_reboot_20260821/p0_q4_single_delta7_p0m_recert/final_result.json`

Same retained Qwen3.8 artifact family used today. Fixed endpoint:
- prompt: `Write a concise technical explanation of CUDA memory bandwidth, quantized transformer inference, and speculative decoding.`
- output tokens: 128
- **D3**
- runs: 107.45 / 107.33 / 107.21 / 107.12 tok/s
- median **107.27 tok/s**
- each run 33 target rounds, 99 drafted, 94 accepted (~94.95%)
- extended fixed run 107.55 tok/s, 66 rounds, 198 drafted, 189 accepted

The broader six-family receipt is workload-dependent: general ~107.555, code ~107.27, JSON ~103.9, Thai ~72.68, Japanese ~89.045, Arabic ~88.895. Therefore 107 is a real fast general/code C1 frontier, not a claim that every language/workload is 107.

The Sep-8 ~42 tok/s C1 measurements used **D1**, long 1024-token generation and ~70% acceptance (~1.70 useful tok/round). They are valid D1 production controls but are NOT an apples-to-apples C1 regression result.

## Immediate execution order — do not improvise around this

0. If a C1 D1 run was already in flight when this note was read, let that one finish and label it `D1_CONTROL_ONLY`. **Do not launch another C1 D1 run.**
1. **Recover the current-source D3 C1 baseline first.** Reproduce the historical fixed endpoint as faithfully as possible using the current binary/source, same artifact and same stock GPU/device. Preserve the old measurement boundary. Then run an equivalent real HTTP/Engine D3 C1 receipt so benchmark and production path are both visible.
2. Report separately:
   - useful output tok/s,
   - target round time,
   - drafted/accepted tokens and acceptance,
   - prompt/output length,
   - exact measurement interval/path.
   Do not mix yield changes with kernel-time changes.
3. If the historical-style D3 result is materially below the old ~107 family, **stop new kernel invention temporarily and trace the regression first** (mode, graph path, head path, KV/context, Vision allocations, scheduler/Engine overhead, active small-T shapes, source changes). Recover the known capability before claiming a new ceiling.
4. Once current D3 C1 is established, **freshly profile the actual D3 C1 path** with Nsight/NCU. Do not substitute C4/T8 profiles. Identify the exact token widths/shapes D3 exercises and current Q4/Q5/GDN/proposal/verification costs.
5. Then pursue at most **one or two materially different C1-relevant architectural candidates**, not a giant knob sweep. Good questions include:
   - Can packed Q4/Q5 fragments live long enough to feed more of the real D3 verification/proposal work before reload/dequantization?
   - Can the exact small-T path used by D3 exploit a different load/dequantize/consume organization without register/shared-memory collapse?
   - Does the already-observed **exact ~1.12x Q5 native-row T2 discriminator** overlap a real D3 C1 shape? If yes, test the relevant real shape; if no, do not force it.
   - Can Q4/Q5/GDN traffic or boundaries be fused/removed in the real D3 path while preserving the established arithmetic/trajectory contract?
6. Every candidate must pass numerical/output/trajectory qualification appropriate to the existing contract, then **real production Engine C1 D3 A/B**. Microbench speedup without C1 movement is not a kernel win for this campaign.
7. A C1 keeper must then be checked for C4+D1 and physical C8 regressions. A C4/C8-only keeper may still be retained separately, but label it `BATCH_SERVING_WIN_NOT_C1`.
8. The current Q5 `partition4`/`NINFER_SERVE_Q5_BATCH` work belongs in that separate bucket unless it proves a real D3 C1 gain. It has useful C4/C8 evidence; do not let it distract from the C1 campaign.
9. **Bound the kernel campaign.** After the D3 baseline is correct, if fresh D3 profiling plus one or two genuinely different architectural attempts fail to move real C1, stop kernel work for now and return to retained-win canonicalization / remaining production-serving gates. Do not spend hours re-sweeping already-dead tile/prefetch/lookahead geometries.

## What counts as success

The goal is not merely to preserve 107. **107.27 is a known floor/reference for the fast C1 family.** Pursue 120, 150, 200+ if genuinely achievable on the same stock hardware/model/quality path. A large C1 gain is the strongest evidence that the inference primitive itself became better rather than aggregate throughput merely benefiting from batching.

## Hard boundaries

- No agents/subagents, no training, no adapter/drafter model.
- No PCI/reset/unbind/rebind, driver/module reload, reboot, clock/power/voltage changes without explicit owner approval.
- No benchmark cheating, semantic degradation, cached generated-answer replay, or workload-specific copy acceleration presented as general C1.
- Preserve inherited uncommitted work. No destructive cleanup.
- No commit/push/default promotion/deployment unless explicitly requested by owner.
- Qualified retained wins must eventually be canonicalized; do not strand them in this dirty continuation worktree.


# OWNER HARD STOP — EXACT C1 D3 RECEIPT FIRST (2026-09-08)

The worker has now switched to D3 but is STILL skipping the required first step by running 768-token corpora / separated-head experiments. **STOP THAT ORDERING.**

Before ANY further C1 corpus, head-mode comparison, Nsight/NCU, Q4/Q5/GDN candidate, C4/C8 qualification, or kernel experiment, reproduce the authoritative historical fixed endpoint as faithfully as current source permits:

- one stock RTX 2080 Ti, one request / C1
- same retained artifact family
- native factory MTP **D3 (`draft_tokens=3`)**
- prompt EXACTLY: `Write a concise technical explanation of CUDA memory bandwidth, quantized transformer inference, and speculative decoding.`
- output tokens EXACTLY **128**
- greedy/no-thinking as historical route requires
- Q4G64 MTP + Marlin Q4, int8 KV
- historical route includes loose logit delta 7; preserve the equivalent current setting if the current binary exposes it
- run **four repeats** and report each tps, target rounds, drafted, accepted, by-position acceptance/digest if available, and the exact measurement boundary
- compare against historical 107.45 / 107.33 / 107.21 / 107.12, median 107.27, each 33 rounds / 99 drafted / 94 accepted

Do NOT substitute a 768/1024-token corpus, a different prompt, D1, C4/C8 aggregate, or a head-mode experiment for this receipt. The current D3 long-corpus results (~45–74 tok/s depending prompt) are useful workload data but DO NOT answer whether the known 107 C1 capability still exists.

If the exact fixed endpoint materially misses the old 107 family, freeze new kernel invention and trace the regression first. If it reproduces, then proceed to the bounded D3-specific kernel campaign.

`--lm-head-draft` / separated-head experiments are not the next task unless the exact historical reproduction itself proves that head-path mismatch is the cause.
