# OWNER NEXT PRIORITIES — DO NOT STOP AT TEXT 262K100

The 262K text gate is now achieved by Q64/Q65 (~104.5 tok/s genuine261900), but this is **not the end state**. Continue directly, sequentially, with NO SUBAGENTS.

## P0 — finish unified multimodal >=100, no benchmark games
Q65 unified candidate currently measures:
- text genuine262: 104.51 tok/s, representative quality pass
- image: 105.28 tok/s
- mixed image+video URL: 101.53 tok/s
- video: **98.97 tok/s**

The video miss is only ~1%, so finish it properly. Do not manipulate max-new/prompt or truncate output merely to cross 100. Profile the actual unified video run against the previously accepted Q23 video lane and identify the regression (target stage0/stage1, proposal, vision/prefill side effects, clocks, split, local-retained-split applicability, launch geometry). Recover >=100 on the same meaningful video workload while retaining coherent temporal facts.

The current Q65 scorer incorrectly marks image/mixed `control=true` if the only special is the normal terminal `<|im_end|>`. Fix the evaluator to distinguish a terminal stop token from embedded/control-token leakage; do not change model output to satisfy a broken scorer. Re-run image/video/mixed and preserve actual decoded text + token IDs.

After the basic three pass, include at least one multi-image and one non-English/mixed-media smoke if practical. Final label must state exactly what was qualified.

## P1 — make Q64/Q45 a real production path, not only copied candidate evidence
Q64 recert is real: 104.53 median genuine262, deterministic, representative quality pass. Q65 reproduces 104.51. But candidate sources were restored after copied builds.

Integrate the retained-local-split mechanism and exact FP16-staged PV path cleanly behind architecture/context-aware production dispatch. Requirements:
- short path must not regress;
- 65K >=100 and genuine262 >=100 retained;
- multimodal path uses the same optimized text decode after visual prefill;
- no hardcoded prompt/token/checkpoint rescue behavior;
- absent eGPU still leaves single-GPU path healthy;
- tests/build and reproducibility evidence.

Do not leave the final result dependent on `.chatgpt/.../bin` diagnostic binaries.

## P2 — REAL concurrent serving: 1 / 2 / 4 / 8 users
The runtime accepts `--sessions 1..8`, but optimized MTP batching is currently specialized for B2/B4. **Eight users are not performance-qualified.** Treat this as a product goal now.

Use realistic help-desk / EMR-like mixes, not eight identical lockstep prompts only. Benchmark at least:
- short ~2K context
- normal ~8K context
- medium ~16K or 32K if memory allows
- 1, 2, 4, 8 active sessions

Report:
- per-user useful tok/s distribution, not only aggregate
- aggregate tok/s
- TTFT
- p50/p95 inter-token gap
- fairness / starvation
- memory per GPU
- prefill interference with ongoing decodes
- quality/determinism appropriate to the selected exact/Q6 policy

Target for 8 simultaneously busy help-desk users: >=20–30 tok/s/user is already useful; >=30–50 each is excellent. Do not claim 8-user capability from capacity alone.

Fresh implementation direction if B8 is weak:
- request-aware persistent/ragged work scheduling for multi-request GQA so idle `(request, head, split)` rectangles are not launched;
- batchwise cross-request pipeline overlap, not the previously failed thousands-of-events per-layer DAG;
- extend MTP batching beyond the B2/B4 special cases only after measuring where B8 actually loses time.

Keep long-context **single-request** Q64 local-split logic separate from multi-request B8 scheduling; they solve different underfill problems.

## P3 — headroom beyond 100, after P0/P1 and preferably after first B8 baseline
100 is a floor, not a ceiling. Once all product gates are clean, keep pushing measured stock-card performance.

Prioritize mechanisms that provide broad reusable headroom rather than prompt-specific approximation. Re-profile the integrated Q64/Q65 production path before choosing work; old pre-Q45/Q64 attribution is stale.

Useful targets:
- dual short and multimodal materially >105 / ideally 110+
- 65K already ~110.9; preserve or improve
- genuine262 move from ~104.5 toward 110+
- single current ~107; seek 110–120+ only if mechanisms are portable and quality-safe
- multi-user aggregate/per-user improvement may be more valuable than another +3 tok/s B1

Do not reopen closed mechanisms without a genuinely different reason. In particular no generic KV4/sink-tail/tree/wider-MTP/static-FFN/column-split/graph-theatre reruns.

## NO SUBAGENTS — AGAIN
No `codex exec`, additional Codex workers, code-mode delegated agents, Claude/OpenCode, autonomous research wrappers, or agent loops. The primary worker does this sequentially itself. Ordinary build/benchmark/Python processes are fine.
