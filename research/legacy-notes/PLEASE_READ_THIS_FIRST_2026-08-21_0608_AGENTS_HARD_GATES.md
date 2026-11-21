# STOP / OWNER OVERRIDE — 2026-08-21 06:08 +07

**READ REPO-ROOT `AGENTS.md` BEFORE DOING ANYTHING ELSE.**

The owner hard gates are now written into `AGENTS.md` and are mandatory. They override looser older notes.

- single RTX 2080 Ti >=80 useful decode tok/s
- dual one-request short/moderate >=100 useful decode tok/s
- dual one-request populated ~65K >=100 useful decode tok/s
- dual one-request genuinely populated ~262K >=100 useful decode tok/s
- current ~72 single / ~65-66 short dual / 58.75 at65K are intermediate failures/frontiers, NOT completion
- useful tok/s = real complete-path emitted/accepted output, not draft/kernel/verifier/aggregate throughput
- NO Codex / codex exec / Claude / OpenCode / sub-agents / agent loops / autonomous coding wrappers
- NO local behavioral model training, finetuning, distillation, adapters, interpolation, custom MTP/DFlash weights
- factory Qwen3.8 native MTP + professionally published upstream exact-target drafts + mechanical conversion only
- do not call one GPU permanently healthy/unhealthy; eGPU/driver boost asymmetry is runtime state, calibrate dynamically
- same-family Q4/Q5 warp/tile/cache/prefetch micro-tuning is exhausted unless a materially new measured hypothesis exists
- materially different kernel/dataflow families remain open: benchmark upstream executable oracles (e.g. SM75 Marlin/AWQ) before writing kernels
- factory native MTP inference/system optimization and short-dual >=100 are P0
- DFlash2 exact-Qwen3.8 sidecar is CLOSED for current NInfer integration economics; do not tune it locally
- when one lane closes, continue directly to the next live mechanism; no watcher/wait state

Any workflow that launches a coding/reasoning agent is a hard violation even if the invocation fails.
