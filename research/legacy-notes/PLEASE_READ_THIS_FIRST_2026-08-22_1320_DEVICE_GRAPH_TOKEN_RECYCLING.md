# 13:20 adviser — bounded Lookahead, device-resident MTP loop, Token Recycling

Read `AGENTS.md` first. Stock GPUs only. No hardware settings/reset/reboot. No agents. No behavioral training. Hard gates remain **80 / 100 / 100 / 100 useful complete-path tok/s**.

## Stop state / current authority
At 13:11 both GPUs were stock-idle and no NInfer/build/research producer was live. Last worker activity was ~12:25 in `p0_aq_lookahead`: official LookaheadDecoding was cloned and only `general_iter1` of an oracle was run. This is a real stop.

Current gates have NOT moved: single ~**76.45–76.5**, best prior fixed experimental P0-M Q4G64/Marlin factory-MTP **77.97**, dual short ~**65.6**, populated65K **58.75**, genuine262K **32.76** t/s.

Recent closures: P0-AH Q4G128/Marlin 74.48 vs 76.45; corrected B1/K256 six-family mean 64.04 vs 64.23 (quality okay, code loss); QuantSpec best trace ceiling 45.28; Humming slower than current fused kernels; QServe core ISA SM80+ and Turing residual already slower; SDFP impossible 42-FFN draft asymptote 55.68; two-replica late-fork ceiling 74.44; full-W4 head 60.98 vs 64.20 with multilingual pathology; proposal candidate-cache and request-local active vocab collapsed acceptance.

## P0-AQ — finish the OFFICIAL Lookahead oracle, but bound it before porting
Prior naive Jacobi/shift is closed. Official LADE is more specific: a 2D Jacobi window, cached n-grams from actual Jacobi trajectories, and simultaneous lookahead + verification branches. Upstream is already under `p0_aq_lookahead/upstream`.

Do not port Python. Finish a bounded oracle: actual upstream n-gram/window semantics; chronological state only; first general/code/JSON, then all six only if credible; charge measured NInfer target-width cost; keep total nodes <=32 initially; compute useful tokens/round and complete-path t/s for 8/16/24/32 nodes. If the optimistic costed ceiling cannot clear 80/100, close immediately. Upstream: https://github.com/hao-ai-lab/LookaheadDecoding

## P0-AR — HIGHEST PRIORITY exact-system idea: device-resident MTP CUDA-Graph round loop
Fresh profile: useful yield ~127/43 = **2.9535/round**; CUDA-Graph body median ~**37.001 ms/round**; complete round ~38.5–39.2 ms. Graph-body-only ceiling is already ~**79.82 t/s**.

Current MTP does every round: host fills ingress -> `cudaGraphLaunch` -> `device.synchronize()` -> host reads acceptance/next drafts -> mutates `SequenceState` -> repeat. The graph is prepared and is not graph-updated every round.

CUDA 13.1 explicitly supports device graph launch, tail launch and tail self-launch (`cudaGetCurrentGraphExec()` + `cudaStreamGraphTailLaunch`). Docs: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#device-graph-launch

First do a tiny feasibility probe, no engine surgery: compile/run a 2-3 iteration device-graph self-tail-launch on stock SM75. If unsupported, close. If supported, inspect current MTP graph node types and try instantiating the otherwise-current captured MTP graph with `cudaGraphInstantiateFlagDeviceLaunch`. Reject early if an incompatible node is unavoidable.

If supported, build a benchmark-only one-request device-resident controller: pre-materialize fixed 8K KV/page capacity; move frontier/anchor/extents/drafts/rope/budget/accepted metadata to device control buffers; append a final controller kernel that consumes egress, updates next-round ingress, and tail-self-launches while work remains. Host launches once and synchronizes once for the discriminator. Preserve speculative state semantics. Product streaming can later use mapped-pinned ring publication; do not make streaming plumbing prerequisite to timing.

Success: >=~1 ms/round whole-path saving. Measure graph body and E2E.

### If P0-AR wins, REOPEN P0-M as a JOINT PACKAGE
P0-M Q4G64/maintained-Marlin factory MTP already measured **77.97 t/s** complete path and was closed only as a standalone below-80 package. Device-resident round chaining is materially distinct. If P0-AR removes the host gap, combine P0-M + P0-AR and run real stock E2E; the old standalone ceiling does not forbid this.

## P0-AS — Token Recycling + cost-aware OPT-Tree, preferably factory-MTP spine + recycled branches
Token Recycling (ACL 2025 Outstanding Paper) stores the target model's otherwise-discarded top-k candidate predictions in a tiny token->successor adjacency table, BFS-builds a speculative tree on recurrence, and target-verifies the tree. No draft model, no datastore, no training; published work reports ~2x on tested models. Paper: https://aclanthology.org/2025.acl-long.338/ Code: https://github.com/Luowaterbi/TokenRecycling

OPT-Tree chooses a dynamic tree maximizing expected accepted length under a node budget. Paper: https://aclanthology.org/2025.tacl-1.8/ Code: https://github.com/Jikai0Wang/OPT-Tree

This is NOT the failed proposal candidate-cache/active-vocab family. Those reused proposal-head rows/candidate sets and reranked autoregressively. Token Recycling caches **target transition predictions** and uses them as speculative branches, then target-verifies the tree.

NInfer already has `include/ninfer/ops/gqa_tree.h` + `src/ops/launcher/gqa_tree.cu`: exact parent-indexed GQA tree-tail attention for **1..32 nodes**, with ordered merge into committed-prefix state. It appears unused in production. Reuse it.

Best likely variant: keep current factory-MTP 3-token path as the primary spine, attach a few historically high-probability target-transition alternatives from the online adjacency cache, and verify one <=32-node tree. This keeps current strong MTP yield while using previously computed target information to rescue misses with no draft-model work.

Oracle before runtime:
1. diagnostic-capture target top-k=8 at each verified position + licensed next token over six-family 256-token traces;
2. replay chronologically, updating adjacency only after that target result is available (no future leakage);
3. simulate plain Token Recycling and MTP-spine+TR at budgets 8/12/16/24/32;
4. compute static TR and OPT-Tree/cost-aware variants;
5. charge **measured NInfer target T=N cost** + real fold/state/control, not free verification;
6. report useful tokens/round, nodes/round, complete-path t/s; implement only if the costed oracle clears single80; separately prove a 65K/262K ceiling before long-context porting.

## P0-AT — FR-Spec only as a small additive mechanism
FR-Spec frequency-ranks a fixed general-purpose draft-vocabulary subset, reducing large draft LM-head work while target verification stays full. No retraining; target output distribution is preserved. Paper: https://arxiv.org/abs/2502.14856 Code: https://github.com/thunlp/FR-Spec

It is distinct from static token-ID prefix truncation, request-local active vocab, and B1. A July-2026 llama.cpp research issue tested it on **Qwen3.6-27B native MTP with a 248,320-token head**: draft LM-head kernel time fell ~85%, but public mixed E2E rose only ~83.9 -> 85.1 t/s (~1.4%). https://github.com/ggml-org/llama.cpp/issues/25187

So first derive the local proposal-head complete-round ceiling; likely too small alone. Only implement after a larger mechanism or if the local ceiling says otherwise. Any frequency map must be fixed/general-purpose from public statistics, never tuned on qualification prompts/private code.

## Priority / autonomy
1. Finish or cheaply close already-started **P0-AQ official Lookahead oracle**.
2. **P0-AR device-resident graph self-launch feasibility + E2E** — highest-priority single80 route.
3. If P0-AR wins, real stock **P0-M + P0-AR** joint E2E; old standalone P0-M closure does not forbid this.
4. **P0-AS Token Recycling / OPT-Tree / MTP-spine tree oracle**, runtime only if costed yield clears a gate.
5. FR-Spec only as small additive work after a larger win.
6. Continue autonomously; do not stop after another lane conclusion. Recompute 80 / 100 / 100 / 100 after accepted work and move to the next materially distinct mechanism.
