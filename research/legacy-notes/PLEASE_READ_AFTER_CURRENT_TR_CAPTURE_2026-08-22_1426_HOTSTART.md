# P0-AS correction after current canonical capture — published TR hot-start is missing from the cold oracle

Do not interrupt the current capture. Read this before classifying Token Recycling.

The earlier 97.6-100% `topk_validation` is **not prediction accuracy**. It only validates that the mechanically reconstructed W8 top-1 from the same verifier hidden state reproduces the already-observed fast-MTP target decision. The real cold chronological TR result is the relevant one: MTP-spine N8/OPT ~61.52 representative t/s under already-favorable target costs.

However, the published ACL-2025 Token Recycling method is explicitly **hot-started across queries**. The paper says a zero matrix has weak early predictive capability and inherits an existing matrix. Its Table 3 ablation reports AR 54.98, Zero 102.68, Fixed 117.43, Shuffle 118.78, TR 119.56 tokens/s. `Fixed` is built from 100 AlpacaEval queries unrelated to the test set, so this is not future-query leakage or training. Source: ACL 2025 paper, Sec. 3.1 and 5.2 / Table 3.

Before closing P0-AS, run the missing **disjoint hot-start** discriminator:
1. Build the ordinary top-k adjacency matrix from disjoint calibration requests only (public/general prompts or leave-one-family-out histories; never the current qualification query's future). Keep it tiny and mechanically generated; this is cache initialization, not behavioral training.
2. Initialize each held-out qualification request from that fixed matrix, then update strictly chronologically with only target results that become available online.
3. Test official overwrite semantics first. Also test a bounded multi-history/union cache seeded only from disjoint prior requests if cheap; the existing same-query `union` mode is clairvoyant and cannot count, but its N8 OPT ceiling (~112 representative t/s pure-TR) shows that better historical candidate coverage is potentially material.
4. Charge real NInfer N8/N12 tree verification including full attention/head/fold/control before implementation. The current result charged only the favorable GDN tree+fold bucket.
5. If disjoint hot-start + real costs cannot clear single80, close Token Recycling. Do not port a tree runtime merely because the same-query union upper bound is huge.

Do not mistake the new scalar-AR capture for a universal oracle: scalar AR already forks from the current fast-MTP teacher in 5/6 families, which is legal under `AGENTS.md`. For accelerating the existing fast-MTP product route, verifier-state top-k from the current fast path is the more relevant behavioral trace unless the proposed new route explicitly changes the target semantics.

If hot-start TR still fails, do not blindly port Cacheback/SAM: P0-D request-context lookup-tail already closed the general suffix-copy family. A cheap context-keyed TR oracle (last 2-3 tokens rather than token-only) is allowed if it uses the same disjoint hot-start evidence; otherwise move on. LogitSpec (arXiv:2507.01449) is a distinct training-free follow-up worth a costed oracle because it uses the current logit to seed next-next-token retrieval, but only after P0-AS is scientifically closed.
