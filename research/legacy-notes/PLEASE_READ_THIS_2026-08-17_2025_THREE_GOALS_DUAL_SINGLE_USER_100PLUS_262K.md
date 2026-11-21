# PLEASE READ — owner clarification: three separate performance goals; dual single-user is extreme speed + native context

Owner/adviser clarification, 2026-08-17 20:25 +07. This corrects any ambiguity introduced by the 19:34 sequencing note. Do not conflate dual multi-user aggregate throughput with the dual **single-request** product goal.

There are THREE separate goals:

1. **Single GPU, single user, INT8:** >=80 useful decode tok/s minimum at the practical one-card context. Native 262K is NOT required on one card.

2. **Dual GPU, ONE active user/request:** this mode exists to provide BOTH native/near-native 262K-class context capacity AND extreme speed. >=100 useful decode tok/s for that single user is the minimum serious performance target; materially above 100 is preferred. A dual implementation that merely fits 262K but runs at 35-50 tok/s is an intermediate correctness implementation, not success. Final qualification must include genuinely populated long-context frontiers (8K/32K/64K/128K/near-262K), not only max-context allocation with a short live history.

3. **Dual GPU, multiple users:** >=100 aggregate useful tok/s is the minimum multi-user performance gate; aim materially higher and report per-user latency/throughput as well as aggregate. This goal does NOT replace goal 2.

Current single-session sharded-MTP work is therefore directly relevant to goal 2. Continue optimizing it after correctness qualification rather than immediately abandoning it for multi-user scheduling. Multi-user pipeline work is also important, but do not call the dual project complete until single-user dual speed + native-context behavior are characterized and aggressively optimized.

The existing 14:02 hard-gate note already states the intended sequence correctly: single-GPU >=80 -> integrate into dual sharding -> dual single-request >=100 while preserving native 262K capacity -> then maximize multi-user throughput. Preserve that interpretation.

The current 128-token short-history qualification is diagnostic only. Acceptance/proposal efficiency and sharded scheduling still need work; do not infer the long-context frontier from it.
