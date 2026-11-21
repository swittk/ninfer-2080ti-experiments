# PLEASE READ — near-term priority: single >=80 and dual 1-request >=100 at populated ~262K

Finish/commit coherent C4 qualification first if still in flight. Then prioritize these two product goals.

## Single GPU: CORRECTED exact frontier / hardware attribution
Do **not** treat the earlier 65.28 -> 70.44 comparison as a software gain. Same-card controls showed the current standard runtime and the old INT8 runtime are essentially identical on the base-clock-limited internal card (~65.2 tok/s), while the old runtime already reaches ~70.30 tok/s on the normally boosting Core-X card. The apparent ~7.9% gain was primarily physical-card clock asymmetry. The current normally-boosting-card exact INT8 frontier is therefore ~70.3-70.4 tok/s, but **single >=80 still needs a real software gain of ~13.5%**. Preserve same-device A/B discipline and always record UUID / loaded clocks. Historical ~72 artifacts inspected used BF16 KV.

## Very high-priority proposal issue
Current 131072-row optimized proposal shortlist covers actual isolated target outputs:
- general 100.0%
- code 99.2%
- JSON 99.2%
- Japanese 78.1%
- Thai 41.4%
For Thai, 75/128 target tokens are absent from the proposal vocabulary, so MTP cannot possibly draft them correctly. Rebuild the SAME-size shortlist from a balanced multilingual/held-out target-trajectory corpus; freeze before evaluation. Also consider static language/domain-specific same-size heads selected once/request.

## Re-open archived top-K exact proposal reranker on today's base
Archived `q4_proposal_target_hybrid_top32_sm75.cu` did: fast Q4 131K screening -> top32 -> gather exact W8 full-LM-head rows -> exact rerank. It reached ~73.3 tok/s / 62.83% acceptance in the older BF16-KV regime. Current proposal B2 reranks against Q4 proposal weights, not the W8 target head. Re-port this exact top32 W8 reranker on today's ~70.4 INT8+A8 base and fixed corpus.

## Other bounded single-GPU reopenings
- old attention-input A8 tests predated today's all-MLP/GDN fragment stack and used BF16 KV; current runtime does not appear to retain equivalent attention A8. Re-test stacked on today's base, exact digest/acceptance required for an engine claim.
- old Q5->Q4 negatives mostly requantized the already-quantized artifact. They do NOT close BF16-source GPTQ/AWQ/OmniQuant-style code/scale selection or per-layer Q4/Q5/W8 allocation using existing formats.
- target-MLP BMMA/B2 is NOT an exact-engine win; its high TPS mostly came from altered target trajectory/acceptance. Only revisit as a quality-qualified noncontiguous sensitivity-selected checkpoint experiment.

## Dual single-request populated ~262K
Profile populated 8K/32K/64K/128K/~262K first. Separate weight/GDN, full-attention KV scan, target head, proposal, boundary.

### Strong exact architecture candidate: TP2 / head-parallel rather than only PP
No serious TP2 experiment found. For MLP: split gate/up rows across GPUs, keep matching halves of down local, reduce only 5120xT partial hidden. Real host-staged transfer probe on these cards: ~24.7 us for 10 KiB, 60.6 us for 40 KiB, 73.7 us for 80 KiB. Thus no-P2P does not automatically kill TP for T4 reductions. Full attention also splits cleanly: 24 Q heads / 4 KV heads -> 12/2 per GPU, naturally halves long-context KV scan.

Before building full TP, make a one-layer T4 TP MLP microbenchmark including real host-staged reductions and compare wall time with production one-GPU layer.

### Lower-risk long-context variant: context-parallel only full attention
Qwen3.8 has only 16 full-attention layers; 48 are GDN. At large context, shard persistent attention KV by sequence across both GPUs, send tiny Q to helper, compute partial online-softmax/output concurrently, merge exact partial stats. Method evidence: arXiv:2411.01783 pass-Q context parallelism for decode/low-bandwidth links. Activate only above measured threshold.

### Re-open wider MTP only at long context
8K MTP5 rejection remains valid for 8K but does NOT close 128K/262K, where target KV scan amortization changes. At populated contexts sweep no-spec/MTP1/2/3/5/7. MagicDec arXiv:2408.11049 is relevant.

### Proposal/drafter KV can be much more aggressive
At long context test sparse/recent+anchor or lower-bit MTP KV first; target correctness is unaffected, only acceptance. Target KV lower/mixed precision requires reference quality calibration; keep GDN recurrent state separate.

### Target head placement/parallelism
Output head is ~1.35 GiB W8 and stage1-owned. First test moving/replicating target head to faster GPU0 after stage1 final hidden (tiny transfer); then vocab-row split across both GPUs with local argmax and deterministic global reduction if worthwhile.

### Single-request async stage0 speculation
C4 cross-round preload proved bubbles are monetizable. For C1, first instrument layer32 boundary hidden -> final correction-token top1/topK predictability. If useful, speculatively run next stage0 on GPU0 while GPU1 finishes current stage1, rollback/discard on miss. Related method: PipeInfer arXiv:2407.11798.

Keep the MTP-acceptance qualification rule authoritative; label exact engine gains separately from drafter/quality gains.
