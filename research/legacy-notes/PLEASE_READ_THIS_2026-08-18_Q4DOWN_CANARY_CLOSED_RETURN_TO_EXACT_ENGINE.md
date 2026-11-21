# Q4-down target canary CLOSED — return to exact/runtime engineering

The BF16-source weight-only audit was valid portable research, but the resulting selective target-model Q5->Q4 candidates are not worth pursuing for the current product goal.

Measured on the normally boosting Core-X TU102:
- baseline canonical 512: 66.20-66.66 tok/s, canonical target trajectory;
- q4down2 (layers 20,57): 65.94 tok/s, target diverges at token 9;
- q4down4 (12,14,20,57): 67.42 tok/s, target diverges at token 122.

Held-out canaries:
- q4down2 preserves general_storage and code_typescript, but diverges Thai at 101, Japanese at 24, Arabic at 16;
- q4down4 diverges canonical/general/code/Thai/Japanese; only Arabic happened to match.
- q4down4 also badly reduced useful performance on code/Thai through changed MTP trajectory/acceptance.

This is not an exact-engine gain. The byte reduction is too small per selected subset and the target semantic/numerical disturbance arrives quickly. Do NOT continue q4down6/q4down8/subset enumeration, prompt-select layers, or claim the 67.42 number as an engine win. Preserve the BF16-source geometry/sensitivity artifacts as research evidence only.

## Immediate next work: exact / target-semantics-preserving
1. Re-port the archived fixed-map Q4/B2 proposal screen -> top-K -> **exact W8 target-head rerank** onto the current trusted single-GPU branch. Do not alter token-map membership. Separate proposal acceptance gains from engine timing.
2. Re-test attention-input A8 on current INT8-KV runtime; exact target digest/acceptance required for engine claim.
3. NCU/SASS grind on dominant current exact kernels on the healthy Core-X card, especially Q4/Q5/W8 bytes, long-scoreboard, register/issue-pipe balance, and Turing INT32/FP32 overlap.
4. Low-cost per-kernel L1/shared carveout A/B.
5. In parallel / next for dual C1: TP2 T4 MLP microbenchmark with real host-staged transfer/reduction and current device-asymmetry accounting; then populated long-context timing and exact context/head parallelism.

Checkpoint co-design can be revisited later only with a real quality target and material byte leverage; it is not the near-term >=80 exact/product path.
