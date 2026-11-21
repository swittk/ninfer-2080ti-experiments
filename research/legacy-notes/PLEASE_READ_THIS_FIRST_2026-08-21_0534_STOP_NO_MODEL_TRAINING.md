# STOP — OWNER OVERRIDE — 2026-08-21 05:34 +07

READ THIS BEFORE ANY MORE WORK.

You partially followed the 05:17 adviser note by finishing the bounded DFlash2 matrix and NInfer economics. Good. But the new `.chatgpt/dual_c1_20260820/native_mtp_v2/root_onpolicy/train.py` / `run.sh` created at 05:28 are AGAINST OWNER DIRECTION.

## ABSOLUTE RULE

DO NOT train, finetune, distill, self-distill, TTT, D-PACE-train, CE-train, interpolate, blend, LoRA, adapter-train, or otherwise create/modify behavioral model/MTP/drafter weights in this NInfer project.

Do not run `root_onpolicy/train.py`. Do not create a replacement training script. Prior local training artifacts are evidence only and are not a forward lane.

Allowed weights only:
1. factory Qwen3.8 target/native-MTP weights;
2. professionally published upstream exact-target speculative checkpoints (e.g. Qwen3.8 DFlash2);
3. mechanical conversion/quantization/export with no behavioral retraining.

NO Codex/Claude/OpenCode/sub-agents either. Primary ChatGPT does the work directly.

## DFlash2 RESULT IS NOW CLOSED AT CURRENT COST

The bounded upstream DFlash2 matrix is complete:
- n4 greedy mean AL 3.168, draft ~11.79 ms/round;
- n7 greedy mean AL 3.745, draft ~15.94 ms/round;
- n4 sampled AL 2.830, draft ~12.08 ms/round;
- n7 sampled AL 3.517, draft ~15.82 ms/round.

NInfer optimistic economics:
- best 65K projected ~50.8 t/s;
- best 262K projected ~37.3 t/s;
- classification `CLOSED_NO_NINFER_INTEGRATION_AT_CURRENT_COST_YIELD`.

Do not integrate or tune DFlash2 now. Preserve it as upstream reference.

## NEXT WORK — SYSTEMS / INFERENCE ONLY

1. Remove/restore the abandoned `dual_stage.cpp` teacher-dump instrumentation. Do not commit it.
2. Delete/ignore the new root-onpolicy training lane; no execution.
3. Re-establish reviewed clean source baseline.
4. Attack FACTORY native-MTP inference:
   - quantify GPU->CPU->GPU accept/proposal control round-trip;
   - model device-resident accept/select/propose path;
   - compare one-model recursive MTP execution with upstream TensorRT-LLM MTP-Eagle details;
   - build SHORT-context factory-MTP dynamic-tree ORACLE first (do not blindly reopen 65K tree);
   - preserve target exact semantics.
5. Attack the SHORT dual-GPU >=100 problem directly. Single exact ~72 while dual representative ~65–66 is unacceptable. Audit same-shape mechanisms vs the public dual2080Ti 101.3 t/s native-MTP stack; model full-target-on-one-GPU + second-GPU speculation coprocessor because our no-P2P host-staged TP may be the wrong short-context architecture.
6. For long context, measure factory-MTP draft-KV cost and only then test training-free Windowed-MTP if material.
7. Entropy-guided MTP topology is oracle-first and training-free only.

Hard gates remain single >=80; dual one-request >=100; populated262K dual >=100. No lowering goals, no model training, no agents.
