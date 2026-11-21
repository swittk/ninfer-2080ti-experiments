# HARD OWNER REQUIREMENT — FINAL PRODUCT MUST SUPPORT MULTIMODAL ON SINGLE AND DUAL

Owner requirement, 2026-08-28. **Final promotion blocker, not optional cleanup.**

## Requirement
The final Qwen3.8-27B NINFER product must support multimodal input on BOTH:
1. single-GPU serving; and
2. the optimized dual-GPU serving/runtime.

Multimodal means the existing supported message surface: `image`/`image_url` and `video`/`video_url`, including mixed text+media prompts. Do NOT ship the optimized dual runtime as text-only.

The owner has requested vision support repeatedly. Text throughput gates do not override this requirement.

## Current verified state
- Shared/single vision stack is intact. On 2026-08-28 these passed on SM75:
  - `ninfer_vision_attention_test`
  - `ninfer_vision_pos_embed_test`
  - `tests/targets/qwen3_6_27b/test_multimodal.py`
- Single CLI exposes `--vision` and supports image/video structured content.
- Current optimized dual layer-sharded runtime is explicitly text-only: `TextShardStage` uses `.vision = false`, and shard artifact binding rejects `features.vision`.
- Therefore dual multimodal is MISSING INTEGRATION today. Shared vision operators existing in the repo is not proof of dual support.

## Final acceptance gates
### Single
- Real E2E image prompt passes on promoted single path.
- Real E2E video prompt passes.
- Mixed text+media works.
- Existing vision tests stay green.
- Keeping vision available does not regress promoted text gates.

### Dual
- Final dual server accepts same image/video message surface as single.
- Vision backbone/merger weights are deliberately placed on selected GPU(s); avoid unjustified duplication.
- Vision encode/merge output is correctly scattered into the text embedding stream before sharded text prefill.
- mRoPE, visual token positions, and chunked prefill remain correct.
- After multimodal prefill, normal dual text sharding, physical KV, TP2 MLP, MTP/spec decode, and long-context routing still work.
- eGPU remains optional; missing second GPU must not break single multimodal.

### Behavioral qualification
Run actual E2E canaries, not only operator tests:
- natural image understanding;
- multi-image comparison;
- video;
- mixed image+video if frontend supports it;
- at least one non-English multimodal prompt if practical.

Compare dual to promoted single/reference under the normal numerical/behavioral-quality contract. Exact bytes are not required unless claiming EXACT, but deterministic coherent model-equivalent behavior is required.

### Context capability
Multimodal must coexist with the native-context-capable dual architecture. Media tokens may consume context, but vision must not fall back to a separate short-context-only engine or impose an artificial low context cap.

## Preferred architecture
Reuse the existing qwen3_6 frontend/vision machinery:
1. prepare text + media once;
2. vision encoder/merger runs on one selected device (likely stage0 unless profiling says otherwise);
3. scatter merged visual embeddings into the text stream;
4. continue through normal dual sharded prefill/decode.

Do not fork tokenizer/chat-template/mRoPE logic for dual. Do not hardcode media token IDs, image sizes, prompt shapes, or benchmark-specific behavior.

## Scheduling / tracking
Current coherent 262K perf work may reach a natural checkpoint first, but BEFORE FINAL PROMOTION/FREEZE, dual multimodal implementation + E2E qualification is mandatory.

Track vision explicitly as:
- `SINGLE_VISION_PASS`
- `DUAL_VISION_IMPLEMENTED_PENDING_QUAL`
- `DUAL_VISION_PASS`

Do not treat empty `PASS_VISION*` marker files as evidence unless they point to actual command/result artifacts.

**FINAL RULE: text speed gates alone are insufficient. A final dual product without working image/video multimodal support is incomplete.**
