find_package(Python3 REQUIRED COMPONENTS Interpreter)

# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the installed product headers stand alone.
add_executable(ninfer_public_api_test test_public_api.cpp)
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)

function(ninfer_add_test name)
  cmake_parse_arguments(arg "NEEDS_SOURCE_DIR" "" "SOURCES;LIBRARIES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
  if(arg_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  else()
    target_link_libraries(${name} PRIVATE ninfer_core)
  endif()
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      NINFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

function(ninfer_add_op_test name)
  ninfer_add_test(${name} ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(ninfer_add_linear_test name source)
  ninfer_add_op_test(${name}
    SOURCES ${source}
            ops/linear/linear_test_common.cpp
    LIBRARIES ninfer_ops)
endfunction()

# Built explicitly by the cross-context determinism harness. It must run concurrently with a
# separate inference process, so it is intentionally not registered as a standalone CTest.
add_executable(ninfer_sm_pressure EXCLUDE_FROM_ALL stress/sm_pressure.cu)
target_link_libraries(ninfer_sm_pressure PRIVATE CUDA::cudart)

function(ninfer_add_fused_linear_test name source common)
  ninfer_add_op_test(${name}
    SOURCES ${source} ${common}
    LIBRARIES ninfer_ops)
endfunction()

# Artifact framing and model-independent runtime contracts.
ninfer_add_test(ninfer_artifact_reader_test
  SOURCES test_ninfer_artifact_reader.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_artifact_materialization_test
  SOURCES test_artifact_materialization.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_device_test       SOURCES test_device.cpp)
ninfer_add_test(ninfer_decode_graph_test SOURCES test_decode_graph.cpp)
ninfer_add_test(ninfer_tensor_test       SOURCES test_tensor.cpp)
ninfer_add_test(ninfer_arena_test        SOURCES test_arena.cpp)
ninfer_add_test(ninfer_request_memory_test SOURCES test_request_memory.cpp)
ninfer_add_test(ninfer_admission_policy_test SOURCES test_admission_policy.cpp)
ninfer_add_test(ninfer_kv_capacity_test SOURCES test_kv_capacity.cpp)
ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES test_sampling_defaults.cpp
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_sampling_defaults_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
ninfer_add_test(ninfer_kv_cache_test
  SOURCES test_kv_cache.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_state_store_test
  SOURCES test_state_store.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_gdn_replay_records_test
  SOURCES test_gdn_replay_records.cpp
  LIBRARIES ninfer_core)
set_tests_properties(
  ninfer_artifact_materialization_test
  ninfer_device_test
  ninfer_decode_graph_test
  ninfer_arena_test
  ninfer_request_memory_test
  ninfer_kv_cache_test
  ninfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)
# Family frontend and one opt-in real-artifact prefix/MTP integration test.
ninfer_add_test(ninfer_qwen3_6_frontend_test
  SOURCES targets/qwen3_6/test_frontend.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_frontend_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl)
ninfer_add_test(ninfer_qwen3_6_runtime_mechanisms_test
  SOURCES targets/qwen3_6/test_runtime_mechanisms.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_runtime_mechanisms_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_qwen3_6_27b_prefix_real_test
  SOURCES targets/qwen3_6_27b/test_engine_prefix_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_prefix_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_load_plan_test
  SOURCES targets/qwen3_6_27b/test_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_27b_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
target_link_libraries(ninfer_qwen3_6_27b_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_6_27b_load_plan_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_dflash_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_dflash_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_dflash_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test
  SOURCES targets/qwen3_6_35b_a3b/test_dflash_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
target_link_libraries(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Opt-in real artifact proof of clean cancellation versus retained prompt state.
ninfer_add_test(ninfer_transient_slots_test SOURCES test_transient_slots.cpp)
ninfer_add_test(ninfer_multi_prefill_real_test SOURCES test_multi_prefill_real.cpp LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_multi_prefill_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_packed_prefill_real_test SOURCES test_packed_prefill_real.cpp LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_packed_prefill_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_target_head_policy_test
  SOURCES test_target_head_policy.cpp
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_target_head_policy_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)

ninfer_add_test(ninfer_cancel_prefix_real_test
  SOURCES test_cancel_prefix_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_cancel_prefix_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_prefill_decode_budget_test
  SOURCES test_prefill_decode_budget.cpp)

# Product protocol contracts.
ninfer_add_test(ninfer_openai_schema_test
  SOURCES test_openai_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_responses_schema_test
  SOURCES test_responses_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_response_store_test
  SOURCES test_response_store.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_anthropic_schema_test
  SOURCES test_anthropic_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES test_tool_call_parser.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_replica_head_cache_test
  SOURCES test_replica_head_cache.cpp
  LIBRARIES ninfer_ops)
set_tests_properties(ninfer_replica_head_cache_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_replica_router_test
  SOURCES test_replica_router.cpp)

ninfer_add_test(ninfer_serve_options_test
  SOURCES test_serve_options.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_request_log_test
  SOURCES test_request_log.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_bench_support_test
  SOURCES test_ninfer_bench_support.cpp
          ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b/ninfer_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b)

# Shared Op numerical tests. Target-composed schedule tests are registered separately below.
set(ninfer_op_tests
  add_bias
  gelu
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  gated_rmsnorm
  l2norm
  gated_delta_net
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  vision_attention
  sampling
  scalar
  cast
  scatter
  position)
foreach(op IN LISTS ninfer_op_tests)
  ninfer_add_op_test(ninfer_${op}_test
    SOURCES ops/test_${op}.cpp
    LIBRARIES ninfer_ops)
endforeach()

# This test directly instantiates the header-defined tiled kernel, so it intentionally does not
# link ninfer_ops (which contains a separate production instantiation of the same CUDA kernel).
ninfer_add_op_test(ninfer_argmax_atomic_preseed_test
  SOURCES ops/test_argmax_atomic_preseed.cu
  LIBRARIES CUDA::cudart)
ninfer_add_test(ninfer_w8_vocab_b2_partial_test
  SOURCES ops/test_w8_vocab_b2_partial.cpp)

# Stateful, fused, and exact-shape Ops remain central even when the current implementation domain
# is one checkpoint/device.
ninfer_add_op_test(ninfer_gqa_attention_test
  SOURCES ops/test_gqa_attention.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gqa_tree_test
  SOURCES ops/test_gqa_tree.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_bidirectional_gqa_attention_test
  SOURCES ops/test_bidirectional_gqa_attention.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_swa_test
  SOURCES ops/test_swa.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_kv_cache_append_prefix_test
  SOURCES ops/test_kv_cache_append_prefix.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_prepare_masked_block_test
  SOURCES ops/test_prepare_masked_block.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_sparse_moe_test
  SOURCES ops/test_sparse_moe.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_mtp_pack_test
  SOURCES ops/test_mtp_pack.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_mtp_round_test
  SOURCES ops/test_mtp_round.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES ops/test_speculative_round.cpp
  LIBRARIES ninfer_ops)
ninfer_add_linear_test(ninfer_linear_q4_a16_test ops/linear/test_q4_a16.cpp)
ninfer_add_linear_test(ninfer_linear_q5_a16_test ops/linear/test_q5_a16.cpp)
ninfer_add_linear_test(ninfer_linear_q6_a16_test ops/linear/test_q6_a16.cpp)
ninfer_add_linear_test(ninfer_linear_w8_a16_test ops/linear/test_w8_a16.cpp)
ninfer_add_linear_test(ninfer_linear_nvfp4_a16_test ops/linear/test_nvfp4_a16.cpp)
ninfer_add_linear_test(ninfer_linear_nvfp4_a4_test ops/linear/test_nvfp4_a4.cpp)
ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES ops/linear/test_bf16_a16.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_attn_input_proj_test
  SOURCES ops/test_attn_input_proj.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_test
  SOURCES ops/test_gdn_input_proj.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_snapshot_test
  SOURCES ops/test_gdn_input_proj_conv_snapshot.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_record_test
  SOURCES ops/test_gdn_input_proj_conv_record.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES ops/test_gated_delta_net_replay_record.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES ops/test_gdn_replay_fold.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_tree_replay_test
  SOURCES ops/test_gdn_tree_replay.cpp
  LIBRARIES ninfer_ops)
ninfer_add_fused_linear_test(
  ninfer_linear_add_q5_a16_test
  ops/linear_add/test_q5_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_add_bf16_a16_test
  ops/linear_add/test_bf16_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_add_w8_a16_test
  ops/linear_add/test_w8_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_op_test(ninfer_linear_add_nvfp4_test
  SOURCES ops/linear_add/test_nvfp4.cpp
  LIBRARIES ninfer_ops)
ninfer_add_fused_linear_test(
  ninfer_linear_pair_w8_a16_test
  ops/linear_pair/test_w8_a16.cpp
  ops/linear_pair/linear_pair_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_q4_a16_test
  ops/linear_swiglu/test_q4_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_w8_a16_test
  ops/linear_swiglu/test_w8_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_nvfp4_test
  ops/linear_swiglu/test_nvfp4.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)

ninfer_add_test(ninfer_qwen3_6_27b_visual_scatter_test
  SOURCES targets/qwen3_6_27b/test_visual_scatter.cpp
  LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_qwen3_6_27b_visual_scatter_test PROPERTIES SKIP_RETURN_CODE 77)

# These tests deliberately compile against the exact target's internal contracts; the target does
# not export impl/ to ordinary consumers.
set(ninfer_qwen3_6_27b_internal_tests
  ninfer_qwen3_6_27b_visual_scatter_test)
foreach(test_target IN LISTS ninfer_qwen3_6_27b_internal_tests)
  target_include_directories(${test_target} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/impl
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
endforeach()

# Pure host preprocessing cache ownership and identity contract.
ninfer_add_test(ninfer_media_prepare_cache_test SOURCES test_media_prepare_cache.cpp)
target_include_directories(ninfer_media_prepare_cache_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "75")
  ninfer_add_test(ninfer_marlin_weight_lifetime_test SOURCES test_marlin_weight_lifetime.cu LIBRARIES ninfer_ops)
  ninfer_add_test(ninfer_sm75_radix_projection_test SOURCES test_sm75_radix_projection.cu LIBRARIES ninfer_ops)
  target_include_directories(ninfer_sm75_radix_projection_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/ops)

  target_include_directories(ninfer_marlin_weight_lifetime_test PRIVATE
    ${PROJECT_SOURCE_DIR}/bench/ops)
  set_tests_properties(ninfer_marlin_weight_lifetime_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

ninfer_add_test(ninfer_engine_reload_real_test SOURCES test_engine_reload_real.cpp LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_engine_reload_real_test PROPERTIES SKIP_RETURN_CODE 77)

# CPU-only bound checks for streaming existing independent Vision segments.
ninfer_add_test(ninfer_vision_batch_plan_test SOURCES test_vision_batch_plan.cpp LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_vision_batch_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)

# Optional real-model golden profile. No GPU work without the explicit artifact.
ninfer_add_test(ninfer_retained_c1_real_test SOURCES test_retained_c1_real.cpp LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_retained_c1_real_test PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME ninfer_retained_profile_guard_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_retained_sm75_profiles.py)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "75")
  ninfer_add_test(ninfer_sm75_prefill_storage_test SOURCES test_sm75_prefill_storage.cpp LIBRARIES ninfer_engine ninfer_core)
  add_test(NAME ninfer_sm75_streamed_prefill_storage_test COMMAND ninfer_sm75_prefill_storage_test streamed)
endif()

ninfer_add_test(ninfer_packed_prefill_policy_test SOURCES test_packed_prefill_policy.cpp)

ninfer_add_test(ninfer_long_context_policy_test SOURCES test_long_context_policy.cpp)

# Same address-invariant GPU test used before the full-model window comparison.
ninfer_add_op_test(ninfer_gqa_recent_view_test SOURCES test_gqa_recent_view.cu LIBRARIES ninfer_ops)

ninfer_add_test(ninfer_dual_draft_reload_real_test SOURCES test_dual_draft_reload_real.cpp LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_dual_draft_reload_real_test PROPERTIES SKIP_RETURN_CODE 77)
target_include_directories(ninfer_dual_draft_reload_real_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)

ninfer_add_op_test(ninfer_gqa_headscale_cache_test SOURCES test_gqa_headscale_cache.cu LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gqa_value_fp16_test SOURCES test_gqa_value_fp16.cu LIBRARIES ninfer_ops)

add_test(NAME ninfer_dual_value_fp16_cli_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_dual_value_fp16_cli.py $<TARGET_FILE:ninfer-dual>)

ninfer_add_test(ninfer_session_copy_test SOURCES test_session_copy.cpp)

ninfer_add_test(ninfer_session_copy_real_test SOURCES test_session_copy_real.cpp LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_session_copy_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_op_test(ninfer_target_candidate_union_test SOURCES test_target_candidate_union.cu LIBRARIES ninfer_ops)

# TP2 packet metadata has no GPU dependency; the optional real-weight check uses
# the production shard kernels and runs only when its explicit source is supplied.
ninfer_add_test(ninfer_tp2_packet_layout_test SOURCES test_tp2_packet_layout.cpp)
ninfer_add_op_test(ninfer_tp2_packet_real_test
  SOURCES test_tp2_packet_real.cu
          ../apps/dual/tp2_generated/tp2_q4_fast.cu
          ../apps/dual/tp2_generated/tp2_q4_slow.cu
          ../apps/dual/tp2_generated/tp2_q5_fast.cu
          ../apps/dual/tp2_generated/tp2_q5_slow.cu
          ../apps/dual/tp2_generated/tp2_q5_alias_fast.cu
          ../apps/dual/tp2_generated/tp2_q5_alias_slow.cu
          ../apps/dual/tp2_generated/tp2_t5_q4_fast.cu
          ../apps/dual/tp2_generated/tp2_t5_q5_fast.cu
          ../apps/dual/tp2_generated/tp2_t5_q4_slow.cu
          ../apps/dual/tp2_generated/tp2_t5_q5_slow.cu
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_tp2_packet_real_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps
  ${PROJECT_SOURCE_DIR}/src/ops/linear_add/q5
  ${PROJECT_SOURCE_DIR}/src/ops/linear_swiglu/q4)

ninfer_add_test(ninfer_sm75_wide_radix_projection_test SOURCES test_sm75_wide_radix_projection.cu LIBRARIES ninfer_ops)
target_include_directories(ninfer_sm75_wide_radix_projection_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/ops)

ninfer_add_op_test(ninfer_resident_radix_ownership_test
  SOURCES test_resident_radix_ownership.cu LIBRARIES ninfer_ops)
target_include_directories(ninfer_resident_radix_ownership_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/ops)

ninfer_add_op_test(ninfer_gqa_proposal_retrieval_test SOURCES test_gqa_proposal_retrieval.cu LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gqa_target_guide_view_test SOURCES test_gqa_target_guide_view.cu LIBRARIES ninfer_ops)

# Exact original-versus-padded vocabulary staging, including graph source reuse.
ninfer_add_op_test(ninfer_w8_vocabulary_padding_test SOURCES test_w8_vocabulary_padding.cu)

# Source-coordinate storage is bit-preserving; no model artifact required.
ninfer_add_op_test(ninfer_joint_kv_owned_storage_test SOURCES test_joint_kv_owned_storage.cu)

# Immutable prefix bytes and independent stage lifetime/error handling.
ninfer_add_test(ninfer_prefix_snapshot_io_test SOURCES test_prefix_snapshot_io.cpp)
find_package(Threads REQUIRED)
target_link_libraries(ninfer_prefix_snapshot_io_test PRIVATE Threads::Threads)

# Ordered mask compaction preserves subset order, capacity and graph storage.
ninfer_add_op_test(ninfer_joint_ordered_union_test SOURCES test_joint_ordered_union.cu)

# Resident prefill expands exact coefficients without a canonical intermediate.
ninfer_add_op_test(ninfer_resident_prefill_direct_test SOURCES test_resident_prefill_direct.cu LIBRARIES ninfer_engine)

# Ready short/ragged admission and per-owner checkpoint/finalization behavior.
ninfer_add_test(ninfer_ready_ragged_prefill_real_test SOURCES test_ready_ragged_prefill_real.cpp LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_ready_ragged_prefill_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_op_test(ninfer_attention_packets_test SOURCES test_attention_packets.cu)

# The optional corpus head must not become a full-vocabulary MTP dependency.
ninfer_add_test(ninfer_mtp_proposal_ownership_test
  SOURCES test_mtp_proposal_ownership.cpp
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_mtp_proposal_ownership_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
target_link_libraries(ninfer_mtp_proposal_ownership_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_mtp_proposal_ownership_test PROPERTIES SKIP_RETURN_CODE 77)

# Full native head equality and two-device source/graph/pinned-buffer ownership.
ninfer_add_op_test(ninfer_parallel_vocab_head_test SOURCES test_parallel_vocab_head.cu
  ${PROJECT_SOURCE_DIR}/apps/dual/parallel_vocab_head.cu)
target_include_directories(ninfer_parallel_vocab_head_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)

add_test(NAME ninfer_parallel_vocab_head_cli_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_parallel_vocab_head_cli.py
          $<TARGET_FILE:ninfer-dual>)

# Native M8 row coverage, exact old attention states, ranges and graph lifetime.
ninfer_add_op_test(ninfer_valid_query_integer_attention_test
  SOURCES test_valid_query_integer_attention.cu
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_direct_attention_metadata_test SOURCES test_direct_attention_metadata.cu)

# Preserve the three-stage reference while checking query-owned completion.
ninfer_add_op_test(ninfer_query_owned_attention_finish_test SOURCES test_query_owned_attention_finish.cu)

# Query preparation preserves each original arithmetic boundary and independent query ownership.
ninfer_add_op_test(ninfer_query_input_lifetime_test SOURCES test_query_input_lifetime.cu)
