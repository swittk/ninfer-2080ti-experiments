add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_load_progress
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_load_progress)

add_executable(ninfer-dual
  dual/main.cpp
  dual/context_parallel_gqa.cu
  dual/parallel_vocab_head.cu
  dual/tp2_mlp.cu
  dual/tp2_generated/tp2_q4_fast.cu
  dual/tp2_generated/tp2_q4_slow.cu
  dual/tp2_generated/tp2_q5_fast.cu
  dual/tp2_generated/tp2_q5_slow.cu
  dual/tp2_generated/tp2_q5_alias_fast.cu
  dual/tp2_generated/tp2_q5_alias_slow.cu
  dual/tp2_generated/tp2_t5_q4_fast.cu
  dual/tp2_generated/tp2_t5_q5_fast.cu
  dual/tp2_generated/tp2_t5_q4_slow.cu
  dual/tp2_generated/tp2_t5_q5_slow.cu)
target_include_directories(ninfer-dual PRIVATE
  ${PROJECT_SOURCE_DIR}/apps
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/ops/linear_add/q5
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer-dual PRIVATE ninfer_engine ninfer_product_prompt_input ${NINFER_CUDART_TARGET})
# Explicit experimental live-prefix covariance/eigenbasis preparation.
target_link_libraries(ninfer-dual PRIVATE CUDA::cublas CUDA::cusolver)
