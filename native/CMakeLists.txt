cmake_minimum_required(VERSION 3.16)
project(gtcrn_native C)

set(GTCRN_WEIGHTS_INDEX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/out/gtcrn_weights_f32"
    CACHE PATH "Directory containing generated weights_index.h")
set(GTCRN_INT8_WEIGHTS_INDEX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/out/gtcrn_weights"
    CACHE PATH "Directory containing generated INT8 weights_index.h")
if(NOT EXISTS "${GTCRN_WEIGHTS_INDEX_DIR}/weights_index.h")
  message(FATAL_ERROR "Generate FP32 weights before configuring native build")
endif()
if(NOT EXISTS "${GTCRN_INT8_WEIGHTS_INDEX_DIR}/weights_index.h")
  message(FATAL_ERROR "Generate INT8 weights before configuring native build")
endif()

add_library(gtcrn_kernels STATIC src/gtcrn_kernels.c src/gtcrn.c)
set_target_properties(gtcrn_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gtcrn_kernels PUBLIC include)
target_compile_features(gtcrn_kernels PUBLIC c_std_99)
if(NOT MSVC)
  target_compile_options(gtcrn_kernels PRIVATE -Wall -Wextra -Werror)
endif()

add_library(kissfft STATIC
  third_party/kissfft/kiss_fft.c
  third_party/kissfft/kiss_fftr.c)
target_include_directories(kissfft PUBLIC third_party/kissfft)
set_target_properties(kissfft PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(gtcrn_native SHARED src/gtcrn_model.c src/gtcrn_audio.c)
target_include_directories(gtcrn_native PUBLIC include PRIVATE "${GTCRN_WEIGHTS_INDEX_DIR}")
target_link_libraries(gtcrn_native PRIVATE gtcrn_kernels kissfft m)
target_compile_features(gtcrn_native PUBLIC c_std_99)
if(NOT MSVC)
  target_compile_options(gtcrn_native PRIVATE -Wall -Wextra -Werror)
endif()

add_executable(gtcrn_cli src/gtcrn_cli.c)
target_include_directories(gtcrn_cli PRIVATE third_party/dr_libs)
target_link_libraries(gtcrn_cli PRIVATE gtcrn_native)

add_library(gtcrn_native_int8 SHARED src/gtcrn_model.c src/gtcrn_audio.c)
target_compile_definitions(gtcrn_native_int8 PUBLIC GTCRN_INT8_WEIGHTS=1 GTCRN_Q15_STATE=1)
target_include_directories(gtcrn_native_int8 PUBLIC include PRIVATE "${GTCRN_INT8_WEIGHTS_INDEX_DIR}")
target_link_libraries(gtcrn_native_int8 PRIVATE gtcrn_kernels kissfft m)
target_compile_features(gtcrn_native_int8 PUBLIC c_std_99)
if(NOT MSVC)
  target_compile_options(gtcrn_native_int8 PRIVATE -Wall -Wextra -Werror)
endif()

add_executable(gtcrn_cli_int8 src/gtcrn_cli.c)
target_include_directories(gtcrn_cli_int8 PRIVATE third_party/dr_libs)
target_link_libraries(gtcrn_cli_int8 PRIVATE gtcrn_native_int8)

# Host-compiles the HAL-independent STM32 adapter to keep its API build-tested.
add_library(gtcrn_stm32_port STATIC stm32h743/gtcrn_stm32_port.c)
target_include_directories(gtcrn_stm32_port PUBLIC stm32h743)
target_link_libraries(gtcrn_stm32_port PUBLIC gtcrn_native_int8)

enable_testing()
add_executable(test_kernels tests/test_kernels.c)
target_link_libraries(test_kernels PRIVATE gtcrn_kernels m)
add_test(NAME kernels COMMAND test_kernels)
add_executable(test_model tests/test_model.c)
target_link_libraries(test_model PRIVATE gtcrn_native)
add_test(NAME model_abi COMMAND test_model)
add_executable(test_audio_stream tests/test_audio_stream.c)
target_link_libraries(test_audio_stream PRIVATE gtcrn_native)
add_test(NAME audio_stream_init COMMAND test_audio_stream)
