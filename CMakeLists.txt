cmake_minimum_required(VERSION 3.20)
project(nerd-dictation-t-one VERSION 1.0.0 LANGUAGES ASM C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Model and resource paths
# ============================================================================

set(T_ONE_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/sherpa-onnx-streaming-t-one-russian-2025-09-08")
set(T_ONE_MODEL_ONNX "${T_ONE_MODEL_DIR}/model.onnx")
set(T_ONE_MODEL_ARCHIVE_PARTS
    "${T_ONE_MODEL_DIR}/model.onnx.tar.bz2.aa"
    "${T_ONE_MODEL_DIR}/model.onnx.tar.bz2.ab"
    "${T_ONE_MODEL_DIR}/model.onnx.tar.bz2.ac"
)
# Tokens are now embedded in models/.../tokens.h header

# ============================================================================
# Extract model from split archive (if needed)
# ============================================================================

if(NOT EXISTS "${T_ONE_MODEL_ONNX}")
    message(STATUS "T-one: extracting model.onnx from split archive...")

    # Check that all archive parts exist
    foreach(part ${T_ONE_MODEL_ARCHIVE_PARTS})
        if(NOT EXISTS "${part}")
            message(FATAL_ERROR "Archive part not found: ${part}")
        endif()
    endforeach()

    # Extract
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E chdir ${T_ONE_MODEL_DIR}
                sh -c "cat model.onnx.tar.bz2.* | tar xjf -"
        RESULT_VARIABLE extract_result
    )

    if(NOT extract_result EQUAL 0)
        message(FATAL_ERROR "Failed to extract model.onnx from archive")
    endif()

    if(NOT EXISTS "${T_ONE_MODEL_ONNX}")
        message(FATAL_ERROR "model.onnx not found after extraction")
    endif()

    message(STATUS "T-one: model.onnx extracted successfully")
endif()

# ============================================================================
# Include res_embed for resource embedding
# ============================================================================

set(RES_EMBED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/res_embed")

if(NOT EXISTS "${RES_EMBED_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "\n"
        "res_embed submodule not found!\n"
        "\n"
        "Please initialize the git submodule:\n"
        "  git submodule update --init --recursive\n"
    )
endif()

# Disable res_embed example build
set(BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)

add_subdirectory(${RES_EMBED_DIR} ${CMAKE_CURRENT_BINARY_DIR}/res_embed)

# ============================================================================
# Include sherpa-onnx as submodule
# ============================================================================

set(SHERPA_ONNX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/sherpa-onnx")

if(NOT EXISTS "${SHERPA_ONNX_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "\n"
        "sherpa-onnx submodule not found!\n"
        "\n"
        "Please initialize the git submodule:\n"
        "  git submodule update --init --recursive\n"
    )
endif()

# sherpa-onnx settings: C API only, minimal build
set(SHERPA_ONNX_ENABLE_C_API ON CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_GPU ON CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_JNI OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_WEBSOCKET OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_TTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# Use system onnxruntime
set(SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE ON CACHE BOOL "" FORCE)
set(ENV{SHERPA_ONNXRUNTIME_INCLUDE_DIR} "/usr/include/onnxruntime/core/session")
set(ENV{SHERPA_ONNXRUNTIME_LIB_DIR} "/usr/lib")

add_subdirectory(${SHERPA_ONNX_DIR} ${CMAKE_CURRENT_BINARY_DIR}/sherpa-onnx)

message(STATUS "T-one backend: using sherpa-onnx from ThirdParty/sherpa-onnx")

# ============================================================================
# Include nerd-dictation-api (C API header)
# ============================================================================

set(NERD_DICTATION_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/nerd-dictation-api")

if(NOT EXISTS "${NERD_DICTATION_API_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "\n"
        "nerd-dictation-api submodule not found!\n"
        "\n"
        "Please initialize the git submodule:\n"
        "  git submodule update --init --recursive\n"
    )
endif()

add_subdirectory(${NERD_DICTATION_API_DIR} ${CMAKE_CURRENT_BINARY_DIR}/nerd-dictation-api)

# ============================================================================
# Source files
# ============================================================================

set(T_ONE_SOURCES
    src/backend.cpp
)

# ============================================================================
# Library
# ============================================================================

add_library(nerd-dictation-t-one SHARED
    ${T_ONE_SOURCES}
)

target_include_directories(nerd-dictation-t-one
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SHERPA_ONNX_DIR}/sherpa-onnx/c-api
)

target_link_libraries(nerd-dictation-t-one
    PUBLIC
    nerd-dictation-api
    sherpa-onnx-c-api
    res_embed
)

# Force whole-archive linking for embedded resources
# This ensures the model data is included even if not directly referenced
if(UNIX AND NOT APPLE)
    set_target_properties(nerd-dictation-t-one PROPERTIES
        LINK_FLAGS "-Wl,--whole-archive ${CMAKE_CURRENT_BINARY_DIR}/libnerd-dictation-t-one_t_one_model_onnx.a -Wl,--no-whole-archive"
    )
endif()

# ============================================================================
# Embed model into library
# ============================================================================

res_embed(TARGET nerd-dictation-t-one NAME "t_one_model_onnx" PATH ${T_ONE_MODEL_ONNX} KEYWORD)

message(STATUS "T-one backend: embedding model.onnx into library (tokens from header)")

# ============================================================================
# Installation
# ============================================================================

install(TARGETS nerd-dictation-t-one
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# C API header is installed from the main project

# Export symbols for shared library
set_target_properties(nerd-dictation-t-one PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Test
# ============================================================================

add_executable(test_transcribe tests/test_transcribe.cpp)
target_link_libraries(test_transcribe PRIVATE
    nerd-dictation-t-one
    nerd-dictation-t-one_t_one_model_onnx
)
