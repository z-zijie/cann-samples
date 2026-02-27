# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

message(STATUS "BUILD_TORCH_OPS ON in fused_infer_attention_score")

# fused_infer_attention_score operation sources
file(GLOB_RECURSE fused_infer_attention_score_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
set(fused_infer_attention_score_SOURCES ${fused_infer_attention_score_NPU_SOURCES})

# Mark .cpp files with special properties
set_source_files_properties(
    ${fused_infer_attention_score_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--npu-arch=dav-3101 -xasc "
)

# Create object library
add_library(fused_infer_attention_score_objects OBJECT ${fused_infer_attention_score_SOURCES})

target_compile_options(fused_infer_attention_score_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(fused_infer_attention_score_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()
