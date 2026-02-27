# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 算子极简调用的cmake配置
function(define_npu_op op_name)

    # 打印编译状态
    message(STATUS "BUILD_TORCH_OPS ON in ${op_name}")

    # 查找当前目录下的所有 cpp 文件
    file(GLOB_RECURSE ${op_name}_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    set(${op_name}_SOURCES ${${op_name}_NPU_SOURCES})

    # 设置编译属性
    set_source_files_properties(
        ${${op_name}_NPU_SOURCES} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "--npu-arch=dav-3101 -xasc"
    )

    # 创建对象库
    add_library(${op_name}_objects OBJECT ${${op_name}_SOURCES})

    # 设置编译选项和头文件目录
    target_compile_options(${op_name}_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_include_directories(${op_name}_objects PRIVATE ${COMMON_INCLUDE_DIRS})
endfunction()