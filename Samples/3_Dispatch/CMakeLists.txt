# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# --- 提取公共变量 ---
set(TARGET_NAME dispatch_demo)
set(DEMO_SRC dispatch.cpp)
set(INSTALL_DIR 3_Dispatch)

# --- 原有设置 ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})

# --- 构建目标 ---
add_executable(${TARGET_NAME} ${DEMO_SRC})
target_include_directories(${TARGET_NAME} PRIVATE ${ASCEND_INCLUDE_DIRS})
target_link_directories(${TARGET_NAME} PRIVATE ${ASCEND_DIR}/lib64)
target_compile_options(${TARGET_NAME} PRIVATE
    "-xasc"
    "--npu-arch=dav-2201"
    "-w"
    "-O3"
)
target_link_libraries(${TARGET_NAME}
    PRIVATE
    m
    dl
    platform
    tiling_api
)

# --- 安装规则 ---
install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION ${INSTALL_DIR}
)