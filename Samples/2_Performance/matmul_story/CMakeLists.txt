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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})

add_executable(quant_matmul_template_aswt ${CMAKE_CURRENT_SOURCE_DIR}/matmul_stubs/examples/quant_matmul_mxfp4/quant_matmul_template_aswt.cpp)

target_include_directories(quant_matmul_template_aswt PRIVATE
    ${ASCEND_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_stubs/include
)
target_link_directories(quant_matmul_template_aswt PRIVATE ${ASCEND_DIR}/lib64)

target_compile_options(quant_matmul_template_aswt PRIVATE
    "-xasc"
    "--npu-arch=dav-3101"
    "-w"
    "-O3"
)

target_link_libraries(quant_matmul_template_aswt
    PRIVATE
    m
    dl
    platform
    tiling_api
)

# ============================================================================
# Install 规则说明
# ============================================================================
# 源码目录结构 (matmul_story 下):
#   common/                           <- 公共头文件与 golden 脚本
#   matmul_stubs/
#     include/                        <- 内核与调度等头文件
#     examples/quant_matmul_mxfp4/     <- MXFP4 样例源码
#   matmul_tutorials/                 <- 预留
#
# 执行 cmake --install build --prefix ./build_out 后，安装目录与源码结构对应:
#
#   <install_prefix>/
#   └── 2_Performance/
#       └── matmul_story/
#           ├── common/
#           │   └── golden/
#           │       └── quant_matmul_mxfp4/
#           │           ├── gen_data.py
#           │           └── verify_result.py
#           └── matmul_stubs/
#               └── examples/
#                   └── quant_matmul_mxfp4/
#                       └── quant_matmul_template_aswt    <- 可执行文件
#
# 使用方式 (在安装根 2_Performance/matmul_story 下):
#   1. 运行 common/golden/quant_matmul_mxfp4/gen_data.py 生成测试数据与 golden
#   2. 运行 matmul_stubs/examples/quant_matmul_mxfp4/quant_matmul_template_aswt 执行算子
#   3. 运行 common/golden/quant_matmul_mxfp4/verify_result.py 验证结果
# ============================================================================

# Install 规则 1: 可执行文件 -> matmul_stubs/examples/quant_matmul_mxfp4/
install(TARGETS quant_matmul_template_aswt
    RUNTIME DESTINATION 2_Performance/matmul_story/matmul_stubs/examples/quant_matmul_mxfp4
)

# Install 规则 2: golden 脚本 -> common/golden/quant_matmul_mxfp4/
# FILES_MATCHING PATTERN "*.py": 只复制 .py 文件，忽略其他文件
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/common/golden/quant_matmul_mxfp4/
    DESTINATION 2_Performance/matmul_story/common/golden/quant_matmul_mxfp4
    FILES_MATCHING
    PATTERN "*.py"
)
