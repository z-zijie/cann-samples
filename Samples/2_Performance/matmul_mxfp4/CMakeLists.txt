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

<<<<<<< HEAD:Samples/2_Performance/matmul_mxfp4/CMakeLists.txt
add_executable(matmul_mxfp4 matmul_mxfp4.cpp)

target_include_directories(matmul_mxfp4 PRIVATE ${ASCEND_INCLUDE_DIRS})
target_link_directories(matmul_mxfp4 PRIVATE ${ASCEND_DIR}/lib64)

target_compile_options(matmul_mxfp4 PRIVATE
=======
add_executable(quant_matmul_mxfp4 quant_matmul_mxfp4.cpp)

target_include_directories(quant_matmul_mxfp4 PRIVATE ${ASCEND_INCLUDE_DIRS})
target_link_directories(quant_matmul_mxfp4 PRIVATE ${ASCEND_DIR}/lib64)

target_compile_options(quant_matmul_mxfp4 PRIVATE
>>>>>>> d24c893... d:Samples/2_Performance/quant_matmul_mxfp4/CMakeLists.txt
    "-xasc"
    "--npu-arch=dav-3101"
    "-w"
    "-O3"
)

<<<<<<< HEAD:Samples/2_Performance/matmul_mxfp4/CMakeLists.txt
target_link_libraries(matmul_mxfp4
=======
target_link_libraries(quant_matmul_mxfp4
>>>>>>> d24c893... d:Samples/2_Performance/quant_matmul_mxfp4/CMakeLists.txt
    PRIVATE
    m
    dl
    platform
    tiling_api
)

<<<<<<< HEAD:Samples/2_Performance/matmul_mxfp4/CMakeLists.txt
install(TARGETS matmul_mxfp4
=======
install(TARGETS quant_matmul_mxfp4
>>>>>>> d24c893... d:Samples/2_Performance/quant_matmul_mxfp4/CMakeLists.txt
    RUNTIME DESTINATION 2_Performance/matmul
)

