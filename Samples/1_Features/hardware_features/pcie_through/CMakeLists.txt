# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cann_sample_check_arch(dav-3510)

set(OP_GRAPH_INC ${ASCEND_DIR}/opp/built-in/op_graph/inc)

# aclnn direct invocation demo
add_executable(pcie_through_gather_v2_aclnn src/pcie_through_gather_v2_aclnn.cpp)
target_include_directories(pcie_through_gather_v2_aclnn PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
)
target_link_directories(pcie_through_gather_v2_aclnn PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
)
target_link_libraries(pcie_through_gather_v2_aclnn PRIVATE
    ascendcl
    nnopbase
    opapi
    runtime
    stdc++
)

# kernel launch demo: uses <<<>>> syntax, compiled as .asc with bisheng
add_executable(pcie_through_gather_v2_kernel_launch src/pcie_through_gather_v2_kernel_launch.asc)
target_include_directories(pcie_through_gather_v2_kernel_launch PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
)
target_compile_options(pcie_through_gather_v2_kernel_launch PRIVATE
    "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
    "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
)
target_link_directories(pcie_through_gather_v2_kernel_launch PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
)
target_link_libraries(pcie_through_gather_v2_kernel_launch PRIVATE
    m
    dl
    platform
    tiling_api
    ascendcl
    runtime
    stdc++
)

install(TARGETS
        pcie_through_gather_v2_aclnn
        pcie_through_gather_v2_kernel_launch
    RUNTIME DESTINATION 1_Features/hardware_features/pcie_through
)
