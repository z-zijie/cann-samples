/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common_utils.h
 * \brief
 */

#ifndef UTILS_COMMON_UTILS_H
#define UTILS_COMMON_UTILS_H

#include "integral_constant.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "std/algorithm.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"


constexpr int64_t MATRIX_INNER_DIM_LIMIT_SIZE = 65536LL;
constexpr int32_t MATMUL_MNK_ALIGN = 16;
constexpr int32_t MATMUL_MNK_ALIGN_INT8 = 32;
constexpr int64_t DOUBLE_BUFFER_COUNT = 2LL;
constexpr int64_t UB_FLOAT_ALIGN_NUM = 8LL;
constexpr int64_t L1_EVENT_ID_OFFSET = 2LL;
constexpr uint32_t UB_ALIGN_SIZE = 32U;
constexpr uint32_t UB_SUB_BANK_LEN = 256U; // SUB0: 256, SUB1: 256B
constexpr uint32_t UB_TWO_BANK_ELEMS_B32 = 128U;
constexpr uint32_t UB_SUB_BANK_ELEMS_B32 = 64U; // SUB0: 64, SUB1: 64
constexpr uint32_t UB_SUB_BANK_NUM = 2U;
constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4U;
constexpr int MNK_M = 0;
constexpr int MNK_N = 1;
constexpr int MNK_K = 2;
constexpr int MNK_B = 3;
constexpr int MNK_M0 = 4;
constexpr int MNK_N0 = 5;

constexpr static uint64_t A_FULL_LOAD_MODE = 1UL;
constexpr static uint64_t B_FULL_LOAD_MODE = 2UL;
constexpr static uint64_t NONE_FULL_LOAD_MODE = 0UL;
constexpr static int64_t PER_BLOCK_SIZE = 128LL;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr uint32_t C0_SIZE_B8 = 32UL;
// FusedMatMul OpType
constexpr static uint64_t OP_TYPE_EMPTY = 0UL;
constexpr static uint64_t OP_TYPE_ADD = 1UL;
constexpr static uint64_t OP_TYPE_MUL = 2UL;
constexpr static uint64_t OP_TYPE_RELU = 5UL;
constexpr uint64_t BLOCK_BYTE_SIZE = 32;
struct MatmulShape {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t b;
};

__host_aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__host_aicore__ inline int64_t CeilAlign(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}


#endif