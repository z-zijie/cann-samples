/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file matmul_utils.h
 * \brief
 */

#ifndef MATMUL_UTILS_H
#define MATMUL_UTILS_H

#include "kernel_operator.h"

namespace x {
namespace matmul {

constexpr static uint16_t ND_DIM = 2;
constexpr static uint16_t TWO_ALIGN = 2;
constexpr static uint64_t DOUBLE_BUFFER_COUNT = 2;
// buffer size
constexpr static int64_t L1_SIZE = 512 * 1024;
constexpr static int64_t L0A_SIZE = 64 * 1024;
constexpr static int64_t L0B_SIZE = 64 * 1024;
constexpr static int64_t L0C_SIZE = 256 * 1024;
// sync flag
constexpr static uint16_t ZERO_FLAG = 0;
constexpr static uint16_t FIRST_FLAG = 1;
// c0 size
constexpr static int64_t B16_C0_SIZE = 16;
constexpr static int64_t B32_C0_SIZE = 8;

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t CeilAlign(uint64_t a, uint64_t b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
__aicore__ inline constexpr static int32_t GetC0Size()
{
    if (sizeof(T) == sizeof(float)) {
        return B32_C0_SIZE;
    }
    return B16_C0_SIZE;
}

} // namespace matmul
} // namespace x

#endif // MATMUL_UTILS_H