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
 * \file matmul_tiling_common.h
 * \brief
 */
#ifndef MATMUL_TILING_COMMON_H
#define MATMUL_TILING_COMMON_H

#include "tiling/platform/platform_ascendc.h"

namespace x {
namespace matmul {
constexpr uint8_t DATA_SIZE_FP32 = 4UL;
constexpr uint8_t DATA_SIZE_FP16 = 2UL;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_K_128_BYTE = 128UL;
constexpr uint64_t BASE_STEP = 1UL;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t DB_OFF_SIZE = 1UL;
constexpr uint64_t NUM_TWO = 2;
constexpr uint64_t STEPKA_THERSHOLD = 4;

enum class DataType : std::uint8_t { FLOAT = 0, FLOAT16, BFLOAT16 };

enum class Format : std::uint8_t { ND = 0, NZ };

struct MatmulCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0CSize{0UL};
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
    uint64_t btSize{0UL};
    platform_ascendc::SocVersion socVersion;
};

struct MatmulArgs {
    bool transA = false;
    bool transB = false;
    uint64_t m = 0UL;
    uint64_t k = 0UL;
    uint64_t n = 0UL;
    Format formatA = Format::ND;
    Format formatB = Format::ND;
    Format formatC = Format::ND;
    DataType dtype = DataType::BFLOAT16;
    uint64_t dtypeSize = DATA_SIZE_FP16;
};

struct MatmulRunInfo {
    uint64_t usedCoreNum = 1UL;
    uint64_t baseM = 1UL;
    uint64_t baseN = 1UL;
    uint64_t baseK = 1UL;
    uint64_t stepM = 1UL;
    uint64_t stepN = 1UL;
    uint64_t stepK = 1UL;
    uint64_t mL1 = 1UL;
    uint64_t nL1 = 1UL;
    uint64_t kL1 = 1UL;
    uint64_t singleCoreK = 1UL;
    uint8_t l1BufferNum = 2UL;
    uint8_t l0cDB = 0UL;
};

} // namespace matmul
} // namespace x

#endif // MATMUL_TILING_COMMON_H