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
 * \file quant_matmul_tiling_common.h
 * \brief
 */
#ifndef QUANT_MATMUL_TILING_COMMON_H
#define QUANT_MATMUL_TILING_COMMON_H

#include "tiling/platform/platform_ascendc.h"

constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t WINDOW_LEN = 4UL;
constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t FP4_C0_SIZE = 64UL
constexpr uint64_t BASEK_LIMIT = 4095UL;
constexpr uint64_t DATA_SIZE_L0C = 4UL;
constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr uint64_t L1_FOUR_BUFFER = 4UL;
constexpr uint64_t STEPK_THERSHOLD = 4UL;
constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
constexpr uint64_t BASEM_BASEN_RATIO = 2UL;
constexpr uint64_t SCALER_FACTOR_MIN = 1UL;
constexpr uint64_t SCALER_FACTOR_MAX = 127UL;
constexpr uint64_t MTE2_MIN_LOAD_SIZE = 32768UL;
constexpr uint64_t MTE2_CACHELINE_SIZE = 128UL;
constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_512 = 512UL;

struct QuantMatmulPlatformInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0cSize{0UL};
    uint64_t l0aSize{0UL};
    uint64_t l0bSize{0UL};
    uint64_t btSize{0UL};
    platform_ascendc::SocVersion socVersion;
};

struct QuantMatmulArgs {
    uint64_t m = 0UL;
    uint64_t k = 0UL;
    uint64_t n = 0UL;
};

struct QuantMatmulRunInfo {
    uint64_t baseM{0UL};
    uint64_t baseN{0UL};
    uint64_t baseK{0UL};
    uint64_t stepKa{0UL};
    uint64_t stepKb{0UL};
    uint64_t depthA1{0UL};
    uint64_t depthB1{0UL};
    uint64_t mL1{0UL};
    uint64_t nL1{0UL};
    uint64_t kL1{0UL};
    uint64_t mTailCnt{0UL};
    uint64_t nTailCnt{0UL};
    uint64_t usedCoreNum{0UL};
    uint64_t l1BufferNum{0UL};
    uint64_t dbL0C{0UL};
    uint64_t mBlockCnt{0UL};
    uint64_t nBlockCnt{0UL};
    uint64_t totalBlockCnt{0UL};
    uint64_t mTailSize{0UL};
    uint64_t nTailSize{0UL};
    uint64_t calRoundCnt{0UL};
    uint64_t tailBlockCnt{0UL};
    uint64_t mBaseTailSplitCnt{1UL};
    uint64_t mTailMain{0UL};
    uint64_t scaleFactorA{0UL};
    uint64_t scaleFactorB{0UL};
    bool isAFullLoad{false};
};

#endif // QUANT_MATMUL_TILING_COMMON_H