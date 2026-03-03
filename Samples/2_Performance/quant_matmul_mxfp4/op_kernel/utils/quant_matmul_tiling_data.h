/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 202 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_tiling_data.h
 * \brief
 */
#ifndef QUANT_MATMUL_TILING_DATA_H
#define QUANT_MATMUL_TILING_DATA_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace ascend_ops {
namespace matmul {

#pragma pack(push, 8)
struct QuantMatmulTilingData {
    uint32_t usedCoreNum = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t mL1 = 0;
    uint32_t nL1 = 0;
    uint32_t kL1 = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint8_t l1BufferNum = 0;
    uint8_t l0cDB = 1; // 默认不开db为1
};
#pragma pack(pop)

} // namespace matmul
} // namespace ascend_ops

#endif // QUANT_MATMUL_TILING_DATA_H
