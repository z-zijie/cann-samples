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
 * \file matmul_coord_utils.h
 * \brief
 */

#ifndef MATMUL_COORD_UTILS_H
#define MATMUL_COORD_UTILS_H

namespace ascend_ops {
namespace matmul {

#include "matmul_integral_constant.h"

constexpr uint32_t OUTER_SIZE = 16;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;

// GetOffsetWithoutLayout
template <class BlockCoord, class ProblemShape>
__aicore__ inline AscendC::Coord<int64_t, int64_t, int64_t>
GetOffsetWithoutLayout(BlockCoord blockCoord, ProblemShape problemShape, bool transA, bool transB)
{
    int64_t m = Get<MNK_M>(problemShape);
    int64_t n = Get<MNK_N>(problemShape);
    int64_t k = Get<MNK_K>(problemShape);
    int64_t mOffset = Get<0>(blockCoord);
    int64_t nOffset = Get<1>(blockCoord);
    int64_t offsetA = transA ? mOffset : mOffset * k;
    int64_t offsetB = transB ? nOffset * k : nOffset;
    int64_t offsetC = mOffset * n + nOffset;
    return {offsetA, offsetB, offsetC};
}

} // namespace matmul
} // namespace ascend_ops

#endif // MATMUL_COORD_UTILS_H