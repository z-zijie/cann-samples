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
 * \file matmul_block_mmad.h
 * \brief
 */
#ifndef MATMUL_BLOCK_MMAD_H
#define MATMUL_BLOCK_MMAD_H

#include "../utils/matmul_integral_constant.h"

namespace ascend_ops {
namespace matmul {
namespace Block {

/**
 * @class BlockMmad
 * @brief Block matrix multiplication class for performing block matrix multiplication operations
 */
template <
    /// The dispatch policy type, which determines the computational pipeline
    class DispatchPolicy,
    /// The shape of L1 tile
    class L1TileShape,
    /// The shape of L0 tile
    class L0TileShape,
    /// Type of matrix A
    class AType,
    /// Layout of matrix A
    class LayoutA,
    /// Type of matrix B
    class BType,
    /// Layout of matrix B
    class LayoutB,
    /// Type of matrix C
    class CType,
    /// Layout of matrix C
    class LayoutC,
    /// Support specialization via the DispatchPolicy type
    typename = void>
class BlockMmad {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

} // namespace Block
} // namespace matmul
} // namespace ascend_ops

#endif // MATMUL_BLOCK_MMAD_H
