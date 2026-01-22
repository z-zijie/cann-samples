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
 * \file matmul_kernel_aswt_impl.h
 * \brief
 */

#ifndef MATMUL_KERNEL_ASWT_IMPL_H
#define MATMUL_KERNEL_ASWT_IMPL_H

#include "../block/matmul_block_mmad_aswt.h"
#include "../block/matmul_block_scheduler_aswt.h"
#include "../utils/matmul_coord_utils.h"

namespace x {
namespace matmul {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockScheduler_>
class MatmulKernelAswtImpl {
public:
    __aicore__ inline MatmulKernelAswtImpl() {}
    __aicore__ inline ~MatmulKernelAswtImpl() {}

    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;

    static constexpr bool TRANS_A = BlockMmad::TRANS_A;
    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    // schedulerOp
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmad::L1TileShape,
                                               typename BlockMmad::L0TileShape, BlockScheduler>::SchedulerOp;
    // come from cann
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    // no need to have tensortrait
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    // shape
    TupleShape problemShape_{};

    struct AddrArguments {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };
    using AddrParams = AddrArguments;

    struct Arguments {
        ProblemShape problemShape;
        AddrArguments addrArgs;
        Arguments() = default;
    };
    struct Params {
        ProblemShape problemShape;
        AddrParams addrParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline static TupleShape ToShapeTuple(const ProblemShape& shape)
    {
        return {shape.m, shape.n, shape.k};
    }

    __aicore__ inline void Init(const Params& params)
    {
        problemShape_ = ToShapeTuple(params.problemShape);
        AddrParams blockMmadParams = params.addrParams;
        // Init GlobalTensor
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr));
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr));
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr));
    }

    __aicore__ inline void operator()(const Params& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // Instantiate mmadOp
        BlockMmad blockMmadOp;
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        // Init
        Init(params);

        BlockSchedulerOp bs(params.problemShape, curBlockIdx, blockNum, params.schParams);
        int64_t tileNum = bs.GetTileNum();
        TupleShape tileL1 = bs.GetTileL1Shape();
        TupleShape tileL0 = bs.GetTileL0Shape();
        int64_t realBlockNum = AscendC::Std::min(tileNum, blockNum);
        if (curBlockIdx >= realBlockNum) {
            return;
        }
        blockMmadOp.Init(problemShape_, tileL1, tileL0, bs.GetL1BuferNum_(), bs.GetL0cDB());
        // Process tiles in ping-pong mode
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            TupleL1L0Shape blockShape = bs.GetBlockShape(tileIdx);
            auto blockCoord = bs.GetBlockCoord(tileIdx);
            // cal offset between blocks
            auto blockOffset = GetOffsetWithoutLayout(blockCoord, problemShape_, TRANS_A, TRANS_B);
            if (Get<0>(blockShape) <= 0 || Get<1>(blockShape) <= 0) {
                return;
            }
            int64_t offsetA = Get<0>(blockOffset);
            int64_t offsetB = Get<1>(blockOffset);
            int64_t offsetC = Get<2>(blockOffset);
            blockMmadOp(cGlobal_[offsetC], aGlobal_[offsetA], bGlobal_[offsetB], blockShape);
        }
    }
};

} // namespace Kernel
} // namespace matmul
} // namespace x

#endif // MATMUL_KERNEL_ASWT_IMPL_H
