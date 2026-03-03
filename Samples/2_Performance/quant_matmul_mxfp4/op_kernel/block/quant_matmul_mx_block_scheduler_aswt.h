/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/* !
 * \file quant_matmul_mx_block_scheduler_aswt.h
 * \brief
 */

#ifndef QUANT_MATMUL_MX_BLOCK_SCHEDULER_ASWT_H
#define QUANT_MATMUL_MX_BLOCK_SCHEDULER_ASWT_H

#include "matmul_block_scheduler.h"
#include "matmul_block_scheduler_policy.h"
#include "../utils/quant_matmul_tiling_data.h"

namespace ascend_ops {
namespace matmul {
namespace Block {

template <class ProblemShape_, class L1TileShape_, class L0TileShape_>
class QuantMatmulMxBlockSchedulerAswtBuiltIn {
public:
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t blockNum_{0};
    int64_t k_{0};
    int64_t tailL1M_{0};
    int64_t tailL1N_{0};
    int64_t tileNum_{1};
    int64_t mainWindow_{1};
    int64_t mainRow_{1};
    int64_t tailWindow_{1};
    int64_t mTileIdx_{1};
    int64_t nTileIdx_{1};
    int64_t lastTileIdx_{-1};
    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t kL1_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t baseK_{0};
    uint8_t l1BuferNum_{0};
    uint8_t l0cDB_{1};

    static constexpr uint64_t WINDOW_LEN = 4UL; // init slide window len
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        const QuantMatmulTilingData* tilingData;
    };

public:
    __aicore__ inline QuantMatmulMxBlockSchedulerAswtBuiltIn(const ProblemShape& shape, int64_t blockNum,
                                                             const Params& params) : blockNum_(blockNum)
    {
        k_ = shape.k;
        mL1_ = params.tilingData->mL1;
        nL1_ = params.tilingData->nL1;
        kL1_ = params.tilingData->kL1;
        baseM_ = params.tilingData->baseM;
        baseN_ = params.tilingData->baseN;
        baseK_ = params.tilingData->baseK;
        l1BuferNum_ = params.tilingData->l1BufferNum;
        l0cDB_ = params.tilingData->l0cDB;
        mTileNum_ = CeilDiv(shape.m, params.tilingData->mL1);
        nTileNum_ = CeilDiv(shape.n, params.tilingData->nL1);
        tileNum_ = mTileNum_ * nTileNum_;
        int64_t tailTileNum = tileNum_ % blockNum_;
        tailL1M_ = shape.m - (mTileNum_ - 1) * params.tilingData->mL1;
        tailL1N_ = shape.n - (nTileNum_ - 1) * params.tilingData->nL1;

        mainWindow_ = WINDOW_LEN < mTileNum_ ? WINDOW_LEN : mTileNum_;
        mainRow_ = mTileNum_ / mainWindow_ - 1;
        tailWindow_ = mTileNum_ - mainRow_ * mainWindow_;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return tileNum_;
    }

    __aicore__ inline uint64_t GetL1BuferNum_()
    {
        return static_cast<uint64_t>(l1BuferNum_);
    }

    __aicore__ inline bool GetL0cDB()
    {
        return l0cDB_ > 1;
    }

    __aicore__ inline AscendC::Shape<int64_t, int64_t, int64_t> GetTileL1Shape()
    {
        return {mL1_, nL1_, kL1_};
    }

    __aicore__ inline AscendC::Shape<int64_t, int64_t, int64_t> GetTileL0Shape()
    {
        return {baseM_, baseN_, baseK_};
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t blkM = mTileIdx_ == (mTileNum_ - 1) ? tailL1M_ : mL1_;
        int64_t blkN = nTileIdx_ == (nTileNum_ - 1) ? tailL1N_ : nL1_;
        int64_t mL0 = blkM;
        int64_t nL0 = blkN;
        // mL1, nL1, k, mL0, nL0
        mL0 = AscendC::Std::min(baseM_, blkM);
        nL0 = AscendC::Std::min(baseN_, blkN);
        return {blkM, blkN, k_, mL0, nL0};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t mOffset = mTileIdx_ * mL1_;
        int64_t nOffset = nTileIdx_ * nL1_;
        return {mOffset, nOffset};
    }

private:
    __aicore__ inline void UpdateMNTileIdx(int64_t tmpIdx)
    {
        if (lastTileIdx_ == tmpIdx) {
            return;
        }
        lastTileIdx_ = tmpIdx;

        int64_t tileIdx = tmpIdx % tileNum_;
        int64_t rowIdx = tileIdx / nTileNum_ / mainWindow_;
        if (rowIdx < mainRow_) {
            mTileIdx_ = rowIdx * mainWindow_ + tileIdx % mainWindow_;
            nTileIdx_ = (tileIdx / mainWindow_) % nTileNum_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIndex = tileIdx - mainRow_ * mainWindow_ * nTileNum_;
            mTileIdx_ = mainRow_ * mainWindow_ + tailIndex % tailWindow_;
            nTileIdx_ = (tailIndex / tailWindow_) % nTileNum_;
        }
        if (rowIdx % 2 != 0) { // mode 2 means even row, need reverse scan
            nTileIdx_ = nTileNum_ - 1 - nTileIdx_;
        }
    }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, QuantMatmulMxBuiltInAswtScheduler> {
    using SchedulerOp = QuantMatmulMxBlockSchedulerAswtBuiltIn<ProblemShape_, L1TileShape_, L0TileShape_>;
};

} // namespace Block
} // namespace matmul
} // namespace ascend_ops

#endif // QUANT_MATMUL_MX_BLOCK_SCHEDULER_ASWT_H