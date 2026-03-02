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
 * \file main.cpp
 * \brief Vector Addition Example
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <tuple>
#include <algorithm>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

typedef float xType;
typedef int32_t indexType;
typedef int64_t countType;

constexpr int64_t SIMT_THREAD_NUM = 2048;

// std::tuple<int64_t, int64_t, int64_t> calc_tiling_params(int64_t totalLength)
// {
//     // TODO UB计算
//     constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
//     constexpr static int64_t PIPELINE_DEPTH = 2;
//     // constexpr static int64_t BUFFER_NUM = 3;
//     auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
//     uint64_t ubSize;
//     ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
//     int64_t coreNum = ascendcPlatform->GetCoreNumAiv();

//     // 使用核数
//     int64_t numBlocks = 1;
//     // int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
//     // numBlocks = std::max(numBlocks, static_cast<int64_t>(1));

//     // 每个核处理的长度
//     int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
//     int64_t tileSize = ubSize / PIPELINE_DEPTH / BUFFER_NUM;
//     return std::make_tuple(numBlocks, blockLength, tileSize);
// }

struct MoeSortTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreElements{0};
    int64_t perCoreLoops{0};
    int64_t perCorePerLoopElements{0};
    int64_t perCoreLastLoopElements{0};
    int64_t lastCoreElements{0};
    int64_t lastCoreLoops{0};
    int64_t lastCorePerLoopElements{0};
    int64_t lastCoreLastLoopElements{0};
    int64_t oneLoopMaxElements{0};
};

struct MoeTokensCountTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreElements{0};
    int64_t perCoreLoops{0};
    int64_t perCorePerLoopElements{0};
    int64_t perCoreLastLoopElements{0};
    int64_t lastCoreElements{0};
    int64_t lastCoreLoops{0};
    int64_t lastCorePerLoopElements{0};
    int64_t lastCoreLastLoopElements{0};
};

struct MoeGatherOutTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreIndicesElements{0};
    int64_t lastCoreIndicesElements{0};
    int64_t perCoreIndicesLoops{0};
    int64_t perCorePerLoopIndicesElements{0};
    int64_t perCoreLastLoopIndicesElements{0};
    int64_t lastCoreIndicesLoops{0};
    int64_t lastCorePerLoopIndicesElements{0};
    int64_t lastCoreLastLoopIndicesElements{0};
    int64_t colsLoops{0};
    int64_t perLoopCols{0};
    int64_t lastLoopCols{0};
    int64_t activeNum{0};
};

struct MoeInitRoutingTilingData {
    MoeSortTilingData sortTilingData;
    MoeTokensCountTilingData countTilingData;
    MoeGatherOutTilingData gatherTilingData;
    int64_t n{0};
    int64_t cols{0};
    int64_t k{0};
    int64_t expertStart{0};
    int64_t expertEnd{0};
    int64_t expertNum{0};
    int64_t expertTokensNumType{0};
};

template <typename X_TYPE, typename EXPERT_IDX_TYPE>
class ExpertIdxSort {
private:
    constexpr static int64_t PIPELINE_DEPTH = 1;
    constexpr static int64_t ONE_REPEAT_SORT_NUM = 32; // 排序元素对齐32，sort api要求
    constexpr static int64_t FP32_ONE_REPEAT_NUM = 64;
    constexpr static float MIN_FP32 = -3.4e38f;
    constexpr static int64_t DST_BLK_STRIDE = 1;
    constexpr static int64_t DST_REP_STRIDE = 8;

    // sort init

    AscendC::GlobalTensor<int32_t> expertIdxGm;
    AscendC::GlobalTensor<int32_t> sortedexpertIdxGm;
    AscendC::GlobalTensor<int32_t> expandedRowIdxGm;

    TQue<QuePosition::VECIN, PIPELINE_DEPTH> sortDataCopyInQueue;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> sortDataCopyOutQueue;
    TBuf<TPosition::VECCALC> tempBuffer;
    TBuf<TPosition::VECCALC> sortedBuffer;

    TPipe *pipe_;
    int64_t blockIdx_ = 0;
    int64_t totalLength = 0;
    int64_t sortNum = 0;
    int64_t tileLength = 0;
    MoeInitRoutingTilingData *tilingData_;

public:
    __aicore__ inline ExpertIdxSort()
    {}

    __aicore__ inline void Init(__gm__ EXPERT_IDX_TYPE *expertIdx,  __gm__ EXPERT_IDX_TYPE *workspace, __gm__ EXPERT_IDX_TYPE *expandedRowIdx,
        MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
    {
        blockIdx_ = AscendC::GetBlockIdx();
        tilingData_ = tilingData;
        pipe_ = tPipe;
        totalLength = tilingData_->sortTilingData.perCoreElements;

        int64_t kvFactor = 2; // key and value
        sortNum = (totalLength + ONE_REPEAT_SORT_NUM - 1) / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;
        int64_t buffSize = sortNum * sizeof(EXPERT_IDX_TYPE) * kvFactor;
        pipe_->InitBuffer(sortDataCopyInQueue, PIPELINE_DEPTH, buffSize);
        pipe_->InitBuffer(sortDataCopyOutQueue, PIPELINE_DEPTH, buffSize);
        pipe_->InitBuffer(tempBuffer, buffSize);
        pipe_->InitBuffer(sortedBuffer, buffSize);

        tileLength = (totalLength * sizeof(EXPERT_IDX_TYPE) + 32 - 1) / 32 * 32 / sizeof(EXPERT_IDX_TYPE);
        expertIdxGm.SetGlobalBuffer(expertIdx, tileLength);
        sortedexpertIdxGm.SetGlobalBuffer(workspace,
                                          (totalLength * sizeof(EXPERT_IDX_TYPE) + 32 - 1) / 32 * 32 / sizeof(EXPERT_IDX_TYPE));
        expandedRowIdxGm.SetGlobalBuffer(expandedRowIdx, tileLength);
    }

    __aicore__ inline void CopyIn()
    {
        LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                        static_cast<uint32_t>(totalLength * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
        DataCopyPad(inLocal[0], expertIdxGm, dataCopyParams, dataCopyPadParams);
        LocalTensor<int32_t> rowIdxLocal = inLocal[sortNum];
        ArithProgression<int32_t>(rowIdxLocal, 0, 1, sortNum);
        sortDataCopyInQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
        DumpTensor(inLocal, 20, 64);
        LocalTensor<int32_t> expertForSourceRowLocal = inLocal[0];
        LocalTensor<float> expertForSourceRowLocalFp32 = expertForSourceRowLocal.ReinterpretCast<float>();
        Cast(expertForSourceRowLocalFp32, expertForSourceRowLocal, RoundMode::CAST_ROUND, tileLength);
        PipeBarrier<PIPE_V>();
        Muls(expertForSourceRowLocalFp32, expertForSourceRowLocalFp32, (float)-1, tileLength);
        PipeBarrier<PIPE_V>();

        int64_t duplicateNum = totalLength % ONE_REPEAT_SORT_NUM;
        if (duplicateNum > 0) {
            int duplicateIndex = totalLength - duplicateNum;
            uint64_t mask0 = UINT64_MAX;
            mask0 = mask0 << duplicateNum;
            mask0 = mask0 & (UINT64_MAX >> (FP32_ONE_REPEAT_NUM - ONE_REPEAT_SORT_NUM));
            uint64_t mask[2] = {mask0, 0};
            Duplicate(expertForSourceRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> concatLocal = expertForSourceRowLocalFp32;
        LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(sortNum));
        Concat(concatLocal, expertForSourceRowLocalFp32, tempTensor, sortNum / ONE_REPEAT_SORT_NUM);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(sortNum));
        LocalTensor<uint32_t> sourceRowLocal;
        sourceRowLocal = inLocal[sortNum].ReinterpretCast<uint32_t>();
        Sort<float, true>(sortedLocal, concatLocal, sourceRowLocal, tempTensor, sortNum / ONE_REPEAT_SORT_NUM);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
        LocalTensor<float> sortedExpertForSourceRowLocal = outLocal[0];
        LocalTensor<uint32_t> expandDstToSrcRowLocal;
        expandDstToSrcRowLocal = outLocal[sortNum].ReinterpretCast<uint32_t>();
        Extract(sortedExpertForSourceRowLocal, expandDstToSrcRowLocal, sortedLocal, sortNum / ONE_REPEAT_SORT_NUM);
        PipeBarrier<PIPE_V>();
        Muls(sortedExpertForSourceRowLocal, sortedExpertForSourceRowLocal, (float)-1, tileLength);
        PipeBarrier<PIPE_V>();

        LocalTensor<int32_t> expertForSourceRowLocalInt32;
        expertForSourceRowLocalInt32 = sortedExpertForSourceRowLocal.ReinterpretCast<int32_t>();
        Cast(expertForSourceRowLocalInt32, sortedExpertForSourceRowLocal, RoundMode::CAST_ROUND, tileLength);
        sortDataCopyOutQueue.EnQue<float>(outLocal);
        sortDataCopyInQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = totalLength * sizeof(int32_t);
        DataCopyPad(sortedexpertIdxGm, outLocal[0], intriParams);
        DataCopyPad(expandedRowIdxGm, outLocal[sortNum], intriParams);
        sortDataCopyOutQueue.FreeTensor(outLocal);
        DumpTensor(sortedexpertIdxGm, 21, 32);
        DumpTensor(expandedRowIdxGm, 22, 32);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ < 1) {
            CopyIn();
            Compute();
            CopyOut();
        }
    }
};

template <typename EXPERT_IDX_TYPE, typename TOKEN_COUNT_TYPE>
class ExpertTokensCount {
private:
    constexpr static int64_t PIPELINE_DEPTH = 1;
    constexpr static int64_t ONE_REPEAT_SORT_NUM = 32; // 排序元素对齐32，sort api要求
    constexpr static int64_t BLOCK_BYTES = 32;
    constexpr static int64_t SIMT_THREAD_NUM = 2048;

    GlobalTensor<EXPERT_IDX_TYPE> sortedExpertIdxGm_;
    GlobalTensor<EXPERT_IDX_TYPE> expertCountTempGm_;
    GlobalTensor<TOKEN_COUNT_TYPE> expertTokensCountGm_;
    GlobalTensor<EXPERT_IDX_TYPE> expertTotalCountGm_;
    GlobalTensor<EXPERT_IDX_TYPE> expandedRowIdxGm_;

    TQue<QuePosition::VECIN, PIPELINE_DEPTH> sortedExpertIdxInQueue_;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> expertCountOutToTempQueue_;
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> expertCountTempInQueue_;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> expertIdxCountOutQueue_;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> expertTotalCountQueue_;

    TPipe *pipe_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t curCoreElements_ = 0;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t actualExpertNum_ = 0;
    int64_t coreLoopsNum_ = 0;
    int64_t perCorePerLoopElements_ = 0;
    int64_t perCoreLastLoopElements_ = 0;
    int64_t actualExpertTotalNum_ = 0;
    int64_t expertNum_ = 0;
    int64_t expertTokensNumType_ = 0;
    int64_t expertCountElements_ = 0;
    MoeTokensCountTilingData *expertTokensCountTilingData_;

public:
    __aicore__ inline ExpertTokensCount()
    {}

    __aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
    {
        if (bytes == 0) {
            return 0;
        }
        return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
    }

    __aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
    {
        return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
    }

    __aicore__ inline void Init(__gm__ EXPERT_IDX_TYPE *expandedRowIdx,  __gm__ TOKEN_COUNT_TYPE *expertTokensCount, __gm__ EXPERT_IDX_TYPE *workspace,
        MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
    {
        blockIdx_ = AscendC::GetBlockIdx();
        pipe_ = tPipe;
        expertTokensCountTilingData_ = &(tilingData->countTilingData);

        needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
        perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
        expertStart_ = tilingData->expertStart;
        expertEnd_ = tilingData->expertEnd;
        actualExpertNum_ = expertEnd_ - expertStart_;
        expertNum_ = tilingData->expertNum;
        expertTokensNumType_ = tilingData->expertTokensNumType;
        expertCountElements_ = actualExpertNum_;

        if (blockIdx_ == needCoreNum_ - 1) {
            curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
            coreLoopsNum_ = expertTokensCountTilingData_->lastCoreLoops;
            perCorePerLoopElements_ = expertTokensCountTilingData_->lastCorePerLoopElements;
            perCoreLastLoopElements_ = expertTokensCountTilingData_->lastCoreLastLoopElements;
        } else {
            curCoreElements_ = expertTokensCountTilingData_->perCoreElements;
            coreLoopsNum_ = expertTokensCountTilingData_->perCoreLoops;
            perCorePerLoopElements_ = expertTokensCountTilingData_->perCorePerLoopElements;
            perCoreLastLoopElements_ = expertTokensCountTilingData_->perCoreLastLoopElements;
        }

        sortedExpertIdxGm_.SetGlobalBuffer(workspace + blockIdx_ * perCoreElements_, curCoreElements_);
        expertTokensCountGm_.SetGlobalBuffer(expertTokensCount, expertCountElements_);
        expertCountTempGm_.SetGlobalBuffer(workspace +
                                           Align(tilingData->n * tilingData->k, sizeof(EXPERT_IDX_TYPE)) * 2, 
                                           actualExpertNum_);
        expertTotalCountGm_.SetGlobalBuffer(workspace + 
                                            Align(tilingData->n * tilingData->k, sizeof(EXPERT_IDX_TYPE)) * 2 + 
                                            Align(actualExpertNum_, sizeof(EXPERT_IDX_TYPE)), actualExpertNum_);
        expandedRowIdxGm_.SetGlobalBuffer(expandedRowIdx + blockIdx_ * perCoreElements_);

        DumpTensor(sortedExpertIdxGm_, 30, 32);
        DumpTensor(expandedRowIdxGm_, 31, 32);

        pipe_->InitBuffer(sortedExpertIdxInQueue_, PIPELINE_DEPTH, 
            AlignBytes(perCorePerLoopElements_, sizeof(EXPERT_IDX_TYPE)));
        pipe_->InitBuffer(expertCountOutToTempQueue_, PIPELINE_DEPTH, 
            AlignBytes(actualExpertNum_, sizeof(EXPERT_IDX_TYPE)));
        pipe_->InitBuffer(expertCountTempInQueue_, PIPELINE_DEPTH, 
            AlignBytes(actualExpertNum_, sizeof(EXPERT_IDX_TYPE)));
        pipe_->InitBuffer(expertIdxCountOutQueue_, PIPELINE_DEPTH, 
            AlignBytes(expertCountElements_, sizeof(TOKEN_COUNT_TYPE)));
        pipe_->InitBuffer(expertTotalCountQueue_, PIPELINE_DEPTH, 
            AlignBytes(1, sizeof(EXPERT_IDX_TYPE)));
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.DeQue<int32_t>();

        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)),
                                    0, 0, 0};
        SetAtomicAdd<int32_t>();
        DataCopyPad(expertCountTempGm_, expertCountOutLocal, copyParams);
        SetAtomicNone();
        expertCountOutToTempQueue_.FreeTensor(expertCountOutLocal);
    }

    __aicore__ inline void ExpertCountCopyIn()
    {
        LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.AllocTensor<int32_t>();

        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                        static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
        DataCopyPad(expertCountTempInLocal, expertCountTempGm_, dataCopyParams, dataCopyPadParams);
        expertCountTempInQueue_.EnQue(expertCountTempInLocal);
    }

    __aicore__ inline void ExpertCountCompute()
    {
        LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.DeQue<int32_t>();
        LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.AllocTensor<int64_t>();
        LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.AllocTensor<int32_t>();
        event_t eventIDMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
            expertCountOutLocal.SetValue(i, expertCount);
            actualExpertTotalNum_ += expertCount;
        }
        expertTotalCountLocal.SetValue(0, static_cast<int32_t>(actualExpertTotalNum_));
        event_t eventIDSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMte3);
        expertIdxCountOutQueue_.EnQue<int64_t>(expertCountOutLocal);
        expertTotalCountQueue_.EnQue<int32_t>(expertTotalCountLocal);
        expertCountTempInQueue_.FreeTensor(expertCountTempInLocal);
    }

    __aicore__ inline void ExpertCountCopyOut()
    {
        LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.DeQue<int64_t>();
        LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.DeQue<int32_t>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                    static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(expertTokensCountGm_, expertCountOutLocal, copyParams);
        copyParams.blockLen = sizeof(int32_t);
        DataCopyPad(expertTotalCountGm_, expertTotalCountLocal, copyParams);
        expertIdxCountOutQueue_.FreeTensor(expertCountOutLocal);
        expertTotalCountQueue_.FreeTensor(expertTotalCountLocal);
    }

    __aicore__ inline void Process();
    // {
        // if (blockIdx_ < needCoreNum_) {
        //     LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
        //     Duplicate(expertCountOutLocal, 0, actualExpertNum_);

        //     __gm__ int32_t *sortedExpertIdxGmAddr = (__gm__ int32_t *)sortedExpertIdxGm_.GetPhyAddr();
        //     __ubuf__ int32_t *expertCountOutLocalAddr = (__ubuf__ int32_t *)expertCountOutLocal.GetPhyAddr();

        //     asc_vf_call<ComputeExpertFirstIndexSimt>(dim3(SIMT_THREAD_NUM), curCoreElements_, expertStart_,
        //                                             expertEnd_, sortedExpertIdxGmAddr, expertCountOutLocalAddr);
        //     asc_vf_call<ComputeExpertCountOutSimt>(dim3(SIMT_THREAD_NUM), curCoreElements_, expertStart_,
        //                                             expertEnd_, sortedExpertIdxGmAddr, expertCountOutLocalAddr,
        //                                             expertCountOutLocalAddr);
        //     DumpTensor(expertTokensCountGm_, 32, 32);                                        
        //     expertCountOutToTempQueue_.EnQue<int32_t>(expertCountOutLocal);
        //     CopyOut();
        // }

        // SyncAll();
        // /* copy expert tokens count result from worksapce to output GM. */
        // if (blockIdx_ == 0) {
        //     ExpertCountCopyIn();
        //     ExpertCountCompute();
        //     ExpertCountCopyOut();
        // }
        // SyncAll();
        // DumpTensor(expertTokensCountGm_, 33, 32);
    // }
};

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertFirstIndexSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __gm__ int32_t *sortedExpertIdGmAddr,
    __ubuf__ int32_t *expertFirstIndexLocalAddr)
{
    auto threadIdx = static_cast<int32_t>(Simt::GetThreadIdx());
    auto threadNum = static_cast<int32_t>(Simt::GetThreadNum());
    for (auto i = threadIdx; i < elementNum; i += threadNum) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId >= expertEnd) {
            break;
        }
        auto prevExpertId = (i == 0 ? -1 : sortedExpertIdGmAddr[i - 1]);
        if (currExpertId != prevExpertId) {
            expertFirstIndexLocalAddr[currExpertId - expertStart] = i;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertCountOutSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __gm__ int32_t *sortedExpertIdGmAddr,
    __ubuf__ int32_t *expertFirstIndexLocalAddr, __ubuf__ int32_t *expertCountOutLocalAddr)
{
    auto threadIdx = static_cast<int32_t>(Simt::GetThreadIdx());
    auto threadNum = static_cast<int32_t>(Simt::GetThreadNum());
    for (auto i = threadIdx; i < elementNum; i += threadNum) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId >= expertEnd) {
            break;
        }
        if (i == elementNum - 1 || currExpertId != sortedExpertIdGmAddr[i + 1]) {
            expertCountOutLocalAddr[currExpertId - expertStart] =
                i + 1 - expertFirstIndexLocalAddr[currExpertId - expertStart];
        }
    }
}

// __aicore__ inline void ExpertTokensCount::Process()
// {
//     if (blockIdx_ < needCoreNum_) {
//         LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
//         Duplicate(expertCountOutLocal, 0, actualExpertNum_);

//         __gm__ int32_t *sortedExpertIdxGmAddr = (__gm__ int32_t *)sortedExpertIdxGm_.GetPhyAddr();
//         __local_mem__ int32_t *expertCountOutLocalAddr = (__local_mem__ int32_t *)expertCountOutLocal.GetPhyAddr();

//         Simt::VF_CALL<ComputeExpertFirstIndexSimt>(Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, curCoreElements_, expertStart_,
//                                                    expertEnd_, sortedExpertIdxGmAddr, expertCountOutLocalAddr);
//         Simt::VF_CALL<ComputeExpertCountOutSimt>(Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, curCoreElements_, expertStart_,
//                                                  expertEnd_, sortedExpertIdxGmAddr, expertCountOutLocalAddr,
//                                                  expertCountOutLocalAddr);

//         expertCountOutToTempQueue_.EnQue<int32_t>(expertCountOutLocal);
//         CopyOut();
//     }
// }

template <typename X_TYPE>
class GatherOut {
private:
    constexpr static int64_t PIPELINE_DEPTH = 1;
    constexpr static int64_t ONE_REPEAT_SORT_NUM = 32; // 排序元素对齐32，sort api要求

    AscendC::GlobalTensor<X_TYPE> xGm_;
    AscendC::GlobalTensor<X_TYPE> expandedXGm_;
    AscendC::GlobalTensor<int32_t> expandedRowIdxGm_;

    AscendC::TQue<QuePosition::VECIN, PIPELINE_DEPTH> expandedRowIdxCopyInQueue_;
    AscendC::TQueBind<TPosition::VECIN, TPosition::VECOUT, PIPELINE_DEPTH> xCopyInQueue_;

    TPipe *pipe_;
    int64_t cols_ = 0;
    int64_t k_ = 0;
    int64_t blockIdx_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t curCoreIndicesLoop_ = 0;
    int64_t curCoreIndicesElements_ = 0;
    int64_t curCorePerLoopIndicesElements_ = 0;
    int64_t curCoreLastLoopIndicesElements_ = 0;
    int64_t perCoreIndicesElements_ = 0;
    int64_t colsLoops_ = 0;
    int64_t perLoopCols_  = 0;
    int64_t lastLoopCols_ = 0;
    MoeInitRoutingTilingData *tilingData_;

public:
    __aicore__ inline GatherOut()
    {}

    __aicore__ inline void Init(__gm__ X_TYPE *x, __gm__ int32_t *workspace, __gm__ int32_t *expandedRowIdx,
        __gm__ X_TYPE *expandedX, MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
    {
        blockIdx_ = AscendC::GetBlockIdx();
        tilingData_ = tilingData;
        pipe_ = tPipe;
        needCoreNum_ = tilingData_->gatherTilingData.needCoreNum;
        k_ = tilingData_->k;
        cols_ = tilingData_->cols;

        perCoreIndicesElements_ = tilingData_->gatherTilingData.perCoreIndicesElements;
        if (blockIdx_ == needCoreNum_ - 1) {
            curCoreIndicesLoop_ = tilingData_->gatherTilingData.lastCoreIndicesLoops;
            curCoreIndicesElements_ = tilingData_->gatherTilingData.lastCoreIndicesElements;
            curCorePerLoopIndicesElements_ = tilingData_->gatherTilingData.lastCorePerLoopIndicesElements;
        } else {
            curCoreIndicesLoop_ = tilingData_->gatherTilingData.perCoreIndicesLoops;
            curCoreIndicesElements_ = perCoreIndicesElements_;
            curCorePerLoopIndicesElements_ = tilingData_->gatherTilingData.perCorePerLoopIndicesElements;
        }
        curCoreLastLoopIndicesElements_ = curCoreIndicesElements_ - (curCoreIndicesLoop_ - 1) * curCorePerLoopIndicesElements_;

        colsLoops_ = tilingData_->gatherTilingData.colsLoops;
        perLoopCols_ = tilingData_->gatherTilingData.perLoopCols;
        lastLoopCols_ = tilingData_->gatherTilingData.lastLoopCols;


        pipe_->InitBuffer(expandedRowIdxCopyInQueue_, PIPELINE_DEPTH, 
            ((curCorePerLoopIndicesElements_ * sizeof(int32_t) + 32 - 1) / 32 * 32));
        pipe_->InitBuffer(xCopyInQueue_, PIPELINE_DEPTH,
            ((cols_ * sizeof(X_TYPE) + 32 - 1) / 32 * 32));

        xGm_.SetGlobalBuffer(x, tilingData_->n * tilingData_->cols);
        expandedRowIdxGm_.SetGlobalBuffer(expandedRowIdx + blockIdx_ * perCoreIndicesElements_,
                                        (curCoreIndicesElements_ * sizeof(int32_t) + 32 - 1) / 32 * 32 / sizeof(int32_t));
        expandedXGm_.SetGlobalBuffer(expandedX + blockIdx_ * perCoreIndicesElements_ * cols_,
                                    curCoreIndicesElements_ * cols_);
    }

    __aicore__ inline void CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements)
    {
        LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(subRowIdxLocal, expandedRowIdxGm_[curExpertLoopOffset], copyParams, padParams);
        expandedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);
    }

    __aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t scaleSrcOffset, int64_t curLoopCols)
    {
        LocalTensor<X_TYPE> xLocal = xCopyInQueue_.AllocTensor<X_TYPE>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(X_TYPE)), 0, 0, 0};
        DataCopyPadExtParams<X_TYPE> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[xSrcOffset], copyParams, padParams);
        xCopyInQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t scaleDstOffset, int64_t curLoopCols)
    {
        LocalTensor<X_TYPE> xLocal = xCopyInQueue_.DeQue<X_TYPE>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopCols * sizeof(X_TYPE)), 0, 0, 0};
        DataCopyPad(expandedXGm_[xDstOffset], xLocal, copyParams);
        xCopyInQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ < needCoreNum_) {
            int64_t curLoopElements = curCorePerLoopIndicesElements_;
            for (int64_t indicesLoop = 0; indicesLoop < curCoreIndicesLoop_; indicesLoop++) {
                if (indicesLoop == curCoreIndicesLoop_ - 1) {
                    curLoopElements = curCoreLastLoopIndicesElements_;
                }
                int64_t curExpertLoopOffset = indicesLoop * curCorePerLoopIndicesElements_;
                event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
                SetFlag<HardEvent::S_MTE2>(event1);
                WaitFlag<HardEvent::S_MTE2>(event1);

                CopyExpertIn(curExpertLoopOffset, curLoopElements);

                LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();
                event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(event2);
                WaitFlag<HardEvent::MTE2_S>(event2);

                for (int64_t indicesIndex = 0; indicesIndex < curLoopElements; indicesIndex++) {
                    int64_t rowIdx = subRowIdxLocal.GetValue(indicesIndex);
                    int64_t xSrcOffset = rowIdx / k_ * cols_;
                    int64_t scaleSrcOffset = rowIdx / k_;
                    int64_t xDstOffset = (curExpertLoopOffset + indicesIndex) * cols_;
                    event_t event3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
                    SetFlag<HardEvent::S_MTE2>(event3);
                    WaitFlag<HardEvent::S_MTE2>(event3);

                    int64_t curLoopCols = perLoopCols_;
                    for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
                        if (colsLoop == colsLoops_ - 1) {
                            curLoopCols = lastLoopCols_;
                        }
                        int64_t colsLoopOffset = colsLoop * perLoopCols_;
                        CopyXIn(xSrcOffset + colsLoopOffset, scaleSrcOffset, curLoopCols);
                        CopyXOut(xDstOffset + colsLoopOffset, indicesIndex, curLoopCols);
                    }
                }
                expandedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
            }
            DumpTensor(expandedXGm_, 100, 32);
        }
    }
};
 	
template <typename X_TYPE, typename EXPERT_IDX_TYPE, typename TOKEN_COUNT_TYPE>
__global__ __aicore__ __vector__ void moe_init_routing(
    __gm__ X_TYPE *x, __gm__ EXPERT_IDX_TYPE *expertIdx,  __gm__ EXPERT_IDX_TYPE *workspace, 
    __gm__ EXPERT_IDX_TYPE *expandedRowIdx, __gm__ X_TYPE *expandedX, 
    __gm__ TOKEN_COUNT_TYPE *expertTokensCountOrCumsum, MoeInitRoutingTilingData tiling)
{
    // TODO: GM_ADDR userWS = GetUserWorkspace(workspace);
    TPipe sortPipe;
    ExpertIdxSort<X_TYPE, EXPERT_IDX_TYPE> sort;
    sort.Init(expertIdx, workspace, expandedRowIdx, &tiling, &sortPipe);
    sort.Process();
    sortPipe.Destroy();

    TPipe countPipe;
    ExpertTokensCount<EXPERT_IDX_TYPE, TOKEN_COUNT_TYPE> countOp;
    countOp.Init(expandedRowIdx, expertTokensCountOrCumsum, workspace, &tiling, &countPipe);
    // countOp.Process();
    countPipe.Destroy();

    // gatherout
    TPipe gatherPipe;
    GatherOut<X_TYPE> gatherOp;
    gatherOp.Init(x, workspace, expandedRowIdx, expandedX, &tiling, &gatherPipe);
    gatherOp.Process();
    gatherPipe.Destroy();
}

template <typename T>
void genInputData(size_t length, std::vector<T>& res) {
    res.resize(length);
    std::iota(res.begin(), res.end(), 0);
}

void genInputExpertIdx(size_t length, std::vector<int32_t>& res) {
    res.resize(length);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 7);
    for (int32_t& elem : res) {
        elem = static_cast<int32_t>(dist(gen));
        std::cout << static_cast<float>(elem) << "\t";
    }
}

template <typename T>
void printData(std::vector<T>& res) {
    for (auto& elem : res) {
        std::cout << static_cast<float>(elem) << "\t";
    }
    std::cout << std::endl;
}

int main()
{
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    int64_t n = 16;
    int64_t k = 2;
    int64_t h = 4;
    int64_t numBlocks, blockLength, tileSize;
    // std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(n * k);
    numBlocks = 1;

    MoeInitRoutingTilingData tilingData;
    tilingData.n = n;
    tilingData.cols = h;
    tilingData.k = k;
    tilingData.expertStart = 0;
    tilingData.expertEnd = 8;
    // sort
    MoeSortTilingData sortTilingData;
    sortTilingData.needCoreNum = numBlocks;
    sortTilingData.perCoreElements =  n * k;
    sortTilingData.perCoreLoops = 1;
    sortTilingData.perCorePerLoopElements = n * k;
    sortTilingData.perCoreLastLoopElements = n * k;
    sortTilingData.lastCoreElements = n * k;
    sortTilingData.lastCoreLoops = 1;
    sortTilingData.lastCorePerLoopElements = n * k;
    sortTilingData.lastCoreLastLoopElements = n * k;
    tilingData.sortTilingData = sortTilingData;
    // count
    MoeTokensCountTilingData countTilingData;
    countTilingData.needCoreNum = numBlocks;
    countTilingData.perCoreElements =  n * k;
    countTilingData.perCoreLoops = 1;
    countTilingData.perCorePerLoopElements = n * k;
    countTilingData.perCoreLastLoopElements = n * k;
    countTilingData.lastCoreElements = n * k;
    countTilingData.lastCoreLoops = 1;
    countTilingData.lastCorePerLoopElements = n * k;
    countTilingData.lastCoreLastLoopElements = n * k;
    tilingData.countTilingData = countTilingData;

    // gatherout
    MoeGatherOutTilingData gatherTilingData;
    gatherTilingData.needCoreNum = numBlocks;
    gatherTilingData.perCoreIndicesElements = n * k;
    gatherTilingData.lastCoreIndicesElements = n * k;
    gatherTilingData.perCoreIndicesLoops = 1;
    gatherTilingData.perCorePerLoopIndicesElements = n * k;
    gatherTilingData. perCoreLastLoopIndicesElements = n * k;
    gatherTilingData.lastCoreIndicesLoops = 1;
    gatherTilingData.lastCorePerLoopIndicesElements = n * k;
    gatherTilingData.lastCoreLastLoopIndicesElements = n * k;
    gatherTilingData.colsLoops = 1;
    gatherTilingData.perLoopCols = h;
    gatherTilingData.lastLoopCols = h;
    tilingData.gatherTilingData = gatherTilingData;

    std::vector<xType> xData;
    genInputData(n * h, xData);
    // printData(xData);

    std::vector<indexType> expertIdxData;
    genInputExpertIdx(n * k, expertIdxData);
    // printData(expertIdxData);

    std::vector<indexType> expandedRowIdxData;
    genInputData(n * k, expandedRowIdxData);

    std::vector<xType> expandedXData;
    // printData(expandedXData);

    xType *xDevice;
    indexType *expertIdxDevice;
    indexType *workspaceDevice;
    xType *expandedXDevice;
    indexType *expandedRowIdxDevice;
    countType *tokenCountDevice;
    xType *expandedXHost;

    size_t xSize = n * h * sizeof(xType);
    size_t expertIdxSize = n * k * sizeof(indexType);
    size_t expandedRowIdxSize = n * k * sizeof(indexType);
    size_t expandedXSize = n * k * h * sizeof(xType);
    size_t tokenCountSize = (tilingData.expertEnd - tilingData.expertStart) * sizeof(countType);

    aclrtMalloc((void **)&xDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expertIdxDevice, expertIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&workspaceDevice, expertIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expandedRowIdxDevice, expandedRowIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expandedXDevice, expandedXSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tokenCountDevice, tokenCountSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void **)&expandedXHost, expandedXSize);

    aclrtMemcpy(xDevice, xSize, xData.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(expertIdxDevice, expertIdxSize, expertIdxData.data(), expertIdxSize, ACL_MEMCPY_HOST_TO_DEVICE);
    // sort场景不需要
    // aclrtMemcpy(expandedRowIdxDevice, expandedRowIdxSize, expandedRowIdxData.data(), expandedRowIdxSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // // Kernel Call
    aclrtSynchronizeStream(stream);

    moe_init_routing<float, int32_t, int64_t><<<numBlocks, nullptr, stream>>>(xDevice, expertIdxDevice, 
        workspaceDevice, expandedRowIdxDevice, expandedXDevice, tokenCountDevice, tilingData);
    aclrtSynchronizeStream(stream);

    // aclrtMemcpy(expandedXData.data(), expandedXSize, expandedXDevice, expandedXSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(expandedXHost, expandedXSize, expandedXDevice, expandedXSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtSynchronizeStream(stream);

    std::cout << "Run completed" << std::endl;
    // printData(expandedXData);
    // for (int i = 0; i < n * k * h; i++) {
    //     printf("Index: %ld, value:%f\n", i, static_cast<float>(expandedXHost[i])); 
    // }

    aclrtFree(xDevice);
    aclrtFree(expertIdxDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(expandedRowIdxDevice);
    aclrtFree(expandedXDevice);
    aclrtFree(expandedXHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}