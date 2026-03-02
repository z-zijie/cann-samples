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
#include "moe_common.h"
#include "moe_mrgsort.h"
#include "moe_mrgsort_out.h"

using namespace AscendC;

typedef float xType;
typedef int32_t indexType;
typedef float scaleType;
typedef float offsetType;
typedef int64_t countType;

constexpr int64_t BLOCK_NUM = 64;
constexpr int64_t UB_BLOCK_SIZE = 32;
constexpr int64_t UB_SIZE = 212992; // TOTAL_UB - UB_FOR_SIMT

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

// 单核排序阶段
struct MoeVBSComputeTilingData {
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

// 多核归并排序阶段
struct MoeVMSMiddleComputeTilingData {
    int64_t needCoreNum{0};
};

// 排序输出阶段
struct MoeSortOutComputeTilingData {
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
    MoeVBSComputeTilingData vbsComputeTilingData;
    MoeVMSMiddleComputeTilingData vmsMiddleComputeTilingData;
    MoeSortOutComputeTilingData sortOutComputeParamsOp;
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

class ExpertIdxSort {
private:
    constexpr static int64_t PIPELINE_DEPTH = 1;
    constexpr static int64_t ONE_REPEAT_SORT_NUM = 32; // 排序元素对齐32，sort api要求
    constexpr static int64_t FP32_ONE_REPEAT_NUM = 64;
    constexpr static float MIN_FP32 = -3.4e38f;
    constexpr static int64_t DST_BLK_STRIDE = 1;
    constexpr static int64_t DST_REP_STRIDE = 8;
    constexpr static int64_t MAX_MRGSORT_LIST = 4;
    constexpr static int64_t WORK_GM_NUM = 2;

    // sort init
    GlobalTensor<int32_t> expertIdxGm;
    GlobalTensor<int32_t> expandedRowIdxGm;
    GlobalTensor<int32_t> sortedExpertForSourceRowGm;
    GlobalTensor<int32_t> expandDstToSrcRowGm;
    GlobalTensor<int32_t> sortedexpertIdxGm;
    GlobalTensor<int32_t> expertCountTempGm;
    GlobalTensor<float> workspaceGms[2];

    TQue<QuePosition::VECIN, PIPELINE_DEPTH> sortDataCopyInQueue;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> sortDataCopyOutQueue;
    TBuf<TPosition::VECCALC> tempBuffer;
    TBuf<TPosition::VECCALC> sortedBuffer;

    TPipe *pipe_;
    int64_t blockIdx_ = 0;
    int64_t totalLength = 0;
    int64_t sortNum = 0;
    int64_t tileLength = 0;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t actualExpertNum = 0;
    int64_t n;
    int64_t k;
    int64_t rowIdxType_ = 0;

    int64_t srcWsIndex = 0;
    int64_t listNum;
    int64_t perListElements;
    int64_t lastListElements;

    int64_t sortTotalLength;
    int64_t sortCoreLoops;
    int64_t sortCoreLoopElements;
    int64_t sortCoreLastLoopElements;

    int64_t perCoreExpert;
    int64_t needInitExpertCore;
    int64_t currentCoreExpert;

    MoeInitRoutingTilingData *tilingData_;
    MoeVBSComputeTilingData *vbsTilingData;
    MoeVMSMiddleComputeTilingData *vmsTilingData;
    MoeSortOutComputeTilingData *sortOutTilingData;

    MoeMrgsort mrgsorter;
    MoeMrgsortParam mrgsortParam;

public:
    __aicore__ inline ExpertIdxSort()
    {}

    __aicore__ inline void Init(__gm__ int32_t *expertIdx,  __gm__ int32_t *workspace, __gm__ int32_t *expandedRowIdx,
        MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
    {
        this->totalLength = tilingData->n * tilingData->k;
        this->vbsTilingData = &(tilingData->vbsComputeTilingData);
        this->vmsTilingData = &(tilingData->vmsMiddleComputeTilingData);
        this->sortOutTilingData = &(tilingData->sortOutComputeParamsOp);

        this->blockIdx_ = GetBlockIdx();
        this->tileLength = this->vbsTilingData->perCorePerLoopElements;
        this->sortTotalLength = this->vbsTilingData->perCoreElements;
        if (this->blockIdx_ == tilingData->vbsComputeTilingData.needCoreNum - 1) {
            this->tileLength = this->vbsTilingData->lastCorePerLoopElements;
            this->sortTotalLength = this->vbsTilingData->lastCoreElements;
        }
        this->n = tilingData->n;
        this->k = tilingData->k;

        expertStart_ = tilingData->expertStart;
        expertEnd_ = tilingData->expertEnd;
        actualExpertNum = expertEnd_ - expertStart_;

        // VBS param init
        if (this->blockIdx_ == this->vbsTilingData->needCoreNum - 1) {
            sortCoreLoops = this->vbsTilingData->lastCoreLoops;
            sortCoreLoopElements = this->vbsTilingData->lastCorePerLoopElements;
            sortCoreLastLoopElements = this->vbsTilingData->lastCoreLastLoopElements;
        } else {
            sortCoreLoops = this->vbsTilingData->perCoreLoops;
            sortCoreLoopElements = this->vbsTilingData->perCorePerLoopElements;
            sortCoreLastLoopElements = this->vbsTilingData->perCoreLastLoopElements;
        }

        this->pipe_ = tPipe;
        expertIdxGm.SetGlobalBuffer(expertIdx +
                                    this->blockIdx_ * tilingData->vbsComputeTilingData.perCoreElements,
                                    this->sortTotalLength);
        sortedexpertIdxGm.SetGlobalBuffer(workspace, Align(this->totalLength, sizeof(int32_t)));
        expandedRowIdxGm.SetGlobalBuffer(expandedRowIdx, Align(this->totalLength, sizeof(int32_t)));

        if (this->blockIdx_ == 0) {
            expertCountTempGm.SetGlobalBuffer(workspace + Align(n * k, sizeof(int32_t)) * 2, actualExpertNum);
            InitGlobalMemory(expertCountTempGm, actualExpertNum, 0);
            event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(event);
            WaitFlag<HardEvent::MTE3_MTE2>(event);
        }

        // key and value
        int64_t kvFactor = 2;
        workspaceGms[0].SetGlobalBuffer((__gm__ float *)workspace + 
                                        Align(this->totalLength, sizeof(int32_t)) * 2 + actualExpertNum, 
                                        Align(this->totalLength, sizeof(int32_t)) * kvFactor);
        workspaceGms[1].SetGlobalBuffer((__gm__ float *)workspace + 
                                        Align(this->totalLength, sizeof(int32_t)) * (kvFactor + 2) + actualExpertNum,
                                        Align(this->totalLength, sizeof(int32_t)) * kvFactor);

        int64_t bufferSize = Ceil(Max(this->sortOutTilingData->oneLoopMaxElements * MAX_MRGSORT_LIST, 
                                sortCoreLoopElements), ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM * 
                                sizeof(int32_t) * kvFactor;
        pipe_->InitBuffer(sortDataCopyInQueue, PIPELINE_DEPTH, bufferSize);
        pipe_->InitBuffer(sortDataCopyOutQueue, PIPELINE_DEPTH, bufferSize);
        pipe_->InitBuffer(sortedBuffer, bufferSize);
        pipe_->InitBuffer(tempBuffer, bufferSize);
    }

    __aicore__ inline void VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum)
    {
        LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
        int64_t inOffset = progress * sortCoreLoopElements;
        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), 
                                         static_cast<uint32_t>(size * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
        DataCopyPad(inLocal[0], expertIdxGm[inOffset], dataCopyParams, dataCopyPadParams);

        LocalTensor<int32_t> rowIdxLocal = inLocal[sortNum];
        int64_t startValue = this->blockIdx_ * this->vbsTilingData->perCoreElements + inOffset;

        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(event);
        WaitFlag<HardEvent::MTE3_S>(event);
        ArithProgression<int32_t>(rowIdxLocal, startValue, 1, size);
        sortDataCopyInQueue.EnQue(inLocal);
    }

    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)
    {
        LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
        LocalTensor<int32_t> expertForSourceRowLocal = inLocal[0];
        LocalTensor<float> expertForSourceRowLocalFp32;

        expertForSourceRowLocalFp32 = expertForSourceRowLocal.ReinterpretCast<float>();
        Cast(expertForSourceRowLocalFp32, expertForSourceRowLocal, RoundMode::CAST_ROUND, sortNum);

        uint16_t repeatTimes = Ceil(sortNum, FLOAT_REG_TENSOR_LENGTH);
        uint32_t sreg = static_cast<uint32_t>(sortNum);
        // __ubuf__
        __local_mem__ float *inUbAddr = (__local_mem__ float *)expertForSourceRowLocalFp32.GetPhyAddr();
        float cmpScalar = static_cast<float>(expertStart_);
        float negone = static_cast<float>(-1);

        __VEC_SCOPE__
        {
            MicroAPI::MaskReg maskRegLoop, cmpMaskReg;
            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

            MicroAPI::RegTensor<float> inRegToFloat, infFloat, vDstReg0;
            Duplicate(infFloat, static_cast<float>(MIN_FP32), pregMain);

            for (uint16_t i = 0; i < repeatTimes; i++) {
                maskRegLoop = MicroAPI::UpdateMask<float>(sreg);
                MicroAPI::DataCopy(inRegToFloat, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
                MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg, inRegToFloat, cmpScalar, maskRegLoop);
                MicroAPI::Muls(inRegToFloat, inRegToFloat, negone, maskRegLoop);
                MicroAPI::Select(vDstReg0, infFloat, inRegToFloat, cmpMaskReg);
                MicroAPI::DataCopy(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, vDstReg0, maskRegLoop);
            }
        }

        int64_t duplicateNum = size % ONE_REPEAT_SORT_NUM;
        if (duplicateNum > 0) {
            int duplicateIndex = size - duplicateNum;
            uint64_t mask0 = UINT64_MAX;
            mask0 = mask0 << duplicateNum;
            mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
            uint64_t mask[2] = {mask0, 0};
            Duplicate(expertForSourceRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        }

        LocalTensor<float> concatLocal = expertForSourceRowLocalFp32;
        LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(sortNum));
        LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
        LocalTensor<uint32_t> sourceRowLocal;
        sourceRowLocal = inLocal[sortNum].ReinterpretCast<uint32_t>();
        Sort<float, true>(outLocal, concatLocal, sourceRowLocal, sortedLocal, sortNum / ONE_REPEAT_SORT_NUM);

        sortDataCopyOutQueue.EnQue<float>(outLocal);
        sortDataCopyInQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)
    {
        LocalTensor<float> outLocal = sortDataCopyOutQueue.DeQue<float>();
        DataCopy(workspaceGms[0][this->blockIdx_ * GetSortLen<float>(this->vbsTilingData->perCoreElements) +
                                GetSortLen<float>(progress * sortCoreLoopElements)],
                outLocal, Align(GetSortLen<float>(size), sizeof(float)));
        sortDataCopyOutQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum)
    {
        VBSCopyIn(progress, size, sortNum);
        UBSortCompute(progress, size, sortNum);
        VBSCopyOut(progress, size, sortNum);
    }

    __aicore__ inline void InitMoeMrgSort(MoeMrgsort *sorter, int64_t listNum, int64_t coreOffset,
                                          int64_t loopOffset)
    {
        GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex][blockIdx_ * coreOffset + loopOffset];
        LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
        LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
        for (int64_t i = 0; i < listNum; i++) {
            LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * i];
            sorter->SetInput(srcWsGm, inLocalT);
        }
        GlobalTensor<float> dstWsGm = workspaceGms[1 - srcWsIndex][blockIdx_ * coreOffset + loopOffset];
        sorter->SetOutput(dstWsGm, outLocal);
        sortDataCopyInQueue.FreeTensor(inLocal);
        sortDataCopyOutQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements,
                                                           int64_t lastListElements)
    {
        int64_t coreOffset = GetSortLen<float>(this->vbsTilingData->perCoreElements);
        mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;

        for (int64_t i = 0; listNum >= 1; i++) {
            int64_t loops = (listNum + MAX_MRGSORT_LIST - 1) / MAX_MRGSORT_LIST;
            int64_t remainListNum = listNum - (loops - 1) * MAX_MRGSORT_LIST;

            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = perListElements;

            int64_t loopOffset = GetSortLen<float>(mrgsortParam.perListElements * MAX_MRGSORT_LIST);
            for (int64_t loop = 0; loop < loops - 1; loop++) {
                InitMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, loop * loopOffset);
                mrgsorter.Init(&mrgsortParam);
                mrgsorter.Process();
            }

            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = lastListElements;
            InitMoeMrgSort(&mrgsorter, remainListNum, coreOffset, (loops - 1) * loopOffset);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();

            listNum = loops;
            lastListElements = perListElements * (remainListNum - 1) + lastListElements;
            perListElements = perListElements * MAX_MRGSORT_LIST;
            srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;
            if (loops == 1) {
                break;
            }
        }
    }

    __aicore__ inline void VBSProcess()
    {
        if (this->blockIdx_ < this->vbsTilingData->needCoreNum) {
            int64_t sortNum = Ceil(sortCoreLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
            for (int64_t loop = 0; loop < sortCoreLoops - 1; loop++) {
                UBSortProcess(loop, sortCoreLoopElements, sortNum);
            }

            sortNum = Ceil(sortCoreLastLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
            UBSortProcess(sortCoreLoops - 1, sortCoreLastLoopElements, sortNum);

            if (sortCoreLoops > 1) {
                OneCoreVMSProcess(sortCoreLoops, sortCoreLoopElements, sortCoreLastLoopElements);
            }
        }
        SyncAll();
    }

    __aicore__ inline void VMSProcess()
    {
        int64_t currentStageNeedCoreNum = this->vmsTilingData->needCoreNum;
        perListElements = this->vbsTilingData->perCoreElements;
        lastListElements = this->vbsTilingData->lastCoreElements;
        listNum = this->vbsTilingData->needCoreNum;

        for (; listNum > MAX_MRGSORT_LIST;) {
            currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
            int64_t coreOffset = GetSortLen<float>(perListElements * MAX_MRGSORT_LIST);
            int64_t remainListNum = listNum - (currentStageNeedCoreNum - 1) * MAX_MRGSORT_LIST;

            if (this->blockIdx_ < currentStageNeedCoreNum - 1) {
                mrgsortParam.perListElements = perListElements;
                mrgsortParam.lastListElements = perListElements;
                mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;
                InitMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, 0);
                mrgsorter.Init(&mrgsortParam);
                mrgsorter.Process();
            } else if (this->blockIdx_ == currentStageNeedCoreNum - 1) {
                mrgsortParam.perListElements = perListElements;
                mrgsortParam.lastListElements = lastListElements;
                mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;
                InitMoeMrgSort(&mrgsorter, remainListNum, coreOffset, 0);
                mrgsorter.Init(&mrgsortParam);
                mrgsorter.Process();
            }
            listNum = currentStageNeedCoreNum;
            currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
            srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;

            lastListElements = perListElements * (remainListNum - 1) + lastListElements;
            perListElements = perListElements * MAX_MRGSORT_LIST;

            SyncAll();
        }
    }

    __aicore__ inline void InitMoeMrgSortOut(MoeMrgsortOut *sorter, int64_t listNum, int64_t coreOffset)
    {
        GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex];
        LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
        LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();

        for (int64_t i = 0; i < listNum; i++) {
            LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * i];
            sorter->SetInput(srcWsGm, inLocalT);
        }

        LocalTensor<float> outLocalV = outLocal[this->sortOutTilingData->oneLoopMaxElements * MAX_MRGSORT_LIST];
        sorter->SetOutput(this->sortedexpertIdxGm, this->expandedRowIdxGm, outLocal, outLocalV);

        LocalTensor<float> tempBuffer =
            sortedBuffer.Get<float>(GetSortLen<float>(this->sortOutTilingData->oneLoopMaxElements) * MAX_MRGSORT_LIST);
        sorter->SetBuffer(tempBuffer);
        sortDataCopyInQueue.FreeTensor(inLocal);
        sortDataCopyOutQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void SortOutProcess()
    {
        if (this->blockIdx_ < 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = lastListElements;
            mrgsortParam.oneLoopMaxElements = this->sortOutTilingData->oneLoopMaxElements;

            MoeMrgsortOut sorter;
            InitMoeMrgSortOut(&sorter, listNum, GetSortLen<float>(perListElements));
            sorter.Init(&mrgsortParam, pipe);
            sorter.Process();
        }
        SyncAll();
    }

    __aicore__ inline void Process()
    {
        VBSProcess();
        VMSProcess();
        SortOutProcess();
    }
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

class ExpertTokensCount {
private:
    constexpr static int64_t PIPELINE_DEPTH = 1;
    constexpr static int64_t BLOCK_BYTES = 32;

    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<int32_t> expertCountTempGm_;
    GlobalTensor<int64_t> expertTokensCountGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;

    TQue<QuePosition::VECIN, PIPELINE_DEPTH> sortedExpertIdxInQueue_;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> expertCountOutToTempQueue_;
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> expertCountTempInQueue_;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> expertIdxCountOutQueue_;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> expertTotalCountQueue_;

    TPipe *pipe_;
    int64_t blockIdx_;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t perCoreElements_ = 0;
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

    __aicore__ inline void Init(__gm__ int32_t *expandedRowIdx,  __gm__ int64_t *expertTokensCount, __gm__ int32_t *workspace,
                                MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
    {
        pipe_ = tPipe;
        blockIdx_ = AscendC::GetBlockIdx();
        expertTokensCountTilingData_ = &(tilingData->countTilingData);


        n_ = tilingData->n;
        k_ = tilingData->k;
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

        expandedRowIdxGm_.SetGlobalBuffer(expandedRowIdx + blockIdx_ * perCoreElements_);
        expertTokensCountGm_.SetGlobalBuffer(expertTokensCount, expertCountElements_);
        sortedExpertIdxGm_.SetGlobalBuffer(workspace + blockIdx_ * perCoreElements_, curCoreElements_);
        expertCountTempGm_.SetGlobalBuffer(workspace + 
                                           Align(n_ * k_, sizeof(int32_t)) * 2,
                                           actualExpertNum_);
        expertTotalCountGm_.SetGlobalBuffer(workspace + 
                                            Align(n_ * k_, sizeof(int32_t)) * 2 + 
                                            Align(actualExpertNum_, sizeof(int32_t)), 
                                            actualExpertNum_);
        // if (blockIdx_ < needCoreNum_) {
        //     printf("-----curCoreElements_%ld", curCoreElements_);
        //     printf("-----sortedExpertIdxGm_%lu", sortedExpertIdxGm_.GetPhyAddr());
        //     printf("-----expertCountTempGm_%lu", expertCountTempGm_.GetPhyAddr());
        //     printf("-----expertTotalCountGm_%lu", expertTotalCountGm_.GetPhyAddr());
        //     DumpTensor(sortedExpertIdxGm_, 104, 8);
        //     DumpTensor(sortedExpertIdxGm_, 104, 32);
        //     DumpTensor(expertCountTempGm_, 104, 8);
        //     DumpTensor(expertTotalCountGm_, 104, 8);
        // }
       
        int64_t sortedExpertIdxInLen = Max(perCorePerLoopElements_, perCoreLastLoopElements_);
        pipe_->InitBuffer(sortedExpertIdxInQueue_, PIPELINE_DEPTH, 
                          AlignBytes(sortedExpertIdxInLen, sizeof(int32_t)));
        pipe_->InitBuffer(expertCountOutToTempQueue_, PIPELINE_DEPTH, 
                          AlignBytes(actualExpertNum_, sizeof(int32_t)));
        pipe_->InitBuffer(expertCountTempInQueue_, PIPELINE_DEPTH, 
                          AlignBytes(actualExpertNum_, sizeof(int32_t)));
        pipe_->InitBuffer(expertIdxCountOutQueue_, PIPELINE_DEPTH, 
                          AlignBytes(expertCountElements_, sizeof(int64_t)));
        pipe_->InitBuffer(expertTotalCountQueue_, PIPELINE_DEPTH, 
                          AlignBytes(1, sizeof(int32_t)));
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
        // TODO:only support count mode now 
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

    __aicore__ inline void Process()
    {
        if (blockIdx_ < needCoreNum_) {
            LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
            Duplicate(expertCountOutLocal, 0, actualExpertNum_);

            __gm__ int32_t *sortedExpertIdxGmAddr = (__gm__ int32_t *)sortedExpertIdxGm_.GetPhyAddr();
            __ubuf__ int32_t *expertCountOutLocalAddr = (__ubuf__ int32_t *)expertCountOutLocal.GetPhyAddr();

            Simt::VF_CALL<ComputeExpertFirstIndexSimt>(Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, curCoreElements_, expertStart_,
                                                       expertEnd_, sortedExpertIdxGmAddr, expertCountOutLocalAddr);
            Simt::VF_CALL<ComputeExpertCountOutSimt>(Simt::Dim3{SIMT_THREAD_NUM, 1, 1}, curCoreElements_, expertStart_,
                                                     expertEnd_, sortedExpertIdxGmAddr, expertCountOutLocalAddr,
                                                     expertCountOutLocalAddr);
                                    
            expertCountOutToTempQueue_.EnQue<int32_t>(expertCountOutLocal);
            CopyOut();
        }

        SyncAll();
        /* copy expert tokens count result from worksapce to output GM. */
        if (blockIdx_ == 0) {
            ExpertCountCopyIn();
            ExpertCountCompute();
            ExpertCountCopyOut();
        }
        SyncAll();
        if (blockIdx_ == 0) {
            DumpTensor(expertTokensCountGm_, 200, 10);
        }
    }
};

template <typename X_TYPE>
class GatherOut {
private:
    constexpr static int64_t PIPELINE_DEPTH = 1;
    constexpr static int64_t BLOCK_BYTES = 32;

    GlobalTensor<X_TYPE> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<X_TYPE> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    TQue<QuePosition::VECIN, PIPELINE_DEPTH> expandedRowIdxCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, PIPELINE_DEPTH> xCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, PIPELINE_DEPTH> scaleCopyInQueue_;

    TPipe *pipe_;
    int64_t cols_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t blockIdx_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t indicesLoops_ = 0;
    int64_t curCoreIndicesElements_ = 0;
    int64_t curCorePerLoopIndicesElements_ = 0;
    int64_t curCoreLastLoopIndicesElements_ = 0;
    int64_t perCoreIndicesElements_ = 0;
    int64_t lastCoreIndicesElements_ = 0;
    int64_t perCorePerLoopIndicesElements_ = 0;
    int64_t lastCorePerLoopIndicesElements_ = 0;
    int64_t colsLoops_ = 0;
    int64_t perLoopCols_  = 0;
    int64_t lastLoopCols_ = 0;
    int64_t expertTotalCount_ = 0;
    MoeInitRoutingTilingData *tilingData_;

public:
    __aicore__ inline GatherOut()
    {}
    
    __aicore__ inline void Init(__gm__ X_TYPE *x, __gm__ float *scale, __gm__ int32_t *workspace, __gm__ int32_t *expandedRowIdx,
        __gm__ X_TYPE *expandedX, __gm__ float *expandedScale, MoeInitRoutingTilingData *tilingData, TPipe *tPipe)
    {
        blockIdx_ = AscendC::GetBlockIdx();
        tilingData_ = tilingData;
        pipe_ = tPipe;
        needCoreNum_ = tilingData_->gatherTilingData.needCoreNum;
        n_ = tilingData_->n;
        k_ = tilingData_->k;
        cols_ = tilingData_->cols;

        colsLoops_ = tilingData_->gatherTilingData.colsLoops;
        perLoopCols_ = tilingData_->gatherTilingData.perLoopCols;
        lastLoopCols_ = tilingData_->gatherTilingData.lastLoopCols;

        expertTotalCountGm_.SetGlobalBuffer(workspace + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align((tilingData_->expertEnd - tilingData_->expertStart), 
                                            sizeof(int32_t)), 1);

        // TODO: all expert is active
        // expertTotalCount_ = expertTotalCountGm_.GetValue(0);
        expertTotalCount_ = n_ * k_;

        perCorePerLoopIndicesElements_ = tilingData_->gatherTilingData.perCorePerLoopIndicesElements;
        lastCorePerLoopIndicesElements_ = tilingData_->gatherTilingData.lastCorePerLoopIndicesElements;
        perCoreIndicesElements_ = Ceil(expertTotalCount_, BLOCK_NUM);
        needCoreNum_ = Ceil(expertTotalCount_, perCoreIndicesElements_);
        lastCoreIndicesElements_ = expertTotalCount_ - (needCoreNum_ - 1) * perCoreIndicesElements_;

        if (blockIdx_ == needCoreNum_ - 1) {
            curCoreIndicesElements_ = lastCoreIndicesElements_;
            curCorePerLoopIndicesElements_ = Min(lastCorePerLoopIndicesElements_, curCoreIndicesElements_);
        } else {
            curCoreIndicesElements_ = perCoreIndicesElements_;
            curCorePerLoopIndicesElements_ = Min(perCorePerLoopIndicesElements_, curCoreIndicesElements_);
        }
        indicesLoops_ = Ceil(curCoreIndicesElements_, curCorePerLoopIndicesElements_);
        curCoreLastLoopIndicesElements_ = curCoreIndicesElements_ - (indicesLoops_ - 1) * curCorePerLoopIndicesElements_;

        pipe_->InitBuffer(expandedRowIdxCopyInQueue_, PIPELINE_DEPTH, AlignBytes(curCorePerLoopIndicesElements_, sizeof(int32_t)));
        pipe_->InitBuffer(xCopyInQueue_, PIPELINE_DEPTH, AlignBytes(perLoopCols_, sizeof(X_TYPE)));
        pipe_->InitBuffer(scaleCopyInQueue_, PIPELINE_DEPTH, AlignBytes(1, sizeof(float)));

        xGm_.SetGlobalBuffer(x, n_ * cols_);
        scaleGm_.SetGlobalBuffer(scale, n_);
        expandedXGm_.SetGlobalBuffer(expandedX + blockIdx_ * perCoreIndicesElements_ * cols_,
                                     curCoreIndicesElements_ * cols_);
        expandedScaleGm_.SetGlobalBuffer(expandedScale + blockIdx_ * perCoreIndicesElements_,
                                         curCoreIndicesElements_);
        expandedRowIdxGm_.SetGlobalBuffer(expandedRowIdx + blockIdx_ * perCoreIndicesElements_,
                                          Align(curCoreIndicesElements_, sizeof(int32_t)));
    }

    __aicore__ inline void CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements)
    {
        LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(subRowIdxLocal, expandedRowIdxGm_[curExpertLoopOffset], copyParams, padParams);
        expandedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);
    }

    __aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t curLoopCols)
    {
        LocalTensor<X_TYPE> xLocal = xCopyInQueue_.AllocTensor<X_TYPE>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(X_TYPE)), 0, 0, 0};
        DataCopyPadExtParams<X_TYPE> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[xSrcOffset], copyParams, padParams);
        xCopyInQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t curLoopCols)
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
            for (int64_t indicesLoop = 0; indicesLoop < indicesLoops_; indicesLoop++) {
                if (indicesLoop == indicesLoops_ - 1) {
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

                    // TODO: not supported yet
                    // if (isInputScale_ == 1) {
                    //     CopyScaleIn(scaleSrcOffset);
                    //     CopyScaleOut(indicesIndex + curExpertLoopOffset);
                    // }

                    int64_t curLoopCols = perLoopCols_;
                    for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
                        if (colsLoop == colsLoops_ - 1) {
                            curLoopCols = lastLoopCols_;
                        }
                        int64_t colsLoopOffset = colsLoop * perLoopCols_;
                        CopyXIn(xSrcOffset + colsLoopOffset, curLoopCols);
                        CopyXOut(xDstOffset + colsLoopOffset, curLoopCols);
                    }
                }
                expandedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
            }
        }
    }
};
 	
template <typename X_TYPE>
__global__ __aicore__ __vector__ void moe_init_routing(
    __gm__ X_TYPE *x, __gm__ int32_t *expertIdx, __gm__ float *scale, __gm__ float *offset, 
    __gm__ int32_t *workspace, 
    __gm__ X_TYPE *expandedX, __gm__ int32_t *expandedRowIdx,
    __gm__ int64_t *expertTokensCountOrCumsum, __gm__ float *expandedScale, 
    MoeInitRoutingTilingData tiling)
{
    // TODO: GM_ADDR userWS = GetUserWorkspace(workspace);
    TPipe sortPipe;
    ExpertIdxSort sort;
    sort.Init(expertIdx, workspace, expandedRowIdx, &tiling, &sortPipe);
    sort.Process();
    sortPipe.Destroy();
    return;

    TPipe countPipe;
    ExpertTokensCount countOp;
    countOp.Init(expandedRowIdx, expertTokensCountOrCumsum, workspace, &tiling, &countPipe);
    countOp.Process();
    countPipe.Destroy();

    // gatherout
    TPipe gatherPipe;
    GatherOut<X_TYPE> gatherOp;
    gatherOp.Init(x, scale, workspace, expandedRowIdx, expandedX, expandedScale, &tiling, &gatherPipe);
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

int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

int64_t CeilAlign(int64_t a, int64_t b)
{
    if (b = 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
};

int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE / bytes;
}

void cal_count_tiling(MoeInitRoutingTilingData *tilingData)
{
    auto *tokensCountTiling = &(tilingData->countTilingData);
    int64_t totalElements = tilingData->n * tilingData->k;
    int64_t perCoreElements = CeilDiv(totalElements, BLOCK_NUM);
    int64_t needCoreNum = CeilDiv(totalElements, perCoreElements);
    int64_t lastCoreElements = totalElements - (needCoreNum - 1) * perCoreElements;
    tokensCountTiling->needCoreNum = needCoreNum;
    tokensCountTiling->perCoreElements = perCoreElements;
    tokensCountTiling->lastCoreElements = lastCoreElements;

    int64_t expertNumElement = tilingData->expertEnd - tilingData->expertStart;
    int64_t maxElementsPerLoop =
        (UB_SIZE - CeilAlign(expertNumElement, UB_BLOCK_SIZE) *
            (static_cast<int64_t>(sizeof(int32_t)) * 2 + static_cast<int64_t>(sizeof(int64_t))) -
            UB_BLOCK_SIZE) / static_cast<int64_t>(sizeof(int32_t));
    int64_t perCoreLoops = CeilDiv(perCoreElements, maxElementsPerLoop);
    int64_t perCorePerLoopElements = CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;
    tokensCountTiling->perCoreLoops = perCoreLoops;
    tokensCountTiling->perCorePerLoopElements = perCorePerLoopElements;
    tokensCountTiling->perCoreLastLoopElements = perCoreLastLoopElements;

    int64_t lastCoreLoops = CeilDiv(lastCoreElements, maxElementsPerLoop);
    int64_t lastCorePerLoopElements = CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;
    tokensCountTiling->lastCoreLoops = lastCoreLoops;
    tokensCountTiling->lastCorePerLoopElements = lastCorePerLoopElements;
    tokensCountTiling->lastCoreLastLoopElements = lastCoreLastLoopElements;
}

void cal_gather_tiling(MoeInitRoutingTilingData *tilingData)
{
    auto *gatherOutTiling = &(tilingData->gatherTilingData);
    int64_t totalLength = tilingData->n * tilingData->k;
    int64_t perCoreIndicesElements = CeilDiv(totalLength, BLOCK_NUM);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = CeilDiv(totalLength, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = totalLength - (needCoreNum - 1) * perCoreIndicesElements;
    int64_t inputXDtypeSize = sizeof(xType);

    int64_t perLoopCols = tilingData->cols;
    int64_t colMultiple = 2 * inputXDtypeSize; // 
    int64_t rowMultiple = 2;
    int64_t perLoopMaxIndicesElements =
        (UB_SIZE - Align(perLoopCols, inputXDtypeSize) * colMultiple - UB_BLOCK_SIZE * 2) / rowMultiple /
        static_cast<int64_t>(sizeof(int32_t));
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = CeilDiv(perLoopCols, 2);
        perLoopMaxIndicesElements =
            (UB_SIZE - Align(perLoopCols, inputXDtypeSize) * colMultiple - UB_BLOCK_SIZE * 2) / rowMultiple /
            static_cast<int64_t>(sizeof(int32_t));
    }
    int64_t colsLoops = CeilDiv(tilingData->cols, perLoopCols);
    int64_t lastLoopCols = tilingData->cols - (colsLoops - 1) * perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum;
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->lastCoreIndicesElements = lastCoreIndicesElements;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreIndicesLoops = lastCoreIndicesLoops;
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;
}

int main()
{
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    int64_t n = 8;
    int64_t k = 2;
    int64_t h = 4;
    int64_t numBlocks = BLOCK_NUM;

    MoeInitRoutingTilingData tilingData;
    tilingData.n = n;
    tilingData.cols = h;
    tilingData.k = k;
    tilingData.expertStart = 0;
    tilingData.expertEnd = 8;
    // sort
    MoeSortTilingData sortTilingData;
    sortTilingData.needCoreNum = 1;
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
    cal_count_tiling(&tilingData);
    // gatherout
    cal_gather_tiling(&tilingData);

    xType *xDevice;
    indexType *expertIdxDevice;
    scaleType *scaleDevice;
    offsetType *offsetDevice;
    indexType *workspaceDevice;
    xType *expandedXDevice;
    indexType *expandedRowIdxDevice;
    countType *tokenCountDevice;
    scaleType *expandedScaleDevice;
    xType *expandedXHost;
    indexType *expandedRowIdxHost;
    countType *tokenCountHost;
    scaleType *expandedScaleHost;

    size_t xSize = n * h * sizeof(xType);
    size_t expertIdxSize = n * k * sizeof(indexType);
    size_t scaleSize = n * sizeof(scaleType);
    size_t offsetSize = n * sizeof(offsetType);

    size_t actualExpertNum_ = tilingData.expertEnd - tilingData.expertStart;
    size_t workspaceSize = Align(n * k, sizeof(indexType)) * 2 + Align(actualExpertNum_, sizeof(indexType)) + 
                           Align(actualExpertNum_, sizeof(countType));

    size_t expandedXSize = n * k * h * sizeof(xType);
    size_t expandedRowIdxSize = n * k * sizeof(indexType);
    size_t tokenCountSize = actualExpertNum_ * sizeof(countType);
    size_t expandedScaleSize = n * sizeof(scaleType);

    aclrtMalloc((void **)&xDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expertIdxDevice, expertIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scaleDevice, scaleSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&offsetDevice, offsetSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMalloc((void **)&expandedXDevice, expandedXSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expandedRowIdxDevice, expandedRowIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&tokenCountDevice, tokenCountSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expandedScaleDevice, expandedScaleSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void **)&expandedXHost, expandedXSize);
    aclrtMallocHost((void **)&expandedRowIdxHost, expandedRowIdxSize);
    aclrtMallocHost((void **)&tokenCountHost, tokenCountSize);
    aclrtMallocHost((void **)&expandedScaleHost, expandedScaleSize);

    std::vector<xType> xData;
    genInputData(n * h, xData);
    // printData(xData);

    std::vector<indexType> expertIdxData;
    genInputExpertIdx(n * k, expertIdxData);
    // printData(expertIdxData);

    // scale and offset is none
    aclrtMemcpy(xDevice, xSize, xData.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(expertIdxDevice, expertIdxSize, expertIdxData.data(), expertIdxSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // // Kernel Call
    aclrtSynchronizeStream(stream);
    moe_init_routing<float><<<numBlocks, nullptr, stream>>>(xDevice, expertIdxDevice, scaleDevice, offsetDevice,
        workspaceDevice, expandedXDevice, expandedRowIdxDevice, tokenCountDevice, expandedScaleDevice, tilingData);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(expandedXHost, expandedXSize, expandedXDevice, expandedXSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(expandedRowIdxHost, expandedRowIdxSize, expandedRowIdxDevice, expandedRowIdxSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(tokenCountHost, tokenCountSize, tokenCountDevice, tokenCountSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(expandedScaleHost, expandedScaleSize, expandedScaleDevice, expandedScaleSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtSynchronizeStream(stream);

    // print output
    std::cout << "Run completed" << std::endl;
    for (int i = 0; i < n * k * h; i++) {
        printf("Index: %ld, value:%f\n", i, static_cast<float>(expandedXHost[i])); 
    }

    aclrtFree(xDevice);
    aclrtFree(expertIdxDevice);
    aclrtFree(scaleDevice);
    aclrtFree(offsetDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(expandedXDevice);
    aclrtFree(expandedRowIdxDevice);
    aclrtFree(tokenCountDevice);
    aclrtFree(expandedScaleDevice);
    aclrtFree(expandedXHost);
    aclrtFree(expandedRowIdxHost);
    aclrtFree(tokenCountHost);
    aclrtFree(expandedScaleHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}