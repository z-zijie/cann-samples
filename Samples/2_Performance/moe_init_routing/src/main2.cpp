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

using namespace AscendC;

typedef float xType;
typedef int32_t indexType;

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

struct MoeInitRoutingSortTilingData {
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

struct MoeInitRoutingTilingData {
    MoeInitRoutingSortTilingData sortTilingData;
    int64_t expertStart;
    int64_t expertEnd;
};

template <typename X_TYPE, typename EXPERT_IDX_TYPE>
class MoeInitRoutingSort {
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
    __aicore__ inline MoeInitRoutingSort()
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
 	
template <typename X_TYPE, typename EXPERT_IDX_TYPE>
__global__ __aicore__ __vector__ void moe_init_routing(
    __gm__ X_TYPE *x, __gm__ EXPERT_IDX_TYPE *expertIdx,  __gm__ EXPERT_IDX_TYPE *workspace, __gm__ EXPERT_IDX_TYPE *expandedRowIdx, __gm__ X_TYPE *expandedX, 
    MoeInitRoutingTilingData tiling,
    int64_t perCoreIndicesElements, // gather
    int64_t lastCoreIndicesElements,
    int64_t perCoreIndicesLoops,
    int64_t perCorePerLoopIndicesElements,
    int64_t perCoreLastLoopIndicesElements,
    int64_t lastCoreIndicesLoops,
    int64_t lastCorePerLoopIndicesElements,
    int64_t lastCoreLastLoopIndicesElements,
    int64_t n, int64_t k, int64_t cols)
{
    AscendC::TPipe sortPipe;
    MoeInitRoutingSort<X_TYPE, EXPERT_IDX_TYPE> sort;
    sort.Init(expertIdx, workspace, expandedRowIdx, &tiling, &sortPipe);
    sort.Process();
    sortPipe.Destroy();

    // // gatherout
    // // Init
    // constexpr static int64_t PIPELINE_DEPTH = 1;
    // AscendC::TPipe pipe_;
    // AscendC::GlobalTensor<T> xGm_;
    // AscendC::GlobalTensor<T> expandedXGm_;
    // AscendC::GlobalTensor<int32_t> expandedRowIdxGm_;

    // AscendC::TQue<QuePosition::VECIN, PIPELINE_DEPTH> expandedRowIdxCopyInQueue_;
    // AscendC::TQueBind<TPosition::VECIN, TPosition::VECOUT, PIPELINE_DEPTH> xCopyInQueue_;

    // int64_t blockIdx_ = AscendC::GetBlockIdx();
    // int64_t curCoreIndicesLoop_;
    // int64_t curCoreIndicesElements_;
    // int64_t curCorePerLoopIndicesElements_;
    // int64_t curCoreLastLoopIndicesElements_;

    // if (blockIdx_ == needCoreNum - 1) {
    //     curCoreIndicesLoop_ = lastCoreIndicesLoops;
    //     curCoreIndicesElements_ = lastCoreIndicesElements;
    //     curCorePerLoopIndicesElements_ = lastCorePerLoopIndicesElements;
    // } else {
    //     curCoreIndicesLoop_ = perCoreIndicesLoops;
    //     curCoreIndicesElements_ = perCoreIndicesElements;
    //     curCorePerLoopIndicesElements_ = perCorePerLoopIndicesElements;
    // }
    // curCoreLastLoopIndicesElements_ = curCoreIndicesElements_ - (curCoreIndicesLoop_ - 1) * curCorePerLoopIndicesElements_;


    // pipe_.InitBuffer(expandedRowIdxCopyInQueue_, PIPELINE_DEPTH, 
    //     ((curCorePerLoopIndicesElements_ * sizeof(int32_t) + 32 - 1) / 32 * 32));
    // pipe_.InitBuffer(xCopyInQueue_, PIPELINE_DEPTH,
    //     ((cols * sizeof(T) + 32 - 1) / 32 * 32));

    // xGm_.SetGlobalBuffer(x, n * cols);
    // expandedRowIdxGm_.SetGlobalBuffer(expandedRowIdx + blockIdx_ * perCoreIndicesElements,
    //                                   (curCoreIndicesElements_ * sizeof(int32_t) + 32 - 1) / 32 * 32 / sizeof(int32_t));
    // expandedXGm_.SetGlobalBuffer(expandedX + blockIdx_ * perCoreIndicesElements * cols,
    //                             curCoreIndicesElements_ * cols);

    // // printf("--------");
    // // DumpTensor(xGm_, 9, 32);
    // // DumpTensor(expandedRowIdxGm_, 10, 32);
    // // Process
    // if (blockIdx_ < needCoreNum) {
    //     int64_t curLoopElements = curCorePerLoopIndicesElements_;
    //     for (int64_t indicesLoop = 0; indicesLoop < curCoreIndicesLoop_; indicesLoop++) {
    //         if (indicesLoop == curCoreIndicesLoop_ - 1) {
    //             curLoopElements = curCoreLastLoopIndicesElements_;
    //         }
    //         int64_t curExpertLoopOffset = indicesLoop * curCorePerLoopIndicesElements_;
    //         event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
 	//         SetFlag<HardEvent::S_MTE2>(event1);
 	//         WaitFlag<HardEvent::S_MTE2>(event1);

    //         // CopyExpertIn
    //         LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
    //         DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
    //         DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    //         DataCopyPad(subRowIdxLocal, expandedRowIdxGm_[curExpertLoopOffset], copyParams, padParams);
    //         expandedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);

    //         subRowIdxLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();
    //         event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
 	//         SetFlag<HardEvent::MTE2_S>(event2);
 	//         WaitFlag<HardEvent::MTE2_S>(event2);

    //         // DumpTensor(subRowIdxLocal, 10, 32);

    //         for (int64_t indicesIndex = 0; indicesIndex < curLoopElements; indicesIndex++) {
    //             int64_t rowIdx = subRowIdxLocal.GetValue(indicesIndex);
    //             // printf("--------rowIdx %ld \n", rowIdx);
    //             int64_t xSrcOffset = rowIdx / k * cols;
    //             int64_t xDstOffset = (curExpertLoopOffset + indicesIndex) * cols;
    //             event_t event3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    //             SetFlag<HardEvent::S_MTE2>(event3);
    //             WaitFlag<HardEvent::S_MTE2>(event3);

    //             // CopyXIn
    //             LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
    //             DataCopyExtParams copyParams0{static_cast<uint16_t>(1), static_cast<uint32_t>(cols * sizeof(T)), 0, 0, 0};
    //             DataCopyPadExtParams<T> padParams0{false, 0, 0, 0};
    //             DataCopyPad(xLocal, xGm_[xSrcOffset], copyParams0, padParams0);
    //             xCopyInQueue_.EnQue(xLocal);

                
    //             // CopyXOut
    //             xLocal = xCopyInQueue_.DeQue<T>();
    //             // DumpTensor(xLocal, 20, 4);
    //             DataCopyExtParams copyParams2{1, static_cast<uint32_t>(cols * sizeof(T)), 0, 0, 0};
    //             DataCopyPad(expandedXGm_[xDstOffset], xLocal, copyParams2);
    //             xCopyInQueue_.FreeTensor(xLocal);

    //         }
    //         expandedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
    //     }
    //     // DumpTensor(expandedXGm_, 20, 32);
    // }
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
    xType *expandedXHost;

    size_t xSize = n * h * sizeof(xType);
    size_t expertIdxSize = n * k * sizeof(indexType);
    size_t expandedRowIdxSize = n * k * sizeof(indexType);
    size_t expandedXSize = n * k * h * sizeof(xType);

    aclrtMalloc((void **)&xDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expertIdxDevice, expertIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&workspaceDevice, expertIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expandedRowIdxDevice, expandedRowIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expandedXDevice, expandedXSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void **)&expandedXHost, expandedXSize);

    aclrtMemcpy(xDevice, xSize, xData.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(expertIdxDevice, expertIdxSize, expertIdxData.data(), expertIdxSize, ACL_MEMCPY_HOST_TO_DEVICE);
    // sort场景不需要
    // aclrtMemcpy(expandedRowIdxDevice, expandedRowIdxSize, expandedRowIdxData.data(), expandedRowIdxSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // // Kernel Call
    int64_t numBlocks, blockLength, tileSize;
    // std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(n * k);
    numBlocks = 1;
    aclrtSynchronizeStream(stream);

    // sort
    MoeInitRoutingTilingData tilingData;
    MoeInitRoutingSortTilingData sortTilingData;
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
    tilingData.expertStart = 0;
    tilingData.expertEnd = 7;

    int64_t perCoreElements = n * k;
    int64_t perCoreLoops = 1;
    int64_t perCorePerLoopElements = n * k;
    int64_t perCoreLastLoopElements = n * k;
    int64_t lastCoreElements = n * k;
    int64_t lastCoreLoops = 1;
    int64_t lastCorePerLoopElements = n * k;
    int64_t lastCoreLastLoopElements = n * k;
    int64_t expertStart = 0;
    int64_t expertEnd = 7;

    // gatherout
    int64_t needCoreNum = numBlocks;
    int64_t perCoreIndicesElements = n * k;
    int64_t lastCoreIndicesElements = n * k;
    int64_t perCoreIndicesLoops = 1;
    int64_t perCorePerLoopIndicesElements = n * k;
    int64_t perCoreLastLoopIndicesElements = n * k;
    int64_t lastCoreIndicesLoops = 1;
    int64_t lastCorePerLoopIndicesElements = n * k;
    int64_t lastCoreLastLoopIndicesElements = n * k;
    moe_init_routing<float, int32_t><<<numBlocks, nullptr, stream>>>(xDevice, expertIdxDevice, workspaceDevice, expandedRowIdxDevice, expandedXDevice, 
        tilingData,
        perCoreIndicesElements, // gather
        lastCoreIndicesElements,
        perCoreIndicesLoops,
        perCorePerLoopIndicesElements,
        perCoreLastLoopIndicesElements,
        lastCoreIndicesLoops,
        lastCorePerLoopIndicesElements,
        lastCoreLastLoopIndicesElements,
        n , k, h);
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