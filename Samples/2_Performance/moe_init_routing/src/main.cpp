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
 	
template <typename T>
__global__ __aicore__ __vector__ void add_kernel(
    __gm__ T *x, __gm__ int32_t *expandedRowIdx, __gm__ T *expandedX, 
    int64_t needCoreNum, 
    int64_t perCoreIndicesElements,
    int64_t lastCoreIndicesElements,
    int64_t perCoreIndicesLoops,
    int64_t perCorePerLoopIndicesElements,
    int64_t perCoreLastLoopIndicesElements,
    int64_t lastCoreIndicesLoops,
    int64_t lastCorePerLoopIndicesElements,
    int64_t lastCoreLastLoopIndicesElements,
    int64_t n, int64_t k, int64_t cols)
{
    // Init
    constexpr static int64_t PIPELINE_DEPTH = 1;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> expandedXGm_;
    AscendC::GlobalTensor<int32_t> expandedRowIdxGm_;

    AscendC::TQue<QuePosition::VECIN, PIPELINE_DEPTH> expandedRowIdxCopyInQueue_;
    AscendC::TQueBind<TPosition::VECIN, TPosition::VECOUT, PIPELINE_DEPTH> xCopyInQueue_;

    int64_t blockIdx_ = AscendC::GetBlockIdx();
    int64_t curCoreIndicesLoop_;
    int64_t curCoreIndicesElements_;
    int64_t curCorePerLoopIndicesElements_;
    int64_t curCoreLastLoopIndicesElements_;

    if (blockIdx_ == needCoreNum - 1) {
        curCoreIndicesLoop_ = lastCoreIndicesLoops;
        curCoreIndicesElements_ = lastCoreIndicesElements;
        curCorePerLoopIndicesElements_ = lastCorePerLoopIndicesElements;
    } else {
        curCoreIndicesLoop_ = perCoreIndicesLoops;
        curCoreIndicesElements_ = perCoreIndicesElements;
        curCorePerLoopIndicesElements_ = perCorePerLoopIndicesElements;
    }
    curCoreLastLoopIndicesElements_ = curCoreIndicesElements_ - (curCoreIndicesLoop_ - 1) * curCorePerLoopIndicesElements_;


    pipe_.InitBuffer(expandedRowIdxCopyInQueue_, PIPELINE_DEPTH, 
        ((curCorePerLoopIndicesElements_ * sizeof(int32_t) + 32 - 1) / 32 * 32));
    pipe_.InitBuffer(xCopyInQueue_, PIPELINE_DEPTH,
        ((cols * sizeof(T) + 32 - 1) / 32 * 32));

    xGm_.SetGlobalBuffer(x, n * cols);
    expandedRowIdxGm_.SetGlobalBuffer(expandedRowIdx + blockIdx_ * perCoreIndicesElements,
                                      (curCoreIndicesElements_ * sizeof(int32_t) + 32 - 1) / 32 * 32 / sizeof(int32_t));
    expandedXGm_.SetGlobalBuffer(expandedX + blockIdx_ * perCoreIndicesElements * cols,
                                curCoreIndicesElements_ * cols);

    // printf("--------");
    // DumpTensor(xGm_, 9, 32);
    // DumpTensor(expandedRowIdxGm_, 10, 32);
    // Process
    if (blockIdx_ < needCoreNum) {
        int64_t curLoopElements = curCorePerLoopIndicesElements_;
        for (int64_t indicesLoop = 0; indicesLoop < curCoreIndicesLoop_; indicesLoop++) {
            if (indicesLoop == curCoreIndicesLoop_ - 1) {
                curLoopElements = curCoreLastLoopIndicesElements_;
            }
            int64_t curExpertLoopOffset = indicesLoop * curCorePerLoopIndicesElements_;
            event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
 	        SetFlag<HardEvent::S_MTE2>(event1);
 	        WaitFlag<HardEvent::S_MTE2>(event1);

            // CopyExpertIn
            LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(subRowIdxLocal, expandedRowIdxGm_[curExpertLoopOffset], copyParams, padParams);
            expandedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);

            subRowIdxLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();
            event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
 	        SetFlag<HardEvent::MTE2_S>(event2);
 	        WaitFlag<HardEvent::MTE2_S>(event2);

            // DumpTensor(subRowIdxLocal, 10, 32);

            for (int64_t indicesIndex = 0; indicesIndex < curLoopElements; indicesIndex++) {
                int64_t rowIdx = subRowIdxLocal.GetValue(indicesIndex);
                // printf("--------rowIdx %ld \n", rowIdx);
                int64_t xSrcOffset = rowIdx / k * cols;
                int64_t xDstOffset = (curExpertLoopOffset + indicesIndex) * cols;
                event_t event3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
                SetFlag<HardEvent::S_MTE2>(event3);
                WaitFlag<HardEvent::S_MTE2>(event3);

                // CopyXIn
                LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
                DataCopyExtParams copyParams0{static_cast<uint16_t>(1), static_cast<uint32_t>(cols * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams0{false, 0, 0, 0};
                DataCopyPad(xLocal, xGm_[xSrcOffset], copyParams0, padParams0);
                xCopyInQueue_.EnQue(xLocal);

                
                // CopyXOut
                xLocal = xCopyInQueue_.DeQue<T>();
                // DumpTensor(xLocal, 20, 4);
                DataCopyExtParams copyParams2{1, static_cast<uint32_t>(cols * sizeof(T)), 0, 0, 0};
                DataCopyPad(expandedXGm_[xDstOffset], xLocal, copyParams2);
                xCopyInQueue_.FreeTensor(xLocal);

            }
            expandedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
        }
        // DumpTensor(expandedXGm_, 20, 32);
    }
}

template <typename T>
void genInputData(size_t length, std::vector<T>& res) {
    res.resize(length);
    std::iota(res.begin(), res.end(), 0);
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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

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

    std::vector<indexType> rowIdxData;
    genInputData(n * k, rowIdxData);
    // printData(rowIdxData);

    std::vector<xType> expandedXData;
    // printData(expandedXData);

    xType *xDevice;
    xType *expandedXDevice;
    indexType *rowIdxDevice;
    xType *expandedXHost;

    size_t xSize = n * h * sizeof(xType);
    size_t rowIdxSize = n * k * sizeof(indexType);
    size_t expandedXSize = n * k * h * sizeof(xType);

    aclrtMalloc((void **)&xDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&rowIdxDevice, rowIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expandedXDevice, expandedXSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void **)&expandedXHost, expandedXSize);

    aclrtMemcpy(xDevice, xSize, xData.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(rowIdxDevice, rowIdxSize, rowIdxData.data(), rowIdxSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // // Kernel Call
    int64_t numBlocks, blockLength, tileSize;
    // std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(n * k);
    numBlocks = 1;
    aclrtSynchronizeStream(stream);

    int64_t needCoreNum = numBlocks;
    int64_t perCoreIndicesElements = n * k;
    int64_t lastCoreIndicesElements = n * k;
    int64_t perCoreIndicesLoops = 1;
    int64_t perCorePerLoopIndicesElements = n * k;
    int64_t perCoreLastLoopIndicesElements = n * k;
    int64_t lastCoreIndicesLoops = 1;
    int64_t lastCorePerLoopIndicesElements = n * k;
    int64_t lastCoreLastLoopIndicesElements = n * k;
    add_kernel<float><<<numBlocks, nullptr, stream>>>(xDevice, rowIdxDevice, expandedXDevice, 
        needCoreNum,
        perCoreIndicesElements,
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
    for (int i = 0; i < n * k * h; i++) {
        printf("Index: %ld, value:%f\n", i, static_cast<float>(expandedXHost[i])); 
    }

    aclrtFree(xDevice);
    aclrtFree(rowIdxDevice);
    aclrtFree(expandedXDevice);
    aclrtFree(expandedXHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}