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
 * \brief
 */

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <limits>

using namespace AscendC;

typedef half dataType;
typedef float scaleType;
typedef int8_t offsetType;
typedef int8_t outputType;

constexpr int64_t BUF_NUM = 1;


struct RmsnormQuantTilingData {
    int64_t a;
    int64_t r;
    int64_t blockFactor; // 单核处理a的行数
    int64_t blockTail; // 尾核处理a的行数
    float epsilon;
};

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + 32 - 1) / 32 * 32;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + 32 - 1) / 32 * 32 / bytes;
}


template <typename DATA_TYPE, typename SCALE_TYPE, typename OFFSET_TYPE, typename OUTPUT_DTYPE>
class RmsNormQuant{
public:
    __aicore__ inline RmsNormQuant(){}

    __aicore__ inline void Init(__gm__ DATA_TYPE *x, __gm__ DATA_TYPE *gamma, __gm__ DATA_TYPE *beta, __gm__ SCALE_TYPE *scale, __gm__ OFFSET_TYPE *offset,  __gm__ OUTPUT_DTYPE *y, RmsnormQuantTilingData *tilingData) {
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ > 0) {
            return;
        }
        tilingData_ = tilingData;
        if (blockIdx_ = GetBlockNum() - 1) {
            a_ = tilingData_->blockFactor;
        } else {
            a_ = tilingData_->blockTail;
        }

        rAlign_ = Align(tilingData_->r, sizeof(DATA_TYPE));

        xGm_.SetGlobalBuffer(x + blockIdx_ * tilingData_->blockFactor * tilingData_->r);
        gammaGm_.SetGlobalBuffer(gamma);
        betaGm_.SetGlobalBuffer(beta);
        scaleGm_.SetGlobalBuffer(scale);
        offsetGm_.SetGlobalBuffer(offset);
        yGm_.SetGlobalBuffer(y + blockIdx_ * tilingData_->blockFactor * tilingData_->r);
        

        pipe_.InitBuffer(xInQueue_, BUF_NUM, rAlign_ * sizeof(DATA_TYPE));

        pipe_.InitBuffer(gammaBuf_, rAlign_ * sizeof(DATA_TYPE) * 3);
        pipe_.InitBuffer(betaBuf_, rAlign_ * sizeof(DATA_TYPE) * 3);

        
        pipe_.InitBuffer(xFloatBuf_, AlignBytes(tilingData_->r, sizeof(float)));
        pipe_.InitBuffer(rmsBuf_, AlignBytes(tilingData_->r, sizeof(float)));
        pipe_.InitBuffer(sumBuf_, 32);
        

        pipe_.InitBuffer(yOutQueue_, BUF_NUM, AlignBytes(tilingData_->r, sizeof(OUTPUT_DTYPE)));

        scale_ = static_cast<float>(scaleGm_.GetValue(0));
        offset_ = offsetGm_.GetValue(0);
    }

    __aicore__ inline void CopyInR() {
        LocalTensor<float> gammaLocalTensor = gammaBuf_.Get<float>();
        LocalTensor<float> betaLocalTensor = betaBuf_.Get<float>();

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = tilingData_->r * sizeof(DATA_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<DATA_TYPE>(0)};

        DataCopyPad(gammaLocalTensor[rAlign_].ReinterpretCast<DATA_TYPE>(), gammaGm_, dataCopyParams, dataCopyPadParams);
        DataCopyPad(betaLocalTensor[rAlign_].ReinterpretCast<DATA_TYPE>(), betaGm_, dataCopyParams, dataCopyPadParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(gammaLocalTensor, gammaLocalTensor[rAlign_].ReinterpretCast<DATA_TYPE>(), RoundMode::CAST_NONE, tilingData_->r);
        Cast(betaLocalTensor, betaLocalTensor[rAlign_].ReinterpretCast<DATA_TYPE>(), RoundMode::CAST_NONE, tilingData_->r);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void CopyInX(int64_t index) {
        LocalTensor<DATA_TYPE> xInLocalTensor = xInQueue_.AllocTensor<DATA_TYPE>();

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = tilingData_->r * sizeof(DATA_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<DATA_TYPE>(0)};

        DataCopyPad(xInLocalTensor, xGm_[index * tilingData_->r], dataCopyParams, dataCopyPadParams);
        xInQueue_.EnQue(xInLocalTensor);
    }

    __aicore__ inline void Compute() {

        LocalTensor<DATA_TYPE> xInLocalTensor = xInQueue_.DeQue<DATA_TYPE>();
        LocalTensor<float> xFloatLocalTensor = xFloatBuf_.Get<float>();
        LocalTensor<float> gammaLocalTensor = gammaBuf_.Get<float>();
        LocalTensor<float> betaLocalTensor = betaBuf_.Get<float>();
        LocalTensor<float> rmsLocalTensor = rmsBuf_.Get<float>();
        LocalTensor<float> sumLocalTensor = sumBuf_.Get<float>();

        LocalTensor<int8_t> yLocalTensor = yOutQueue_.AllocTensor<int8_t>();

        Cast(xFloatLocalTensor, xInLocalTensor, RoundMode::CAST_NONE, tilingData_->r);
        Mul(rmsLocalTensor, xFloatLocalTensor, xFloatLocalTensor, tilingData_->r);
        ReduceSum(sumLocalTensor, rmsLocalTensor, xInLocalTensor.template ReinterpretCast<float>(), tilingData_->r);
        pipe_barrier(PIPE_ALL);
        float sum = sumLocalTensor.GetValue(0);
        pipe_barrier(PIPE_ALL);
        Duplicate(rmsLocalTensor, sum, tilingData_->r);
        Muls(rmsLocalTensor, rmsLocalTensor, static_cast<float>(1.0f / tilingData_->r), tilingData_->r);
        Adds(rmsLocalTensor, rmsLocalTensor, tilingData_->epsilon, tilingData_->r);
        Sqrt(rmsLocalTensor, rmsLocalTensor, tilingData_->r);

        Div(rmsLocalTensor, xFloatLocalTensor, rmsLocalTensor, tilingData_->r);
        Mul(rmsLocalTensor, rmsLocalTensor, gammaLocalTensor, tilingData_->r);
        Add(rmsLocalTensor, rmsLocalTensor, betaLocalTensor, tilingData_->r);

        Cast(xFloatLocalTensor.template ReinterpretCast<half>(), rmsLocalTensor, RoundMode::CAST_NONE, tilingData_->r);
        Cast(yLocalTensor, xFloatLocalTensor.template ReinterpretCast<half>(), RoundMode::CAST_RINT, tilingData_->r);
        
        xInQueue_.FreeTensor(xInLocalTensor);
        yOutQueue_.EnQue<int8_t>(yLocalTensor);
        
    }

    __aicore__ inline void CopyOut(int64_t index) {
        LocalTensor<int8_t> yLocalTensor = yOutQueue_.DeQue<int8_t>();

        DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_->r * sizeof(int8_t)),
        0, 0, 0};
        DataCopyPad(yGm_[index * tilingData_->r], yLocalTensor, dataCopyParams);
        yOutQueue_.FreeTensor(yLocalTensor);
    }

    __aicore__ inline void Process() {
        CopyInR();
        for (int64_t index = 0; index < a_; index++) {
            CopyInX(index);
            Compute();
            CopyOut(index);
        }
    }



private:
    TPipe pipe_;
    RmsnormQuantTilingData * tilingData_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECOUT, 1> yOutQueue_;

    TBuf<QuePosition::VECCALC> gammaBuf_;
    TBuf<QuePosition::VECCALC> betaBuf_;

    TBuf<QuePosition::VECCALC> xFloatBuf_;

    TBuf<QuePosition::VECCALC> sumBuf_;

    TBuf<QuePosition::VECCALC> rmsBuf_;

    GlobalTensor<DATA_TYPE> xGm_;
    GlobalTensor<DATA_TYPE> gammaGm_;
    GlobalTensor<DATA_TYPE> betaGm_;
    GlobalTensor<SCALE_TYPE> scaleGm_;
    GlobalTensor<OFFSET_TYPE> offsetGm_;
    GlobalTensor<OUTPUT_DTYPE> yGm_;

    int64_t blockIdx_ = 0;
    int64_t a_ = 0;
    int64_t rAlign_ = 0;

    SCALE_TYPE scale_ = 0;
    OFFSET_TYPE offset_ = 0;



    
};


template <typename DATA_TYPE, typename SCALE_TYPE, typename OFFSET_TYPE, typename OUTPUT_DTYPE>
__global__ __aicore__ __vector__ void rms_norm_quant(__gm__ DATA_TYPE *x, __gm__ DATA_TYPE *gamma, __gm__ DATA_TYPE *beta, __gm__ SCALE_TYPE *scale, __gm__ OFFSET_TYPE *offset,  __gm__ OUTPUT_DTYPE *y, RmsnormQuantTilingData tiling) {
    RmsNormQuant<DATA_TYPE, SCALE_TYPE, OFFSET_TYPE, OUTPUT_DTYPE> op;
    op.Init(x, gamma, beta,scale, offset, y, &tiling);
    op.Process();
}

void CHECK_ACL(aclError __ret) {
    if (__ret != ACL_ERROR_NONE) 
        std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl;
}

size_t segmentProduct(const std::vector<size_t>& vec, size_t i, size_t j) {
    if (i < 0 || j > vec.size() || i > j) {
        std::cerr << "Invalid indices" << std::endl;
        return 0;
    }

    size_t product = 1;
    for (size_t k = i; k < j; ++k) {
        product *= vec[k];
    }
    return product;
}

template <typename T>
void genInputData(size_t totalSize, std::vector<T>& res) {
    res.resize(totalSize);
    std::iota(res.begin(), res.end(), 0);
}

int32_t main() {
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    size_t a = 1;
    size_t r = 32;
    float espilon = 0.0001f;


    std::vector<size_t> xShape = {a, r};
    std::vector<size_t> gammaShape = {r};
    std::vector<size_t> betaShape = {r};
    std::vector<size_t> scaleShape = {1, };
    std::vector<size_t> offsetShape = {1, };
    std::vector<size_t> yShape = {a, r};

    size_t xEleNum = segmentProduct(xShape, 0, xShape.size());
    size_t gammaEleNum = segmentProduct(gammaShape, 0, gammaShape.size());
    size_t betaEleNum = segmentProduct(betaShape, 0, betaShape.size());
    size_t scaleEleNum = segmentProduct(offsetShape, 0, offsetShape.size());
    size_t offsetEleNum = segmentProduct(scaleShape, 0, scaleShape.size());
    size_t yEleNum = segmentProduct(yShape, 0, yShape.size());
    size_t xSize = xEleNum * sizeof(dataType);
    size_t gammaSize = gammaEleNum * sizeof(dataType);
    size_t betaSize = betaEleNum * sizeof(dataType);
    size_t scaleSize = scaleEleNum * sizeof(scaleType);
    size_t offsetSize = offsetEleNum * sizeof(offsetType);
    size_t ySize = yEleNum * sizeof(outputType);

    std::vector<dataType> xData;
    genInputData(xEleNum, xData);

    for (auto i = 0; i < 10; i++) {
        std::cout << "x[" << i << "] = " << static_cast<float>(xData[i]) << std::endl;
    }

    std::vector<dataType> gammaData;
    genInputData(gammaEleNum, gammaData);

    std::vector<dataType> betaData;
    genInputData(betaEleNum, betaData);

    std::vector<scaleType> scaleData = {1.0f};
    std::vector<offsetType> offsetData = {0};

    dataType *xDevice;
    dataType *gammaDevice;
    dataType *betaDevice;
    scaleType *scaleDevice;
    offsetType *offsetDevice;

    RmsnormQuantTilingData tilingData;
    tilingData.blockTail = a;
    tilingData.blockFactor = a;
    tilingData.r = r;
    tilingData.a = a;
    tilingData.epsilon = espilon;

    outputType *yHost;
    outputType *yDevice;

    // uint64_t maxCoreNum = 4; // It's just an example. In fact, max_core_num should be obtained based on hardware information.
    // uint64_t tempUsedBlockNum = (static_cast<uint64_t>(outerDimSize) + threadNum - 1) / threadNum;
    // uint64_t numBlock = std::min(tempUsedBlockNum, maxCoreNum);

    // 申请device内存
    CHECK_ACL(aclrtMalloc((void **)&xDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&gammaDevice, gammaSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&betaDevice, betaSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&scaleDevice, scaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&offsetDevice, offsetSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, ySize, ACL_MEM_MALLOC_HUGE_FIRST));

    // 申请输出host内存
    CHECK_ACL(aclrtMallocHost((void **)&yHost, ySize));

    // 将host数据拷贝到divice
    CHECK_ACL(aclrtMemcpy(xDevice, xSize, xData.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(gammaDevice, gammaSize, gammaData.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(betaDevice, betaSize, betaData.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(scaleDevice, scaleSize, scaleData.data(), scaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(offsetDevice, offsetSize, offsetData.data(), offsetSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, xSize, xData.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // 调用算子
    rms_norm_quant<dataType, scaleType, offsetType, outputType><<<1, 0, stream>>>(xDevice, gammaDevice, betaDevice, scaleDevice, offsetDevice,  yDevice, tilingData);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(yHost, ySize, yDevice, ySize, ACL_MEMCPY_DEVICE_TO_HOST));

    printf("start print y: \n");
    for (auto i = 0; i < 10; i++) {
        // std::cout << *((outputType *)(yHost) + i) << " ";
        std::cout << static_cast<int>(yHost[i]) << " ";
    }
    std::cout << std::endl;

    // 释放空间
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(gammaDevice));
    CHECK_ACL(aclrtFree(betaDevice));
    CHECK_ACL(aclrtFree(scaleDevice));
    CHECK_ACL(aclrtFree(offsetDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFreeHost(yHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}
