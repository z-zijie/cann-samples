/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_quant.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_QUANT_H
#define MOE_DISTRIBUTE_DISPATCH_V2_QUANT_H

#include "moe_distribute_v2_constant.h"
#include "moe_distribute_v2_base.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "quantize_functions.h"
#endif

namespace Mc2Kernel {
using namespace AscendC;
using namespace MoeDistributeV2Base;

template <typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist, bool IsNeedAllgather>
class MoeDistributeDispatchV2Quant{
public:
    uint32_t axisH_{0};
    uint32_t hOutSizeAlign_{0};

    LocalTensor<float> floatLocalTemp_;
    LocalTensor<float> smoothScalesTensor_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;

    TBuf<> smoothScalesBuf_;

    DataCopyParams scalesInParams_;
    DataCopyPadParams scalesPadParams_;
    DataCopyParams scaleOutParams_;

    __aicore__ inline MoeDistributeDispatchV2Quant() = default;

    __aicore__ inline void SetQuantInitParams(LocalTensor<float> floatLocalTemp, LocalTensor<float> smoothScalesTensor, 
                                              TBuf<> smoothScalesBuf, GlobalTensor<uint8_t> dynamicScalesOutGMTensor) 
    {
        floatLocalTemp_ = floatLocalTemp;
        smoothScalesBuf_ = smoothScalesBuf;
        smoothScalesTensor_ = smoothScalesTensor;
        dynamicScalesOutGMTensor_ = dynamicScalesOutGMTensor;

        scalesInParams_ = {1U, static_cast<uint16_t>(axisH_ * sizeof(float)), 0U, 0U};
        scalesPadParams_ = {true, 0, 0, 0};
    }

    __aicore__ inline void QuantInit(uint32_t &hAlignSize_, uint32_t &hOutSize_, uint32_t scaleInBytes_, 
                                     int32_t &tokenQuantAlign_, uint32_t &hScaleIdxSize_, uint32_t &scaleOutBytes, uint32_t axisH)
    {
        axisH_ = axisH;
        hOutSizeAlign_ = Ceil(hOutSize_, UB_ALIGN) * UB_ALIGN; // scale起始放置偏移
        hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN; //用于搬入token数据xInQueue_大小申请

        #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if constexpr (QuantMode == MX_QUANT) {
            hOutSizeAlign_ = Align256(axisH_) * sizeof(ExpandXOutType);
            hAlignSize_ = Align128(axisH_) * sizeof(XType); // MX量化计算scale时每次搬入128个数据
            hOutSizeAlign_ += Align2(Ceil32(axisH_)); 
            scaleOutBytes = Align2(Ceil32(axisH_)) * sizeof(fp8_e8m0_t); // MX量化每32个值生成一个scale，且scale数量需为偶数
        }
        #endif
        uint32_t hScaleSizeAlign = Ceil(hOutSizeAlign_, UB_ALIGN) * UB_ALIGN; //保证后面填充三元组的起始地址对齐32
        tokenQuantAlign_ = hScaleSizeAlign / sizeof(int32_t);
        // 实际搬运大小，搬运Align32(token_align + scaleOutBytes) + 3*4B(三元组)
        hScaleIdxSize_ = hScaleSizeAlign + EXPAND_IDX_INFO * sizeof(int32_t);
    }

    __aicore__ inline void QuantProcess(LocalTensor<ExpandXOutType>& outLocal, LocalTensor<XType>& inLocal, uint32_t expertIndex,
                                        uint32_t scalesCount_, GlobalTensor<float> &scalesGMTensor_)
    {
        #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if constexpr (QuantMode == MX_QUANT) {
            QuantDynamicMxFp8(outLocal, inLocal);
        }
        #endif
    }

    #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    __aicore__ inline void QuantDynamicMxFp8(LocalTensor<ExpandXOutType>& outLocal, LocalTensor<XType>& inLocal)
    {
        if constexpr (Std::IsSame<ExpandXOutType, fp8_e4m3fn_t>::value ||
            Std::IsSame<ExpandXOutType, fp8_e5m2_t>::value) {
            uint32_t mxScaleNum = Align2(Ceil32(axisH_));
            __ubuf__ XType* srcAddr = (__ubuf__ XType*)inLocal.GetPhyAddr();
            __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)floatLocalTemp_.GetPhyAddr();
            __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)floatLocalTemp_[Align32(mxScaleNum)].GetPhyAddr();
            __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)outLocal.GetPhyAddr();
            __ubuf__ uint16_t* mxScaleLocalAddr = (__ubuf__ uint16_t*)outLocal[Align256<uint32_t>(axisH_)].GetPhyAddr();

            quant::ComputeMaxExp(srcAddr, maxExpAddr, axisH_); // 计算最大Exp
            quant::ComputeScale<ExpandXOutType>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum); // 计算scales并填充
            quant::ComputeData<XType, ExpandXOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_); // 计算量化后的expandx并填充
        }
    }
    #endif

    __aicore__ inline void CopyScalesToOut(uint32_t currentTokenIndex, uint32_t scaleOutBytes,
                                           LocalTensor<ExpandXOutType> &quantTok, DataCopyExtParams &scaleOutParams)
    {
        if constexpr (((QuantMode > UNQUANT) && (QuantMode != STATIC_QUANT))) {
            LocalTensor<uint8_t> scaleLT;
            #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            if constexpr (QuantMode == MX_QUANT) {
                scaleLT = quantTok[Align256<uint32_t>(axisH_)].template ReinterpretCast<uint8_t>();
            }
            #endif
            DataCopyPad(dynamicScalesOutGMTensor_[currentTokenIndex * scaleOutBytes], scaleLT, scaleOutParams);
        }
    }
};
}
#endif // MOE_DISTRIBUTE_DISPATCH_V2_QUANT_H