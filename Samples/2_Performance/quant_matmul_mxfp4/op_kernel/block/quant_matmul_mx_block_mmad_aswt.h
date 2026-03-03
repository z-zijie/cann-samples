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

/*!
 * \file quant_matmul_mx_block_mmad_aswt.h
 * \brief
 */

#ifndef QUANT_MATMUL_MX_BLOCK_MMAD_ASWT_H
#define QUANT_MATMUL_MX_BLOCK_MMAD_ASWT_H

#include "matmul_block_mmad.h"
#include "../policy/matmul_dispatch_policy.h"
#include "../utils/matmul_layout_utils.h"
#include "../utils/matmul_common_utils.h"
#include "../utils/matmul_tuple_utils.h"

namespace ascend_ops {
namespace matmul {
namespace Block {

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_>
class BlockMmad<
    DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
    AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<QuantMatmulMxMultiBlockWithAswt<>, DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using MxL0AType = AscendC::mx_fp8_e4m3_t;
    using MxL0BType = AscendC::mx_fp8_e4m3_t;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using DispatchPolicy = DispatchPolicy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{AscendC::BLOCK_CUBE};
    uint64_t baseN_{AscendC::BLOCK_CUBE};
    uint64_t baseK_{AscendC::BLOCK_CUBE};
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(AType);
    constexpr static uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
    constexpr static int32_t C0_SIZE = 32;
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};

    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        }
    }

public:
    __aicore__ inline void Init(const TupleShape& shape, const TupleShape& tileL1, const TupleShape& tileL0,
                                uint64_t l1BufNum, bool l0cDB)
    {
        m_ = Get<DIMENSION_M>(shape);
        n_ = Get<DIMENSION_N>(shape);
        k_ = Get<DIMENSION_K>(shape);
        mL1_ = Get<DIMENSION_M>(tileL1);
        nL1_ = Get<DIMENSION_N>(tileL1);
        kL1_ = Get<DIMENSION_K>(tileL1);
        baseM_ = Get<DIMENSION_M>(tileL0);
        baseN_ = Get<DIMENSION_N>(tileL0);
        baseK_ = Get<DIMENSION_K>(tileL0);
        l1BufNum_ = l1BufNum;
        enableL0cPingPong_ = l0cDB;
        // init tensor
        aL1OneBuffer_ = mL1_ * kL1_;
        scaleAL1OneBuffer_ = mL1_ * CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        bL1OneBuffer_ = nL1_ * kL1_;
        scaleBL1OneBuffer_ = nL1_ * CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        kL1Iter_ = CeilDiv(k_, kL1_);
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            // 2 buffer: L1 space is : A0|B0|AScale0|BScale0|...|A1|B1|AScale1|BScale1|...
            uint64_t l1Offset = (L1_SIZE >> 1) * (bufferId & 1);
            aL1Offset_[bufferId] = l1Offset;
            bL1Offset_[bufferId] = l1Offset + aL1OneBuffer_;
            scaleAL1Offset_[bufferId] = bL1Offset_[bufferId] + bL1OneBuffer_;
            scaleBL1Offset_[bufferId] = scaleAL1Offset_[bufferId] + scaleAL1OneBuffer_;
        }
    }

    __aicore__ inline void CopyInScaleA1(const AscendC::GlobalTensor<fp8_e8m0_t>& scaleAGlobal,
                                         const AscendC::LocalTensor<fp8_e8m0_t>& scaleAL1Local, uint64_t curML1,
                                         uint64_t curKL1, uint64_t kL1OffsetLength)
    {
        AscendC::GlobalTensor<half> scaleAGlobalB16;
        scaleAGlobalB16.SetGlobalBuffer(((__gm__ half*)(scaleAGlobal.GetPhyAddr())));
        auto scaleAL1LocalB16 = scaleAL1Local.template ReinterpretCast<half>();
        uint64_t offsetScaleA = kL1OffsetLength / MXFP_DIVISOR_SIZE;

        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.dValue = curML1;
        dn2nzParams.nValue = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
        dn2nzParams.srcDnMatrixStride = 0;
        dn2nzParams.srcDValue = CeilDiv(k_, MXFP_DIVISOR_SIZE);
        dn2nzParams.dstNzC0Stride = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(scaleAL1LocalB16, scaleAGlobalB16[offsetScaleA], dn2nzParams);
    }

    __aicore__ inline void CopyInScaleB1(const AscendC::GlobalTensor<fp8_e8m0_t>& scaleBGlobal,
                                         const AscendC::LocalTensor<fp8_e8m0_t>& scaleBL1Local, uint64_t curNL1,
                                         uint64_t curKL1, uint64_t kL1OffsetLength)
    {
        AscendC::GlobalTensor<half> scaleBGlobalB16;
        scaleBGlobalB16.SetGlobalBuffer(((__gm__ half*)(scaleBGlobal.GetPhyAddr())));
        auto scaleBL1LocalB16 = scaleBL1Local.template ReinterpretCast<half>();
        uint64_t offsetScaleB = kL1OffsetLength / MXFP_DIVISOR_SIZE;

        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.dValue = curNL1;
        dn2nzParams.nValue = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
        dn2nzParams.srcDnMatrixStride = 0;
        dn2nzParams.srcDValue = CeilDiv(k_, MXFP_DIVISOR_SIZE);
        dn2nzParams.dstNzC0Stride = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(scaleBL1LocalB16, scaleBGlobalB16[offsetScaleB], dn2nzParams);
    }

    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<AType>& aGlobal,
                                    const AscendC::LocalTensor<AType>& aL1Local, uint64_t curML1, uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = curML1;
        nd2nzParams.dValue = curKL1;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = k_;
        nd2nzParams.dstNzC0Stride = CeilAlign(curML1, AscendC::BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(aL1Local, aGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<BType>& bGlobal,
                                    const AscendC::LocalTensor<BType>& bL1Local, uint64_t curNL1, uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = curNL1;
        nd2nzParams.dValue = curKL1;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = k_;
        nd2nzParams.dstNzC0Stride = CeilAlign(curNL1, AscendC::BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bL1Local, bGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInA2(const AscendC::LocalTensor<MxL0AType>& l0aLocal,
                                    const AscendC::LocalTensor<AType>& aL1Local,
                                    const AscendC::LocalTensor<fp8_e8m0_t>& scaleAL1Local, uint64_t iter,
                                    uint64_t curML1, uint64_t curKL1, uint64_t mL0, uint64_t kL0)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        AscendC::LoadData2DMxParams loadData2DMxParams;
        uint64_t m1 = CeilDiv(mL0, AscendC::BLOCK_CUBE);

        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = CeilDiv(iter * baseK_, C0_SIZE);
        loadDataParams.mStep = m1;
        loadDataParams.kStep = CeilDiv(kL0, C0_SIZE);
        loadDataParams.srcStride = loadDataParams.mStep;
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;

        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.yStartPosition = CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.xStep = m1;
        loadData2DMxParams.yStep = CeilDiv(kL0, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.srcStride = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        AscendC::LoadData(l0aLocal, aL1Local, scaleAL1Local, loadDataParams, loadData2DMxParams);
    }

    __aicore__ inline void CopyInB2(const AscendC::LocalTensor<MxL0BType>& l0bLocal,
                                    const AscendC::LocalTensor<BType>& bL1Local,
                                    const AscendC::LocalTensor<fp8_e8m0_t>& scaleBL1Local, uint64_t iter,
                                    uint64_t curNL1, uint64_t curKL1, uint64_t nL0, uint64_t kL0)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        AscendC::LoadData2DMxParams loadData2DMxParams;
        uint64_t n1 = CeilDiv(nL0, AscendC::BLOCK_CUBE);

        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = CeilDiv(iter * baseK_, C0_SIZE);
        loadDataParams.mStep = n1;
        loadDataParams.kStep = CeilDiv(kL0, C0_SIZE);
        loadDataParams.srcStride = loadDataParams.mStep;
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;

        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.yStartPosition = CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.xStep = n1;
        loadData2DMxParams.yStep = CeilDiv(kL0, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.srcStride = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        AscendC::LoadData(l0bLocal, bL1Local, scaleBL1Local, loadDataParams, loadData2DMxParams);
    }

    __aicore__ inline void Mmad(uint64_t l0cOffset, uint64_t l0abOffset, uint64_t m, uint64_t n, uint64_t k,
                                bool isFirstLoop, bool isLastLoop)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = CeilAlign(k, MXFP_DIVISOR_SIZE);
        mmadParams.disableGemv = true;
        mmadParams.cmatrixSource = false;
        mmadParams.cmatrixInitVal = isFirstLoop;
        mmadParams.unitFlag = enableL0cPingPong_ ? 0 : (isLastLoop ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION);
        AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], mmadParams);
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<CType>& cGlobal, AscendC::LocalTensor<float>& c1Local,
                                   uint64_t baseM, uint64_t baseN)
    {
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = baseN;
        intriParams.mSize = baseM;
        intriParams.dstStride = n_;
        intriParams.srcStride = CeilAlign(baseM, AscendC::BLOCK_CUBE);
        // set mode according to dtype
        if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
            intriParams.quantPre = QuantMode_t::F322BF16;
        } else if (AscendC::IsSameType<CType, half>::value) {
            intriParams.quantPre = QuantMode_t::F322F16;
        } else if (AscendC::IsSameType<CType, float>::value) {
            intriParams.quantPre = QuantMode_t::NoQuant;
        }
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = enableL0cPingPong_ ? 0 : FINAL_ACCUMULATION;
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(cGlobal, c1Local, intriParams);
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<CType> cGlobal, AscendC::GlobalTensor<AType> aGlobal,
                                      AscendC::GlobalTensor<BType> bGlobal,
                                      AscendC::GlobalTensor<fp8_e8m0_t> scaleAGlobal,
                                      AscendC::GlobalTensor<fp8_e8m0_t> scaleBGlobal, TupleL1L0Shape tileShape)
    {
        uint64_t curML1 = Get<MNK_M>(tileShape);
        uint64_t curNL1 = Get<MNK_N>(tileShape);
        uint64_t curML0 = Get<MNK_M0>(tileShape);
        uint64_t curNL0 = Get<MNK_N0>(tileShape);
        uint64_t ml1Align = CeilAlign(curML1, AscendC::BLOCK_CUBE);
        uint64_t nl1Align = CeilAlign(curNL1, AscendC::BLOCK_CUBE);
        uint64_t kl1Offset = 0;
        uint64_t l0cOffset = (l0cPingPong_ & 0x1) * HALF_L0C_SIZE;
        if (enableL0cPingPong_) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
        }
        uint64_t curKL1 = kL1_;
        uint64_t kL1OffsetLength = 0;
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1OffsetLength) : kL1_;
            uint64_t l1BufId = abL1LoopCnt_ & 0x1;
            // copy data to l1 buffer
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

            uint64_t offsetScaleAL1 = scaleAL1Offset_[l1BufId];
            CopyInScaleA1(scaleAGlobal, scaleAL1Local_[offsetScaleAL1], curML1, curKL1, kL1OffsetLength);

            uint64_t offsetScaleBL1 = scaleBL1Offset_[l1BufId];
            CopyInScaleB1(scaleBGlobal, scaleBL1Local_[offsetScaleBL1], curNL1, curKL1, kL1OffsetLength);

            uint64_t offsetA = kL1OffsetLength;
            uint64_t offsetAL1 = aL1Offset_[l1BufId];
            CopyInA1(aGlobal[offsetA], aL1Local_[offsetAL1], curML1, curKL1);

            uint64_t offsetB = kL1OffsetLength;
            uint64_t offsetBl1 = bL1Offset_[l1BufId];
            CopyInB1(bGlobal[offsetB], bL1Local_[offsetBl1], curNL1, curKL1);

            kL1OffsetLength += curKL1;

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0Iter = CeilDiv(curKL1, baseK_);
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                // copy data to l0 buffer
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                CopyInA2(l0aLocal_[l0Offset], aL1Local_[offsetAL1], scaleAL1Local_[offsetScaleAL1], iter1, curML1,
                         curKL1, curML0, curK0);
                CopyInB2(l0bLocal_[l0Offset], bL1Local_[offsetBl1], scaleBL1Local_[offsetScaleBL1], iter1, curNL1,
                         curKL1, curNL0, curK0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                // compute
                bool isFirstLoop = iter0 == 0 && iter1 == 0;
                bool isLastLoop = iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter;
                Mmad(l0cOffset, l0Offset, curML0, curNL0, curK0, isFirstLoop, isLastLoop);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                l0PingPong_++;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }
        AscendC::LocalTensor<float> c1Local = c1Local_[l0cOffset];
        if (enableL0cPingPong_) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
        }
        // copy data to global memory
        CopyOut(cGlobal, c1Local, curML0, curNL0);
        if (enableL0cPingPong_) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
            l0cPingPong_++;
        }
    }

private:
private:
    uint64_t aL1OneBuffer_ = 0;
    uint64_t bL1OneBuffer_ = 0;
    uint64_t scaleAL1OneBuffer_ = 0;
    uint64_t scaleBL1OneBuffer_ = 0;
    uint64_t aL1Offset_[2] = {0UL};
    uint64_t bL1Offset_[2] = {0UL};
    uint64_t scaleAL1Offset_[2] = {0UL};
    uint64_t scaleBL1Offset_[2] = {0UL};
    AscendC::LocalTensor<MxL0AType> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<MxL0BType> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, L0C_SIZE};
    AscendC::LocalTensor<AType> aL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleAL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleBL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
};

} // namespace Block
} // namespace matmul
} // namespace ascend_ops

#endif // QUANT_MATMUL_MX_BLOCK_MMAD_ASWT_H