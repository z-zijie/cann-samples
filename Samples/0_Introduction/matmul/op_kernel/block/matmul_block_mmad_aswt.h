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
 * \file block_matmul_pingpong_without_que.h
 * \brief
 */

#ifndef MATMUL_BLOCK_MMAD_ASWT_H
#define MATMUL_BLOCK_MMAD_ASWT_H

#include "matmul_block_mmad.h"
#include "../policy/matmul_dispatch_policy.h"
#include "../utils/matmul_layout_utils.h"
#include "../utils/matmul_common_utils.h"
#include "../utils/matmul_tuple_utils.h"

namespace x {
namespace matmul {
namespace Block {

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_>
class BlockMmad<DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
                AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<MatmulMultiBlockWithAswt<>, DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
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
    constexpr static bool TRANS_A = TagToTrans<LayoutA>::value;
    constexpr static bool TRANS_B = TagToTrans<LayoutB>::value;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(AType);
    constexpr static uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
    // C0_SIZE equals 8 in order to adapt to the fp32 matrix
    constexpr static int32_t C0_SIZE = GetC0Size<AType>();
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
        bL1Init_ = aL1OneBuffer_ * l1BufNum_;
        bL1OneBuffer_ = nL1_ * kL1_;
        kL1Iter_ = CeilDiv(k_, kL1_);
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
    }

    // For FP32: L1 copy needs no modification
    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<AType>& aGlobal,
                                    const AscendC::LocalTensor<AType>& al1Local, uint64_t curML1, uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = TRANS_A ? curKL1 : curML1;
        uint64_t dDim = TRANS_A ? curML1 : curKL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = TRANS_A ? m_ : k_;
        nd2nzParams.dstNzC0Stride = (nDim + AscendC::BLOCK_CUBE - 1) / AscendC::BLOCK_CUBE * AscendC::BLOCK_CUBE;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<BType>& bGlobal,
                                    const AscendC::LocalTensor<BType>& bl1Local, uint64_t curNL1, uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = TRANS_B ? curNL1 : curKL1;
        uint64_t dDim = TRANS_B ? curKL1 : curNL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = TRANS_B ? k_ : n_;
        nd2nzParams.dstNzC0Stride = (nDim + AscendC::BLOCK_CUBE - 1) / AscendC::BLOCK_CUBE * AscendC::BLOCK_CUBE;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInA2(const AscendC::LocalTensor<AType>& a2Local,
                                    const AscendC::LocalTensor<AType>& al1Local, uint64_t curML1, uint64_t curKL1,
                                    uint64_t mL0, uint64_t kL0)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        if constexpr (!TRANS_A) {
            // (M, K) use LoadData2D
            loadDataParams.mStep = CeilDiv(mL0, AscendC::BLOCK_CUBE);
            if constexpr (AscendC::IsSameType<AType, half>::value || AscendC::IsSameType<AType, bfloat16_t>::value) {
                loadDataParams.kStep = CeilDiv(kL0, AscendC::BLOCK_CUBE);
            } else {
                loadDataParams.kStep = CeilDiv(kL0, C0_SIZE);
            }
            loadDataParams.srcStride = CeilDiv(curML1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else {
            // (K, M)
            loadDataParams.mStep = CeilDiv(kL0, AscendC::BLOCK_CUBE);
            if constexpr (AscendC::IsSameType<AType, half>::value || AscendC::IsSameType<AType, bfloat16_t>::value) {
                loadDataParams.kStep = CeilDiv(mL0, AscendC::BLOCK_CUBE);
                loadDataParams.dstStride = loadDataParams.kStep;
            } else {
                // actually div 8 then align to 2
                loadDataParams.kStep = CeilDiv(mL0, AscendC::BLOCK_CUBE) * TWO_ALIGN;
                loadDataParams.dstStride = loadDataParams.kStep >> 1;
            }
            loadDataParams.srcStride = CeilDiv(curKL1, AscendC::BLOCK_CUBE);
            loadDataParams.ifTranspose = true;
        }
        AscendC::LoadData<AType>(a2Local, al1Local, loadDataParams);
    }

    __aicore__ inline void CopyInB2(const AscendC::LocalTensor<BType>& b2Local,
                                    const AscendC::LocalTensor<BType>& bl1Local, uint64_t curNL1, uint64_t curKL1,
                                    uint64_t nL0, uint64_t kL0)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;

        if constexpr (TRANS_B) {
            // (N, K) use LoadData2D
            loadDataParams.mStep = CeilDiv(nL0, AscendC::BLOCK_CUBE);
            if constexpr (AscendC::IsSameType<BType, half>::value || AscendC::IsSameType<BType, bfloat16_t>::value) {
                loadDataParams.kStep = CeilDiv(kL0, AscendC::BLOCK_CUBE);
            } else {
                loadDataParams.kStep = CeilDiv(kL0, C0_SIZE);
            }
            loadDataParams.srcStride = CeilDiv(curNL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else {
            // (K, N) use LoadData2D
            loadDataParams.mStep = CeilDiv(kL0, AscendC::BLOCK_CUBE);
            if constexpr (AscendC::IsSameType<AType, half>::value || AscendC::IsSameType<AType, bfloat16_t>::value) {
                loadDataParams.kStep = CeilDiv(nL0, AscendC::BLOCK_CUBE);
                loadDataParams.dstStride = loadDataParams.kStep;
            } else {
                loadDataParams.kStep = CeilDiv(nL0, AscendC::BLOCK_CUBE) * TWO_ALIGN;
                loadDataParams.dstStride = loadDataParams.kStep >> 1;
            }
            loadDataParams.srcStride = CeilDiv(curKL1, AscendC::BLOCK_CUBE);
            loadDataParams.ifTranspose = true;
        }

        if constexpr (AscendC::IsSameType<BType, bfloat16_t>::value) {
            AscendC::LoadData(b2Local, bl1Local, loadDataParams);
        } else {
            AscendC::LoadData<BType>(b2Local, bl1Local, loadDataParams);
        }
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
        intriParams.reluPre = 0;
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = enableL0cPingPong_ ? 0 : FINAL_ACCUMULATION;
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(cGlobal, c1Local, intriParams);
    }

    __aicore__ inline void Mmad(uint64_t l0cOffset, uint64_t l0abOffset, uint64_t m, uint64_t n, uint64_t k,
                                bool isFirstLoop, bool isLastLoop)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.disableGemv = true;
        mmadParams.cmatrixSource = false;
        mmadParams.cmatrixInitVal = isFirstLoop;
        mmadParams.unitFlag = enableL0cPingPong_ ? 0 : (isLastLoop ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION);
        AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], mmadParams);
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<CType> cGlobal, AscendC::GlobalTensor<AType> aGlobal,
                                      AscendC::GlobalTensor<BType> bGlobal, TupleL1L0Shape tileShape)
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
        kL1_ = Min(k_, kL1_);
        uint64_t curKL1 = kL1_;
        uint64_t kL1OffsetLength = 0;
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1OffsetLength) : kL1_;
            uint64_t l1BufId = abL1LoopCnt_ & 0x1;
            // copy data to l1 buffer
            uint64_t offsetA = TRANS_A ? kL1OffsetLength * m_ : kL1OffsetLength;
            uint64_t offsetAl1 = aL1OneBuffer_ * l1BufId;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            CopyInA1(aGlobal[offsetA], l1Local_[offsetAl1], curML1, curKL1);
            uint64_t offsetB = TRANS_B ? kL1OffsetLength : kL1OffsetLength * n_;
            uint64_t offsetBl1 = bL1Init_ + bL1OneBuffer_ * l1BufId;
            CopyInB1(bGlobal[offsetB], l1Local_[offsetBl1], curNL1, curKL1);

            kL1OffsetLength += curKL1;
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0Iter = CeilDiv(curKL1, baseK_);
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                // copy data to l0 buffer
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                CopyInA2(l0aLocal_[l0Offset], l1Local_[offsetAl1], curML1, curKL1, curML0, curK0);
                offsetAl1 += TRANS_A ? baseK_ * C0_SIZE : ml1Align * baseK_;
                CopyInB2(l0bLocal_[l0Offset], l1Local_[offsetBl1], curNL1, curKL1, curNL0, curK0);
                offsetBl1 += TRANS_B ? baseK_ * nl1Align : baseK_ * C0_SIZE;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                // compute
                bool isFirstLoop = iter0 == 0 && iter1 == 0;
                bool isLastLoop = iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
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
    uint64_t bL1Init_ = 0;
    uint64_t aL1OneBuffer_ = 0;
    uint64_t bL1OneBuffer_ = 0;
    AscendC::LocalTensor<AType> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<BType> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, L0C_SIZE};
    AscendC::LocalTensor<AType> l1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
};

} // namespace Block
} // namespace matmul
} // namespace x

#endif // MATMUL_BLOCK_MMAD_ASWT_H