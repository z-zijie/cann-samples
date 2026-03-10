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
 * \file block_mmad_mx.h
 * \brief Mmad 层：单 tile 流水计算，由 Scheduler 驱动多次调用
 */

#ifndef MATMUL_BLOCK_MMAD_MX_BASE_H
#define MATMUL_BLOCK_MMAD_MX_BASE_H

#include "../../../common/kernel_operator.h"
#include "../../../common/kernel_utils/layout_utils.h"
#include "../../../common/kernel_utils/common_utils.h"
#include "../../../common/kernel_utils/tuple_utils.h"
#include "../utils/quant_matmul_constant.h"

namespace Block {
template <
    class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
    class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_>
class BlockMmadMx {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using MxL0AType = AType_;
    using MxL0BType = BType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool transA = false;
    static constexpr bool transB = true;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1{0};  // 0 表示使用默认 baseK_*4
    };

    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t baseM_{256};
    uint64_t baseN_{256};
    uint64_t baseK_{128};
    uint64_t kL1_{1024};
    uint64_t kL1TileNum;
    uint64_t tailKL1;
    bool isBias_{false};
    uint64_t C0_SIZE{64};

    __aicore__ inline void Init(const TupleShape& problemShape, const BlockShape& l0TileShape,
                                const L1Params& l1Params, bool isBias, bool enableL0CPingPong)
    {
        (void)enableL0CPingPong;
        m_ = static_cast<uint64_t>(Get<IDX_M_IDX>(problemShape));
        n_ = static_cast<uint64_t>(Get<IDX_N_IDX>(problemShape));
        k_ = static_cast<uint64_t>(Get<IDX_K_IDX>(problemShape));
        baseM_ = static_cast<uint64_t>(Get<IDX_M_IDX>(l0TileShape));
        baseN_ = static_cast<uint64_t>(Get<IDX_N_IDX>(l0TileShape));
        baseK_ = static_cast<uint64_t>(Get<IDX_K_IDX>(l0TileShape));
        if (baseK_ == 0) {
            baseK_ = 128 / sizeof(fp4x2_e2m1_t);
        }
        kL1_ = (l1Params.kL1 > 0) ? l1Params.kL1 : (baseK_ * 4);
        kL1TileNum = CeilDiv(k_, kL1_);
        tailKL1 = k_ - ((kL1TileNum - 1) * kL1_);
        isBias_ = isBias;
    }
    // 左矩阵GM->L1, ND2NZ
    __aicore__ inline void CopyInL1A(
        const AscendC::GlobalTensor<AType>& aGlobal, const AscendC::LocalTensor<AType>& al1Local,
        uint64_t curML1, uint64_t curKL1, uint64_t k_) // curML1、curKL1为分形矩阵行列，k为大矩阵列
    {
        AscendC::Nd2NzParams nd2nzParams;
        uint64_t nDim = curML1;
        uint64_t dDim = curKL1;

        nd2nzParams.ndNum = 1;                                              // Nd src矩阵的个数 [0, 4095]
        nd2nzParams.nValue = nDim;                                          // 每个Nd src矩阵的行数 [0, 16384]
        nd2nzParams.dValue = (dDim + 1) >> 1;                               // 每个Nd src矩阵的列数 [0, 65535], 最小单位b8搬运，因此列数需要除2
        nd2nzParams.srcNdMatrixStride = 1;                                  // 相邻Nd src矩阵之间起始地址的偏移 [0, 65535]
        nd2nzParams.srcDValue = (k_ + 1) >> 1;                               // 每个Nd src矩阵相邻行之间的偏移 [1, 65535], 同理除2
        // transA == false
        nd2nzParams.dstNzC0Stride = CeilAlign(nDim, AscendC::BLOCK_CUBE);   // n/16
        nd2nzParams.dstNzNStride = 1;                                       // 每个Nd src矩阵相邻行转为Nz之后在dst矩阵的行偏移，[1, 16384]，单位为C0_SIZE(32B)
        nd2nzParams.dstNzMatrixStride = 1;                                  // 相邻Nz dst矩阵之间起始地址的偏移 [0, 65535]
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
    }

    // 右矩阵GM-L1，ND2NZ
    __aicore__ inline void CopyInL1B(
        const AscendC::GlobalTensor<AType>& bGlobal, const AscendC::LocalTensor<AType>& bl1Local,
        uint64_t curNL1, uint64_t curKL1, uint64_t k_)
    {
        AscendC::Nd2NzParams nd2nzParams;
        uint64_t nDim = curNL1;
        uint64_t dDim = curKL1;

        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = (dDim + 1) >> 1;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = (k_ + 1) >> 1;
        // transB == true
        nd2nzParams.dstNzC0Stride = CeilAlign(nDim, AscendC::BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
    }

    // 左矩阵scale GM->L1, DN2NZ
    __aicore__ inline void CopyInL1ScaleA(
        const AscendC::GlobalTensor<fp8_e8m0_t>& aScaleGlobal, const AscendC::LocalTensor<fp8_e8m0_t>& aScalel1Local,
        uint64_t curML1, uint64_t curKL1, uint64_t k_)
    {
        uint64_t nDim = curML1;
        uint64_t dDim = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);

        AscendC::GlobalTensor<half> TmpGlobalB16;
        TmpGlobalB16.SetGlobalBuffer(((__gm__ half*)(aScaleGlobal.GetPhyAddr())));
        auto aScaleL1LocalImpl = aScalel1Local.template ReinterpretCast<half>();

        // transA == false
        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.nValue = dDim;  // 行数
        dn2nzParams.dValue = nDim;  // 列数
        dn2nzParams.srcDnMatrixStride = 0;
        dn2nzParams.srcDValue = CeilDiv(k_, MXFP_DIVISOR_SIZE);
        dn2nzParams.dstNzC0Stride = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(aScaleL1LocalImpl, TmpGlobalB16, dn2nzParams);
    }

    __aicore__ inline void CopyInL1ScaleB(
        const AscendC::GlobalTensor<fp8_e8m0_t>& bScaleGlobal, const AscendC::LocalTensor<fp8_e8m0_t>& bScalel1Local,
        uint64_t curNL1, uint64_t curKL1, uint64_t k_)
    {
        uint64_t nDim = curNL1;
        uint64_t dDim = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);

        AscendC::GlobalTensor<half> TmpGlobalB16;
        TmpGlobalB16.SetGlobalBuffer(((__gm__ half*)(bScaleGlobal.GetPhyAddr())));
        auto bScaleL1LocalImpl = bScalel1Local.template ReinterpretCast<half>();

        // transB == true
        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.nValue = dDim; // 行数
        dn2nzParams.dValue = nDim; // 列数
        dn2nzParams.srcDnMatrixStride = 0;
        dn2nzParams.srcDValue = CeilDiv(k_, MXFP_DIVISOR_SIZE);
        dn2nzParams.dstNzC0Stride = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(bScaleL1LocalImpl, TmpGlobalB16, dn2nzParams);
    }

    // 左矩阵L1->L01, LoadData2D
    __aicore__ inline void CopyInL0A(
        const AscendC::LocalTensor<MxL0AType>& al0Local, const AscendC::LocalTensor<AType>& al1Local,
        const AscendC::LocalTensor<fp8_e8m0_t>& scaleAl1Local, 
        uint64_t curML1, uint64_t curKL1, uint64_t iter, uint64_t curKL0)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        uint64_t m1 = CeilDiv(curML1, AscendC::BLOCK_CUBE);
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = CeilDiv(baseK_ * iter, C0_SIZE);
        loadDataParams.mStep = m1;
        loadDataParams.kStep = CeilDiv(curKL0, C0_SIZE);
        loadDataParams.srcStride = loadDataParams.mStep;
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData2DMxParams loadDataMxParams;
        loadDataMxParams.xStartPosition = 0;
        loadDataMxParams.yStartPosition = CeilDiv(baseK_ * iter, MXFP_DIVISOR_SIZE);
        loadDataMxParams.xStep = m1;
        loadDataMxParams.yStep = CeilDiv(curKL0, MXFP_DIVISOR_SIZE);
        loadDataMxParams.srcStride = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
        loadDataMxParams.dstStride = CeilDiv(curKL0, MXFP_DIVISOR_SIZE);
        AscendC::LoadData(al0Local, al1Local, scaleAl1Local, loadDataParams, loadDataMxParams);
    }

    // 右矩阵L1->L01, LoadData2D
    __aicore__ inline void CopyInL0B(
        const AscendC::LocalTensor<MxL0BType>& bl0Local, const AscendC::LocalTensor<BType>& bl1Local,
        const AscendC::LocalTensor<fp8_e8m0_t>& scaleBl1Local, 
        uint64_t curNL1, uint64_t curKL1, uint64_t iter, uint64_t curKL0)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        uint64_t n1 = CeilDiv(curNL1, AscendC::BLOCK_CUBE);
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = CeilDiv(baseK_ * iter, C0_SIZE);
        loadDataParams.mStep = n1;
        loadDataParams.kStep = CeilDiv(curKL0, C0_SIZE); // C0_size = 32 / 0.5 = 64
        loadDataParams.srcStride = loadDataParams.mStep;
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData2DMxParams loadDataMxParams;
        loadDataMxParams.xStartPosition = 0;
        loadDataMxParams.yStartPosition = CeilDiv(baseK_ * iter, MXFP_DIVISOR_SIZE); // mxfp_divisor_size
        loadDataMxParams.xStep = n1;
        loadDataMxParams.yStep = CeilDiv(curKL0, MXFP_DIVISOR_SIZE);
        loadDataMxParams.srcStride = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
        loadDataMxParams.dstStride = CeilDiv(curKL0, MXFP_DIVISOR_SIZE);
        AscendC::LoadData(bl0Local, bl1Local, scaleBl1Local, loadDataParams, loadDataMxParams);
    }
    __aicore__ inline void CopyOut(
        const AscendC::GlobalTensor<float> &cGlobal, const AscendC::LocalTensor<float> &cL0Local_,
        uint64_t curML0, uint64_t curNL0, uint64_t n)
    {
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = curNL0;
        intriParams.mSize = curML0;
        intriParams.dstStride = n;
        intriParams.srcStride = CeilAlign(curML0, AscendC::BLOCK_CUBE);
        // // 根据输出Dtype设置不同的量化模式
        // if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        //     intriParams.quantPre = QuantMode_t::F322BF16;
        // } else if (AscendC::IsSameType<T, half>::value) {
        //     intriParams.quantPre = QuantMode_t::F322F16;
        // } else if (AscendC::IsSameType<T, float>::value) {
        //     intriParams.quantPre = QuantMode_t::NoQuant;
        // }
        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.reluPre = 0;
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = 0;
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(cGlobal, cL0Local_, intriParams);
    }
    // 单 tile 计算，由 Scheduler 驱动多次调用。传入的 global 张量已按当前 tile 偏移。
    __aicore__ inline void operator()(
        AscendC::GlobalTensor<AType> aGlobal,
        AscendC::GlobalTensor<BType> bGlobal,
        AscendC::GlobalTensor<fp8_e8m0_t> scaleAGlobal,
        AscendC::GlobalTensor<fp8_e8m0_t> scaleBGlobal,
        AscendC::GlobalTensor<BiasType> biasGlobal,
        AscendC::GlobalTensor<CType> cGlobal,
        const BlockShape& singleShape)
    {
        (void)biasGlobal;
        uint64_t curML1 = static_cast<uint64_t>(Get<IDX_M_TILEIDX>(singleShape));
        uint64_t curNL1 = static_cast<uint64_t>(Get<IDX_N_TILEIDX>(singleShape));
        uint64_t curML0 = curML1;
        uint64_t curNL0 = curNL1;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);

        uint64_t offsetA = 0;
        uint64_t offsetB = 0;
        uint64_t offsetScaleA = 0;
        uint64_t offsetScaleB = 0;

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
            uint64_t curKL1 = iter0 == (kL1TileNum - 1) ? tailKL1 : kL1_;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            uint64_t offsetAL1 = 0;
            uint64_t offsetBL1 = baseM_ * kL1_;
            uint64_t offsetScaleAL1 = (baseM_ + baseN_) * kL1_ / 2;
            uint64_t offsetScaleBL1 = (baseM_ + baseN_) * kL1_ / 2 + baseM_ * CeilDiv(kL1_, GROUP_SIZE);
            CopyInL1A(aGlobal[offsetA], aL1Local_[offsetAL1], curML1, curKL1, k_);
            CopyInL1B(bGlobal[offsetB], bL1Local_[offsetBL1], curNL1, curKL1, k_);
            CopyInL1ScaleA(scaleAGlobal[offsetScaleA], scaleAL1Local_[offsetScaleAL1], curML1, curKL1, k_);
            CopyInL1ScaleB(scaleBGlobal[offsetScaleB], scaleBL1Local_[offsetScaleBL1], curNL1, curKL1, k_);
            offsetA += curKL1;
            offsetScaleA += CeilDiv(curKL1, GROUP_SIZE);
            offsetB += curKL1;
            offsetScaleB += CeilDiv(curKL1, GROUP_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);

            uint64_t kL0TileNum = CeilDiv(curKL1, baseK_);
            uint64_t tailKL0 = curKL1 - (kL0TileNum - 1) * baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0TileNum; ++iter1) {
                uint64_t curKL0 = iter1 == (kL0TileNum - 1) ? tailKL0 : baseK_;

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
                uint64_t l0Offset = 0;
                CopyInL0A(aL0Local_[l0Offset], aL1Local_[offsetAL1], scaleAL1Local_[offsetScaleAL1], curML1, curKL1, iter1, curKL0);
                CopyInL0B(bL0Local_[l0Offset], bL1Local_[offsetBL1], scaleBL1Local_[offsetScaleBL1], curNL1, curKL1, iter1, curKL0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);

                bool isFirstLoop = iter0 == 0 && iter1 == 0;
                Mmad(cL0Local_, aL0Local_[l0Offset], bL0Local_[l0Offset], curML0, curNL0, curKL0, isFirstLoop);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);

        CopyOut(cGlobal, cL0Local_, curML0, curNL0, n_);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    }
private:
    __aicore__ inline void Mmad(
        const AscendC::LocalTensor<float> &c1Local, const AscendC::LocalTensor<MxL0AType> &al0Local, const AscendC::LocalTensor<MxL0BType> &bl0Local,
        uint64_t mL0, uint64_t nL0, uint64_t kL0, bool isFirstLoop)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = mL0;
        mmadParams.n = nL0;
        mmadParams.k = kL0;
        mmadParams.disableGemv = true;
        mmadParams.cmatrixSource = false;
        mmadParams.cmatrixInitVal = isFirstLoop;
        mmadParams.unitFlag = 0;
        AscendC::Mmad(c1Local, al0Local, bl0Local, mmadParams);
    }
private:
    AscendC::LocalTensor<AType> aL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleAL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleBL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<MxL0AType> aL0Local_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<MxL0BType> bL0Local_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> cL0Local_{AscendC::TPosition::CO1, 0, L0C_SIZE};
};
}
#endif