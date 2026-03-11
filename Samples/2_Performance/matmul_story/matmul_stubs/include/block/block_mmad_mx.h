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
 * \brief
 */

#ifndef MATMUL_BLOCK_MMAD_MX_QUANT_H
#define MATMUL_BLOCK_MMAD_MX_QUANT_H
#include "kernel_utils/layout_utils.h"
#include "kernel_utils/common_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "../utils/quant_matmul_constant.h"
#include "include/experimental/tensor_api/tensor.h"
#include "../tile/tile_mmad_mx.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"
#include "../tile/copy_scale_gm_to_l1.h"
#include "../tile/copy_l0c_to_gm.h"
#include "../tile/pad_mx_kl1.h"

namespace Block {
using namespace AscendC;
using namespace AscendC::Te;

template<typename T>
struct MyMakeNzLayout {
__aicore__ inline static decltype(auto) Execute(size_t row, size_t column)
{
    return AscendC::Te::MakeNzLayout<T>(row, column);
}
};

template<typename T>
struct MyMakeZnLayout {
__aicore__ inline static decltype(auto) Execute(size_t row, size_t column)
{
    return AscendC::Te::MakeZnLayout<T>(row, column);
}
};

// struct TileL1L0Param {
//     uint64_t curM = 0;
//     uint64_t curN = 0;
//     uint64_t curAlignM = 0;
//     uint64_t curAlignN = 0;
//     uint64_t curGmAKL1 = 0; 
//     uint64_t curGmBKL1 = 0;
//     uint64_t curPadAKL1 = 0;  // pad to 64 align
//     uint64_t curPadBKL1 = 0;  // pad to 64 align
//     uint64_t curKL0 = 0;
// };

template <
    class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
    class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_,
    class Enable = void>
class BlockMmadMx {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy_>, "Should not be here!");
};

template <
    class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
    class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_>
class BlockMmadMx<
    DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_,
    LayoutBias_, TileCopy_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_base_of_v<QuantMatmulMxMultiBlockWithAswt<>, DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<QuantMatmulMxMultiBlockWithAswt<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE>,
                                   DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using MxL0AType = typename GetL0DataType<AType, true>::Type;
    using MxL0BType = typename GetL0DataType<BType, true>::Type;
    using BiasType = BiasType_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t HALF_L0C_SIZE = L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    constexpr static int32_t BIAS_C0 = AscendC::AuxGetC0Size<BiasType>();
    constexpr static uint64_t BLOCK_CUBE = 16UL;
    constexpr static uint64_t MXFP_GROUP_SIZE = 32UL;
    constexpr static uint64_t MXFP_DIVISOR_SIZE = 64UL;
    constexpr static uint64_t MXFP_MULTI_BASE_SIZE = 2;
    constexpr static uint64_t SCALE_BUFFER_NUM = 2;
    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};

    using MakeLayoutAL1 = AscendC::Std::conditional_t<transA, MyMakeZnLayout<AType>, MyMakeNzLayout<AType>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<transB, MyMakeZnLayout<BType>, MyMakeNzLayout<BType>>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    __aicore__ inline BlockMmadMx()
    {
        #pragma unroll
        for(uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        // Fixpipe output layout: column-first layout matches CO1->GM nz2nd conversion expectations here.
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmadMx()
    {
        #pragma unroll
        for(uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
        // Restore default layout transform for other kernels.
        AscendC::SetMMLayoutTransform(false);
    }

public:
    __aicore__ inline void Init(const TupleShape &problemShape, const BlockShape &l0TileShape,
                                const L1Params &l1Params, bool isBias, bool dbL0C)
    {
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        baseM_ = Get<IDX_M_IDX>(l0TileShape);
        baseN_ = Get<IDX_N_IDX>(l0TileShape);
        baseK_ = Get<IDX_K_IDX>(l0TileShape);
        isBias_ = isBias;
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0C;
        // L1 buffer layout depends on `fullLoadMode`:
        // - Non-full-load: A/B and their scales are double-buffered across `l1BufNum_`.
        // - A-full-load: A and scaleA are kept resident, only B/scaleB are double-buffered.
        bL1OneBuffer_ = baseN_ * kL1_;
        if constexpr (AscendC::IsSameType<AType, fp4x2_e2m1_t>::value) {
            bL1OneBuffer_ = (baseN_ * kL1_) >> 1;
        }
        scaleBL1OneBuffer_ = baseN_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        if constexpr (AscendC::IsSameType<AType, fp4x2_e2m1_t>::value) {
            scaleBL1OneBuffer_ = (baseN_ * Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE) >> 1;
        }
        if (isBias_) {
            biasL1OneBuffer_ = baseN_ * sizeof(BiasType);
        }
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            // Non-full-load mode:
            // every K-slice loads both A and B into L1, then pushes them down to L0.
            aL1OneBuffer_ = baseM_ * Align(kL1_, MXFP_DIVISOR_SIZE);
            if constexpr (AscendC::IsSameType<AType, fp4x2_e2m1_t>::value) {
                aL1OneBuffer_ = (baseM_ * Cmct::Gemm::Align(kL1_, MXFP_DIVISOR_SIZE)) >> 1;
            }
            scaleAL1OneBuffer_ = baseM_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
            for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                // L1 space example:
                // - 2 buffers: A0|B0|AScale0|BScale0|bias0|...|A1|B1|AScale1|BScale1|bias1|...
                // - 4 buffers: A0A2|B0B2|AScale0|BScale0|bias0|...|A1A3|B1B3|AScale1|BScale1|bias1|...
                uint64_t l1Offset = L1_SIZE * (bufferId & 1);
                l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
                l1BufferBOffset_[bufferId] =
                    l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);
            }
            for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
                // l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
                // l1BufferScaleAOffset_[bufferId] = (l1BufferScaleAOffset_[bufferId] + 1) >> 1;
                // l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
                // l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
                if constexpr (AscendC::IsSameType<AType, fp4x2_e2m1_t>::value) {
                    l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + (bL1OneBuffer_ << 1) * (l1BufNum_ >> 1);
                } else {
                    l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
                }
                l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
                l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
            }
        } else {
            // A-full-load mode:
            // keep the entire A tile (and its scales) resident in L1 and only rotate B.
            uint64_t mAlign = Align(baseM_, BLOCK_CUBE);
            uint64_t kAlign = Align(k_, MXFP_DIVISOR_SIZE);
            aL1OneBuffer_ = mAlign * kAlign;
            if constexpr (AscendC::IsSameType<AType, fp4x2_e2m1_t>::value) {
                aL1OneBuffer_ = (baseM_ * Cmct::Gemm::Align(kL1_, MXFP_DIVISOR_SIZE)) >> 1;
            }
            scaleAL1OneBuffer_ = baseM_ * CeilDiv(k_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
            // 2 buffer: L1 space is : B0|BScale0|bias0|A|AScale|...|B1|BScale1|bias1|
            // 4 buffer: L1 space is : B0B2|BScale0|bias0|A|AScale|...|B1B3|BScale1|bias1|...
            l1BufferAOffset_[0] = bL1OneBuffer_ * (l1BufNum_ >> 1) + ((scaleBL1OneBuffer_ + biasL1OneBuffer_) << 1);
            l1BufferScaleAOffset_[0] = (l1BufferAOffset_[0] + aL1OneBuffer_ + 1) >> 1;
            uint64_t b1Offset = l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_ >= (L1_SIZE >> 1)
                                ? l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_ : (L1_SIZE >> 1);
            b1Offset <<= 1;
            for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer_ * (bufferId >> 1);
            }
            for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
                // l1BufferScaleBOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
                // l1BufferScaleBOffset_[bufferId] = (l1BufferScaleBOffset_[bufferId] + 1) >> 1;
                // l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
                if constexpr (AscendC::IsSameType<AType, fp4x2_e2m1_t>::value) {
                    l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + (bL1OneBuffer_ << 1) * (l1BufNum_ >> 1);
                } else {
                    l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
                }
                l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
                l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
            }
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    // __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<AType> &aGlobal,
    //     const AscendC::LocalTensor<AType> &al1Local, TileL1L0Param &tileL1L0Param)
    // {
    //     AscendC::Nd2NzParams nd2nzParams;
    //     nd2nzParams.ndNum = 1;
    //     uint64_t nDim = tileL1L0Param.curM;
    //     uint64_t dDim = tileL1L0Param.curGmAKL1;

    //     nd2nzParams.nValue = nDim;
    //     nd2nzParams.dValue = (dDim + 1) >> 1;
    //     nd2nzParams.srcNdMatrixStride = 0;
    //     nd2nzParams.srcDValue = (k_ + 1) >> 1;
    //     nd2nzParams.dstNzC0Stride = tileL1L0Param.curAlignM;
    //     nd2nzParams.dstNzNStride = 1;
    //     nd2nzParams.dstNzMatrixStride = 0;
    //     AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
    // }

    // __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<BType> &bGlobal,
    //     const AscendC::LocalTensor<BType> &bl1Local, TileL1L0Param &tileL1L0Param)
    // {
    //     AscendC::Nd2NzParams nd2nzParams;
    //     nd2nzParams.ndNum = 1;
    //     uint64_t nDim = tileL1L0Param.curN;
    //     uint64_t dDim = tileL1L0Param.curGmBKL1;

    //     nd2nzParams.nValue = nDim;
    //     nd2nzParams.dValue = (dDim + 1) >> 1;
    //     nd2nzParams.srcNdMatrixStride = 0;
    //     nd2nzParams.srcDValue = (k_ + 1) >> 1;
    //     nd2nzParams.dstNzC0Stride = tileL1L0Param.curAlignN;
    //     nd2nzParams.dstNzNStride = 1;
    //     nd2nzParams.dstNzMatrixStride = 0;
    //     AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
    // }

    // __aicore__ inline void CopyInBias(const AscendC::GlobalTensor<BiasType> &biasGlobal,
    //                                   const AscendC::LocalTensor<BiasType> &cl1Local, uint64_t curNL1)
    // {
    //     // No need to add sync flag for bias L1 loading because bias loading operation can be covered by A/B/ScaleA/ScaleB load.
    //     AscendC::DataCopyPadParams padParams;
    //     // 单位为Byte
    //     AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(curNL1 * sizeof(BiasType)), 0, 0};
    //     AscendC::DataCopyPad(cl1Local, biasGlobal, biasParam, padParams);
    // }

    // __aicore__ inline void CopyInScaleA(const GlobalTensor<fp8_e8m0_t> &aScaleGlobal,
    //                                     const LocalTensor<fp8_e8m0_t> &aScaleL1Local, uint64_t curML1, uint64_t curKL1,
    //                                     uint64_t kL1Offset)
    // {
    //     if (DispatchPolicy::fullLoadMode != 0 && kL1Offset != 0) {
    //         return;
    //     }
    //     uint64_t curScaleKL1 = curKL1;
    //     if (kL1Offset + curScaleKL1 > k_) {
    //         curScaleKL1 = k_ - kL1Offset;
    //     }
    //     uint64_t nDim = curML1;
    //     uint64_t dDim = CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);

    //     uint64_t offsetScaleAGM = kL1Offset / MXFP_DIVISOR_SIZE;

    //     GlobalTensor<half> aScaleGlobalB16;
    //     aScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(aScaleGlobal.GetPhyAddr())));
    //     auto aScaleL1LocalImpl = aScaleL1Local.template ReinterpretCast<half>();

    //     AscendC::Dn2NzParams dn2nzParams;
    //     dn2nzParams.dnNum = 1;
    //     dn2nzParams.dValue = nDim;
    //     dn2nzParams.nValue = dDim;
    //     dn2nzParams.srcDnMatrixStride = 0;
    //     dn2nzParams.srcDValue = CeilDiv(k_, MXFP_DIVISOR_SIZE);
    //     dn2nzParams.dstNzC0Stride = CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
    //     dn2nzParams.dstNzNStride = 1;
    //     dn2nzParams.dstNzMatrixStride = 0;
    //     AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[offsetScaleAGM], dn2nzParams);
    // }

    // __aicore__ inline void CopyInScaleB(const GlobalTensor<fp8_e8m0_t> &bScaleGlobal,
    //                                     const LocalTensor<fp8_e8m0_t> &bScaleL1Local, uint64_t curNL1,
    //                                     uint64_t kL1Offset)
    // {
    //     uint64_t curScaleKL1 = scaleKL1_;
    //     if (kL1Offset + curScaleKL1 > k_) {
    //         curScaleKL1 = k_ - kL1Offset;
    //     }
    //     uint64_t nDim = curNL1;
    //     uint64_t dDim = CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);

    //     GlobalTensor<half> bScaleGlobalB16;
    //     bScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(bScaleGlobal.GetPhyAddr())));
    //     auto bScaleL1LocalImpl = bScaleL1Local.template ReinterpretCast<half>();

    //     uint64_t offsetScaleBGM = kL1Offset / MXFP_DIVISOR_SIZE;

    //     AscendC::Dn2NzParams dn2nzParams;
    //     dn2nzParams.dnNum = 1;
    //     dn2nzParams.dValue = nDim;
    //     dn2nzParams.nValue = dDim;
    //     dn2nzParams.srcDnMatrixStride = 0;
    //     dn2nzParams.srcDValue = CeilDiv(k_, MXFP_DIVISOR_SIZE);
    //     dn2nzParams.dstNzC0Stride = CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
    //     dn2nzParams.dstNzNStride = 1;
    //     dn2nzParams.dstNzMatrixStride = 0;
    //     AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[offsetScaleBGM], dn2nzParams);
    // }

    // __aicore__ inline void CopyInC2(const AscendC::LocalTensor<BiasType> &biasL1Local,
    //                                 const AscendC::LocalTensor<float> &biasBt, uint64_t nl1Align, bool needBias)
    // {
    //     if (!needBias) {
    //         return;
    //     }
    //     // s32场景要对齐到2 因此是align(nl1Align / 8, 2)
    //     uint64_t btAlign = AscendC::BLOCK_CUBE / BIAS_C0;
    //     uint16_t bustLenth = Align(nl1Align / BIAS_C0, btAlign);
    //     AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(bustLenth), 0, 0};
    //     // 当dstlocal位于C2时，C2中至少为fp32*16
    //     AscendC::DataCopy(biasBt, biasL1Local, biasParam);
    // }

    // __aicore__ inline void CopyInL0A(
    //     const AscendC::LocalTensor<MxL0AType>& l0aLocal, const AscendC::LocalTensor<AType>& al1Local,
    //     const AscendC::LocalTensor<fp8_e8m0_t>& scaleAl1Local, uint64_t iter, TileL1L0Param& tileL1L0Param,
    //     uint64_t curScaleKL1)
    // {
    //     AscendC::LoadData2DParamsV2 loadDataParams;
    //     AscendC::LoadData2DMxParams loadData2DMxParams;
    //     uint64_t m1 = CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE);
    //     loadDataParams.mStartPosition = 0;
    //     loadDataParams.kStartPosition = CeilDiv(iter * baseK_, C0_SIZE);
    //     loadDataParams.mStep = m1;
    //     loadDataParams.kStep = CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
    //     loadDataParams.srcStride = loadDataParams.mStep;
    //     loadDataParams.dstStride = loadDataParams.mStep;
    //     loadDataParams.ifTranspose = false;

    //     loadData2DMxParams.xStartPosition = 0;
    //     loadData2DMxParams.yStartPosition = CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
    //     loadData2DMxParams.xStep = m1;
    //     loadData2DMxParams.yStep = CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
    //     loadData2DMxParams.srcStride = CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
    //     loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
    //     AscendC::LoadData(l0aLocal, al1Local, scaleAl1Local, loadDataParams, loadData2DMxParams);
    // }

    // __aicore__ inline void CopyInL0B(const AscendC::LocalTensor<MxL0BType> &l0bLocal,
    //                                  const AscendC::LocalTensor<BType> &bl1Local,
    //                                  const AscendC::LocalTensor<fp8_e8m0_t> &scaleBl1Local, uint64_t iter,
    //                                  TileL1L0Param &tileL1L0Param)
    // {
    //     AscendC::LoadData2DParamsV2 loadDataParams;
    //     AscendC::LoadData2DMxParams loadData2DMxParams;
    //     uint64_t n1 = CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE);
    //     loadDataParams.mStartPosition = 0;
    //     loadDataParams.kStartPosition = CeilDiv(iter * baseK_, C0_SIZE);
    //     loadDataParams.mStep = n1;
    //     loadDataParams.kStep = CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
    //     loadDataParams.srcStride = loadDataParams.mStep;
    //     loadDataParams.dstStride = loadDataParams.mStep;
    //     loadDataParams.ifTranspose = false;

    //     loadData2DMxParams.xStartPosition = 0;
    //     loadData2DMxParams.yStartPosition = CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
    //     loadData2DMxParams.xStep = n1;
    //     loadData2DMxParams.yStep = CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
    //     loadData2DMxParams.srcStride = CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
    //     loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
    //     AscendC::LoadData(l0bLocal, bl1Local, scaleBl1Local, loadDataParams, loadData2DMxParams);
    // }

    // __aicore__ inline void CopyOut(const AscendC::GlobalTensor<CType> &cGlobal, AscendC::LocalTensor<float> &c1Local,
    //                                uint64_t baseM, uint64_t baseN)
    // {
    //     AscendC::DataCopyCO12DstParams intriParams;
    //     intriParams.nSize = baseN;
    //     intriParams.mSize = baseM;
    //     intriParams.dstStride = n_;
    //     intriParams.srcStride = Align(baseM, AscendC::BLOCK_CUBE);
    //     // set mode according to dtype
    //     if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
    //         intriParams.quantPre = QuantMode_t::F322BF16;
    //     } else if (AscendC::IsSameType<CType, half>::value) {
    //         intriParams.quantPre = QuantMode_t::F322F16;
    //     } else if (AscendC::IsSameType<CType, float>::value) {
    //         intriParams.quantPre = QuantMode_t::NoQuant;
    //     }
    //     intriParams.nz2ndEn = true;
    //     intriParams.unitFlag = FINAL_ACCUMULATION;  // 3 unitflag
    //     AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    //     AscendC::DataCopy(cGlobal, c1Local, intriParams);
    // }

    // __aicore__ inline void UpdateKL1(TileL1L0Param &tileL1L0Param, uint64_t iter0)
    // {
    //     tileL1L0Param.curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;
    //     tileL1L0Param.curPadBKL1 = CeilAlign(tileL1L0Param.curGmBKL1, MXFP_DIVISOR_SIZE);
    //     tileL1L0Param.curGmAKL1 = tileL1L0Param.curGmBKL1;
    //     tileL1L0Param.curPadAKL1 = tileL1L0Param.curPadBKL1;  // pad to 64 align
    // }

    // __aicore__ inline void UpdateKL0(TileL1L0Param &tileL1L0Param, uint64_t iter1)
    // {
    //     if (iter1 * baseK_ + baseK_ > tileL1L0Param.curPadBKL1) {
    //         tileL1L0Param.curKL0 = tileL1L0Param.curPadBKL1 - iter1 * baseK_;
    //     } else {
    //         tileL1L0Param.curKL0 = baseK_;
    //     }
    // }

    // __aicore__ inline void GetAlignMN(TileL1L0Param& tileL1L0Param)
    // {
    //     tileL1L0Param.curAlignM = CeilAlign(tileL1L0Param.curM, BLOCK_CUBE);
    //     tileL1L0Param.curAlignN = CeilAlign(tileL1L0Param.curN, BLOCK_CUBE);
    // }

    // __aicore__ inline void CopyScalesInL1(AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
    //                                       AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal, TileL1L0Param &tileL1L0Param,
    //                                       uint64_t l1Iter, uint64_t scaleL1BufId)
    // {
    //     uint64_t kL1Offset = l1Iter * kL1_;
    //     if constexpr (DispatchPolicy::fullLoadMode == 0) {
    //         if (l1Iter % (scaleKL1_ / kL1_) == 0) {
    //             AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
    //             CopyInScaleA(scaleAGlobal, scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId]], tileL1L0Param.curM,
    //                          scaleKL1_, kL1Offset);
    //             CopyInScaleB(scaleBGlobal, scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId]], tileL1L0Param.curN,
    //                          kL1Offset);
    //         }
    //     } else {
    //         if (l1Iter % (scaleKL1_ / kL1_) == 0) {
    //             AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
    //             CopyInScaleB(scaleBGlobal, scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId]], tileL1L0Param.curN,
    //                          kL1Offset);
    //         }
    //         if (abL1LoopCnt_ == 0) {
    //             CopyInScaleA(scaleAGlobal, scaleAL1Local_[l1BufferScaleAOffset_[0]], tileL1L0Param.curM, k_, kL1Offset);
    //         }
    //     }
    // }

    // __aicore__ inline void CopyAInL1(AscendC::GlobalTensor<AType> aGlobal, TileL1L0Param tileL1L0Param, uint64_t offsetA,
    //                                  uint64_t offsetAL1, uint64_t l1Iter)
    // {
    //     if constexpr (DispatchPolicy::fullLoadMode == 0) {
    //         CopyInA1(aGlobal[offsetA], aL1Local_[offsetAL1], tileL1L0Param);
    //     } else {
    //         offsetAL1 = l1BufferAOffset_[0] + l1Iter * kL1_ * tileL1L0Param.curAlignM;
    //         if (abL1LoopCnt_ < kL1Iter_) {
    //             CopyInA1(aGlobal[offsetA], aL1Local_[offsetAL1], tileL1L0Param);
    //         }
    //     }
    // }

    // __aicore__ inline void CopyBInL1(AscendC::GlobalTensor<BType> bGlobal, TileL1L0Param tileL1L0Param, uint64_t l1BufId,
    //                                  uint64_t l1Iter)
    // {
    //     uint64_t offsetB = l1Iter * kL1_;
    //     CopyInB1(bGlobal[offsetB], bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
    // }

    // __aicore__ inline void Iterate(TileL1L0Param &tileL1L0Param, MmadParams &mmadParams, uint64_t l1Iter, uint64_t l1BufId,
    //                                uint64_t scaleL1BufId, uint64_t offsetAl1, uint64_t l0cOffset)
    // {
    //     uint64_t kL0Iter = CeilDiv(tileL1L0Param.curGmBKL1, baseK_);
    //     for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
    //         UpdateKL0(tileL1L0Param, iter1);
    //         // Load data to L0 and open DB
    //         uint64_t l0Offset = (HALF_L0_SIZE << 1) * (l0PingPong_ & 0x1);
    //         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
    //         uint64_t offsetScaleL1 = BLOCK_CUBE * (l1Iter % (scaleKL1_ / kL1_)) * (kL1_ / MXFP_GROUP_SIZE);
    //         if constexpr (DispatchPolicy::fullLoadMode == 0) {
    //             CopyInL0A(l0aLocal_[l0Offset], aL1Local_[offsetAl1],
    //                       scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId] + offsetScaleL1], iter1, tileL1L0Param,
    //                       scaleKL1_);
    //         } else {
    //             offsetAl1 = l1BufferAOffset_[0] + l1Iter * kL1_ * tileL1L0Param.curAlignM;
    //             uint64_t offsetScaleAL1 = BLOCK_CUBE * l1Iter * (kL1_ / MXFP_GROUP_SIZE);
    //             CopyInL0A(l0aLocal_[l0Offset], aL1Local_[offsetAl1],
    //                       scaleAL1Local_[l1BufferScaleAOffset_[0] + offsetScaleAL1], iter1, tileL1L0Param, k_);
    //         }
    //         // copy bias to bt
    //         CopyInC2(biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)], biasBt_[baseN_ * biasBufId_],
    //                  Align(mmadParams.n, AscendC::BLOCK_CUBE), NeedBias(l1Iter, iter1));
    //         CopyInL0B(l0bLocal_[l0Offset], bL1Local_[l1BufferBOffset_[l1BufId]],
    //                   scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId] + offsetScaleL1], iter1, tileL1L0Param);
    //         AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
    //         mmadParams.k = CeilAlign(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
    //         mmadParams.unitFlag =
    //             (l1Iter + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
    //         mmadParams.cmatrixInitVal = (l1Iter == 0 && iter1 == 0 && !isBias_);
    //         Mmad(mmadParams, l0cOffset, l0Offset, baseN_ * biasBufId_, NeedBias(l1Iter, iter1));
    //         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
    //         l0PingPong_++;
    //     }
    // }

    // __aicore__ inline void operator()(AscendC::GlobalTensor<AType> aGlobal,
    //                                   AscendC::GlobalTensor<BType> bGlobal,
    //                                   AscendC::GlobalTensor<fp8_e8m0_t> scaleAGlobal,
    //                                   AscendC::GlobalTensor<fp8_e8m0_t> scaleBGlobal,
    //                                   AscendC::GlobalTensor<BiasType> biasGlobal,
    //                                   AscendC::GlobalTensor<CType> cGlobal,
    //                                   BlockShape singleShape)
    // {
    //     TileL1L0Param tileL1L0Param;
    //     tileL1L0Param.curM = Get<IDX_M_TILEIDX>(singleShape);
    //     tileL1L0Param.curN = Get<IDX_N_TILEIDX>(singleShape);
    //     GetAlignMN(tileL1L0Param);
    //     AscendC::MmadParams mmadParams;
    //     mmadParams.m = tileL1L0Param.curM;
    //     mmadParams.n = tileL1L0Param.curN;
    //     mmadParams.disableGemv = true;
    //     uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
    //     for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
    //         // Load data to L1 and open DB
    //         uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
    //         uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
    //         uint64_t offsetA = iter0 * kL1_;
    //         uint64_t offsetAl1 = l1BufferAOffset_[l1BufId];
    //         CopyScalesInL1(scaleAGlobal, scaleBGlobal, tileL1L0Param, iter0, scaleL1BufId);
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    //         biasBufId_ = abL1LoopCnt_ & 1;
    //         UpdateKL1(tileL1L0Param, iter0);
    //         CopyAInL1(aGlobal, tileL1L0Param, offsetA, offsetAl1, iter0);
    //         if (isBias_ && iter0 == 0) {
    //             CopyInBias(biasGlobal, biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)],
    //                        tileL1L0Param.curN);
    //         }
    //         CopyBInL1(bGlobal, tileL1L0Param, l1BufId, iter0);
    //         AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

    //         Iterate(tileL1L0Param, mmadParams, iter0, l1BufId, scaleL1BufId, offsetAl1, l0cOffset);
    //         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    //         if ((iter0 + 1) % (scaleKL1_ / kL1_) == 0 || iter0 == kL1Iter_ - 1) {
    //             AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
    //             scaleLoopCnt_++;
    //         }
    //         abL1LoopCnt_++;
    //     }
    //     // Copy out to GM
    //     AscendC::LocalTensor<float> c1Local = c1Local_[l0cOffset];
    //     // 数据搬出到GM或ub
    //     CopyOut(cGlobal, c1Local, mmadParams.m, mmadParams.n);
    //     if (enableL0cPingPong_) {
    //         l0cPingPong_++;
    //     }
    // }

    template <
        typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
        typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TensorScaleA gmScaleA, TensorScaleB gmScaleB, TensorBias gmBias, TensorC gmC,
        BlockShape singleShape)
    {
        auto curM = Get<IDX_M_TILEIDX>(singleShape);
        auto curN = Get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::MakeL0CLayout(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(MakeL0CmemPtr((__cc__ float*)l0cOffset), layoutL0C);
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            auto curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;
            auto curPadBKL1 = Cmct::Gemm::CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE); // pad to 64 align
            auto curGmAKL1 = curGmBKL1;
            auto curPadAKL1 = curPadBKL1;

            // A, B GM->L1
            auto layoutAL1 = MakeLayoutAL1::Execute(curM, curPadAKL1);
            auto layoutBL1 = MakeLayoutBL1::Execute(curPadBKL1, curN);
            auto tensorAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL1memPtr((__cbuf__ AType*)l1BufferAOffset_[l1BufId]), layoutAL1);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL1memPtr((__cbuf__ BType*)l1BufferBOffset_[l1BufId]), layoutBL1);
            // 先slice再copy
            auto gmBlockA = gmA(AscendC::Te::MakeCoord(0, iter0 * kL1_), AscendC::Te::MakeShape(curM, curGmAKL1));
            Cmct::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
            AscendC::Te::Copy(
                AscendC::Te::CopyAtom<
                    AscendC::Te::CopyTraits<AscendC::Te::CopyGM2L1, AscendC::Te::DataCopyTraitDefault>>{},
                tensorAL1, gmBlockA);
            auto gmBlockB = gmB(AscendC::Te::MakeCoord(iter0 * kL1_, 0), AscendC::Te::MakeShape(curGmBKL1, curN));
            Cmct::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
            AscendC::Te::Copy(
                AscendC::Te::CopyAtom<
                    AscendC::Te::CopyTraits<AscendC::Te::CopyGM2L1, AscendC::Te::DataCopyTraitDefault>>{},
                tensorBL1, gmBlockB);

            // scaleA, scaleB GM->L1
            uint64_t kL1Offset = iter0 * kL1_;
            if (iter0 % (scaleKL1_ / kL1_) == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));

                uint64_t curScaleKL1 = scaleKL1_;
                if (kL1Offset + curScaleKL1 > k_) {
                    curScaleKL1 = k_ - kL1Offset;
                }

                // L1上的K需要填完整的K（scaleKL1_），不能是尾块，GM上的填实际大小（可能是尾块）
                auto layoutScaleAL1 =
                    AscendC::Te::MakeZzLayout<fp8_e8m0_t>(curM, Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_GROUP_SIZE));
                auto tensorScaleAL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL1memPtr((__cbuf__ fp8_e8m0_t*)l1BufferScaleAOffset_[scaleL1BufId]),
                    layoutScaleAL1);
                auto gmBlockScaleA = gmScaleA(
                    AscendC::Te::MakeCoord(0, iter0 * kL1_ / MXFP_GROUP_SIZE),
                    AscendC::Te::MakeShape(
                        curM, Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));

                AscendC::Te::Copy(
                    AscendC::Te::CopyAtom<AscendC::Te::CopyTraits<Cmct::Gemm::Tile::CopyScaleGM2L1>>{}, tensorScaleAL1,
                    gmBlockScaleA);

                auto layoutScaleBL1 =
                    AscendC::Te::MakeNnLayout<fp8_e8m0_t>(Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_GROUP_SIZE), curN);
                auto tensorScaleBL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL1memPtr((__cbuf__ fp8_e8m0_t*)l1BufferScaleBOffset_[scaleL1BufId]),
                    layoutScaleBL1);
                auto gmBlockScaleB = gmScaleB(
                    AscendC::Te::MakeCoord(iter0 * kL1_ / MXFP_GROUP_SIZE, 0),
                    AscendC::Te::MakeShape(
                        Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));
                AscendC::Te::Copy(
                    AscendC::Te::CopyAtom<AscendC::Te::CopyTraits<Cmct::Gemm::Tile::CopyScaleGM2L1>>{}, tensorScaleBL1,
                    gmBlockScaleB);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0Iter = Cmct::Gemm::CeilDiv(curGmBKL1, baseK_);
            for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                auto curKL0 = (iter1 * baseK_ + baseK_ > curPadBKL1) ? (curPadBKL1 - iter1 * baseK_) : baseK_;
                // Load data to L0 and open DB(unit: B8)
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);

                // A, B L1->L0
                auto layoutAL0 = AscendC::Te::MakeNzLayout<AType>(curM, curKL0);
                auto layoutBL0 = AscendC::Te::MakeZnLayout<BType>(curKL0, curN);
                auto tensorAL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr((__ca__ AType*)l0Offset), layoutAL0);
                auto tensorBL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr((__cb__ BType*)l0Offset), layoutBL0);
                auto tensorBlockAL1 =
                    tensorAL1(AscendC::Te::MakeCoord(0, iter1 * baseK_), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(
                    AscendC::Te::CopyAtom<
                        AscendC::Te::CopyTraits<AscendC::Te::CopyL12L0, AscendC::Te::LoadDataTraitDefault>>{},
                    tensorAL0, tensorBlockAL1);
                auto tensorBlockBL1 =
                    tensorBL1(AscendC::Te::MakeCoord(iter1 * baseK_, 0), AscendC::Te::MakeShape(curKL0, curN));
                if constexpr(transB) {  // TODO: 后续asc自动推导后删除通过trait传递转置属性
                    AscendC::Te::Copy(
                        AscendC::Te::CopyAtom<
                            AscendC::Te::CopyTraits<AscendC::Te::CopyL12L0, AscendC::Te::LoadDataTraitDefault>>{},
                        tensorBL0, tensorBlockBL1);
                } else {
                    AscendC::Te::Copy(
                        AscendC::Te::CopyAtom<
                            AscendC::Te::CopyTraits<AscendC::Te::CopyL12L0, AscendC::Te::LoadData2BTrait>>{},
                        tensorBL0, tensorBlockBL1);
                }

                // scaleA, scaleB L1->L0
                auto coordScaleKL1 = (iter0 % (scaleKL1_ / kL1_)) * Cmct::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * 2;
                auto layoutScaleAL0 =
                    AscendC::Te::MakeZzLayout<fp8_e8m0_t>(curM, Cmct::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * 2);
                auto tensorScaleAL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr((__ca__ fp8_e8m0_t*)(l0Offset)), layoutScaleAL0);
                auto layoutScaleAL1 =
                    AscendC::Te::MakeZzLayout<fp8_e8m0_t>(curM, Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * 2);
                auto tensorScaleAL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL1memPtr((__cbuf__ fp8_e8m0_t*)l1BufferScaleAOffset_[scaleL1BufId]),
                    layoutScaleAL1);
                auto tensorBlockScaleAL1 = tensorScaleAL1(
                    AscendC::Te::MakeCoord(0, coordScaleKL1),
                    AscendC::Te::MakeShape(curM, Cmct::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * 2));
                AscendC::Te::Copy(
                    AscendC::Te::CopyAtom<AscendC::Te::CopyTraits<Cmct::Gemm::Tile::CopyL12L0MxScaleA3510>>{},
                    tensorScaleAL0, tensorBlockScaleAL1, AscendC::Te::MakeCoord(0, iter1 * baseK_));

                auto layoutScaleBL0 =
                    AscendC::Te::MakeNnLayout<fp8_e8m0_t>(Cmct::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * 2, curN);
                auto tensorScaleBL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr((__cb__ fp8_e8m0_t*)(l0Offset)), layoutScaleBL0);
                auto layoutScaleBL1 =
                    AscendC::Te::MakeNnLayout<fp8_e8m0_t>(Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * 2, curN);
                auto tensorScaleBL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL1memPtr((__cbuf__ fp8_e8m0_t*)l1BufferScaleBOffset_[scaleL1BufId]),
                    layoutScaleBL1);
                auto tensorBlockScaleBL1 = tensorScaleBL1(
                    AscendC::Te::MakeCoord(coordScaleKL1, 0),
                    AscendC::Te::MakeShape(Cmct::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * 2, curN));
                AscendC::Te::Copy(
                    AscendC::Te::CopyAtom<AscendC::Te::CopyTraits<Cmct::Gemm::Tile::CopyL12L0MxScaleB3510>>{},
                    tensorScaleBL0, tensorBlockScaleBL1, AscendC::Te::MakeCoord(iter1 * baseK_, 0));

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                // mmad_mx
                uint8_t mmadUnitFlag =
                    (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
                bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0 && !isBias_);
                AscendC::Te::Mad(
                    AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<Tile::MmadMx>>{}.with(
                        static_cast<uint16_t>(curM),
                        static_cast<uint16_t>(Cmct::Gemm::CeilAlign(curKL0, MXFP_DIVISOR_SIZE)),
                        static_cast<uint16_t>(curN), mmadUnitFlag, false, mmadCmatrixInitVal),
                    tensorL0C, tensorAL0, tensorBL0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                l0PingPong_++;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            if ((iter0 + 1) % (scaleKL1_ / kL1_) == 0 || iter0 == kL1Iter_ - 1) {
                // Scale buffers may live longer than A/B buffers when one scale chunk
                // covers multiple `kL1_` compute iterations.
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
                scaleLoopCnt_++;
            }
            abL1LoopCnt_++;
        }
        // C L0C->GM
        AscendC::Te::Copy(
            AscendC::Te::CopyAtom<
                AscendC::Te::CopyTraits<AscendC::Te::CopyL0C2GM, AscendC::Te::MxFixpipeTrait<CType>>>{},
            gmC, tensorL0C);
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    __aicore__ inline bool NeedBias(uint64_t kIter0, uint64_t kIter1)
    {
        // Bias is added once at the beginning of accumulation.
        return isBias_ && kIter0 == 0 && kIter1 == 0;
    }

    __aicore__ inline void Mmad(
        AscendC::MmadParams &mmadParams, uint64_t l0cOffset, uint64_t l0abOffset, uint64_t biasOffset, bool needBias)
    {
        // MMAD supports two sources for the initial accumulator state:
        // - bias from BT/C2 on the first step
        // - the existing partial sum in CO1 on all following steps
        mmadParams.cmatrixSource = needBias;
        if (needBias) {
            AscendC::Mmad(
                c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], biasBt_[biasOffset], mmadParams);
        } else {
            mmadParams.cmatrixSource = false;
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], mmadParams);
        }
    }

private:
    uint16_t biasBufId_ = 0;
    uint64_t biasL1OneBuffer_ = 0UL;
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[4] = {0UL}; // default 4 buffer
    uint64_t l1BufferBOffset_[4] = {0UL}; // default 4 buffer
    uint64_t l1BufferScaleAOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferScaleBOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferBiasOffset_[2] = {0UL}; // default 2 buffer
    AscendC::LocalTensor<MxL0AType> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<MxL0BType> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, L0C_SIZE};
    AscendC::LocalTensor<float> biasBt_{AscendC::TPosition::C2, 0, BT_SIZE};
    AscendC::LocalTensor<AType> aL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<BiasType> biasL1Local_{AscendC::TPosition::A1, 0, L1_SIZE / sizeof(BiasType)};
    AscendC::LocalTensor<fp8_e8m0_t> scaleAL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleBL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
};
}  // namespace Block
#endif