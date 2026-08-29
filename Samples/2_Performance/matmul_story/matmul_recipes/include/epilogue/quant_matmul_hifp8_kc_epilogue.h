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
 * \file quant_matmul_hifp8_kc_epilogue.h
 * \brief KC MIX epilogue (AIV side): read the fp32 accumulator from UB (written by the AIC
 *        fixpipe DUAL_DST_SPLIT_M), multiply by the perchannel scale ((n,) vector Mul) and the
 *        pertoken scale ((m,) per-row DIST_BRC_B32 broadcast Mul), cast to bf16, write to GM.
 *        The dequant core uses VF instructions, following blaze VFDoDequant (block_epilogue_dequant.h).
 */

#ifndef QUANT_MATMUL_HIFP8_KC_EPILOGUE_H
#define QUANT_MATMUL_HIFP8_KC_EPILOGUE_H

#include "kernel_operator.h"

namespace QuantMatmulHifp8Kc {

constexpr uint32_t KC_BLOCK = 32;
constexpr uint32_t KC_F_ALIGN = KC_BLOCK / sizeof(float);
constexpr uint32_t KC_B_ALIGN = KC_BLOCK / sizeof(bfloat16_t);
constexpr uint32_t CV_RATIO = 2;

// V_MTE3/MTE3_V event ids of the output ping/pong (ids are reusable across event pairs;
// pong uses 2 to distinguish from ping)
constexpr uint32_t OUT_BUF0_EVT = 0;
constexpr uint32_t OUT_BUF1_EVT = 2;

// Register Cast trait fp32 → bf16 (following blaze DQ_CT_FP32_2_HALF)
constexpr AscendC::Reg::CastTrait DQ_CT_FP32_2_BF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

// Per-row VF dequant (function-level SIMT-VF, following the ops-math arch35 VF style):
// fp32 row (UB) → RegTensor, × x2Scale (perchannel, UB vector) → × x1Scale
// (pertoken, DIST_BRC_B32 broadcasts the row scalar) → bf16 (DIST_PACK_B32) back to out ping/pong.
// Works only on __ubuf__ raw pointers and scalar params, iterating eleNumPerVf blocks and
// narrowing the mask per block.
__simd_vf__ inline void VFDoDequantRow(
    __ubuf__ float* l0cUbAddr, __ubuf__ float* x2ScaleUbAddr, __ubuf__ float* x1ScaleUbAddr,
    __ubuf__ bfloat16_t* outUbAddr, uint32_t mIdx, uint32_t nAligned, uint32_t eleNumPerVf, uint16_t nLoopCnt)
{
    for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
        uint32_t remain = nAligned - vfBlockIdx * eleNumPerVf;
        AscendC::Reg::MaskReg maskN = AscendC::Reg::UpdateMask<float>(remain);
        uint32_t blockOffset = vfBlockIdx * eleNumPerVf;

        AscendC::Reg::RegTensor<float> l0cReg;
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(
            l0cReg, l0cUbAddr + mIdx * nAligned + blockOffset);

        AscendC::Reg::RegTensor<float> x2ScaleReg;
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(x2ScaleReg, x2ScaleUbAddr + blockOffset);

        AscendC::Reg::RegTensor<float> mulScaleReg;
        AscendC::Reg::Mul(mulScaleReg, l0cReg, x2ScaleReg, maskN);

        AscendC::Reg::RegTensor<float> x1ScaleReg;
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(x1ScaleReg, x1ScaleUbAddr + mIdx);

        AscendC::Reg::RegTensor<float> mulPtScaleReg;
        AscendC::Reg::Mul(mulPtScaleReg, mulScaleReg, x1ScaleReg, maskN);

        AscendC::Reg::RegTensor<bfloat16_t> outReg;
        AscendC::Reg::Cast<bfloat16_t, float, DQ_CT_FP32_2_BF16>(outReg, mulPtScaleReg, maskN);
        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
            outUbAddr + blockOffset, outReg, maskN);
    }
}

class QuantMatmulHifp8KcEpilogue {
public:
    struct Params {
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR outGmAddr{nullptr};
        int64_t m{0};
        int64_t n{0};
        int64_t baseM{0};
        int64_t baseN{0};
    };

    __aicore__ inline QuantMatmulHifp8KcEpilogue()
    {}
    __aicore__ inline ~QuantMatmulHifp8KcEpilogue()
    {}

    __aicore__ inline void Init(const Params& params)
    {
        m_ = params.m;
        n_ = params.n;
        baseM_ = params.baseM;
        baseN_ = params.baseN;

        x2ScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.x2ScaleGmAddr), n_);
        x1ScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.x1ScaleGmAddr), m_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.outGmAddr), m_ * n_);

        nAlignF_ = (static_cast<uint32_t>(baseN_) + KC_F_ALIGN - 1) / KC_F_ALIGN * KC_F_ALIGN;
        nAlignB_ = (static_cast<uint32_t>(baseN_) + KC_B_ALIGN - 1) / KC_B_ALIGN * KC_B_ALIGN;
        uint32_t mForSingleVec = (static_cast<uint32_t>(baseM_) + CV_RATIO - 1) / CV_RATIO;
        mAlignF_ = (mForSingleVec + KC_F_ALIGN - 1) / KC_F_ALIGN * KC_F_ALIGN;
        subBlockIdx_ = AscendC::GetSubBlockIdx();

        // UB layout (one per AIV, all regions 32B aligned):
        // [0]                : L0C fp32 accumulator (written by AIC fixpipe SPLIT_M), capacity ceil(baseM/2) * nAlignF
        // floats [x2ScaleUbOffset_] : x2 perchannel scale (nAlignF floats) [x1ScaleUbOffset_] : x1 pertoken scale of
        // this sub-block segment (mAlignF floats) [out0UbOffset_]    : output row bf16 ping (nAlignB bf16)
        // [out1UbOffset_]    : output row bf16 pong (nAlignB bf16, alternated by row parity so
        //                      the MTE3 GM write overlaps the next row's V compute)

        l0cUbBytes_ = mForSingleVec * nAlignF_ * sizeof(float);
        x2ScaleUbOffset_ = l0cUbBytes_;
        x1ScaleUbOffset_ = x2ScaleUbOffset_ + nAlignF_ * sizeof(float);
        out0UbOffset_ = x1ScaleUbOffset_ + mAlignF_ * sizeof(float);
        out1UbOffset_ = out0UbOffset_ + nAlignB_ * sizeof(bfloat16_t);
    }

    __aicore__ inline void operator()(int64_t mPos, int64_t nPos, int64_t curM, int64_t curN)
    {
        if ASCEND_IS_AIC {
            return;
        }

        // DUAL_DST_SPLIT_M: sub-block 0 handles the first ceil(curM/2) rows, sub-block 1 the rest.
        // Each AIV's UB stores its own half starting at local offset 0, row pitch = CeilAlign(curN, 8) floats.
        int64_t halfM = (curM + 1) / 2;
        int64_t mInVec = (subBlockIdx_ == 1) ? (curM - halfM) : halfM;
        int64_t mOffset = static_cast<int64_t>(subBlockIdx_) * halfM;
        if (mInVec <= 0) {
            return;
        }

        uint32_t nAligned = (static_cast<uint32_t>(curN) + KC_F_ALIGN - 1) / KC_F_ALIGN * KC_F_ALIGN;
        uint32_t curNBytesF = static_cast<uint32_t>(curN) * sizeof(float);
        uint32_t curNBytesB = static_cast<uint32_t>(curN) * sizeof(bfloat16_t);

        AscendC::DataCopyExtParams copyNF{1, curNBytesF, 0, 0, 0};
        AscendC::DataCopyExtParams copyMF{1, static_cast<uint32_t>(mInVec * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padF{false, 0, 0, 0};
        AscendC::DataCopyExtParams copyNB{1, curNBytesB, 0, 0, 0};

        AscendC::LocalTensor<bfloat16_t> out0Ub(AscendC::TPosition::VECCALC, out0UbOffset_, nAlignB_);
        AscendC::LocalTensor<bfloat16_t> out1Ub(AscendC::TPosition::VECCALC, out1UbOffset_, nAlignB_);

        // One DMA per tile: x2Scale (consumed by V) + x1Scale sub-block segment (consumed by VF
        // DIST_BRC_B32). Both use MTE2_V to wait for the DMA landing before the V-stream
        // Reg::DataCopy reads them.
        AscendC::LocalTensor<float> x2ScaleUb(AscendC::TPosition::VECCALC, x2ScaleUbOffset_, nAlignF_);
        AscendC::DataCopyPad(x2ScaleUb, x2ScaleGm_[nPos], copyNF, padF);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

        AscendC::LocalTensor<float> x1ScaleUb(AscendC::TPosition::VECCALC, x1ScaleUbOffset_, mAlignF_);
        AscendC::DataCopyPad(x1ScaleUb, x1ScaleGm_[mPos + mOffset], copyMF, padF);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // Prime the first round: both out buffers are writable (no prior MTE3).
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OUT_BUF0_EVT);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OUT_BUF1_EVT);

        DequantizeRows(mPos, mOffset, nPos, mInVec, nAligned, out0Ub, out1Ub, copyNB);

        // Consume the remaining MTE3_V of both buffers, keeping the flags balanced per tile.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OUT_BUF0_EVT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OUT_BUF1_EVT);
        // The next tile's scale DMA must wait until this tile's last V read completes
        // (x1 BRC + x2 Mul).
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    }

private:
    // Row loop (ping/pong GM writes): each row first waits for the previous MTE3 of the same
    // buffer (row-2), then runs the VF dequant, then hands the result to MTE3 via the
    // V_MTE3/MTE3_V pair (overlapping with the next row's V compute).
    // The VF consumes scales/results on the V stream; the x1/x2 MTE2 syncs are done in operator().
    template <typename OutTensor>
    __aicore__ inline void DequantizeRows(
        int64_t mPos, int64_t mOffset, int64_t nPos, int64_t mInVec, uint32_t nAligned, const OutTensor& out0Ub,
        const OutTensor& out1Ub, const AscendC::DataCopyExtParams& copyNB)
    {
        uint32_t eleNumPerVf = asc_get_vf_len() / sizeof(float);
        uint16_t nLoopCnt = static_cast<uint16_t>((nAligned + eleNumPerVf - 1) / eleNumPerVf);
        auto l0cUbAddr = GetUbAddr<float>(0);
        auto x2ScaleUbAddr = GetUbAddr<float>(x2ScaleUbOffset_);
        auto x1ScaleUbAddr = GetUbAddr<float>(x1ScaleUbOffset_);
        for (int64_t mIdx = 0; mIdx < mInVec; mIdx++) {
            int64_t rowOffset = (mPos + mOffset + mIdx) * n_ + nPos;
            uint32_t outEvt = (mIdx & 1) == 0 ? OUT_BUF0_EVT : OUT_BUF1_EVT;
            auto outUb = (mIdx & 1) == 0 ? out0Ub : out1Ub;

            // Wait for the previous MTE3 of this buffer to finish writing GM (ping/pong alternates,
            // the other buffer's MTE3 is still in flight).
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outEvt);

            auto outUbAddr =
                (mIdx & 1) == 0 ? GetUbAddr<bfloat16_t>(out0UbOffset_) : GetUbAddr<bfloat16_t>(out1UbOffset_);
            // Function-level VF call (ops-math arch35 style: standalone __simd_vf__ function + asc_vf_call).
            asc_vf_call<VFDoDequantRow>(
                l0cUbAddr, x2ScaleUbAddr, x1ScaleUbAddr, outUbAddr, static_cast<uint32_t>(mIdx), nAligned, eleNumPerVf,
                nLoopCnt);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outEvt);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outEvt);
            AscendC::DataCopyPad(outGm_[rowOffset], outUb, copyNB);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outEvt);
        }
    }

    // UB address resolution: bank 0 base + byte offset (same as blaze GetUbAddr).
    // Read-only raw pointers (Reg::DataCopy accepts __ubuf__ pointers); row/buffer addresses
    // are 32B-aligned as laid out in Init.
    template <class T>
    __aicore__ inline static __ubuf__ T* GetUbAddr(uint64_t byteOffset)
    {
        return reinterpret_cast<__ubuf__ T*>(asc_get_phy_buf_addr(0) + byteOffset);
    }
    int64_t m_{0};
    int64_t n_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    uint32_t nAlignF_{0};
    uint32_t nAlignB_{0};
    uint32_t mAlignF_{0};
    uint32_t subBlockIdx_{0};

    uint64_t l0cUbBytes_{0};
    uint64_t x2ScaleUbOffset_{0};
    uint64_t x1ScaleUbOffset_{0};
    uint64_t out0UbOffset_{0};
    uint64_t out1UbOffset_{0};

    AscendC::GlobalTensor<float> x2ScaleGm_;
    AscendC::GlobalTensor<float> x1ScaleGm_;
    AscendC::GlobalTensor<bfloat16_t> outGm_;
};

} // namespace QuantMatmulHifp8Kc

#endif // QUANT_MATMUL_HIFP8_KC_EPILOGUE_H
