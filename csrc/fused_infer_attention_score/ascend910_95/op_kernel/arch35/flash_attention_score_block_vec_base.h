/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_block_vec_base.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_BLOCK_VEC_BASE_H_
#define FLASH_ATTENTION_SCORE_BLOCK_VEC_BASE_H_
#include "util_regbase.h"
#include "infer_flash_attention_comm.h"
#include "flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "vf/vf_flashupdate_new.h"
#include "vf/vf_div_cast.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
TEMPLATES_DEF_BASE
class FABlockVecBase {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
    static constexpr uint32_t vec1HalfS1BaseSize = s1BaseSize >> 1;
    static constexpr uint32_t vec1S2CopyCountDn = s1BaseSize >> 5;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ScmBlock = s1BaseSize * 8;
    static constexpr uint32_t vec1ScmBlockFp32 = s1BaseSize * 4;
    static constexpr uint32_t vec1ScmBlockFp8 = s1BaseSize * 16;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = false;
    static constexpr bool useDn = false;
    static constexpr bool hasPse = pseMode != PseTypeEnum::PSE_NONE_TYPE;
    static constexpr bool hasPseOuter = (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) ||
                                        (pseMode == PseTypeEnum::PSE_OUTER_MUL_ADD_TYPE);
    static constexpr bool containAllOptionalInput = hasPse && hasAtten && hasDrop;
    static constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;
    static constexpr TPosition bmm2OutPos = GetC2Position(
        dVTemplateType, UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, pseMode, hasAtten, hasDrop,
        s1BaseSize == 64), (s2BaseSize == 256 && s1BaseSize == 64));
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
    static constexpr bool isW8In = false;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;
    using pseShiftW8InType = typename AscendC::Conditional<isInfer, half, OUTPUT_T>::type;
    using pseShiftType = typename AscendC::Conditional<isW8In, pseShiftW8InType, INPUT_T>::type;
    static constexpr int64_t FP8_QUANT_KV_BLOCK_SIZE = isInfer ? 256 : 128;
    // ==================== Functions ======================
    __aicore__ inline FABlockVecBase() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const optiling::FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo) {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            tilingData = tiling;
            GetDerived()->InitCubeVecSharedParams(sharedParams, aicIdx, subBlockIdx);
            this->GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
            attenMaskInfoPtr = &attenMaskInfo;
            pseInfoPtr = &pseInfo;
        }
    }
    __aicore__ inline void InitCommonGlobalBuffer(
        __gm__ uint8_t *attenMask, __gm__ uint8_t *&workspace, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);

    TPipe *tPipe;
    const optiling::FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;

    /* =====================可选GM变量==================== */
    __gm__ uint8_t *pseSlope;
    using pseGmType = typename std::conditional<hasPse, GlobalTensor<pseShiftType>, int8_t>::type;
    pseGmType pseGm;
    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;
    attenMaskGmType attenMaskGmInt;

    using vec2ResGmType = typename std::conditional<splitD, GlobalTensor<float>, int8_t>::type;
    vec2ResGmType vec2ResGm[3];

    /* =====================V侧UB变量==================== */
    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue[2];
    TQue<QuePosition::VECIN, 1> pseInQue;
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf[3];
    TBuf<> softmaxSumBuf[3];
    TBuf<> softmaxExpBuf[3];
    TBuf<> vselrIndexesBuf[4];
    TBuf<> mm2InBuf;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域 */
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;
    /* =================初始化后不变的信息================= */
    PseInfo *pseInfoPtr;
    AttenMaskInfo *attenMaskInfoPtr;
    T negativeFloatScalar;
    T positiveFloatScalar;
    // Bmm2阶段subblock在Gm上的偏移
    int64_t bmm2SubBlockOffset = 0;
    int64_t vec2SubBlockOffset = 0;
    __aicore__ inline ChildClass* GetDerived() {
        return static_cast<ChildClass*>(this);
    }
protected:
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = INPUT_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
private:
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void ProcessVec1Dn(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ProcessVec1Nd(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void ProcessVec2DSplit(GlobalTensor<T> &mmRes, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ProcessVec2NoGlobalUpdate(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, int64_t vec2CalcSize);

    __aicore__ inline bool SoftmaxInvalidLineCheck(LocalTensor<T> &maxUb, uint32_t negativeIntScalar, SoftMaxShapeInfo &softmaxShapeInfo);
    __aicore__ inline void InvalidLineProcess(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb);

    __aicore__ inline int64_t ComputeOffsetForSoftmax(RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx);
    __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar);
    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo, int64_t dSizeAligned64);
};

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::InitCommonGlobalBuffer(
    __gm__ uint8_t *attenMask, __gm__ uint8_t *&workspace, ConstInfo<isInfer, hasRope> &constInfo) 
{
    if ASCEND_IS_AIV {
        if constexpr (hasAtten) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }
        if constexpr (!bmm2Write2Ub) {
            int64_t bmm2ResBlock = tilingData->inputParamsRegbase.dSizeV;
            if constexpr (splitD) {
                bmm2ResBlock = (int64_t)dVTemplateType;
            }
            int64_t vec2ResultSize = (s1BaseSize) * constInfo.dBasicBlock;
            int64_t vec2Offset = CeilDiv(vec2ResultSize, 128) * 128 * sizeof(T);
            int64_t mm2ResultSize = (s1BaseSize) * bmm2ResBlock; // 使用Cube计算的总大小， Gm上的数据按照实际的dSize存储
            if constexpr (splitD) {
                vec2SubBlockOffset = constInfo.subBlockIdx * vec2ResultSize >> 1;
                vec2ResGm[0].SetGlobalBuffer((__gm__ T *)(workspace));
                workspace += vec2Offset;
                vec2ResGm[1].SetGlobalBuffer((__gm__ T *)(workspace));
                workspace += vec2Offset;
                vec2ResGm[2].SetGlobalBuffer((__gm__ T *)(workspace));
                workspace += vec2Offset;
            }
            bmm2SubBlockOffset = constInfo.subBlockIdx * mm2ResultSize >> 1; // s1BaseSize一定可以被2整除
        }
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer, hasRope> &constInfo)
{
    ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo, constInfo);
}

// =================================Private Functions=================================

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline bool FABlockVecBase<TEMPLATE_BASE_ARGS>::SoftmaxInvalidLineCheck(
    LocalTensor<T> &maxUb, uint32_t negativeIntScalar, SoftMaxShapeInfo &softmaxShapeInfo)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    bool isUpdateNeedCheck = false;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, softmaxShapeInfo.srcK);
    for (uint32_t i = 0; i < softmaxShapeInfo.srcM; i++) {
        T maxValue = maxUb.GetValue(i);
        uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);
        if (checkValue == negativeIntScalar) {
            isUpdateNeedCheck = true;
            break;
        }
    }
    SetMaskNorm();
    ResetMask();
    return isUpdateNeedCheck;
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::InvalidLineProcess(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb)
{
    if (constInfo.softMaxCheckRes) {
        SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1),
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1)};
        bool res = SoftmaxInvalidLineCheck(maxUb, NEGATIVE_MIN_VAULE_FP32, softmaxShapeInfo);
        if (!res) {
            constInfo.softMaxCheckRes = false;
        } else {
            if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                SoftmaxSumUpdate<T>(sumUb, maxUb, runInfo.halfS1RealSize, this->negativeFloatScalar,
                    this->positiveFloatScalar);
            }
        }
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec1Nd(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer, hasRope> &constInfo)
{
    bmm1ResBuf.WaitCrossCore();
    float slopes = 0.0f;
    float posShift = 0.0f;

    LocalTensor<pseShiftType> pseUb;
    LocalTensor<uint8_t> dropMaskUb;

    // attenMaskUb, srcTensor, maskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
    // attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize, constInfo
    LocalTensor<uint8_t> attenMaskUb;
    if constexpr (hasAtten == true) {
        AttenMaskCopyIn<hasAtten, isFd>(this->attenMaskInQue[runInfo.taskIdMod2], this->attenMaskInQue[1 - runInfo.taskIdMod2],
            this->attenMaskGmInt, runInfo, constInfo, *attenMaskInfoPtr);
        attenMaskUb = this->attenMaskInQue[runInfo.taskIdMod2].template DeQue<uint8_t>();
    }
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
    LocalTensor<uint8_t> apiTmpBuffer;
    if constexpr (IsSameType<INPUT_T, float>::value) {
        apiTmpBuffer = this->sumBrdcst.template AllocTensor<uint8_t>();
    } else {
        apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    }

    int64_t stage1Offset = 0;
    if constexpr (!IsSameType<INPUT_T, float>::value) {
        stage1Offset = runInfo.taskIdMod2;
    }
    float descaleQK = 1.0;

    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
    if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx) {
        if (likely(runInfo.s2RealSize == 128)) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 64) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize < 128) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb);
        } else {
            if constexpr (s2BaseSize == 256) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten, pseMode, hasDrop>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                    attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                    pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                    constInfo.keepProb);
            }
        }
    } else {
        if (likely(runInfo.s2RealSize == 128)) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 64) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize < 128) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb);
        } else {
            if constexpr (s2BaseSize == 256) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten, pseMode, hasDrop>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                    attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                    pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                    constInfo.keepProb);
            }
        }
    }
    bmm1ResBuf.SetCrossCore();
    //AscendC::DumpTensor(stage1CastTensor, 919191, 2048);
    if constexpr (hasAtten) {
        this->attenMaskInQue[runInfo.taskIdMod2].template FreeTensor(attenMaskUb);
    }
    // =================== DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
    LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

    if (likely(runInfo.halfS1RealSize != 0)) {
        if constexpr (IsSameType<INPUT_T, float>::value) {
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1ScmBlockFp32], stage1CastTensor,
                    {16, (uint16_t)runInfo.halfS1RealSize, (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
                    (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});
        }else {
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1ScmBlock], stage1CastTensor,
                    {s2BaseSize / 16, (uint16_t)runInfo.halfS1RealSize,
                    (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
                    (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});
        }
    }
    
    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
    outputBuf.SetCrossCore();
    // ======================================================
    if (runInfo.s2LoopCount != runInfo.s2LoopStartIdx) {
        UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize);
    }
    if constexpr (IsSameType<INPUT_T, float>::value) {
        this->sumBrdcst.template FreeTensor(apiTmpBuffer);
    }
    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
        if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
            this->InvalidLineProcess(runInfo, constInfo, sumUb, maxUb);
        }
    }
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        // GetDerived()->SoftmaxDataCopyOut(runInfo, constInfo, sumUb, maxUb);
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline int64_t FABlockVecBase<TEMPLATE_BASE_ARGS>::ComputeOffsetForSoftmax(
    RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx)
{
    return vec2S1Idx * runInfo.vec2S1BaseSize;
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec2OnUb(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo) {
    if (unlikely(runInfo.vec2S1BaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }


    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    
    int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
    float deSCaleVValue;
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
        float deSCalePreVValue = 1.0f;
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            
            FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    1.0, 1.0);
            
        } else {
            
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                1.0, 1.0);
            
        }
    }
    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(
                vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize, (uint16_t)dTemplateAlign64, deSCaleVValue);
        }
        //AscendC::DumpTensor(vec2ResUb, 515151, 256);
        GetDerived()->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec2NoGlobalUpdate(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
    int64_t vec2CalcSize) {
    LocalTensor<INPUT_T> vec2ResUb = this->stage2OutBuf.template Get<INPUT_T>()[runInfo.taskIdMod2 * vec2CalcSize];
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[runInfo.taskIdMod2]);
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    DivCast<T, INPUT_T, dTemplateAlign64>(vec2ResUb, bmm2ResBuf.template GetTensor<T>(), sumUb, runInfo.vec2S1RealSize);
    bmm2ResBuf.SetCrossCore();
    Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, 0);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[runInfo.taskIdMod2]);
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec2DSplit(
    GlobalTensor<T> &mmRes, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo) {
    // bmm2 result is on GM and global update data on UB
    runInfo.vec2S1BaseSize = 8192 / constInfo.dBasicBlock;
    int64_t vec2LoopLimit = CeilDiv(runInfo.halfS1RealSize, runInfo.vec2S1BaseSize);
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    LocalTensor<T> bmm2Ub = mm2InBuf.template Get<T>();

    event_t mte2ToV = static_cast<event_t>(tPipe->FetchEventID(HardEvent::MTE2_V));
    event_t vToMte2 = static_cast<event_t>(tPipe->FetchEventID(HardEvent::V_MTE2));
    event_t mte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    float deSCaleVValue;
    for (int64_t vec2S1Idx = 0; vec2S1Idx < vec2LoopLimit; vec2S1Idx++) {
        runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
        if (vec2S1Idx == vec2LoopLimit - 1) {
            runInfo.vec2S1RealSize = runInfo.halfS1RealSize - vec2S1Idx * runInfo.vec2S1BaseSize;
        }
        int64_t vec2CalcSize = runInfo.vec2S1RealSize * constInfo.dBasicBlock;
        // Gm地址偏移是按照实际DSize实现的，虽然我们设置了KC=192
        int64_t mm2ResInnerOffset = vec2S1Idx * runInfo.vec2S1BaseSize * dTemplateAlign64;
        SetFlag<HardEvent::V_MTE2>(vToMte2);
        WaitFlag<HardEvent::V_MTE2>(vToMte2);
        if (constInfo.dSizeV == dTemplateAlign64) {
            DataCopy(bmm2Ub, mmRes[bmm2SubBlockOffset + mm2ResInnerOffset], vec2CalcSize);
        } else {
            DataCopyParams dataCopyParams;
            DataCopyPadParams dataCopyPadParams;
            dataCopyParams.blockCount = runInfo.vec2S1RealSize;
            dataCopyParams.dstStride = (constInfo.dBasicBlock - constInfo.dSizeV) * sizeof(T) / blockBytes;
            dataCopyParams.srcStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T);
            dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
            DataCopyPad(bmm2Ub, mmRes[bmm2SubBlockOffset + mm2ResInnerOffset],
                        dataCopyParams, dataCopyPadParams);
        }

        // 经过了跳读，UB上每行是按照dTemplateAlign64对齐的
        int64_t vec2ResInnerOffset = vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dBasicBlock;
        if (vec2LoopLimit > 1) {
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            DataCopy(vec2ResUb, this->vec2ResGm[runInfo.multiCoreIdxMod3][vec2SubBlockOffset + vec2ResInnerOffset], vec2CalcSize);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
            DataCopy(vec2ResUb, bmm2Ub, vec2CalcSize);
        } else {
            int64_t vec2ExpBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
            float deSCalePreVValue = 1.0f;
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>()[vec2ExpBufOffset];
            if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx + 1) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, 0xFF, true>(
                        vec2ResUb, bmm2Ub, vec2ResUb, expUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                } else {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, 0xFF, false>(
                        vec2ResUb, bmm2Ub, vec2ResUb, expUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                }
            } else {
                int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> sumUb =
                    this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx + 1) {
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, 0xFF, true>(vec2ResUb, bmm2Ub,
                        vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                } else {
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, 0xFF, false>(vec2ResUb, bmm2Ub,
                        vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                }
            }
        }

        if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
            if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
                int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> sumUb =
                    this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                LastDivNew<T, INPUT_T, OUTPUT_T, 0xFF>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock, deSCaleVValue);
            }
            GetDerived()->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
        } else if (vec2LoopLimit > 1) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
            DataCopy(this->vec2ResGm[runInfo.multiCoreIdxMod3][vec2SubBlockOffset + vec2ResInnerOffset], vec2ResUb, vec2CalcSize);
        }
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec2(
    mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    bmm2ResBuf.WaitCrossCore();

    if constexpr (bmm2Write2Ub) {
        ProcessVec2OnUb(bmm2ResBuf, runInfo, constInfo);
    } else if constexpr (splitD) {
        GlobalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        ProcessVec2DSplit(mmRes, runInfo, constInfo);
    } else {
        // bmm2 result is on GM and global update data on UB
        runInfo.vec2S1BaseSize = 8192 / dTemplateAlign64;
        int64_t vec2LoopLimit = CeilDiv(runInfo.halfS1RealSize, runInfo.vec2S1BaseSize);
        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        LocalTensor<T> bmm2Ub = mm2InBuf.template Get<T>();
        GlobalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();

        event_t mte2ToV = static_cast<event_t>(tPipe->FetchEventID(HardEvent::MTE2_V));
        event_t vToMte2 = static_cast<event_t>(tPipe->FetchEventID(HardEvent::V_MTE2));
        float deSCaleVValue;

        for (int64_t vec2S1Idx = 0; vec2S1Idx < vec2LoopLimit; vec2S1Idx++) {
            runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
            if (vec2S1Idx == vec2LoopLimit - 1) {
                runInfo.vec2S1RealSize = runInfo.halfS1RealSize - vec2S1Idx * runInfo.vec2S1BaseSize;
            }

            int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
            // Gm地址偏移是按照实际DSize实现的，虽然我们设置了KC=192
            int64_t mm2ResInnerOffset = vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dSizeV;
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            if (constInfo.dSizeV == dTemplateAlign64) {
                DataCopy(bmm2Ub, mmRes[bmm2SubBlockOffset + mm2ResInnerOffset], vec2CalcSize);
            } else {
                DataCopyParams dataCopyParams;
                DataCopyPadParams dataCopyPadParams;
                dataCopyParams.blockCount = runInfo.vec2S1RealSize;
                dataCopyParams.dstStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T) / blockBytes;
                dataCopyParams.srcStride = 0;
                dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
                DataCopyPad(bmm2Ub, mmRes[bmm2SubBlockOffset + mm2ResInnerOffset],
                            dataCopyParams, dataCopyPadParams);
            }
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            // 经过了跳读，UB上每行是按照dTemplateAlign64对齐的
            LocalTensor vec2ResInner = vec2ResUb[vec2S1Idx * runInfo.vec2S1BaseSize * dTemplateAlign64];

            if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
                DataCopy(vec2ResInner, bmm2Ub, vec2CalcSize);
            } else {
                int64_t vec2ExpBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                float deSCalePreVValue = 1.0f;
                LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>()[vec2ExpBufOffset];
                if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                    if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx + 1) {
                        FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(
                            vec2ResInner, bmm2Ub, vec2ResInner, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                            deSCaleVValue, deSCalePreVValue);
                    } else {
                        FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                            vec2ResInner, bmm2Ub, vec2ResInner, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                            deSCaleVValue, deSCalePreVValue);
                    }
                } else {
                    int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                    LocalTensor<float> sumUb =
                        softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                    if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx + 1) {
                        FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(vec2ResInner, bmm2Ub,
                            vec2ResInner, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue,
                            deSCalePreVValue);
                    } else {
                        FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResInner, bmm2Ub,
                            vec2ResInner, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue,
                            deSCalePreVValue);
                    }   
                }
            }

            if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
                    int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                    LocalTensor<float> sumUb =
                        softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                    LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(
                        vec2ResInner, vec2ResInner, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue);
                }
                GetDerived()->CopyOutAttentionOut(runInfo, constInfo, vec2ResInner, vec2S1Idx, vec2CalcSize);
            }
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    }
    return;
}

TEMPLATES_DEF_BASE_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb,
    int64_t vec2S1Idx, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, int64_t dSizeAligned64)
{
    if constexpr (isInfer && hasAtten) {
        if (!constInfo.isRowInvalid || \
            attenMaskInfoPtr->compressMode != static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
            return;
        }
        int64_t vec2MaxBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
        LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2MaxBufOffset];
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        bool isRowInvalidNeedUpdate = false;
        for (uint32_t i = 0; i < runInfo.vec2S1RealSize; i++) {
            float maxValue = maxTensor.GetValue(i);
            uint32_t checkValue = *(uint32_t*)&maxValue;
            if (checkValue == NEGATIVE_MIN_VAULE_FP32) {
                isRowInvalidNeedUpdate = true;
                break;
            }
        }
        if (isRowInvalidNeedUpdate) {
            
            RowInvalidUpdateVF<float>(vec2ResUb, maxTensor,  runInfo.vec2S1RealSize, constInfo.dSizeV, static_cast<uint32_t>(dSizeAligned64));
            
        }
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::Bmm2DataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (splitD) {
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    if constexpr (!IsSameType<INPUT_T, VEC2_RES_T>::value) {
        attenOut.SetAddr(vec2ResUb.address_);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                int64_t vec2MaxBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2MaxBufOffset];
                InvalidLineUpdate<T, dTemplateAlign64>(vec2ResUb, vec2ResUb, maxTensor, runInfo.vec2S1RealSize,
                    dSizeAligned64, this->negativeFloatScalar, 0.0);
            }
        }
        
        RowInvalid(vec2ResUb, vec2S1Idx, runInfo, constInfo, dSizeAligned64);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
        
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    } else {
        
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.taskIdMod2]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.taskIdMod2]);
        attenOut = vec2ResUb;
        
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    if constexpr (IsSameType<INPUT_T, float>::value) {
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 3;
    } else {
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4;
    }
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2S1RealSize;

    int64_t attenOutOffset = constInfo.dSizeV;
    
    

    if constexpr (isInfer) {
        
        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
            attenOut, dataCopyParams); 
        
    } else {
        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
            attenOut, dataCopyParams); // 
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::SoftmaxInitBuffer()
{
    tPipe->InitBuffer(softmaxSumBuf[0], 256); // [64, 1]
    tPipe->InitBuffer(softmaxSumBuf[1], 256); // [64, 1]
    tPipe->InitBuffer(softmaxSumBuf[2], 256); // [64, 1]
    tPipe->InitBuffer(maxBrdcst, 1, 2048); // [64, 8]
    tPipe->InitBuffer(sumBrdcst, 1, 2048); // [64, 8]
    tPipe->InitBuffer(softmaxMaxBuf[0], 256); // [64, 1]
    tPipe->InitBuffer(softmaxMaxBuf[1], 256); // [64, 1]
    tPipe->InitBuffer(softmaxMaxBuf[2], 256); // [64, 1]
    tPipe->InitBuffer(softmaxExpBuf[0], 256); // [64, 1]
    tPipe->InitBuffer(softmaxExpBuf[1], 256); // [64, 1]
    tPipe->InitBuffer(softmaxExpBuf[2], 256); // [64, 1]
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer, hasRope> &constInfo)
{
    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    if constexpr (!bmm2Write2Ub) {
        tPipe->InitBuffer(mm2InBuf, 32768); // bmm2结果在Gm，vector2开启多层循环，每次处理32KB
    }
    if constexpr (s2BaseSize == 256) { // s1BaseSize = 128
        if constexpr (s1BaseSize == 128) { // s1BaseSize = 128 s2BaseSize = 256
            tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
            SoftmaxInitBuffer();
            
            tPipe->InitBuffer(stage1OutQue[0], 1, 33024);
            tPipe->InitBuffer(stage1OutQue[1], 1, 33024);
            
        } else { // s1BaseSize = 64 s2BaseSize = 256
            SoftmaxInitBuffer();
            tPipe->InitBuffer(commonTBuf, 512); // 实际上只需要512Bytes
            tPipe->InitBuffer(stage2OutBuf, 32 * dTemplateAlign64 * sizeof(T));
            tPipe->InitBuffer(stage1OutQue[0], 1, 16896);
            tPipe->InitBuffer(stage1OutQue[1], 1, 16896);
            if constexpr (hasAtten) {
                tPipe->InitBuffer(attenMaskInQue[0], 1, 8192);
                tPipe->InitBuffer(attenMaskInQue[1], 1, 8192);
            }
            
        }
    } else { // s1BaseSize = 128 s2BaseSize = 128
        SoftmaxInitBuffer();
        
        

        if constexpr (hasAtten) {
            tPipe->InitBuffer(attenMaskInQue[0], 1, 8192);
            tPipe->InitBuffer(attenMaskInQue[1], 1, 8192);
        }

        if constexpr (!IsSameType<INPUT_T, float>::value) {
            tPipe->InitBuffer(commonTBuf, 512); // 实际上只需要512Bytes
        }
        

        if constexpr (bmm2Write2Ub) {
            // 小于128Bmm2结果和Vec2结果都在UB
            tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
        } else if constexpr (dTemplateAlign64 <= 256) {
            // bmm2结果在Gm，Vector2结果在UB，开启多层循环，每次处理32KB
            tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
        } else {
            // bmm2结果在Gm，Vector2结果也在Gm，开启多层循环，每次处理32KB
            tPipe->InitBuffer(stage2OutBuf, 32768);
        }
        if constexpr (IsSameType<INPUT_T, float>::value) {
            tPipe->InitBuffer(stage1OutQue[0], 1, 33280);
        } else {
            tPipe->InitBuffer(stage1OutQue[0], 1, 16640);
            tPipe->InitBuffer(stage1OutQue[1], 1, 16640);
        }
    }
    // GetDerived()->InitUniqueLocalBuffer(constInfo);
    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();

    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);

}


TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::GetExtremeValue(
    T &negativeScalar, T &positiveScalar)
{
    if constexpr (IsSameType<T, float>::value) {
        uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
        negativeScalar = *((float *)&tmp1);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode ==
                static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                uint32_t tmp2 = POSITIVE_MAX_VALUE_FP32;
                positiveScalar = *((float *)&tmp2);
            }
        }
    } else {
        uint16_t tmp1 = NEGATIVE_MIN_VAULE_FP16;
        negativeScalar = *((half *)&tmp1);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode ==
                static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                uint16_t tmp2 = POSITIVE_MAX_VALUE_FP16;
                positiveScalar = *((half *)&tmp2);
            }
        }
    }
}

TEMPLATES_DEF
class FABlockVecDummy {
public:
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr TPosition bmm2OutPos = GetC2Position(
        dVTemplateType, BaseApi::UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, pseMode, hasAtten, hasDrop,
        s1BaseSize == 64), (s2BaseSize == 256 && s1BaseSize == 64));
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;

    __aicore__ inline FABlockVecDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, 
        ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const optiling::FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx,
        AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo) {};
    __aicore__ inline void InitGlobalBuffer(
         __gm__ uint8_t *attenMask, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
        ConstInfo<isInfer, hasRope> &constInfo) {}

    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer, hasRope> &constInfo) {
    }
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo) {}

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo) {}
};
}
#endif // FLASH_ATTENTION_SCORE_BLOCK_VEC_BASE_H_