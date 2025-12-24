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
 * \file infer_flash_attention_kvcache.h
 * \brief
 */
#ifndef INFER_FLASH_ATTENTION_KVCACHE_H
#define INFER_FLASH_ATTENTION_KVCACHE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "infer_flash_attention_comm.h"
#include "infer_flash_attention_sparse.h"

using namespace matmul;

TEMPLATE_INTF
__aicore__ inline void InitQueryLeftPaddingSize(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope>& constInfo, int64_t& actualS1Size)
{
    if (!constInfo.isQHasLeftPadding) {
        runParam.queryLeftPaddingSize = 0;
    } else {
        int64_t qLeftPaddingSize = constInfo.s1Size - actualS1Size - constInfo.queryRightPaddingSize;
        runParam.queryLeftPaddingSize = qLeftPaddingSize > 0 ? qLeftPaddingSize : 0;
        if (qLeftPaddingSize < 0) {
            actualS1Size = 0;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void InitKVLeftPaddingSize(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope>& constInfo, int64_t& actualS2Size)
{
    if (!constInfo.isKVHasLeftPadding) {
        runParam.kvLeftPaddingSize = 0;
    } else {
        int64_t kvLeftPaddingSize = constInfo.s2Size - actualS2Size - constInfo.kvRightPaddingSize;
        runParam.kvLeftPaddingSize = kvLeftPaddingSize > 0 ? kvLeftPaddingSize : 0;
        if (kvLeftPaddingSize < 0) {
            actualS2Size = 0;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(RunParamStr<isInfer>& runParam, 
    const ConstInfo<isInfer, hasRope> &constInfo, const AttenMaskInfo &attenMaskInfo, int32_t sIdx,
    GlobalTensor<INPUT_T>& keyGm, __gm__ int64_t *actualSeqQlenAddr,
    __gm__ int64_t * actualSeqKvlenAddr)
{
    // TensorList场景获取不同batch的KvSeq长度
    if (constInfo.isKvContinuous == 0) {
        ListTensorDesc keyListTensorDesc((__gm__ void*)keyGm.GetPhyAddr());
        AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;
        uint64_t dimInfo[4];
        kvTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(kvTensorDesc, sIdx);
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            runParam.s2InCurrentBatch = kvTensorDesc.GetShape(2);
        } else {
            runParam.s2InCurrentBatch = kvTensorDesc.GetShape(1);
        }
    }
    int64_t actualS1Size = 0;
    int64_t actualS2Size = 0;
    int64_t actualSeqMin = 1;
    int64_t actualSeqKVMin = 1;
    if (constInfo.isActualLenDimsNull) {
        actualS1Size = constInfo.s1Size;
        if (constInfo.isGqa) {
            actualS1Size = constInfo.gS1;
        }
    } 
    if (constInfo.isActualLenDimsKVNull) {
        actualS2Size = (constInfo.isKvContinuous == 1) ? constInfo.s2Size :
            runParam.s2InCurrentBatch;
    } else {
        
        actualS2Size = (constInfo.actualSeqLenKVSize == actualSeqKVMin) ? 
            actualSeqKvlenAddr[0] : actualSeqKvlenAddr[sIdx];
        
    }

    InitQueryLeftPaddingSize<TEMPLATE_INTF_ARGS>(runParam, constInfo, actualS1Size);
    InitKVLeftPaddingSize<TEMPLATE_INTF_ARGS>(runParam, constInfo, actualS2Size);

    runParam.actualS1Size = actualS1Size;
    runParam.actualS2Size = actualS2Size;
    GetSparseParam<TEMPLATE_INTF_ARGS>(constInfo, attenMaskInfo, runParam);

    runParam.actualS1Size = 
        (runParam.actualS1Size > runParam.actualS2Size + runParam.preTokensPerBatch) ?
        runParam.actualS2Size + runParam.preTokensPerBatch : runParam.actualS1Size;

    // 计算S1的尾块大小，非对齐
    runParam.actualS1Size = (runParam.nextTokensPerBatch >= 0) ? runParam.actualS1Size :
        (runParam.actualS1Size + runParam.nextTokensPerBatch);

    
    runParam.qBOffset = sIdx * constInfo.s1Size * constInfo.n2GD + runParam.queryLeftPaddingSize * constInfo.dSize;
    

    // 推理的TND场景的mask和pse都是padding过的
    runParam.b1SSOffset = runParam.boIdx * constInfo.s1S2;
    // 推理的mask的sequence length可能大于qk的sequence length
    runParam.b1SSAttenMaskOffset = runParam.boIdx * (uint64_t)attenMaskInfo.attenMaskS1Size * (uint64_t)attenMaskInfo.attenMaskS2Size;
}

TEMPLATE_INTF
__aicore__ inline void GetKeyCoreOffsetParam(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo, 
    int32_t sIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t keyInnerOffsetSize = 0;
    
    uint64_t headStrideK = 0;
    if (constInfo.isKvContinuous == 1) {
        headStrideK = constInfo.s2D;
        keyInnerOffsetSize = sIdx * constInfo.n2Size * headStrideK +
            runParam.kvLeftPaddingSize * constInfo.dSize;
    } else {
        headStrideK = constInfo.dSize * runParam.s2InCurrentBatch;
        keyInnerOffsetSize = 0;
    }
    runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * headStrideK;
    
}

TEMPLATE_INTF
__aicore__ inline void GetValueCoreOffsetParam(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo, 
    int32_t sIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t valueInnerOffsetSize = 0;
    
    uint64_t headStrideV = 0;
    if (constInfo.isKvContinuous == 1) {
        headStrideV = constInfo.s2Dv;
        valueInnerOffsetSize = sIdx * constInfo.n2Size * headStrideV +
            runParam.kvLeftPaddingSize * constInfo.dSizeV;
    } else {
        headStrideV = constInfo.dSizeV * runParam.s2InCurrentBatch;
        valueInnerOffsetSize = 0;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * headStrideV;
    
    if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
        GetKeyCoreOffsetParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, sIdx, actualSeqKvlenAddr);
    } else {
        runParam.keyCoreOffset = runParam.valueCoreOffset;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    const AttenMaskInfo &attenMaskInfo,
    GlobalTensor<INPUT_T>& keyGm, __gm__ int64_t *actualSeqQlenAddr,
    __gm__ int64_t *actualSeqKvlenAddr)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(
        runParam, constInfo, attenMaskInfo, runParam.boIdx, keyGm, actualSeqQlenAddr, actualSeqKvlenAddr);
    GetValueCoreOffsetParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, actualSeqKvlenAddr);

}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo, bool lastBN, 
    int64_t nextGs1Idx)
{
    constexpr int32_t s1BaseSize = static_cast<int32_t>(s1TemplateType);
    int32_t s1LoopTimes = CeilDiv(runParam.actualS1Size, s1BaseSize);
    // 不是最后一个bn, 赋值souterBlockNum
    if (!lastBN) {
        runParam.s1LoopTimes = s1LoopTimes;
    } else { // 最后一个bn, 从数组下一个元素取值
        runParam.s1LoopTimes = nextGs1Idx == 0 ? s1LoopTimes : nextGs1Idx;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    uint32_t sOuterLoopIdx)
{
    int64_t cubeSOuterOffset = sOuterLoopIdx * (uint32_t)s1TemplateType;
    if (runParam.actualS1Size == 0) {
        runParam.s1RealSize = 0;
    } else {
        runParam.s1RealSize = Min((uint32_t)s1TemplateType, runParam.actualS1Size - cubeSOuterOffset);
    }

    cubeSOuterOffset += (runParam.nextTokensPerBatch < 0) ? -runParam.nextTokensPerBatch : 0;

    
    runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
        
    
    
    runParam.firstHalfS1RealSize = runParam.halfS1RealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfS1RealSize = runParam.s1RealSize - runParam.halfS1RealSize;
        runParam.sOuterOffset = cubeSOuterOffset + runParam.firstHalfS1RealSize;
    } else {
        runParam.sOuterOffset = cubeSOuterOffset;
    }
    runParam.cubeSOuterOffset = cubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    int32_t sIdx, __gm__ int64_t *actualSeqQlenAddr, PseInfo& pseInfo)
{
    
    int64_t actualSeqLen = 0;
    int64_t seqOffset = 0;
    
    actualSeqLen = constInfo.s1Size;
    seqOffset = sIdx * constInfo.s1Size;
    
    if ASCEND_IS_AIC {

        if (constInfo.isGqa) {
            runParam.tensorQOffset = runParam.qBOffset + runParam.n2oIdx * constInfo.gD * actualSeqLen +
                runParam.cubeSOuterOffset * constInfo.dSize;
        } else {
            
            runParam.tensorQOffset = runParam.qBOffset + runParam.n2oIdx * constInfo.gS1D +
                runParam.goIdx * constInfo.s1D + runParam.cubeSOuterOffset * constInfo.dSize;
            
        }
        
    } else {
        int64_t attentionOutSeqOffset = seqOffset * constInfo.n2GDv;

        if (constInfo.isGqa && constInfo.s1Size > 1) { 
            
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.n2oIdx * constInfo.gDv * actualSeqLen +
                runParam.sOuterOffset * constInfo.dSizeV;
            
        } else if(constInfo.isGqa) { //IFA
                runParam.attentionOutOffset = attentionOutSeqOffset + runParam.n2oIdx * constInfo.gDv * actualSeqLen +	
                    runParam.sOuterOffset * constInfo.dSizeV;
        } else {
            
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.n2oIdx * constInfo.gS1Dv +
                runParam.goIdx * constInfo.s1Dv + (runParam.sOuterOffset + runParam.queryLeftPaddingSize) *
                constInfo.dSizeV;
            
        }
        
        int64_t softmaxLseSeqOffset = seqOffset * constInfo.n2G;
        
        if (constInfo.isGqa) {
            runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.n2oIdx * constInfo.gSize * actualSeqLen +
                runParam.sOuterOffset;
        } else {
            
            runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.n2oIdx * constInfo.gS1 +
                runParam.goIdx * constInfo.s1Size + runParam.sOuterOffset + runParam.queryLeftPaddingSize;
            
        }
        
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    uint32_t sOuterLoopIdx, __gm__ int64_t *actualSeqQlenAddr, PseInfo& pseInfo)
{
    // 后续的函数依赖 sOuterOffset
    ComputeSouterParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, sOuterLoopIdx);

    // 使用转换后的左上角的pretoken nexttoken
    if (runParam.nextTokensPerBatch < 0 && runParam.sOuterOffset < ((runParam.nextTokensPerBatch * (-1)) /
        runParam.halfS1RealSize * runParam.halfS1RealSize)) {
        return true;
    }

    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, actualSeqQlenAddr, pseInfo);
    return false;
}

TEMPLATE_INTF
__aicore__ inline int64_t ClipSInnerTokenCube(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeS2LoopInfo(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo)
{
    constexpr int32_t s2BaseSize = static_cast<int32_t>(s2TemplateType);

    int64_t sInnerFirstToken = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset - runParam.preTokensPerBatch,
        0, runParam.actualS2Size);
    runParam.s2LineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset + runParam.nextTokensPerBatch +
        runParam.s1RealSize, 0, runParam.actualS2Size);

    runParam.s2LoopStartIdx = sInnerFirstToken / s2BaseSize;
    runParam.s2LoopEndIdx = (runParam.s2LineEndIdx + s2BaseSize - 1) / s2BaseSize;

    if (runParam.s2LoopEndIdx <= runParam.s2LoopStartIdx) {
        return true;
    }
    if constexpr (hasAtten) {
        runParam.s2LineStartIdx = runParam.s2LoopStartIdx * s2BaseSize;
    } else {
        runParam.s2LineStartIdx = sInnerFirstToken;
    }
    return false;
}

TEMPLATE_INTF
__aicore__ inline void ComputeOffset(const RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope> &constInfo, uint32_t sInnerLoopIdx, RunInfo<isInfer> &runInfo)
{
    if ASCEND_IS_AIV {
        
        if (!constInfo.isGqa) {
            runInfo.vecCoreOffset = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
        } else {
            runInfo.vecCoreOffset = 0;
        }
    } else {
        
        runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;

        if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseD;
        } else {
            runInfo.keyOffset = runInfo.valueOffset;
        }
        
    }
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr<isInfer>& runParam, RunInfo<isInfer> &runInfo)
{
    runInfo.keyOffset = runParam.keyOffset;
    runInfo.boIdx = runParam.boIdx;
    runInfo.preTokensPerBatch = runParam.preTokensPerBatch;
    runInfo.nextTokensPerBatch = runParam.nextTokensPerBatch;
    runInfo.b1SSAttenMaskOffset = runParam.b1SSAttenMaskOffset;
    runInfo.b1SSOffset = runParam.b1SSOffset;
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.softmaxLseOffset = runParam.softmaxLseOffset;
    runInfo.s2InCurrentBatch = runParam.s2InCurrentBatch;
    runInfo.queryLeftPaddingSize = runParam.queryLeftPaddingSize;
    runInfo.kvLeftPaddingSize = runParam.kvLeftPaddingSize;
}

#endif  // INFER_FLASH_ATTENTION_KVCACHE_H