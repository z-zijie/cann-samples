/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FIA_TILING_H
#define FIA_TILING_H

#include <iostream>
// must define this before tiling.h, otherwise will compile error
typedef uint8_t char_t; 
// #include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "fia_tiling_struct.h"
#include "../op_kernel/arch35/flash_attention_score_tiling_regbase.h"


struct FIAShapeInfo {
    uint32_t b = 0;
    uint32_t n = 0;
    uint32_t s = 0;
    uint32_t d = 0;
    uint32_t h = 0;
    uint32_t t = 0;
};
constexpr uint32_t FLOAT32SIZE = 4;

class FiaDemoTiling {
public:
    FiaDemoTiling(){ }

    uint8_t RunBigKernelTilingWithParams(ContextParamsForTiling& contextKeyParams);
    uint8_t SetPlatMemoryInfo(ContextParamsForTiling& contextKeyParams);
    uint8_t SetAttributeInfo(ContextParamsForTiling& contextKeyParams);
    uint8_t CheckSingleAttribute(ContextParamsForTiling& contextKeyParams, FIAShapeInfo& queryShapeInfo,
    FIAShapeInfo& keyShapeInfo, FIAShapeInfo& valueShapeInfo);

    bool SetInputLayout(string& layout);
    bool SetShape(FIAShapeInfo& shapeInfo, ContextParamsForTiling& contextKeyParams, const std::string inputName);
    bool SetHeadNumRatio(ContextParamsForTiling& contextKeyParams);
    bool SetMaskTypeAndShape(ContextParamsForTiling& contextKeyParams);
    bool SetSparseMode(ContextParamsForTiling& contextKeyParams,uint32_t qS);
    void SetSparseType(uint32_t qS);
    void SetSparseModeData(ContextParamsForTiling& contextKeyParams, const at::IntArrayRef attenMaskShape, 
                                        const int32_t& sparseMode, const int64_t& preTokens, const int64_t& nextTokens);

    void SetTilingData(ContextParamsForTiling& contextKeyParams, FIAShapeInfo& queryShapeInfo, FIAShapeInfo& valueShapeInfo);
    void SetTilingDataAttribute(ContextParamsForTiling& contextKeyParams);                                
    void InferConstantization();
    void InferSplitCoreMode();
    void InferTilingMod(const ContextParamsForTiling& contextKeyParams, std::vector<int64_t>& actualSeqLengths, 
                                    std::vector<int64_t>& actualSeqLengthsKV, uint32_t actualSeqArrayLen, uint32_t d);

    uint8_t AdjustTilingData(ContextParamsForTiling& contextKeyParams, const FIAShapeInfo& queryShapeInfo, const FIAShapeInfo& valueShapeInfo);

    bool AdjustCVTilingCVDiff(const ContextParamsForTiling& contextKeyParams, uint32_t& sOuterFactor, uint32_t& sInnerFactor, 
                                        uint32_t& softmaxSOuterFactor, const FIAShapeInfo& queryShapeInfo, const FIAShapeInfo& valueShapeInfo);
    uint8_t SetQKVStartIdx(ContextParamsForTiling& contextKeyParams);

    void FlashAttentionSplitNBSeq(std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV, bool isAttenMaskUsed);
    void GetPreNextTokensLeftUp(int64_t actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp);
    void TilingDataconvert();
    void SetAttenMaskCompressMode();
    void SetLayoutType();
    void FlashAttentionInitOutputSplit(int64_t totalSize);

    void FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) const;
    int64_t GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner, int64_t sOuter, int64_t token);
    int64_t GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV, uint32_t sOuterSize, uint32_t sInnerSize, 
                                                    int64_t preTokensLeftUp, int64_t nextTokensLeftUp, bool isAttenMaskUsed);
    void ComputeSplitNBSeq(uint32_t batchSize, const size_t tilingElementArrayLen, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV,
                                          uint32_t sOuterSize, uint32_t sInnerSize, double coreWightTarget, uint32_t& curCore);

    int64_t GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums); 
    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t actualUsedCoreNum);

    uint8_t ComputeTilingData(ContextParamsForTiling& contextKeyParams, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV); 


    uint8_t CheckCrossoverAttribute(ContextParamsForTiling& contextKeyParams, FIAShapeInfo& queryShapeInfo);
    bool CheckMaskShapeCrossSparse(ContextParamsForTiling& contextKeyParams, const int32_t* sparseMode, uint32_t sQ, const uint32_t sK, const uint32_t batchSize);
    bool CheckMaskShape(ContextParamsForTiling& contextKeyParams, const int32_t* sparseMode,
                                                      int64_t& attenMaskBatch, int64_t& attenMaskS1, int64_t& attenMaskS2, bool& checkMask, const uint32_t sQ,
                                                    const uint32_t sK, const uint32_t batchSize, std::string& strMaskShape);
    bool ParseActualSeqLengths(ContextParamsForTiling& contextKeyParams,
                               FIAShapeInfo& queryShapeInfo, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV);
    bool CheckMultiFeatureCrossover(ContextParamsForTiling& contextKeyParams,
                                FIAShapeInfo& queryShapeInfo, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV);

    ContextParamsForTiling* contextKeyParamsPtr = nullptr;
    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;

    int64_t actSeqLenDims = 0;
    int64_t actSeqLenKVDims = 0;
    bool enableActSeqLen = false;
    bool enableActSeqLenKV = false;
    bool enableMask = false;
    bool isDefaultSparseMode = false;
    bool enableTensorList = false;

    uint32_t m_gOfMla = 0;
    uint32_t m_qn = 0;
    bool enableIFA = false;
    uint32_t gSize = 1;
    uint32_t S2 = 0;
    int64_t sparsePreTokens = 0;
    int64_t sparseNextTokens = 0;
    int32_t sparseModeVal = 0;
    uint8_t sparseType = 0;
    bool isDNoTail = true;
    uint32_t dataTypeSize = FLOAT32SIZE;
    bool isConstantization = false;
    bool enableMatmulNorm = false;

    uint32_t m_sOuterFactor = 0;
    uint32_t m_sInnerFactor = 0;
    uint32_t sOuterFactorTiling = 0;
    uint32_t softmaxSInnerFactorTiling = 0;
    uint32_t softmaxSOuterFactorTiling = 0;
    int64_t actualSharedPrefixLen = 0;

    uint8_t attenMaskShapeType = 0; // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)

    InputLayout inputLayout = InputLayout::BSH;
    SplitCoreMode splitCoreMode = SplitCoreMode::SPLIT_NBS_VECTOR;

    uint32_t needInit = 0U;
    int64_t middleActualSeqLengths = 0;
    int64_t maxActualseqKV = 0;

    optiling::FlashAttentionScoreSimplifiedTilingData faTilingAdapter;
};



#endif