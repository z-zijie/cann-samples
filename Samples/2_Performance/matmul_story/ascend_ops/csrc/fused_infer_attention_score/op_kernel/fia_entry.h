/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_entry.h
 * \brief
 */

#ifndef PROMPT_FLASH_ATTENTION_ENTRY_310_H_
#define PROMPT_FLASH_ATTENTION_ENTRY_310_H_
#include "kernel/flash_attention_score_tiling.h"
namespace optiling {};
#include "./kernel/flash_attention_score_kernel_infer.h"
#include "fia_enum.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

__aicore__ inline void CopyTiling(FlashAttentionScoreSimplifiedTilingData *tilingData, __gm__ uint8_t *tilingGM)
{
    int64_t *ptr = reinterpret_cast<int64_t *>(tilingData);
    auto tiling32 = reinterpret_cast<__gm__ int64_t *>(tilingGM);
    for (int64_t i = 0; i < sizeof(FlashAttentionScoreSimplifiedTilingData) / sizeof(int64_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

template <uint8_t inOutLayoutType=0, bool hasAttenMask=0, uint16_t config=3>
inline __aicore__ void FlashAttentionEntry(__gm__ uint8_t *query, __gm__ uint8_t *key,
    __gm__ uint8_t *value, __gm__ uint8_t *attenMask, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    FlashAttentionScoreSimplifiedTilingData  tilingDataTemp;
    CopyTiling(&tilingDataTemp, tiling);
    FlashAttentionScoreSimplifiedTilingData* __restrict tilingData = &tilingDataTemp;
    
    PARSE_PARAMS_NoQuant(inOutLayoutType, config, hasAttenMask);
    constexpr uint64_t qkvSizeRsv2 =
        MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                static_cast<uint64_t>(dTemplateType),
            static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) *
        2;
    constexpr uint64_t vec1ResultSize =
        static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
    TPipe tPipe;
    // using TemplateType = 
    if ASCEND_IS_AIC {  // CUBE 实现
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,
            BaseApi::FABlockCube<bfloat16_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                hasAttenMask,
                false,
                false,
                true,
                false,
                false>,
            BaseApi::FABlockCubeDummy<bfloat16_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                hasAttenMask,
                false,
                false,
                true,
                false,
                false>>::type;
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,
            BaseApi::FABlockVecDummy<bfloat16_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                hasAttenMask,
                false,
                false,
                true,
                false,
                false>,
            BaseApi::FABlockVecInfer<bfloat16_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                hasAttenMask,
                false,
                false,
                true,
                false,
                false>>::type;

        BaseApi::FlashAttentionScoreKernelInfer<CubeBlockType, VecBlockType> op;
        op.InitBaseAPI(query, key, value, attenMask, attentionOut, user, nullptr, &tPipe);
        op.Process();

    } else {  // VECTOR 实现
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,
            BaseApi::FABlockCube<bfloat16_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                hasAttenMask,
                false,
                false,
                true,
                false,
                false>,
            BaseApi::FABlockCubeDummy<bfloat16_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                hasAttenMask,
                false,
                false,
                true,
                false,
                false>>::type;
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,
            BaseApi::FABlockVecDummy<bfloat16_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                hasAttenMask,
                false,
                false,
                true,
                false,
                false>,
            BaseApi::FABlockVecInfer<bfloat16_t,
                float,
                bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION,
                outputLayoutType,
                s1TemplateType,
                s2TemplateType,
                dTemplateType,
                dVTemplateType,
                hasAttenMask,
                false,
                false,
                true,
                false,
                false>>::type;
        BaseApi::FlashAttentionScoreKernelInfer<CubeBlockType, VecBlockType> op;
        op.InitBaseAPI(query, key, value, attenMask, attentionOut, user, tilingData, &tPipe);
        op.Process();
    }
}
#endif  // end of PROMPT_FLASH_ATTENTION_ENTRY_310_H_