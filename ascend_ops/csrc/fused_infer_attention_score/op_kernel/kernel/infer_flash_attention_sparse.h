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
 * \file infer_flash_attention_sparse.h
 * \brief
 */
#ifndef INFER_FLASH_ATTENTION_SPARSE_H
#define INFER_FLASH_ATTENTION_SPARSE_H

#include "infer_flash_attention_comm.h"

TEMPLATE_INTF
__aicore__ inline void GetSparseParam(const ConstInfo<isInfer, hasRope> &constInfo,
    const AttenMaskInfo &attenMaskInfo, RunParamStr<isInfer> &runParam)
{
    if constexpr (hasAtten) {
        runParam.preTokensPerBatch = attenMaskInfo.preTokens;
        runParam.nextTokensPerBatch = attenMaskInfo.nextTokens;
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            runParam.preTokensPerBatch = SPARSE_MODE_INT_DEFAULT;

            runParam.nextTokensPerBatch = runParam.actualS2Size - runParam.actualS1Size;

        }
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            runParam.preTokensPerBatch = attenMaskInfo.preTokens - runParam.actualS2Size +
                runParam.actualS1Size;
            runParam.nextTokensPerBatch = attenMaskInfo.nextTokens + runParam.actualS2Size -
                runParam.actualS1Size;
        }
    }
}

#endif  // INFER_FLASH_ATTENTION_SPARSE_H