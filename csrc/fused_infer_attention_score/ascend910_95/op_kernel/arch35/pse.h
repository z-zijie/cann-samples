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
 * \file pse.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_PSE_H
#define FLASH_ATTENTION_SCORE_PSE_H

#include "kernel_operator.h"
#include "util_regbase.h"

namespace regbaseutil {
constexpr static int64_t pseS1S2 = 0;
constexpr static int64_t pse1S2 = 1;
constexpr static int64_t pseSlopeBn = 2;
constexpr static int64_t pseSlopeN = 3;

constexpr static uint8_t pseEncodeALibiS2Full = 0x11;

enum class PseTypeEnum {
    PSE_OUTER_MUL_ADD_TYPE = 0, // default
    PSE_OUTER_ADD_MUL_TYPE,
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE,
    PSE_NONE_TYPE = 9
};

struct PseInfo {
    int64_t pseBSize;         // pse输入batch大小
    int64_t pseS1Size;        // for alibi
    int64_t pseS2ComputeSize; // for alibi, do not need assignment
    int64_t pseS2Size;        // for alibi
    int64_t readS2Size;       // for alibi, do not need assignment
    uint32_t pseLayoutType;   // pse输入shape的layout
    uint32_t pseEncodeType;   // for distinguish alibi
    uint32_t pseType;         // 0: outer, mul-add   1:outer, add-mul   2:inner, mul-add   3:inner, mul-add-sqrt
    uint32_t pseStride;
    int64_t qStartIdx;
    int64_t kvStartIdx;
};

template <typename INPUT_T, bool hasPse>
__aicore__ inline void DataCopyInCommon(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor,
                                        int64_t offset, int64_t s1Size, int64_t s2Size, int64_t actualS2Len,
                                        int32_t s2BaseSize)
{

}

template <typename INPUT_T, bool hasPse>
__aicore__ inline void DataCopyIn(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor, int64_t offset,
                                  int64_t s1Size, int64_t s2Size, int64_t s2BaseSize, int64_t actualS2Len)
{

}

template <typename INPUT_T, bool hasPse>
__aicore__ inline void DataCopyInAlign8(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor, int64_t offset,
                                  int64_t s1Size, int64_t s2Size, int64_t actualS2Len)
{

}

template <bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline int64_t PseComputeOffset(const RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer, hasRope> &constInfo, PseInfo &pseInfo)
{
    return 0;
}

template <bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline int64_t PseAlibiComputeOffset(const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, PseInfo &pseInfo)
{
        return 0; 
}

template <bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline bool NeedPseAlibiCompute(const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, PseInfo &pseInfo)
{
        return false;
}

template <typename T, typename INPUT_T, bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline void PseAlibiCopyIn(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor,
                                      const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, PseInfo &pseInfo)
{

}

template <typename T, typename INPUT_T, bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline void PseCopyIn(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor,
                                 const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, PseInfo &pseInfo)
{

}

template <typename T, typename INPUT_T, bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline void PseCopyIn(TQue<QuePosition::VECIN, 1> &pseInQue, GlobalTensor<INPUT_T> &srcTensor,
                                 const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, PseInfo &pseInfo)
{

}

template <typename T, typename INPUT_T, bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline void ComputeInnerPseOffset(float &slopes, float &posShift, const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
                                             PseInfo &pseInfo, __gm__ uint8_t *pseSlope)
{

}
}
#endif
