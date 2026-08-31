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
 * \file moe_kernel_common.h
 * \brief
 */

#ifndef MOE_COMMON_H
#define MOE_COMMON_H

#include "kernel_operator.h"

using namespace AscendC;

constexpr static int64_t BLOCK_BYTES = 32;
constexpr static int64_t PIPELINE_DEPTH = 1;
constexpr static int64_t DOUBLE_BUFFER = 2;
constexpr static int64_t SIMT_THREAD_NUM = 2048;
constexpr static int64_t SIMT_X_THREAD_NUM = 64;
constexpr static int64_t SIMT_Y_THREAD_NUM = 32;
constexpr static int64_t SIMT_DCACHE_SIZE = 64 * 1024; // UB给SIMT预留64k的DCache空间
constexpr static int64_t KV_FACTOR = 2; // sort key and value
constexpr static int64_t SORT_BUFFER_FACTOR = 6;

constexpr static int64_t ONE_REPEAT_SORT_NUM = 32; // 排序元素对齐32，sort api要求
constexpr static int64_t SORT_API_MAX_ELEM = 32 * 255; // AscendC::Sort全排序模式最多支持一次排序(32*255rep)个元素
constexpr static int64_t MRG_LIST_NUM = 4;
constexpr static int64_t MRG_SORT_API_MAX_ELEM = 1024;
constexpr static int64_t FP32_ONE_REPEAT_NUM = 64;
constexpr static float MIN_FP32 = -3.4e38f;
constexpr static int64_t MERGE_LIST_TWO = 2;
constexpr static int64_t MERGE_LIST_THREE = 3;
constexpr static int64_t MERGE_LIST_FOUR = 4;
constexpr static int64_t MERGE_LIST_IDX_TWO = 2;
constexpr static int64_t MERGE_LIST_IDX_THREE = 3;

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

// 孤立单点 fence（等价于原 SetWaitFlag 单点 fence）：附近没有可自然包住的
// producer/consumer 流水指令，故保留空 Lock/Unlock 对表达 PIPE_A -> PIPE_B 的先后关系——
// producer 侧 Lock/Unlock 让 A 管道在此点前的指令完成后释放，consumer 侧 Lock/Unlock 让 B 管道等待该释放。
// 注意：此类二进制锁在「目标管道先到、源尚未执行」时存在漏等窗口，须在 950 真机重点验证时序；
// 若异常应回退为原 SetFlag/WaitFlag 并列入豁免清单，或补预置相位。
// 能自然包住真实流水指令的位置（如 CopyX 的 DataCopyPad、Compute 的 Duplicate 等），
// 已在各 .asc 调用点就地 Lock/真实指令/Unlock，不再走这里的 helper。
__aicore__ inline void SyncMte3ToMte2(uint8_t mutex)
{
    AscendC::Mutex::Lock<PIPE_MTE3>(mutex);
    AscendC::Mutex::Unlock<PIPE_MTE3>(mutex);
    AscendC::Mutex::Lock<PIPE_MTE2>(mutex);
    AscendC::Mutex::Unlock<PIPE_MTE2>(mutex);
}

__aicore__ inline void SyncMte3ToS(uint8_t mutex)
{
    AscendC::Mutex::Lock<PIPE_MTE3>(mutex);
    AscendC::Mutex::Unlock<PIPE_MTE3>(mutex);
    AscendC::Mutex::Lock<PIPE_S>(mutex);
    AscendC::Mutex::Unlock<PIPE_S>(mutex);
}

__aicore__ inline void SyncSToMte2(uint8_t mutex)
{
    AscendC::Mutex::Lock<PIPE_S>(mutex);
    AscendC::Mutex::Unlock<PIPE_S>(mutex);
    AscendC::Mutex::Lock<PIPE_MTE2>(mutex);
    AscendC::Mutex::Unlock<PIPE_MTE2>(mutex);
}

__aicore__ inline void SyncMte2ToS(uint8_t mutex)
{
    AscendC::Mutex::Lock<PIPE_MTE2>(mutex);
    AscendC::Mutex::Unlock<PIPE_MTE2>(mutex);
    AscendC::Mutex::Lock<PIPE_S>(mutex);
    AscendC::Mutex::Unlock<PIPE_S>(mutex);
}

__simd_vf__ inline void SortVf(__ubuf__ float *inUbAddr, int64_t expertStart, uint32_t sreg, uint16_t repeatTimes)
{
    float cmpScalar = static_cast<float>(expertStart);
    float negone = static_cast<float>(-1);

    MicroAPI::RegTensor<float> inRegToFloat, infFloat, vDstReg0;
    MicroAPI::MaskReg maskRegLoop, cmpMaskReg;
    MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    Duplicate(infFloat, static_cast<float>(MIN_FP32), pregMain);

    for (uint16_t i = 0; i < repeatTimes; i++) {
        maskRegLoop = MicroAPI::UpdateMask<float>(sreg);
        MicroAPI::DataCopy(inRegToFloat, inUbAddr + i * VECTOR_REG_WIDTH / sizeof(float));
        MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg, inRegToFloat, cmpScalar, maskRegLoop);
        MicroAPI::Muls(inRegToFloat, inRegToFloat, negone, maskRegLoop);
        MicroAPI::Select(vDstReg0, infFloat, inRegToFloat, cmpMaskReg);
        MicroAPI::DataCopy(inUbAddr + i * VECTOR_REG_WIDTH / sizeof(float), vDstReg0, maskRegLoop);
    }
}

#endif // MOE_COMMON_H