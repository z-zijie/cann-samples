/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <acl/acl.h>

#include <cstdint>

// 返回 true 表示 Host 侧调用成功。v0/v1 会在返回前同步 stream，v2～v11 可在 Kernel 提交后返回；
// 调用方在读取输出或释放输入输出前仍应同步 stream。requestedAicCoreNum 为 0 时使用设备全部 AIC 核。
bool FlashAttnLiteNPU(
    uint8_t* dQ, uint8_t* dK, uint8_t* dV, uint8_t* dOut, uint32_t batchSize, uint32_t seqLen, float softmaxScale,
    uint32_t requestedAicCoreNum, aclrtStream stream);
