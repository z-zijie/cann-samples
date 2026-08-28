/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file kernel_args.h
 * \brief AICPU 与 AICore 内核共享的 Tiling/参数结构体定义
 */

#pragma once

#include <cstdint>

namespace KernelInfo {
struct TilingInfo {
    int8_t type;
    int8_t mode;
    int8_t len;
};

struct KernelArgs {
    uint32_t* xDevice;
    uint32_t* yDevice;
    uint32_t* zDevice;
    TilingInfo* ti; // Parameters shared with aicore are used for synchronizing tiling selection
};
} // namespace KernelInfo
