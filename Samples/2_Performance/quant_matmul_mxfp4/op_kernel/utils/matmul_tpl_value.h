/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file matmul_tpl_value.h
 * \brief
 */

#ifndef MATMUL_TPL_VALUE_H
#define MATMUL_TPL_VALUE_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace ascend_ops {
namespace matmul {

#define MATMUL_ASWT 0
#define MATMUL_STREAMK 1

#define MATMUL_FIXPIPE_CUBE 0
#define MATMUL_FIXPIPE_MIX 1

struct MatmulTplValue {
    uint8_t computeMode = MATMUL_ASWT;
    uint8_t copyOutMode = MATMUL_FIXPIPE_CUBE;
};

} // namespace matmul
} // namespace ascend_ops

#endif // MATMUL_TPL_VALUE_H
