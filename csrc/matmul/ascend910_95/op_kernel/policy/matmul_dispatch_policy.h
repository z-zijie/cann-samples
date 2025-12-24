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
 * \file matmul_dispatch_policy.h
 * \brief
 */
#ifndef MATMUL_DISPATCH_POLICY_H
#define MATMUL_DISPATCH_POLICY_H

#include "../utils/matmul_integral_constant.h"

namespace x {
namespace matmul {

/* block schedule policies */
struct KernelMultiBlockOnKAxis {}; // Multi-tile pipelined transfer with K-axis caching
struct KernelMultiBlockStreamK {}; // Multi-tile transfer with K-axis spliting and caching

/**
 * @struct MatmulMultiBlockWithAswt
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0>>
struct MatmulMultiBlockWithAswt {
    using ScheduleType = KernelMultiBlockOnKAxis;
    using SingleShape = SingleCoreShape;
};

} // namespace matmul
} // namespace x

#endif // MATMUL_DISPATCH_POLICY_H
