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
 * \file matmul_block_scheduler_policy.h
 * \brief
 */

#ifndef MATMUL_BLOCK_SCHEDULER_POLICY_H
#define MATMUL_BLOCK_SCHEDULER_POLICY_H

namespace ascend_ops {
namespace matmul {

struct BuiltInAswtScheduler {};
struct BuiltInStreamKScheduler {};
struct QuantMatmulMxBuiltInAswtScheduler {};

} // namespace matmul
} // namespace ascend_ops
#endif // MATMUL_BLOCK_SCHEDULER_POLICY_H