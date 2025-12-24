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
 * \file matmul_block_scheduler.h
 * \brief
 */

#ifndef MATMUL_BLOCK_SCHEDULER_UTILS_H
#define MATMUL_BLOCK_SCHEDULER_UTILS_H

namespace x {
namespace matmul {
namespace Block {

// Base template definition for BlockSchedulerSelector
template <class ProblemShape, class L1TileShape, class L0TileShape, class BlockScheduler = void>
struct BlockSchedulerSelector;

} // namespace Block
} // namespace matmul
} // namespace x

#endif // MATMUL_BLOCK_SCHEDULER_UTILS_H