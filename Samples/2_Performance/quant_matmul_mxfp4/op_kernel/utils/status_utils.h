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
 * \file status_utils.h
 * \brief
 */

#ifndef UTILS_STATUS_UTILS_H
#define UTILS_STATUS_UTILS_H

namespace Cmct {
namespace Gemm {
enum class Status {
    success,
    batchErrorExcceedsLimit,
    mnkErrorExceedsLimit,
    mkErrorMatrixExceedsLimit,
    kmErrorMatrixExceedsLimit,
    knErrorMatrixExceedsLimit,
    nkErrorMatrixExceedsLimit,
    bf16BiasErrorInvalidDataType,
    tileShapeErrorExceedsLimit,
    l1L0ErrorExceedsLimit,
    l1L0ErrorNotAlign,
    l1MnL0MnErrorNotSame,
    l1kErrorSmallerL0k,
    l1kErrorL0kNotAlign,
    l1L0ErrorNotAlignInt8
};

} // namespace Gemm
} // namespace Cmct
#endif