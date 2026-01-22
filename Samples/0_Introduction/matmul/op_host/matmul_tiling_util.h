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
 * \file matmul_tiling_util.h
 * \brief
 */
#ifndef MATMUL_TILING_UTIL_H
#define MATMUL_TILING_UTIL_H

namespace x {
namespace matmul {

class TilingUtil {
public:
    template <typename T>
    static T CeilDivision(T num1, T num2)
    {
        return (num1 + num2 - 1) / num2;
    }

    template <typename T>
    static T CeilAlign(T num1, T num2)
    {
        return TilingUtil::CeilDivision(num1, num2) * num2;
    }

    template <typename T>
    static T FloorAlign(T num1, T num2)
    {
        return num1 / num2 * num2;
    }
};

} // namespace matmul
} // namespace x

#endif // MATMUL_TILING_UTIL_H
