/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_layout_utils.h
 * \brief
 */
#ifndef MATMUL_LAYOUT_UTILS_H
#define MATMUL_LAYOUT_UTILS_H

#include "matmul_integral_constant.h"

namespace ascend_ops {
namespace matmul {

namespace layout {
struct RowMajor {};
struct ColumnMajor {};
} // namespace layout

// TagToTrans
template <typename T>
struct TagToTrans {
    static_assert(AscendC::Std::always_false_v<T>, "TagToTrans is not implemented for this layout");
};

template <>
struct TagToTrans<layout::RowMajor> {
    static constexpr bool value = false;
};

template <>
struct TagToTrans<layout::ColumnMajor> {
    static constexpr bool value = true;
};

enum class TransposeCombo : unsigned {
    NN = 0,
    NT = 1,
    TN = 2,
    TT = 3
};

// 创建一个通用的分派宏，处理所有4种组合
#define DISPATCH_TRANSPOSE_COMBINATION(TRANS_A, TRANS_B, ...)                                                          \
    do {                                                                                                               \
        constexpr unsigned kTransAWeight = 2;                                                                          \
        constexpr unsigned kTransBWeight = 1;                                                                          \
        const auto combo =                                                                                             \
            static_cast<TransposeCombo>((TRANS_A ? kTransAWeight : 0U) + (TRANS_B ? kTransBWeight : 0U));              \
        switch (combo) {                                                                                               \
            case TransposeCombo::NN: /* transA=false, transB=false */                                                  \
            {                                                                                                          \
                constexpr bool transA = false;                                                                         \
                constexpr bool transB = false;                                                                         \
                __VA_ARGS__                                                                                            \
            } break;                                                                                                   \
            case TransposeCombo::NT: /* transA=false, transB=true */                                                   \
            {                                                                                                          \
                constexpr bool transA = false;                                                                         \
                constexpr bool transB = true;                                                                          \
                __VA_ARGS__                                                                                            \
            } break;                                                                                                   \
            case TransposeCombo::TN: /* transA=true, transB=false */                                                   \
            {                                                                                                          \
                constexpr bool transA = true;                                                                          \
                constexpr bool transB = false;                                                                         \
                __VA_ARGS__                                                                                            \
            } break;                                                                                                   \
            case TransposeCombo::TT: /* transA=true, transB=true */                                                    \
            {                                                                                                          \
                constexpr bool transA = true;                                                                          \
                constexpr bool transB = true;                                                                          \
                __VA_ARGS__                                                                                            \
            } break;                                                                                                   \
            default:                                                                                                   \
                break;                                                                                                 \
        }                                                                                                              \
    } while (0)

} // namespace matmul
} // namespace ascend_ops

#endif // MATMUL_LAYOUT_UTILS_H