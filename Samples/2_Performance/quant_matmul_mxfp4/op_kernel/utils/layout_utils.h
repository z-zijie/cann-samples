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
 * \file layout_utils.h
 * \brief
 */

#ifndef UTILS_LAYOUT_UTILS_H
#define UTILS_LAYOUT_UTILS_H

// dependency of matmul_utils.h
#include "matmul/tiling.h"
// CubeFormat
#include "matmul/matmul_config.h"
// AscendC::CeilAlign
#include "../../impl/adv_api/detail/matmul/utils/matmul_utils.h"
#include "./integral_constant.h"

namespace layout {
struct RowMajor {};
struct RowMajorAlign {}; // ND_ALIGN align to 32
struct ColumnMajor {};
struct ColumnMajorAlign {};
struct Nz {};
struct Zn {};
} // namespace layout

constexpr int32_t C0_BYTE_SIZE = 32;
constexpr int32_t C0_SIZE_FP16 = C0_BYTE_SIZE / sizeof(half);       // 16
constexpr int32_t C0_SIZE_BF16 = C0_BYTE_SIZE / sizeof(bfloat16_t); // 16
constexpr int32_t C0_SIZE_FP32 = C0_BYTE_SIZE / sizeof(float);      // 8
constexpr int32_t C0_NUM_PER_FRACTAL = 16;
constexpr int32_t C0_SIZE_L0C = 16;

// TagToFormat
template <typename T>
struct TagToFormat {
    static_assert(AscendC::Std::always_false_v<T>, "TagToFormat is not implemented for this layout");
};

template <>
struct TagToFormat<layout::RowMajor> {
    using tag = layout::RowMajor;
    static constexpr CubeFormat format = CubeFormat::ND;
};

template <>
struct TagToFormat<layout::RowMajorAlign> {
    using tag = layout::RowMajorAlign;
    static constexpr CubeFormat format = CubeFormat::ND_ALIGN;
};

template <>
struct TagToFormat<layout::ColumnMajor> {
    using tag = layout::ColumnMajor;
    static constexpr CubeFormat format = CubeFormat::ND;
};

template <>
struct TagToFormat<layout::ColumnMajorAlign> {
    using tag = layout::ColumnMajorAlign;
    static constexpr CubeFormat format = CubeFormat::ND_ALIGN;
};

template <>
struct TagToFormat<layout::Zn> {
    using tag = layout::Zn;
    static constexpr CubeFormat format = CubeFormat::NZ;
};

template <>
struct TagToFormat<layout::Nz> {
    using tag = layout::Nz;
    static constexpr CubeFormat format = CubeFormat::NZ;
};

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
struct TagToTrans<layout::RowMajorAlign> {
    static constexpr bool value = false;
};

template <>
struct TagToTrans<layout::ColumnMajor> {
    static constexpr bool value = true;
};

template <>
struct TagToTrans<layout::ColumnMajorAlign> {
    static constexpr bool value = false;
};

template <>
struct TagToTrans<layout::Zn> {
    static constexpr bool value = true;
};

template <>
struct TagToTrans<layout::Nz> {
    static constexpr bool value = false;
};

#endif