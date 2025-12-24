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
 * \file rms_norm_torch.cpp
 * \brief
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "kernels/rms_norm_kernels.h"

namespace x {
namespace RmsNorm {

// Register the operator's schema
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, float? eps=None) -> (Tensor, Tensor)");
}

// Meta function implementation of RMSNorm
/*
 * IMPORTANT: The function signature below MUST exactly match the operator schema declared for this operator.
 * Any change (types, order, const-qualifiers, return type, or parameter names used by dispatch) will break registration/dispatch.
 * Do not modify this signature without updating the operator schema accordingly.
 */
std::tuple<torch::Tensor, torch::Tensor> rms_norm_meta(
    const torch::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<torch::Tensor>& weight_opt,
    std::optional<double> eps)
{
    auto input_sym_sizes = input.sym_sizes();
    int64_t input_ndim = input_sym_sizes.size();
    int64_t normalized_ndim = normalized_shape.size();

    TORCH_CHECK(
        normalized_ndim >= 1,
        "Expected normalized_shape to be at least 1-dimensional, i.e., ",
        "containing at least one element, but got normalized_shape = ",
        normalized_shape);
    TORCH_CHECK(
        normalized_ndim <= input_ndim,
        "Shape mismatch: normalized_shape length (",
        normalized_ndim,
        ") cannot be greater than input tensor's dimension (",
        input_ndim, ").");
    for (size_t i = 0; i < normalized_ndim; ++i) {
        TORCH_CHECK(
            input_sym_sizes[input_ndim - normalized_ndim + i] == normalized_shape[i],
            "Shape mismatch: Expected input to have shape [..., ", normalized_shape,
            "] at the end, but got input of shape ", input_sym_sizes, ".");
    }

    if (weight_opt.has_value()) {
        const torch::Tensor& weight = weight_opt.value();
        TORCH_CHECK(
            weight.sym_sizes().size() == normalized_ndim,
            "Shape mismatch: weight should have the same number of dimensions as normalized_shape. ",
            "Expected weight to have shape ", normalized_shape,
            ", but got weight of shape ", weight.sym_sizes(), ".");
        for (size_t i = 0; i < normalized_ndim; ++i) {
            TORCH_CHECK(
                weight.sym_sizes()[i] == normalized_shape[i],
                "Shape mismatch: Expected weight to have shape ", normalized_shape,
                ", but got weight of shape ", weight.sym_sizes(), ".");
        }
    }

    torch::Tensor output = torch::empty_symint(input_sym_sizes, input.options());
    std::vector<c10::SymInt> rstd_shape;
    size_t keep_dim_count = input_ndim - normalized_ndim;
    rstd_shape.reserve(keep_dim_count);
    for (int64_t i = 0; i < keep_dim_count; ++i) {
        rstd_shape.push_back(input_sym_sizes[i]);
    }

    auto rstd_options = input.options();
    if (input.scalar_type() == torch::kHalf || input.scalar_type() == torch::kBFloat16) {
        rstd_options = rstd_options.dtype(torch::kFloat);
    } else {
        rstd_options = rstd_options.dtype(input.scalar_type());
    }
    torch::Tensor rstd = torch::empty_symint(rstd_shape, rstd_options);
    return std::make_tuple(output, rstd);
}

// Register the meta implementation
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("rms_norm", TORCH_FN(rms_norm_meta));
}

// NPU implementation of RMSNorm
/*
 * IMPORTANT: The function signature below MUST exactly match the operator schema declared for this operator.
 * Any change (types, order, const-qualifiers, return type, or parameter names used by dispatch) will break registration/dispatch.
 * Do not modify this signature without updating the operator schema accordingly.
 */
std::tuple<torch::Tensor, torch::Tensor> rms_norm_npu(
    const torch::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<torch::Tensor>& weight,
    std::optional<double> eps)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // TODO: NPU kernel launch code
    return std::make_tuple(input, input);
}

// Register the NPU implementation
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("rms_norm", rms_norm_npu);
}

}  // namespace RmsNorm
}  // namespace x
