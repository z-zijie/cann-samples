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
 * \file matmul_torch.cpp
 * \brief
 */

#include <torch/all.h>

#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "kernel_operator.h"
#include "op_host/matmul_tiling_engine.h"
#include "op_kernel/block/matmul_block_mmad_aswt.h"
#include "op_kernel/block/matmul_block_scheduler_policy.h"
#include "op_kernel/kernel/matmul_kernel_aswt_impl.h"
#include "op_kernel/policy/matmul_dispatch_policy.h"
#include "op_kernel/utils/matmul_common_utils.h"
#include "op_kernel/utils/matmul_dtype_utils.h"
#include "op_kernel/utils/matmul_layout_utils.h"

namespace x {
namespace matmul {

constexpr static uint8_t ND_DIM = 2;

template <typename MatmulKernelImpl>
__global__ __aicore__ void MatmulKernel(__gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* output,
                                        const MatmulTilingData matmulTilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    using Params = typename MatmulKernelImpl::Params;
    Params params = {
        {matmulTilingData.m,matmulTilingData.n, matmulTilingData.k},
        {input, weight, output},
        {&matmulTilingData}
    };
    MatmulKernelImpl matmulKernelImpl;
    matmulKernelImpl(params);
    return;
}

template <typename T>
void MatmulApi(aclrtStream stream, const at::Tensor& input, const at::Tensor& weight, const torch::Tensor& output,
               bool transA, bool transB)
{
    MatmulTilingData matmulTilingData;
    MatmulTplValue matmulTplValue;
    MatmulTilingEngine matmulTilingEngine;
    matmulTilingEngine.GetTiling(input, weight, transA, transB, matmulTilingData, matmulTplValue);

    uint32_t blockDim = matmulTilingData.usedCoreNum;
    __gm__ uint8_t* inputPtr = (__gm__ uint8_t*)input.data_ptr<T>();
    __gm__ uint8_t* weightPtr = (__gm__ uint8_t*)weight.data_ptr<T>();
    __gm__ uint8_t* outputPtr = (__gm__ uint8_t*)output.data_ptr<T>();

    using aType = typename TagToAscendDtype<T>::Type;
    using bType = typename TagToAscendDtype<T>::Type;
    using cType = typename TagToAscendDtype<T>::Type;

    DISPATCH_TRANSPOSE_COMBINATION(transA, transB, {
        using layoutA = std::conditional_t<transA, layout::ColumnMajor, layout::RowMajor>;
        using layoutB = std::conditional_t<transB, layout::ColumnMajor, layout::RowMajor>;
        using layoutC = layout::RowMajor;
        using L1TileShape = AscendC::Shape<_0, _0, _0>;
        using L0TileShape = AscendC::Shape<_0, _0, _0>;

        using BlockScheduler = BuiltInAswtScheduler;
        using DispatchPolicy = MatmulMultiBlockWithAswt<>;
        using BlockMmad =
            Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, aType, layoutA, bType, layoutB, cType, layoutC>;
        using ProblemShape = MatmulShape;
        using MatmulKernelImpl = Kernel::MatmulKernelAswtImpl<ProblemShape, BlockMmad, BlockScheduler>;

        MatmulKernel<MatmulKernelImpl><<<blockDim, nullptr, stream>>>(inputPtr, weightPtr, outputPtr, matmulTilingData);
    });
}

torch::Tensor CreateOutputTensor(const torch::Tensor& input, const torch::Tensor& weight, bool transA, bool transB)
{
    TORCH_CHECK(input.dim() == ND_DIM, "Input tensor must be 2D");
    TORCH_CHECK(weight.dim() == ND_DIM, "Weight tensor must be 2D");

    int64_t m = transA ? input.size(1) : input.size(0);
    int64_t kA = transA ? input.size(0) : input.size(1);
    int64_t kB = transB ? weight.size(1) : weight.size(0);
    int64_t n = transB ? weight.size(0) : weight.size(1);

    TORCH_CHECK(m > 0, "Input tensor of m has to be positive, but got", m);
    TORCH_CHECK(kA > 0, "Input tensor of k has to be positive, but got", kA);
    TORCH_CHECK(kB > 0, "Weight tensor of k has to be positive, but got", kB);
    TORCH_CHECK(n > 0, "Weight tensor of n has to be positive, but got", n);
    TORCH_CHECK(kA == kB, "Reduce dim must same with input tensor and weight tensor");
    return torch::empty({m, n}, input.options());
}

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("matmul(Tensor input, Tensor weight, bool trans_a = False, bool trans_b = False) -> Tensor");
}

torch::Tensor MatmulNpu(const torch::Tensor& input, const torch::Tensor& weight, bool transA, bool transB)
{
    TORCH_CHECK(torch_npu::utils::is_npu(input), "Input tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(weight), "Weight tensor must be on NPU device");

    auto output = CreateOutputTensor(input, weight, transA, transB);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() -> int {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "MatmulNpu",
                                        [&] { MatmulApi<scalar_t>(stream, input, weight, output, transA, transB); });
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("Matmul", acl_call);
    return output;
}

// Register Ascend implementations for matmulv3
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("matmul", MatmulNpu);
}

} // namespace matmul
} // namespace x
