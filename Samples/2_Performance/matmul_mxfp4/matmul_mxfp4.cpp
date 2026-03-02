/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_mxfp4.cpp
 * \brief
 */

#include "kernel_operator.h"
// #include "op_host/matmul_tiling_engine.h"
#include "op_kernel/block/matmul_block_mmad_aswt.h"
#include "op_kernel/block/matmul_block_scheduler_policy.h"
#include "op_kernel/kernel/matmul_kernel_aswt_impl.h"
#include "op_kernel/policy/matmul_dispatch_policy.h"
#include "op_kernel/utils/matmul_common_utils.h"
#include "op_kernel/utils/matmul_dtype_utils.h"
#include "op_kernel/utils/matmul_layout_utils.h"

namespace ascend_ops {
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

#define CHECK_COND(cond, message, return_expr)                                                                         \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::cerr << "ERROR: " << message << std::endl;                                                            \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

// 打印使用说明
void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
}

// 解析命令行参数
void parseArguments(int argc, char* argv[], int& m, int& k, int& n)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        exit(1);
    }
    if (argc < 4) {
        throw std::invalid_argument("ERROR: Lacks Arguments");
    }
    try {
        m = std::stoi(argv[1]);
        k = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument("ERROR: m k n must be Integer");
    }

    if (m <= 0 || k <= 0 || n <= 0) {
        throw std::invalid_argument("ERROR: m k n must be positive");
    }
}

int main(int argc, char* argv[])
{
    // 获取输入Shape
    int m, k, n;
    try {
        parseArguments(argc, argv, m, k, n);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // 资源初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclInit(nullptr);
    CHECK_COND(ret == ACL_SUCCESS, "aclInit failed.", return 1);
    ret = aclrtSetDevice(deviceId);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSetDevice failed.", return 1);
    ret = aclrtCreateStream(&stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtCreateStream failed.", return 1);

    // 构造Host侧输入输出
    std::vector<float> hostInput(m * k, 0);
    std::vector<float> hostWeight(k * n, 0);
    std::vector<float> hostOutput(m * n, 0);
    std::vector<float> goldenOutput(m * n, 0);
    matmul::FillRandomData<float>(hostInput, -2.0f, 2.0f);
    matmul::FillRandomData<float>(hostWeight, -2.0f, 2.0f);

    // 申请Device侧地址
    GM_ADDR deviceInput = nullptr;
    GM_ADDR deviceWeight = nullptr;
    GM_ADDR deviceOutput = nullptr;
    auto sizeInput = hostInput.size() * sizeof(float);
    auto sizeWeight = hostWeight.size() * sizeof(float);
    auto sizeOutput = hostOutput.size() * sizeof(float);
    ret = aclrtMalloc((void**)&deviceInput, sizeInput, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceInput failed.", return 1);
    ret = aclrtMalloc((void**)&deviceWeight, sizeWeight, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceWeight failed.", return 1);
    ret = aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceOutput failed.", return 1);

    // 输入数据Host To Device
    ret = aclrtMemcpy(deviceInput, sizeInput, hostInput.data(), sizeInput, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceInput failed.", return 1);
    ret = aclrtMemcpy(deviceWeight, sizeWeight, hostWeight.data(), sizeWeight, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceWeight failed.", return 1);

    // 调用算子Kernel
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "get ascendcPlatform failed.", return 1);
    uint32_t numBlocks = ascendcPlatform->GetCoreNumAic();
    matmul::MatmulKernel<float><<<numBlocks, nullptr, stream>>>(deviceInput, deviceWeight, deviceOutput, m, k, n);

    // 同步等待算子执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed.", return 1);

    // 输出数据Device To Host
    ret = aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceOutput failed.", return 1);

    // 计算golden，对比精度
    matmul::ComputeGolden<float>(m, k, n, hostInput, hostWeight, goldenOutput);
    std::vector<uint64_t> errorIndices = matmul::Compare<float>(hostOutput, goldenOutput);
    if (errorIndices.size() == 0) {
        std::cout << "matmul run successfully!" << std::endl;
    } else {
        for (uint64_t i : errorIndices) {
            uint64_t errIdx = errorIndices[i];
            std::cout << "error index: " << errIdx << ", output: " << hostOutput[errIdx]
                      << ", golden: " << goldenOutput[errIdx] << std::endl;
        }
        std::cout << "matmul run failed!" << std::endl;
    }

    // 资源释放
    std::unique_ptr<void, aclError (*)(void*)> DeviceInputAddr(deviceInput, aclrtFree);
    std::unique_ptr<void, aclError (*)(void*)> DeviceWeightAddr(deviceWeight, aclrtFree);
    std::unique_ptr<void, aclError (*)(void*)> DeviceOutputAddr(deviceOutput, aclrtFree);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}


} // namespace matmul
} // namespace ascend_ops
