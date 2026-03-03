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
#include <cstdlib>
#include "kernel_operator.h"
#include "op_host/matmul_tiling_engine.h"
#include "op_kernel/block/matmul_block_mmad_aswt.h"
#include "op_kernel/block/matmul_block_scheduler_policy.h"
#include "op_kernel/kernel/matmul_kernel_aswt_impl.h"
#include "op_kernel/policy/matmul_dispatch_policy.h"
#include "op_kernel/utils/matmul_common_utils.h"
#include "op_kernel/utils/matmul_dtype_utils.h"
#include "op_kernel/utils/matmul_layout_utils.h"

// todo: 这里宏可以移到公共文件中
// 暂时没有 aclrtGetErrorString
#define ACLRT_CHECK_WITH_MSG(call, msg)                                                             \
    do {                                                                                            \
        aclError err = call;                                                                        \
        if (err != ACL_SUCCESS) {                                                                   \
            std::cerr << "*** ACLRT Error in " << __FILE__ << " at line " << __LINE__ << std::endl; \
            std::cerr << msg << std::endl;                                                          \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while (0)

#define ACLRT_KERNEL_CHECK(msg)                                                                     \
    do {                                                                                            \
        aclError err = aclrtGetLastError(0);                                                        \
        if (err != ACL_SUCCESS) {                                                                   \
            std::cerr << "*** ACLRT Error in " << __FILE__ << " at line " << __LINE__ << std::endl; \
            std::cerr << msg << std::endl;                                                          \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while (0)

#define CHECK_COND(cond, msg)                           \
    do {                                                \
        if (!(cond)) {                                  \
            std::cerr << "ERROR: " << msg << std::endl; \
            exit(EXIT_FAILURE);                         \
        }                                               \
    } while (0)

constexpr static uint32_t MX_GROUP_SIZE = 32;
constexpr static uint32_t MX_DIVISOR_SIZE = 64;

__global__ __aicore__ void MatmulKernel(__gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* output,
                                        const MatmulTilingData matmulTilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);


}

template <typename MatmulKernelImpl>
__global__ __aicore__ void MatmulKernel(
    __gm__ uint8_t* dA, __gm__ uint8_t* dB, __gm__ uint8_t* dScaleA, __gm__ uint8_t* dScaleB, __gm__ uint8_t* dC,
    const QuantMatmulTilingData quantMatmulTilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    using Params = typename MatmulKernelImpl::Params;
    Params params = {
        {matmulTilingData.m, matmulTilingData.n, matmulTilingData.k}, {input, weight, output}, {&matmulTilingData}};
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

int main(int argc, char* argv[])
{
    // get inputShape
    uint32_t m, k, n;
    try {
        parseArguments(argc, argv, m, k, n);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // init
    int32_t deviceId = 0;
    aclrtStream stream;
    uint32_t deviceCount;
    ACLRT_CHECK_WITH_MSG(aclrtGetDeviceCount(&deviceCount), "Failed to get ACLRT devices");
    CHECK_COND(deviceCount < 1U, "No ACLRT devices found");
    ACLRT_CHECK_WITH_MSG(aclInit(nullptr), "aclInit failed.");
    ACLRT_CHECK_WITH_MSG(aclrtSetDevice(deviceId), "aclrtSetDevice failed.");
    ACLRT_CHECK_WITH_MSG(aclrtCreateStream(&stream), "aclrtCreateStream failed.");

    // host data
    // std::vector<uint8_t> hA((m * k + 1) >> 1, 0);
    // std::vector<uint8_t> hB((k * n + 1) >> 1, 0);
    // std::vector<uint8_t> hScaleA((m * (k + 64) - 1 / 64), 0);
    // std::vector<uint8_t> hScaleB((n * (k + 64) - 1 / 64), 0);
    // std::vector<float> hC(m * n, 0);
    uint8_t* hA = nullptr;
    uint8_t* hB = nullptr;
    uint8_t* hScaleA = nullptr;
    uint8_t* hScaleB = nullptr;
    float* hC = nullptr;
    // matmul::FillRandomData<float>(hostInput, -2.0f, 2.0f);
    // matmul::FillRandomData<float>(hostWeight, -2.0f, 2.0f);

    // device addr
    uint8_t* dA = nullptr;
    uint8_t* dB = nullptr;
    uint8_t* dScaleA = nullptr;
    uint8_t* dScaleB = nullptr;
    float* dC = nullptr;

    // fp4 needs to be divided by 2.
    size_t sizeA = ((m * k + 1) >> 1) * sizeof(uint8_t);
    size_t sizeB = ((k * n + 1) >> 1) * sizeof(uint8_t);
    size_t sizeScaleA = ((m * (k + MX_DIVISOR_SIZE) - 1 / MX_DIVISOR_SIZE)) * sizeof(uint8_t);
    size_t sizeScaleB = ((n * (k + MX_DIVISOR_SIZE) - 1 / MX_DIVISOR_SIZE)) * sizeof(uint8_t);
    size_t sizeC = m * n * sizeof(float);

    // malloc host 锁页内存
    ACLRT_CHECK_WITH_MSG(aclrtMallocHost((void**)&hA, sizeA), "aclrtMallocHost failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMallocHost((void**)&hB, sizeB), "aclrtMallocHost failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMallocHost((void**)&hScaleA, sizeScaleA), "aclrtMallocHost failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMallocHost((void**)&hScaleB, sizeScaleB), "aclrtMallocHost failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMallocHost((void**)&hC, sizeC), "aclrtMallocHost failed.");

    // malloc device memory
    ACLRT_CHECK_WITH_MSG(aclrtMalloc((void**)&dA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMalloc((void**)&dB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMalloc((void**)&dScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMalloc((void**)&dScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMalloc((void**)&dC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed.");

    // memcpy h2d
    // ACLRT_CHECK_WITH_MSG(aclrtMemcpy(dA, sizeA, hA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy failed.");
    // ACLRT_CHECK_WITH_MSG(aclrtMemcpy(dB, sizeB, hB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy failed.");
    // ACLRT_CHECK_WITH_MSG(aclrtMemcpy(dScaleA, sizeScaleA, hScaleA, sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy failed.");
    // ACLRT_CHECK_WITH_MSG(aclrtMemcpy(dScaleB, sizeScaleB, hScaleB, sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMemcpyAsync(dA, sizeA, hA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE, stream), "aclrtMemcpyAsync failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMemcpyAsync(dB, sizeB, hB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE, stream), "aclrtMemcpyAsync failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMemcpyAsync(dScaleA, sizeScaleA, hScaleA, sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE, stream), "aclrtMemcpyAsync failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMemcpyAsync(dScaleB, sizeScaleB, hScaleB, sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE, stream), "aclrtMemcpyAsync failed.");

    // get platform info
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "Get ascendcPlatform failed.");
    uint32_t numBlocks = ascendcPlatform->GetCoreNumAic();

    // kernel launch
    QuantMatmulMxfp4Kernel<<<numBlocks, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dC, tilingData);

    ACLRT_KERNEL_CHECK("Kernel launch fail");

    // memcpy d2h
    // ACLRT_CHECK_WITH_MSG(aclrtMemcpy(hC.data(), sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMemcpyAsync(hC, sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST, stream), "aclrtMemcpyAsync failed.");

    // Sync
    ACLRT_CHECK_WITH_MSG(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream failed.");

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
    aclrtFreeHost(hA);
    aclrtFreeHost(hB);
    aclrtFreeHost(hScaleA);
    aclrtFreeHost(hScaleB);
    aclrtFreeHost(hC);
    aclrtFree(dA);
    aclrtFree(dB);
    aclrtFree(dScaleA);
    aclrtFree(dScaleB);
    aclrtFree(dC);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
