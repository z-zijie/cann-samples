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
 * \file quant_matmul_mxfp4.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdlib>
#include "kernel_operator.h"
#include "data_utils.h"
#include "op_kernel/block/block_mmad_mx.h"
#include "op_kernel/block/block_scheduler_policy.h"
#include "op_kernel/block/block_scheduler_mxfp4.h"
#include "op_kernel/kernel/quant_matmul_mx_kernel_aswt_impl.h"
#include "op_kernel/policy/dispatch_policy.h"
#include "op_kernel/utils/common_utils.h"
#include "op_kernel/utils/layout_utils.h"
#include "op_kernel/utils/quant_matmul_tiling_data.h"

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
        aclError err = aclrtGetLastError(aclrtLastErrLevel::ACL_RT_THREAD_LEVEL);                   \
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

// constexpr static uint32_t MX_GROUP_SIZE = 32;
constexpr static uint32_t MX_DIVISOR_SIZE = 64;

__global__ __aicore__ void QuantMatmulMxfp4Kernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dScaleA, GM_ADDR dScaleB, GM_ADDR dBias, GM_ADDR dC,
    const QuantMatmulTilingData quantMatmulTilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    using AType = fp4x2_e2m1_t;
    using BType = fp4x2_e2m1_t;
    using BiasType = float;
    using CType = float;

    using layoutA = layout::RowMajor;
    using layoutB = layout::ColumnMajor;
    using layoutC = layout::RowMajor;
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    using BlockScheduler = QuantMatmulMxAswtScheduler;
    using DispatchPolicy = QuantMatmulMxMultiBlockWithAswt<>;
    using BlockMmad = Block::BlockMmadMx<
        DispatchPolicy, L1TileShape, L0TileShape, AType, layoutA, BType, layoutB, CType, layoutC, BiasType, layoutC, void>;
    using ProblemShape = MatmulShape;
    using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernelAswtImpl<ProblemShape, BlockMmad, BlockScheduler>;

    using Params = typename QuantMatmulKernelImpl::Params;

    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    QBMMTiling qbmmParams{quantMatmulTilingData.baseM, quantMatmulTilingData.baseN, quantMatmulTilingData.baseK,
                          static_cast<uint32_t>(quantMatmulTilingData.isBias),
                          static_cast<uint32_t>(quantMatmulTilingData.dbL0C)};

    Params params = {
        {quantMatmulTilingData.m, quantMatmulTilingData.n, quantMatmulTilingData.k, 1},
        {dA, dB, dC, dBias, dScaleA, dScaleB},
        {quantMatmulTilingData.stepK * quantMatmulTilingData.baseK, quantMatmulTilingData.scaleKL1, quantMatmulTilingData.nBufferNum},
        {quantMatmulTilingData.baseM, quantMatmulTilingData.baseN, quantMatmulTilingData.mTailTile, quantMatmulTilingData.nTailTile,
         quantMatmulTilingData.mBaseTailSplitCnt, quantMatmulTilingData.nBaseTailSplitCnt, quantMatmulTilingData.mTailMain,
         quantMatmulTilingData.nTailMain},
        qbmmParams};
    QuantMatmulKernelImpl quantMatmulKernelImpl;
    quantMatmulKernelImpl(params);
}

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

    if (k % 2 != 0) {
        throw std::invalid_argument("ERROR: k must be an even number");
    }

    if ((k / 32) % 2 != 0) {
        throw std::invalid_argument("ERROR: k should satisfy that ceil(k, 32) is an even number");
    }
}

int main(int argc, char* argv[])
{
    // get inputShape
    int m, k, n;
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
    CHECK_COND(deviceCount > 0U, "No ACLRT devices found");
    ACLRT_CHECK_WITH_MSG(aclInit(nullptr), "aclInit failed.");
    ACLRT_CHECK_WITH_MSG(aclrtSetDevice(deviceId), "aclrtSetDevice failed.");
    ACLRT_CHECK_WITH_MSG(aclrtCreateStream(&stream), "aclrtCreateStream failed.");

    // host data
    uint8_t* hA = nullptr;
    uint8_t* hB = nullptr;
    uint8_t* hScaleA = nullptr;
    uint8_t* hScaleB = nullptr;
    float* hBias = nullptr;
    float* hC = nullptr;

    // device addr
    GM_ADDR dA = nullptr;
    GM_ADDR dB = nullptr;
    GM_ADDR dScaleA = nullptr;
    GM_ADDR dScaleB = nullptr;
    GM_ADDR dBias = nullptr;
    GM_ADDR dC = nullptr;

    // fp4 needs to be divided by 2.
    size_t sizeA = ((m * k + 1) >> 1) * sizeof(uint8_t);
    size_t sizeB = ((k * n + 1) >> 1) * sizeof(uint8_t);
    size_t sizeScaleA = ((m * (k + MX_DIVISOR_SIZE) - 1 / MX_DIVISOR_SIZE)) * sizeof(uint8_t);
    size_t sizeScaleB = ((n * (k + MX_DIVISOR_SIZE) - 1 / MX_DIVISOR_SIZE)) * sizeof(uint8_t);
    size_t sizeBias = n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    QuantMatmulTilingData quantMatmulTilingData;
    // QuantMatmulTilingEngine quantMatmulTilingEngine;
    // quantMatmulTilingEngine.GetTiling(m, n, k, true, false, quantMatmulTilingData);
    quantMatmulTilingData.m = m;
    quantMatmulTilingData.n = n;
    quantMatmulTilingData.k = k;
    quantMatmulTilingData.baseM = 256;
    quantMatmulTilingData.baseN = 256;
    quantMatmulTilingData.baseK = 256;
    quantMatmulTilingData.scaleKL1 = 8192;
    quantMatmulTilingData.stepK = 2;
    quantMatmulTilingData.nBufferNum = 2;
    quantMatmulTilingData.isBias = 0;
    quantMatmulTilingData.dbL0C = 1;

    quantMatmulTilingData.mTailTile = 1;
    quantMatmulTilingData.nTailTile = 1;
    quantMatmulTilingData.mBaseTailSplitCnt = 1;
    quantMatmulTilingData.nBaseTailSplitCnt = 1;
    quantMatmulTilingData.mTailMain = 0;
    quantMatmulTilingData.nTailMain = 0;

    // malloc pinned memory
    ACLRT_CHECK_WITH_MSG(aclrtMallocHost((void**)&hA, sizeA), "aclrtMallocHost failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMallocHost((void**)&hB, sizeB), "aclrtMallocHost failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMallocHost((void**)&hScaleA, sizeScaleA), "aclrtMallocHost failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMallocHost((void**)&hScaleB, sizeScaleB), "aclrtMallocHost failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMallocHost((void**)&hBias, sizeBias), "aclrtMallocHost failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMallocHost((void**)&hC, sizeC), "aclrtMallocHost failed.");

    ReadFile("./input/input_a.bin", sizeA, hA, sizeA);
    ReadFile("./input/input_b.bin", sizeB, hB, sizeB);
    ReadFile("./input/input_scaleA.bin", sizeScaleA, hScaleA, sizeScaleA);
    ReadFile("./input/input_scaleB.bin", sizeScaleB, hScaleB, sizeScaleB);

    // malloc device memory
    ACLRT_CHECK_WITH_MSG(aclrtMalloc((void**)&dA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMalloc((void**)&dB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMalloc((void**)&dScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMalloc((void**)&dScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMalloc((void**)&dBias, sizeBias, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMalloc((void**)&dC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed.");

    // memcpy h2d
    ACLRT_CHECK_WITH_MSG(aclrtMemcpyAsync(dA, sizeA, hA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE, stream), "aclrtMemcpyAsync failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMemcpyAsync(dB, sizeB, hB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE, stream), "aclrtMemcpyAsync failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMemcpyAsync(dScaleA, sizeScaleA, hScaleA, sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE, stream), "aclrtMemcpyAsync failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMemcpyAsync(dScaleB, sizeScaleB, hScaleB, sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE, stream), "aclrtMemcpyAsync failed.");
    ACLRT_CHECK_WITH_MSG(aclrtMemcpyAsync(dBias, sizeBias, hBias, sizeBias, ACL_MEMCPY_HOST_TO_DEVICE, stream), "aclrtMemcpyAsync failed.");

    // get platform info
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "Get ascendcPlatform failed.");
    uint32_t numBlocks = ascendcPlatform->GetCoreNumAic();

    // kernel launch
    QuantMatmulMxfp4Kernel<<<numBlocks, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dBias, dC, quantMatmulTilingData);

    ACLRT_KERNEL_CHECK("Kernel launch fail");

    // memcpy d2h
    ACLRT_CHECK_WITH_MSG(aclrtMemcpyAsync(hC, sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST, stream), "aclrtMemcpyAsync failed.");

    // Sync
    ACLRT_CHECK_WITH_MSG(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream failed.");

    WriteFile("./output/npu_out.bin", hC, sizeC);

    // 资源释放
    aclrtFreeHost(hA);
    aclrtFreeHost(hB);
    aclrtFreeHost(hScaleA);
    aclrtFreeHost(hScaleB);
    aclrtFreeHost(hBias);
    aclrtFreeHost(hC);
    aclrtFree(dA);
    aclrtFree(dB);
    aclrtFree(dScaleA);
    aclrtFree(dScaleB);
    aclrtFree(dBias);
    aclrtFree(dC);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
