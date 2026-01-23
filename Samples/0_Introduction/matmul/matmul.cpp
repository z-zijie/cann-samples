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

#include "kernel_operator.h"
#include "op_host/matmul_tiling_engine.h"
#include "op_kernel/block/matmul_block_mmad_aswt.h"
#include "op_kernel/block/matmul_block_scheduler_policy.h"
#include "op_kernel/kernel/matmul_kernel_aswt_impl.h"
#include "op_kernel/policy/matmul_dispatch_policy.h"
#include "op_kernel/utils/matmul_common_utils.h"
#include "op_kernel/utils/matmul_layout_utils.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include "acl/acl.h"

namespace x {
namespace matmul {

constexpr static uint8_t ND_DIM = 2;

template <typename MatmulKernelImpl>
__global__ __aicore__ void MatmulKernel(__gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* output,
                                        const x::matmul::MatmulTilingData matmulTilingData)
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
void MatmulApi(aclrtStream stream, __gm__ uint8_t* inputPtr, __gm__ uint8_t* weightPtr, __gm__ uint8_t* outputPtr,
               uint32_t m, uint32_t k, uint32_t n, bool transA, bool transB)
{
    x::matmul::MatmulTilingData matmulTilingData;
    MatmulTplValue matmulTplValue;
    MatmulTilingEngine matmulTilingEngine;
    matmulTilingEngine.GetTiling(m, k, n, transA, transB, matmulTilingData, matmulTplValue);

    uint32_t blockDim = matmulTilingData.usedCoreNum;

    DISPATCH_TRANSPOSE_COMBINATION(transA, transB, {
        using layoutA = std::conditional_t<transA, layout::ColumnMajor, layout::RowMajor>;
        using layoutB = std::conditional_t<transB, layout::ColumnMajor, layout::RowMajor>;
        using layoutC = layout::RowMajor;
        using L1TileShape = AscendC::Shape<_0, _0, _0>;
        using L0TileShape = AscendC::Shape<_0, _0, _0>;

        using BlockScheduler = BuiltInAswtScheduler;
        using DispatchPolicy = MatmulMultiBlockWithAswt<>;
        using BlockMmad =
            Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, T, layoutA, T, layoutB, T, layoutC>;
        using ProblemShape = MatmulShape;
        using MatmulKernelImpl = Kernel::MatmulKernelAswtImpl<ProblemShape, BlockMmad, BlockScheduler>;

        MatmulKernel<MatmulKernelImpl><<<blockDim, nullptr, stream>>>(inputPtr, weightPtr, outputPtr, matmulTilingData);
    });
}

} // namespace matmul
} // namespace x

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// 打印使用说明
void printUsage(const std::string& programName) {
    std::cerr << "Usage: " << programName << " m k n [A_transpose] [B_transpose]" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m              row of matrix A" << std::endl;
    std::cerr << "  k              col of matrix A" << std::endl;
    std::cerr << "  n              col of matrix B" << std::endl;
    std::cerr << "  A_transpose              if matrix A is transposed(optional)" << std::endl;
    std::cerr << "  B_transpose              if matrix B is transposed(optional)" << std::endl;
    std::cerr << "Example: " << programName << " 100 50 200 true false" << std::endl;
}

// 解析命令行参数
void parseArguments(int argc, char* argv[], int& m, int& k, int& n, bool& transposeA, bool& transposeB) {
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

    if (argc >= 5) {
        if (std::string(argv[4]) == "true") {
            transposeA = true;
        } else if (std::string(argv[4]) == "false") {
            transposeA = false;
        } else {
            throw std::invalid_argument("ERROR: A_transpose must be true or false");
        }
    }

    if (argc >= 6) {
        if (std::string(argv[5]) == "true") {
            transposeB = true;
        } else if (std::string(argv[5]) == "false") {
            transposeB = false;
        } else {
            throw std::invalid_argument("ERROR: B_transpose must be true or false");
        }
    }
}

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclCreatStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream) {
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

namespace golden {
    template <typename Element>
    void FillRandomData(std::vector<Element>& data, Element min, Element max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_integral<Element>::value) {
            std::uniform_int_distribution<Element> dist(min, max);
            for (auto& ele : data) ele = dist(gen);
        } else if constexpr (std::is_floating_point<Element>::value) {
            std::uniform_real_distribution<Element> dist(min, max);
            for (auto& ele : data) ele = dist(gen);
        }
    }

    template <typename Element>
    void ComputeGolden(int m, int k, int n, std::vector<Element>& hostInput, std::vector<Element>& hostWeight,
                std::vector<Element>& goldenOutput) {
        for (uint32_t row; row < m; ++row) {
            for (uint32_t col; col < n; ++col) {
                size_t offsetGolden = row * n + col;
                Element sum = 0;
                for (uint32_t iter = 0; iter < k; ++iter) {
                    size_t offsetInput = row * k + iter;
                    size_t offsetWeight = iter * n + col;
                    sum += hostInput[offsetInput] * hostWeight[offsetWeight];
                }
                goldenOutput[offsetGolden] = sum;
            }
        }
    }
    template <typename Element>
    std::vector<uint64_t> Compare(std::vector<Element>& hostOutput, std::vector<Element>& goldenOutput) {
        std::vector<uint64_t> errorIndices;
        const float rtol = 1.0f / 256;
        for (uint64_t i = 0; i < hostOutput.size(); ++i) {
            Element actualValue = hostOutput[i];
            Element expectValue = goldenOutput[i];
            Element diff = std::fabs(actualValue - expectValue);
            if (diff > rtol * std:: max(1.0f, std::fabs(expectValue))) {
                errorIndices.push_back(i);
            }
        }
        return errorIndices;
    }
}
int main(int argc, char* argv[]) {
    int m, k, n;
    bool transposeA = false;
    bool transposeB = false;
    try {
        parseArguments(argc, argv, m, k, n, transposeA, transposeB);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    int32_t deviceId = 0;
    aclrtStream stream;
    Init(deviceId, &stream);

    // host Input
    std::vector<float> hostInput(m * k, 0);
    std::vector<float> hostWeight(k * n, 0);
    std::vector<float> hostOutput(m * n, 0);
    std::vector<float> goldenOutput(m * n, 0);
    golden::FillRandomData<float>(hostInput, -2.0f, 2.0f);
    golden::FillRandomData<float>(hostWeight, -2.0f, 2.0f);

    void* deviceInput = nullptr;
    void* deviceWeight = nullptr;
    void* deviceOutput = nullptr;
    auto sizeInput = hostInput.size() * sizeof(float);
    auto sizeWeight = hostWeight.size() * sizeof(float);
    auto sizeOutput = hostOutput.size() * sizeof(float);
    aclrtMalloc(&deviceInput, sizeInput, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&deviceWeight, sizeWeight, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(deviceInput, sizeInput, hostInput.data(), sizeInput, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(deviceWeight, sizeWeight, hostWeight.data(), sizeWeight, ACL_MEMCPY_HOST_TO_DEVICE);
    x::matmul::MatmulApi<float>(stream, (__gm__ uint8_t*)deviceInput, (__gm__ uint8_t*)deviceWeight, (__gm__ uint8_t*)deviceOutput,
                    m, k, n, transposeA, transposeB);
    aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST);

    golden::ComputeGolden<float>(m, k, n, hostInput, hostWeight, goldenOutput);

    std::vector<uint64_t> errorIndices = golden::Compare<float>(hostOutput, goldenOutput);
    std::cout << "============="  << std::endl;
    for (int i = 0; i < m * n; ++i) {
        std::cout << "golden " << i << " is: " << goldenOutput[i] << std::endl;
    }
    std::cout << "=============" << std::endl;
    for (int i = 0; i < m * n; ++i) {
        std::cout << "output " << i << " is: " << hostOutput[i] << std::endl;
    }
    std::unique_ptr<void, aclError (*)(void*)> DeviceInputAddr(deviceInput, aclrtFree);
    std::unique_ptr<void, aclError (*)(void*)> DeviceWeightAddr(deviceWeight, aclrtFree);
    std::unique_ptr<void, aclError (*)(void*)> DeviceOutputAddr(deviceOutput, aclrtFree);

    Finalize(deviceId, stream);
    return 0;
}