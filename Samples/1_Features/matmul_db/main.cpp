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
 * \file matmul.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include "acl/acl.h"

#include "tiling/platform/platform_ascendc.h"
#include "matmul_golden.h"
#include "matmul_utils.h"
#include "include/experimental/tensor_api/tensor.h"

namespace AscendC::Te {
    constexpr LoadDataTrait LOAD_DATA_B_TRAIT{true};

    struct LoadData2BTrait {
        using TraitType = LoadDataTrait;
        static constexpr const TraitType value = LOAD_DATA_B_TRAIT;
    };
}
namespace matmul {

using namespace AscendC::Te;

// 矩阵乘Kernel的简单样例
template <typename T>
__global__ __aicore__ void MatmulKernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, uint32_t m, uint32_t k, uint32_t n)
{
    // 设定kernel 运行类型 cube only
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    constexpr static uint16_t CUBE_BLOCK_SIZE = 16;
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT /sizeof(T);
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT /sizeof(T);

    // 初始化变量
    uint64_t baseM = 256;
    uint64_t baseN = 256;
    uint64_t baseK = 128 / sizeof(T);
    uint64_t kL1 = 512 / sizeof(T);
    uint64_t mTileNum = CeilDiv(m, baseM);
    uint64_t nTileNum = CeilDiv(n, baseN);
    uint64_t tileNum = mTileNum * nTileNum;
    uint64_t kL1TileNum = CeilDiv(k, kL1);
    uint64_t tailKL1 = k - (kL1TileNum - 1) * kL1;
    uint64_t tailBaseM = CeilAlign((m - (mTileNum - 1) * baseM), CUBE_BLOCK_SIZE);
    uint64_t tailBaseN = CeilAlign((n - (nTileNum - 1) * baseN), CUBE_BLOCK_SIZE);

    uint64_t curBlockIdx = AscendC::GetBlockIdx();
    uint64_t blockNum = AscendC::GetBlockNum();

    // 初始化db变量
    uint64_t l0PingPong_ = 0;
    uint64_t l1PingPong = 0;
    uint64_t l0CPingPong = 0;
    uint64_t l1BufferAOffset[2] = {0UL};
    uint64_t l1BufferBOffset[2] = {0UL};

    // 构造gm tensor
    auto layoutA = AscendC::Te::MakeNDLayout<T>(m, k);
    auto layoutB = AscendC::Te::MakeNDLayout<T>(k, n);
    auto layoutC = AscendC::Te::MakeNDLayout<T>(m, n);

    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(aGm)), layoutA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(bGm)), layoutB);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(cGm)), layoutC);

    // 设置首次MTE1_MTE2 M_MTE1
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);

    // 多核处理不同块
    for (uint64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
        // 获取当前处理的基本块index
        uint64_t mTileIdx = tileIdx / nTileNum;
        uint64_t nTileIdx = tileIdx % nTileNum;
        uint64_t curM = mTileIdx == (mTileNum - 1) ? tailBaseM : baseM;
        uint64_t curN = nTileIdx == (nTileNum - 1) ? tailBaseN : baseN;
        uint64_t actualCurM = mTileIdx == (mTileNum - 1) ? (m - mTileIdx * baseM) : baseM;
        uint64_t actualCurN = nTileIdx == (nTileNum - 1) ? (n - nTileIdx * baseN) : baseN;

        // slice gm to l1
        uint64_t l0cOffset = (l0CPingPong & 1) * HALF_L0C_SIZE;


        auto gmBlockA_ = gmA(AscendC::Te::MakeCoord(mTileIdx * baseM, 0L), AscendC::Te::MakeShape(curM, k));
        auto gmBlockB_ = gmB(AscendC::Te::MakeCoord(0L, nTileIdx * baseN), AscendC::Te::MakeShape(k, curN));
        auto gmBlockC_ = gmC(
            AscendC::Te::MakeCoord(mTileIdx * baseM, nTileIdx * baseN), AscendC::Te::MakeShape(actualCurM, actualCurN));

        auto layoutL0C = AscendC::Te::MakeL0CLayout(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(MakeL0CmemPtr((__cc__ float*)l0cOffset), layoutL0C);

        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
            uint64_t l1BufId = l1PingPong & 1;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

            auto curGmBKL1 = (iter0 + 1 == kL1TileNum) ? (k - iter0 * kL1) : kL1;
            auto curGmAKL1 = curGmBKL1;

            // A, B GM->L1
            uint64_t l1Offset = (AscendC::TOTAL_L1_SIZE >> 1) * l1BufId;
            l1BufferAOffset[l1BufId] =  l1Offset;
            l1BufferBOffset[l1BufId] =  l1Offset + baseM * kL1 * sizeof(T);

            auto layoutAL1 = AscendC::Te::MakeNzLayout<T>(curM, curGmAKL1);
            auto layoutBL1 = AscendC::Te::MakeNzLayout<T>(curGmBKL1, curN);
            auto tensorAL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr((__cbuf__ T*)l1BufferAOffset[l1BufId]), layoutAL1);
            auto tensorBL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr((__cbuf__ T*)l1BufferBOffset[l1BufId]), layoutBL1);

            // slice l1 to l0
            auto gmBlockA = gmBlockA_(AscendC::Te::MakeCoord(0, iter0 * kL1), AscendC::Te::MakeShape(curM, curGmAKL1));
            AscendC::Te::Copy(
                AscendC::Te::CopyAtom<
                    AscendC::Te::CopyTraits<AscendC::Te::CopyGM2L1, AscendC::Te::DataCopyTraitDefault>>{},
                tensorAL1, gmBlockA);
            auto gmBlockB = gmBlockB_(AscendC::Te::MakeCoord(iter0 * kL1, 0), AscendC::Te::MakeShape(curGmBKL1, curN));
            AscendC::Te::Copy(
                AscendC::Te::CopyAtom<
                    AscendC::Te::CopyTraits<AscendC::Te::CopyGM2L1, AscendC::Te::DataCopyTraitDefault>>{},
                tensorBL1, gmBlockB);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0IterNum = CeilDiv(curGmBKL1, baseK);
            uint64_t tailKL0 = curGmBKL1 - (kL0IterNum - 1) * baseK;
            for (uint16_t iter1 = 0; iter1 < kL0IterNum; ++iter1) {
                uint64_t l0BufId = l0PingPong_ & 1;
                uint64_t l0Offset = HALF_L0_SIZE * l0BufId;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);

                uint64_t curKL0 = (iter1 + 1 == kL0IterNum) ? tailKL0 : baseK;

                // A, B L1->L0
                auto layoutAL0 = AscendC::Te::MakeNzLayout<T>(curM, curKL0);
                auto layoutBL0 = AscendC::Te::MakeZnLayout<T>(curKL0, curN);
                auto tensorAL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr((__ca__ T*)l0Offset), layoutAL0);
                auto tensorBL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr((__cb__ T*)l0Offset), layoutBL0);

                // slice l0 to mmad
                auto tensorBlockAL1 =
                    tensorAL1(AscendC::Te::MakeCoord(0, iter1 * baseK), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(
                    AscendC::Te::CopyAtom<
                        AscendC::Te::CopyTraits<AscendC::Te::CopyL12L0, AscendC::Te::LoadDataTraitDefault>>{},
                    tensorAL0, tensorBlockAL1);
                auto tensorBlockBL1 =
                    tensorBL1(AscendC::Te::MakeCoord(iter1 * baseK, 0), AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(
                    AscendC::Te::CopyAtom<AscendC::Te::CopyTraits<AscendC::Te::CopyL12L0, LoadData2BTrait>>{},
                    tensorBL0, tensorBlockBL1);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);

                // mmad
                MmadParams para;
                para.cmatrixInitVal = (iter1 == 0 && iter0 == 0) ? true : false;
                AscendC::Te::Mad(
                    MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{}, tensorL0C, tensorAL0, tensorBL0, para);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                l0PingPong_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            l1PingPong++;
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);

        // L0C->GM
        AscendC::Te::Copy(
            AscendC::Te::CopyAtom<AscendC::Te::CopyTraits<AscendC::Te::CopyL0C2GM, AscendC::Te::FixpipeTraitDefault>>{},
            gmBlockC_, tensorL0C);
        l0CPingPong++;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
}

} // namespace matmul

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