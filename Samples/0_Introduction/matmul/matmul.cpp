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

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include "acl/acl.h"

#include "tiling/platform/platform_ascendc.h"
#include "matmul_utils.h"

namespace x {
namespace matmul {

template <typename T>
__aicore__ inline void CopyInA1(const AscendC::GlobalTensor<T>& aGlobal, const AscendC::LocalTensor<T>& al1Local,
                                uint64_t curML1, uint64_t curKL1, uint64_t k)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    uint64_t nDim = curML1;
    uint64_t dDim = curKL1;

    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = dDim;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = k;
    nd2nzParams.dstNzC0Stride = CeilAlign(nDim, AscendC::BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
}

template <typename T>
__aicore__ inline void CopyInB1(const AscendC::GlobalTensor<T>& bGlobal, const AscendC::LocalTensor<T>& bl1Local,
                                uint64_t curNL1, uint64_t curKL1, uint64_t n)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    uint64_t nDim = curKL1;
    uint64_t dDim = curNL1;

    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = dDim;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = n;
    nd2nzParams.dstNzC0Stride = CeilAlign(nDim, AscendC::BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
}

template <typename T>
__aicore__ inline void CopyInA2(const AscendC::LocalTensor<T>& l0aLocal, const AscendC::LocalTensor<T>& al1Local,
                                uint64_t curML1, uint64_t curKL1, uint64_t mL0, uint64_t kL0)
{
    AscendC::LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;

    // (M, K) use LoadData2D
    loadDataParams.mStep = CeilDiv(mL0, AscendC::BLOCK_CUBE);
    if constexpr (AscendC::IsSameType<T, half>::value || AscendC::IsSameType<T, bfloat16_t>::value) {
        loadDataParams.kStep = CeilDiv(kL0, AscendC::BLOCK_CUBE);
    } else {
        loadDataParams.kStep = CeilDiv(kL0, GetC0Size<T>());
    }
    loadDataParams.srcStride = CeilDiv(curML1, AscendC::BLOCK_CUBE);
    loadDataParams.dstStride = loadDataParams.mStep;
    loadDataParams.ifTranspose = false;
    AscendC::LoadData<T>(l0aLocal, al1Local, loadDataParams);
}

template <typename T>
__aicore__ inline void CopyInB2(const AscendC::LocalTensor<T>& l0bLocal, const AscendC::LocalTensor<T>& bl1Local,
                                uint64_t curNL1, uint64_t curKL1, uint64_t nL0, uint64_t kL0)
{
    AscendC::LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;

    // (K, N) use LoadData2D
    loadDataParams.mStep = CeilDiv(kL0, AscendC::BLOCK_CUBE);
    if constexpr (AscendC::IsSameType<T, half>::value || AscendC::IsSameType<T, bfloat16_t>::value) {
        loadDataParams.kStep = CeilDiv(nL0, AscendC::BLOCK_CUBE);
        loadDataParams.dstStride = loadDataParams.kStep;
    } else {
        loadDataParams.kStep = CeilDiv(nL0, AscendC::BLOCK_CUBE) * TWO_ALIGN;
        loadDataParams.dstStride = loadDataParams.kStep >> 1;
    }
    loadDataParams.srcStride = CeilDiv(curKL1, AscendC::BLOCK_CUBE);
    loadDataParams.ifTranspose = true;
    AscendC::LoadData<T>(l0bLocal, bl1Local, loadDataParams);
}

template <typename T>
__aicore__ inline void Mmad(const AscendC::LocalTensor<float>& c1Local, const AscendC::LocalTensor<T>& l0aLocal,
                            const AscendC::LocalTensor<T>& l0bLocal, uint64_t m, uint64_t n, uint64_t k,
                            bool isFirstLoop)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.disableGemv = true;
    mmadParams.cmatrixSource = false;
    mmadParams.cmatrixInitVal = isFirstLoop;
    mmadParams.unitFlag = 0;
    AscendC::Mmad(c1Local, l0aLocal, l0bLocal, mmadParams);
}

template <typename T>
__aicore__ inline void CopyOut(const AscendC::GlobalTensor<T>& cGlobal, const AscendC::LocalTensor<float>& c1Local,
                               uint64_t baseM, uint64_t baseN, uint64_t n)
{
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.nSize = baseN;
    intriParams.mSize = baseM;
    intriParams.dstStride = n;
    intriParams.srcStride = CeilAlign(baseM, AscendC::BLOCK_CUBE);
    // set mode according to dtype
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        intriParams.quantPre = QuantMode_t::F322BF16;
    } else if (AscendC::IsSameType<T, half>::value) {
        intriParams.quantPre = QuantMode_t::F322F16;
    } else if (AscendC::IsSameType<T, float>::value) {
        intriParams.quantPre = QuantMode_t::NoQuant;
    }
    intriParams.reluPre = 0;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    AscendC::DataCopy(cGlobal, c1Local, intriParams);
}

template <typename T>
__global__ __aicore__ void MatmulKernel(__gm__ uint8_t* aGm, __gm__ uint8_t* bGm, __gm__ uint8_t* cGm, uint32_t m,
                                        uint32_t k, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    AscendC::GlobalTensor<T> aGlobal;
    AscendC::GlobalTensor<T> bGlobal;
    AscendC::GlobalTensor<T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(aGm));
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bGm));
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cGm));
    AscendC::LocalTensor<T> l0aLocal{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<T> l0bLocal{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> l0cLocal{AscendC::TPosition::CO1, 0, L0C_SIZE};
    AscendC::LocalTensor<T> l1Local{AscendC::TPosition::A1, 0, L1_SIZE};

    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(T);

    uint64_t baseM = 256;
    uint64_t baseN = 256;
    uint64_t baseK = 128 / sizeof(T);
    uint64_t kL1 = 512 / sizeof(T);
    uint64_t mTileNum = CeilDiv(m, baseM);
    uint64_t nTileNum = CeilDiv(n, baseN);
    uint64_t tileNum = mTileNum * nTileNum;
    uint64_t tailBaseM = m - (mTileNum - 1) * baseM;
    uint64_t tailBaseN = n - (nTileNum - 1) * baseN;
    uint64_t kL1TileNum = CeilDiv(k, kL1);
    uint64_t tailKL1 = k - (kL1TileNum - 1) * kL1;

    uint64_t l1PingPong = 0;
    uint64_t l0PingPong = 0;
    uint64_t curBlockIdx = AscendC::GetBlockIdx();
    uint64_t blockNum = AscendC::GetBlockNum();

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);

    for (uint64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
        uint64_t mTileIdx = tileIdx / nTileNum;
        uint64_t nTileIdx = tileIdx % nTileNum;
        uint64_t mL1 = mTileIdx == (mTileNum - 1) ? tailBaseM : baseM;
        uint64_t nL1 = nTileIdx == (nTileNum - 1) ? tailBaseN : baseN;
        uint64_t mL0 = mL1;
        uint64_t nL0 = nL1;
        uint64_t mOffset = mTileIdx * baseM;
        uint64_t nOffset = nTileIdx * baseN;
        uint64_t offsetA = mOffset * k;
        uint64_t offsetB = nOffset;
        uint64_t offsetC = mOffset * n + nOffset;

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
            uint64_t curKL1 = (iter0 + 1 == kL1TileNum) ? tailKL1 : kL1;
            uint64_t l1BufId = l1PingPong & 0x1;
            // copy data to l1 buffer
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            uint64_t offsetAL1 = baseM * kL1 * l1BufId;
            CopyInA1<T>(aGlobal[offsetA], l1Local[offsetAL1], mL1, curKL1, k);
            offsetA += curKL1;
            uint64_t offsetBL1 = baseM * kL1 * 2 + baseN * kL1 * l1BufId;
            CopyInB1<T>(bGlobal[offsetB], l1Local[offsetBL1], nL1, curKL1, n);
            offsetB += curKL1 * n;
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0TileNum = CeilDiv(curKL1, baseK);
            uint64_t tailKL0 = curKL1 - (kL0TileNum - 1) * baseK;
            for (uint64_t iter1 = 0; iter1 < kL0TileNum; ++iter1) {
                uint64_t curKL0 = (iter1 + 1 == kL0TileNum) ? tailKL0 : baseK;
                uint64_t l0BufId = l0PingPong & 0x1;
                // copy data to l0 buffer
                uint64_t l0Offset = HALF_L0_SIZE * l0BufId;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                CopyInA2<T>(l0aLocal[l0Offset], l1Local[offsetAL1], mL1, curKL1, mL0, curKL0);
                offsetAL1 += CeilAlign(mL1, AscendC::BLOCK_CUBE) * baseK;
                CopyInB2<T>(l0bLocal[l0Offset], l1Local[offsetBL1], nL1, curKL1, nL0, curKL0);
                offsetBL1 += baseK * GetC0Size<T>();
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                // compute
                bool isFirstLoop = iter0 == 0 && iter1 == 0;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                Mmad(l0cLocal, l0aLocal[l0Offset], l0bLocal[l0Offset], mL0, nL0, curKL0, isFirstLoop);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                l0PingPong++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            l1PingPong++;
        }
        // copy data to global memory
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
        CopyOut(cGlobal[offsetC], l0cLocal, mL0, nL0, n);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
}

} // namespace matmul
} // namespace x

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
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

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

namespace golden {
template <typename Element>
void FillRandomData(std::vector<Element>& data, Element min, Element max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_integral<Element>::value) {
        std::uniform_int_distribution<Element> dist(min, max);
        for (auto& elem : data) elem = dist(gen);
    } else if constexpr (std::is_floating_point<Element>::value) {
        std::uniform_real_distribution<Element> dist(min, max);
        for (auto& elem : data) elem = dist(gen);
    }
}

template <typename Element>
void ComputeGolden(int m, int k, int n, std::vector<Element>& hostInput, std::vector<Element>& hostWeight,
                   std::vector<Element>& goldenOutput)
{
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
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
std::vector<uint64_t> Compare(std::vector<Element>& hostOutput, std::vector<Element>& goldenOutput)
{
    std::vector<uint64_t> errorIndices;
    const float rtol = 1.0f / 256;
    for (uint64_t i = 0; i < hostOutput.size(); ++i) {
        Element actualValue = hostOutput[i];
        Element expectValue = goldenOutput[i];
        Element diff = std::fabs(actualValue - expectValue);
        if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
            errorIndices.push_back(i);
        }
    }
    return errorIndices;
}
} // namespace golden
int main(int argc, char* argv[])
{
    int m, k, n;
    try {
        parseArguments(argc, argv, m, k, n);
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

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t blockDim = ascendcPlatform->GetCoreNumAic();
    x::matmul::MatmulKernel<float><<<blockDim, nullptr, stream>>>(
        (__gm__ uint8_t*)deviceInput, (__gm__ uint8_t*)deviceWeight, (__gm__ uint8_t*)deviceOutput, m, k, n);

    auto ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST);
    golden::ComputeGolden<float>(m, k, n, hostInput, hostWeight, goldenOutput);

    std::vector<uint64_t> errorIndices = golden::Compare<float>(hostOutput, goldenOutput);
    if (errorIndices.size() == 0) {
        std::cout << "run success!" << std::endl;
    } else {
        for (auto i : errorIndices) {
            errIdx = errorIndices[i];
            std::cout << "error index: " << errIdx << ", output: " << hostOutput[errIdx]
                      << ", golden: " << goldenOutput[errIdx] << std::endl;
        }
        std::cout << "run failed!" << std::endl;
    }
    std::unique_ptr<void, aclError (*)(void*)> DeviceInputAddr(deviceInput, aclrtFree);
    std::unique_ptr<void, aclError (*)(void*)> DeviceWeightAddr(deviceWeight, aclrtFree);
    std::unique_ptr<void, aclError (*)(void*)> DeviceOutputAddr(deviceOutput, aclrtFree);

    Finalize(deviceId, stream);
    return 0;
}