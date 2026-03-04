/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <random>

#include "kernel_operator.h"
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "data_utils.h"

using namespace AscendC;
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}
__aicore__ inline uint64_t CeilAlign(uint64_t a, uint64_t b)
{
    return CeilDiv(a, b) * b;
}

class MmadMx {
    public:
    __aicore__ inline MmadMx()
    {

    }
    __aicore__ inline ~MmadMx() {

    }
    using AType = fp4x2_e2m1_t;
    using BType = fp4x2_e2m1_t;
    using MxL0AType = fp4x2_e2m1_t;
    using MxL0BType = fp4x2_e2m1_t;
    constexpr static int64_t L1_SIZE = 512 * 1024;
    constexpr static int64_t L0A_SIZE = 64 * 1024;
    constexpr static int64_t L0B_SIZE = 64 * 1024;
    constexpr static int64_t L0C_SIZE = 256 * 1024;
    
    __aicore__ inline void Init(uint64_t m, uint64_t k, uint64_t n,
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR aScaleGM, GM_ADDR bScaleGM, GM_ADDR cGM)
    {
        m_ = m;
        k_ = k;
        n_ = n;
        baseM_ = 256;
        baseK_ = 128 / sizeof(fp4x2_e2m1_t);
        baseN_ = 256;
        kL1_ = baseK_ * 4;
        mTileNum = CeilDiv(m, baseM_);
        nTileNum = CeilDiv(n, baseN_);
        TileNum = mTileNum * nTileNum;
        kL1TileNum = CeilDiv(k, kL1_);
        tailM = m - ((mTileNum - 1) * baseM_);
        tailN = n - ((nTileNum - 1) * baseN_);
        tailKL1 = k - ((kL1TileNum - 1) * kL1_);
        C0_SIZE = 64;

        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(aGM));
        bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(bGM));
        cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGM));
        scaleAGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t*>(aScaleGM));
        scaleBGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t*>(bScaleGM));
    }
// 左矩阵GM->L1, ND2NZ
__aicore__ inline void CopyInL1A(
    const AscendC::GlobalTensor<AType>& aGlobal, const AscendC::LocalTensor<AType>& al1Local,
    uint64_t curML1, uint64_t curKL1, uint64_t k_) // curML1、curKL1为分形矩阵行列，k为大矩阵列
{
    AscendC::Nd2NzParams nd2nzParams;
    uint64_t nDim = curML1;
    uint64_t dDim = curKL1;

    nd2nzParams.ndNum = 1;                                              // Nd src矩阵的个数 [0, 4095]
    nd2nzParams.nValue = nDim;                                          // 每个Nd src矩阵的行数 [0, 16384]
    nd2nzParams.dValue = (dDim + 1) >> 1;                               // 每个Nd src矩阵的列数 [0, 65535], 最小单位b8搬运，因此列数需要除2
    nd2nzParams.srcNdMatrixStride = 1;                                  // 相邻Nd src矩阵之间起始地址的偏移 [0, 65535]
    nd2nzParams.srcDValue = (k_ + 1) >> 1;                               // 每个Nd src矩阵相邻行之间的偏移 [1, 65535], 同理除2
    // transA == false
    nd2nzParams.dstNzC0Stride = CeilAlign(nDim, AscendC::BLOCK_CUBE);   // n/16
    nd2nzParams.dstNzNStride = 1;                                       // 每个Nd src矩阵相邻行转为Nz之后在dst矩阵的行偏移，[1, 16384]，单位为C0_SIZE(32B)
    nd2nzParams.dstNzMatrixStride = 1;                                  // 相邻Nz dst矩阵之间起始地址的偏移 [0, 65535]
    AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
}

// 右矩阵GM-L1，ND2NZ
__aicore__ inline void CopyInL1B(
    const AscendC::GlobalTensor<AType>& bGlobal, const AscendC::LocalTensor<AType>& bl1Local,
    uint64_t curNL1, uint64_t curKL1, uint64_t k_)
{
    AscendC::Nd2NzParams nd2nzParams;
    uint64_t nDim = curNL1;
    uint64_t dDim = curKL1;

    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = (dDim + 1) >> 1;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = (k_ + 1) >> 1;
    // transB == true
    nd2nzParams.dstNzC0Stride = CeilAlign(nDim, AscendC::BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
}

// 左矩阵scale GM->L1, DN2NZ
__aicore__ inline void CopyInL1ScaleA(
    const GlobalTensor<fp8_e8m0_t>& aScaleGlobal, const LocalTensor<fp8_e8m0_t>& aScalel1Local,
    uint64_t curML1, uint64_t curKL1, uint64_t k_)
{
    uint64_t nDim = curML1;
    uint64_t dDim = CeilDiv(curKL1, 64);

    GlobalTensor<half> TmpGlobalB16;
    TmpGlobalB16.SetGlobalBuffer(((__gm__ half*)(aScaleGlobal.GetPhyAddr())));
    auto aScaleL1LocalImpl = aScalel1Local.template ReinterpretCast<half>();
    
    // transA == false
    AscendC::Dn2NzParams dn2nzParams;
    dn2nzParams.dnNum = 1;
    dn2nzParams.nValue = dDim;  // 行数
    dn2nzParams.dValue = nDim;  // 列数
    dn2nzParams.srcDnMatrixStride = 0;
    dn2nzParams.srcDValue = CeilDiv(k_, 64);
    dn2nzParams.dstNzC0Stride = CeilDiv(curKL1, 64);
    dn2nzParams.dstNzNStride = 1;
    dn2nzParams.dstNzMatrixStride = 0;
    AscendC::DataCopy(aScaleL1LocalImpl, TmpGlobalB16, dn2nzParams);
}

__aicore__ inline void CopyInL1ScaleB(
    const GlobalTensor<fp8_e8m0_t>& bScaleGlobal, const LocalTensor<fp8_e8m0_t>& bScalel1Local,
    uint64_t curNL1, uint64_t curKL1, uint64_t k_)
{
    uint64_t nDim = curNL1;
    uint64_t dDim = CeilDiv(curKL1, 64);

    GlobalTensor<half> TmpGlobalB16;
    TmpGlobalB16.SetGlobalBuffer(((__gm__ half*)(bScaleGlobal.GetPhyAddr())));
    auto bScaleL1LocalImpl = bScalel1Local.template ReinterpretCast<half>();

    // transB == true
    AscendC::Dn2NzParams dn2nzParams;
    dn2nzParams.dnNum = 1;
    dn2nzParams.nValue = dDim; // 行数
    dn2nzParams.dValue = nDim; // 列数
    dn2nzParams.srcDnMatrixStride = 0;
    dn2nzParams.srcDValue = CeilDiv(k_, 64);
    dn2nzParams.dstNzC0Stride = CeilDiv(curKL1, 64);
    dn2nzParams.dstNzNStride = 1;
    dn2nzParams.dstNzMatrixStride = 0;
    AscendC::DataCopy(bScaleL1LocalImpl, TmpGlobalB16, dn2nzParams);
}

// 左矩阵L1->L01, LoadData2D
__aicore__ inline void CopyInL0A(
    const AscendC::LocalTensor<MxL0AType>& al0Local, const AscendC::LocalTensor<AType>& al1Local,
    const AscendC::LocalTensor<fp8_e8m0_t>& scaleAl1Local, 
    uint64_t curML1, uint64_t curKL1, uint64_t iter, uint64_t curKL0)
{
    AscendC::LoadData2DParamsV2 loadDataParams;
    uint64_t m1 = CeilDiv(curML1, AscendC::BLOCK_CUBE);
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = CeilDiv(baseK_ * iter, C0_SIZE);
    loadDataParams.mStep = m1;
    loadDataParams.kStep = CeilDiv(curKL0, C0_SIZE);
    loadDataParams.srcStride = loadDataParams.mStep;
    loadDataParams.dstStride = loadDataParams.mStep;
    loadDataParams.ifTranspose = false;

    AscendC::LoadData2DMxParams loadDataMxParams;
    loadDataMxParams.xStartPosition = 0;
    loadDataMxParams.yStartPosition = CeilDiv(baseK_ * iter, 64);
    loadDataMxParams.xStep = m1;
    loadDataMxParams.yStep = CeilDiv(curKL0, 64);
    loadDataMxParams.srcStride = CeilDiv(curKL1, 64);
    loadDataMxParams.dstStride = CeilDiv(curKL0, 64);
    AscendC::LoadData(al0Local, al1Local, scaleAl1Local, loadDataParams, loadDataMxParams);
}

// 右矩阵L1->L01, LoadData2D
__aicore__ inline void CopyInL0B(
    const AscendC::LocalTensor<MxL0BType>& bl0Local, const AscendC::LocalTensor<BType>& bl1Local,
    const AscendC::LocalTensor<fp8_e8m0_t>& scaleBl1Local, 
    uint64_t curNL1, uint64_t curKL1, uint64_t iter, uint64_t curKL0)
{
    AscendC::LoadData2DParamsV2 loadDataParams;
    uint64_t n1 = CeilDiv(curNL1, AscendC::BLOCK_CUBE);
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = CeilDiv(baseK_ * iter, C0_SIZE);
    loadDataParams.mStep = n1;
    loadDataParams.kStep = CeilDiv(curKL0, C0_SIZE); // C0_size = 32 / 0.5 = 64
    loadDataParams.srcStride = loadDataParams.mStep;
    loadDataParams.dstStride = loadDataParams.mStep;
    loadDataParams.ifTranspose = false;

    AscendC::LoadData2DMxParams loadDataMxParams;
    loadDataMxParams.xStartPosition = 0;
    loadDataMxParams.yStartPosition = CeilDiv(baseK_ * iter, 64); // mxfp_divisor_size
    loadDataMxParams.xStep = n1;
    loadDataMxParams.yStep = CeilDiv(curKL0, 64);
    loadDataMxParams.srcStride = CeilDiv(curKL1, 64);
    loadDataMxParams.dstStride = CeilDiv(curKL0, 64);
    AscendC::LoadData(bl0Local, bl1Local, scaleBl1Local, loadDataParams, loadDataMxParams);
}
__aicore__ inline void Mmad(
    const AscendC::LocalTensor<float> &c1Local, const AscendC::LocalTensor<MxL0AType> &al0Local, const AscendC::LocalTensor<MxL0BType> &bl0Local,
    uint64_t mL0, uint64_t nL0, uint64_t kL0, bool isFirstLoop)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = mL0;
    mmadParams.n = nL0;
    mmadParams.k = kL0;
    mmadParams.disableGemv = true;
    mmadParams.cmatrixSource = false;
    mmadParams.cmatrixInitVal = isFirstLoop;
    mmadParams.unitFlag = 0;
    AscendC::Mmad(c1Local, al0Local, bl0Local, mmadParams);
}
    __aicore__ inline void CopyOut(
        const AscendC::GlobalTensor<float> &cGlobal, const AscendC::LocalTensor<float> &cL0Local_,
        uint64_t curML0, uint64_t curNL0, uint64_t n)
    {
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = curNL0;
        intriParams.mSize = curML0;
        intriParams.dstStride = n;
        intriParams.srcStride = CeilAlign(curML0, AscendC::BLOCK_CUBE);
        // // 根据输出Dtype设置不同的量化模式
        // if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        //     intriParams.quantPre = QuantMode_t::F322BF16;
        // } else if (AscendC::IsSameType<T, half>::value) {
        //     intriParams.quantPre = QuantMode_t::F322F16;
        // } else if (AscendC::IsSameType<T, float>::value) {
        //     intriParams.quantPre = QuantMode_t::NoQuant;
        // }
        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.reluPre = 0;
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = 0;
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(cGlobal, cL0Local_, intriParams);
    }
__aicore__ inline void KernelRun()
{

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);

    uint64_t curBlockIdx = AscendC::GetBlockIdx();  // 获取当前核的index
    uint64_t blockNum = AscendC::GetBlockNum();     // 获取任务配置的核数

    for (uint64_t tileIdx = curBlockIdx; tileIdx < TileNum; tileIdx += blockNum) {
        uint64_t mTileIdx = tileIdx / nTileNum;
        uint64_t nTileIdx = tileIdx % nTileNum;
        uint64_t curML1 = mTileIdx == (mTileNum - 1) ? tailM : baseM_;
        uint64_t curNL1 = nTileIdx == (nTileNum - 1) ? tailN : baseN_;
        uint64_t curML0 = curML1;
        uint64_t curNL0 = curNL1;
        uint64_t mOffset = mTileIdx * baseM_;
        uint64_t nOffset = nTileIdx * baseN_;
        uint64_t offsetA = mOffset * k_; // Offset in GM
        uint64_t offsetB = nOffset * k_;
        uint64_t offsetC = mOffset * n_ + nOffset;
        uint64_t offsetScaleA = mOffset * CeilDiv(k_, 32);
        uint64_t offsetScaleB = nOffset * CeilDiv(k_, 32);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) { // iter KL1 -> k_
            uint64_t curKL1 = iter0 == (kL1TileNum - 1) ? tailKL1 : kL1_;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            uint64_t offsetAL1 = 0;   // Offset in L1
            uint64_t offsetBL1 = baseM_ * kL1_;
            uint64_t offsetScaleAL1 = (baseM_ + baseN_) * kL1_ / 2;
            uint64_t offsetScaleBL1 = (baseM_ + baseN_) * kL1_ / 2 + baseM_ * CeilDiv(kL1_, 32);
            CopyInL1A(aGlobal[offsetA], aL1Local_[offsetAL1], curML1, curKL1, k_);
            CopyInL1B(bGlobal[offsetB], bL1Local_[offsetBL1], curNL1, curKL1, k_);
            CopyInL1ScaleA(scaleAGlobal[offsetScaleA], scaleAL1Local_[offsetScaleAL1], curML1, curKL1, k_);
            CopyInL1ScaleB(scaleBGlobal[offsetScaleB], scaleBL1Local_[offsetScaleBL1], curNL1, curKL1, k_);
            offsetA += curKL1;
            offsetScaleA += CeilDiv(curKL1, 32);
            offsetB += curKL1;
            offsetScaleB += CeilDiv(curKL1, 32);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);

            uint64_t kL0TileNum = CeilDiv(curKL1, baseK_);
            uint64_t tailKL0 = curKL1 - (kL0TileNum - 1) * baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0TileNum; ++iter1) {
                uint64_t curKL0 = iter1 == (kL0TileNum - 1) ? tailKL0 : baseK_;

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
                uint64_t l0Offset = 0;
                CopyInL0A(aL0Local_[l0Offset], aL1Local_[offsetAL1], scaleAL1Local_[offsetScaleAL1], curML1, curKL1, iter1, curKL0);
                CopyInL0B(bL0Local_[l0Offset], bL1Local_[offsetBL1], scaleBL1Local_[offsetScaleBL1], curNL1, curKL1, iter1, curKL0);
                // offsetAL1 += 0;
                // offsetBL1 += 0;
                // offsetScaleAL1 += 0;
                // offsetScaleBL1 += 0;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);

                // Mmad
                bool isFirstLoop = iter0 == 0 && iter1 == 0;
                Mmad(cL0Local_, aL0Local_[l0Offset], bL0Local_[l0Offset], curML0, curNL0, curKL0, isFirstLoop);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);

        CopyOut(cGlobal[offsetC], cL0Local_, curML0, curNL0, n_);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
}

    AscendC::GlobalTensor<AType> aGlobal;
    AscendC::GlobalTensor<BType> bGlobal;
    AscendC::GlobalTensor<float> cGlobal;
    AscendC::GlobalTensor<fp8_e8m0_t> scaleAGlobal;
    AscendC::GlobalTensor<fp8_e8m0_t> scaleBGlobal;
    AscendC::LocalTensor<AType> aL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleAL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleBL1Local_{AscendC::TPosition::A1, 0, L1_SIZE};
    AscendC::LocalTensor<MxL0AType> aL0Local_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<MxL0BType> bL0Local_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> cL0Local_{AscendC::TPosition::CO1, 0, L0C_SIZE};
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;

    // Mmad计算大小 baseM_ * baseN_
    uint64_t baseM_{256};
    uint64_t baseN_{256};
    uint64_t baseK_{256}; //128 / sizeof(fp4x2_e2m1_t)
    uint64_t kL1_{1024}; // baseK_ * 4
    uint64_t mTileNum;
    uint64_t nTileNum;
    uint64_t TileNum;
    uint64_t kL1TileNum;
    uint64_t tailM;
    uint64_t tailN;
    uint64_t tailKL1;
    uint64_t C0_SIZE;

    constexpr static uint64_t BLOCK_CUBE = 16UL;
    constexpr static uint64_t MXFP_DIVISOR_SIZE = 64UL;
};


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

__global__ __aicore__ void Kernel(uint64_t m, uint64_t k, uint64_t n,
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR aScaleGM, GM_ADDR bScaleGM, GM_ADDR cGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    MmadMx mmadMx;
    mmadMx.Init(m, k, n, aGM, bGM, aScaleGM, bScaleGM, cGM);
    mmadMx.KernelRun();
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
    std::vector<uint8_t> hostA((m * k + 1) >> 1, 0);
    std::vector<uint8_t> hostB((k * n + 1) >> 1, 0);
    std::vector<uint8_t> hostScaleA(m * ((k + 1) / 64) * 2, 0);
    std::vector<uint8_t> hostScaleB(n * ((k + 1) / 64) * 2, 0);
    std::vector<float> hostOutput(m * n, 0);
    auto sizeA = static_cast<size_t>(1) * hostA.size() * sizeof(uint8_t);
    auto sizeB = static_cast<size_t>(1) * hostB.size() * sizeof(uint8_t);
    auto sizeScaleA = static_cast<size_t>(1) * hostScaleA.size() * sizeof(uint8_t);
    auto sizeScaleB = static_cast<size_t>(1) * hostScaleB.size() * sizeof(uint8_t);
    auto sizeOutput = static_cast<size_t>(1) * hostOutput.size() * sizeof(float);
    ReadFile("./input/input_a.bin", sizeA, hostA.data(), sizeA);
    ReadFile("./input/input_b.bin", sizeB, hostB.data(), sizeB);
    ReadFile("./input/input_scaleA.bin", sizeScaleA, hostScaleA.data(), sizeScaleA);
    ReadFile("./input/input_scaleB.bin", sizeScaleB, hostScaleB.data(), sizeScaleB);
    // ((m * (k + 64) - 1 / 64)) 

    // 申请Device侧地址
    GM_ADDR deviceA = nullptr;
    GM_ADDR deviceB = nullptr;
    GM_ADDR deviceScaleA = nullptr;
    GM_ADDR deviceScaleB = nullptr;
    GM_ADDR deviceOutput = nullptr;
    ret = aclrtMalloc((void**)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
    std::unique_ptr<void, aclError (*)(void*)> DeviceAAddr(deviceA, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceA failed.", return 1);
    ret = aclrtMalloc((void**)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST);
    std::unique_ptr<void, aclError (*)(void*)> DeviceBAddr(deviceB, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceB failed.", return 1);
    ret = aclrtMalloc((void**)&deviceScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_FIRST);
    std::unique_ptr<void, aclError (*)(void*)> DeviceScaleAAddr(deviceScaleA, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceScaleA failed.", return 1);
    ret = aclrtMalloc((void**)&deviceScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_FIRST);
    std::unique_ptr<void, aclError (*)(void*)> DeviceScaleBAddr(deviceScaleB, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceScaleB failed.", return 1);
    ret = aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST);
    std::unique_ptr<void, aclError (*)(void*)> DeviceOutputAddr(deviceOutput, aclrtFree);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceOutput failed.", return 1);

    // 输入数据Host To Device
    ret = aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceA failed.", return 1);
    ret = aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceB failed.", return 1);
    ret = aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceScaleA failed.", return 1);
    ret = aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceScaleB failed.", return 1);

    // 调用算子Kernel
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "get ascendcPlatform failed.", return 1);
    uint32_t blockDim = ascendcPlatform->GetCoreNumAic();
    // matmul::MatmulKernel<<<blockDim, nullptr, stream>>>(deviceA, deviceB, deviceOutput, m, k, n);
    Kernel<<<blockDim, nullptr, stream>>>(m, k, n, deviceA, deviceB, deviceScaleA, deviceScaleB, deviceOutput);
    
    // 同步等待算子执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed.", return 1);

    // 输出数据Device To Host
    ret = aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceOutput failed.", return 1);

    WriteFile("./output/output_y.bin", hostOutput.data(), sizeOutput);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}