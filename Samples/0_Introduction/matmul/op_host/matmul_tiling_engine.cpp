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
 * \file matmul_tiling_engine.cpp
 * \brief
 */
#include <cstdio>

#include "matmul_tiling_engine.h"
#include "matmul_tiling_util.h"

namespace x {
namespace matmul {

void PrintTilingData(const MatmulTilingData& tilingData)
{
    printf(">>>>>>>>>> MatmulTilingData <<<<<<<<<<\n");
    printf("usedCoreNum: %u, m: %u, n: %u, k: %u, mL1: %u, nL1: %u, kL1: %u, baseM: %u, baseN: %u, baseK: %u, "
           "skSingleCoreK: %u, l1BufferNum: %u, l0cDB: %u\n",
           tilingData.usedCoreNum, tilingData.m, tilingData.n, tilingData.k, tilingData.mL1, tilingData.nL1,
           tilingData.kL1, tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.skSingleCoreK,
           tilingData.l1BufferNum, tilingData.l0cDB);
}

void MatmulTilingEngine::InitCompileInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    compileInfo_.aicNum = ascendcPlatform->GetCoreNumAic();
    compileInfo_.aivNum = ascendcPlatform->GetCoreNumAiv();
    compileInfo_.socVersion = ascendcPlatform->GetSocVersion();
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo_.l1Size);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo_.l0ASize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo_.l0BSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo_.l0CSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo_.l2Size);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::BT, compileInfo_.btSize);

    printf(">>>>>>>>>> MatmulCompileInfo <<<<<<<<<<\n");
    printf("aicNum: %lu, aivNum: %lu, ubSize: %lu, l1Size: %lu, l0ASize: %lu, l0BSize: %lu, l0CSize: %lu, "
           "l2Size: %lu, btSize: %lu\n",
           compileInfo_.aicNum, compileInfo_.aivNum, compileInfo_.ubSize, compileInfo_.l1Size, compileInfo_.l0ASize,
           compileInfo_.l0BSize, compileInfo_.l0CSize, compileInfo_.l2Size, compileInfo_.btSize);
}

void MatmulTilingEngine::InitShapeArgs(uint32_t m, uint32_t k, uint32_t n, bool transA, bool transB)
{
    args_.transA = transA;
    args_.transB = transB;
    args_.m = transA ? k : m;
    args_.k = transA ? m : k;
    args_.n = transB ? k : n;

    printf(">>>>>>>>>> MatmulArgs <<<<<<<<<<\n");
    printf("m: %lu, k: %lu, n: %lu, dtype: %u, dtypeSize: %u, transA: %d, transB: %d\n", args_.m, args_.k, args_.n,
           args_.dtype, args_.dtypeSize, args_.transA, args_.transB);
}

void MatmulTilingEngine::InitRunInfo()
{
    runInfo_.usedCoreNum = compileInfo_.aicNum;
    runInfo_.baseM = BASIC_BLOCK_SIZE_256;
    runInfo_.baseN = BASIC_BLOCK_SIZE_256;
    runInfo_.baseK = BASIC_BLOCK_K_128_BYTE / args_.dtypeSize;
    runInfo_.stepM = BASE_STEP;
    runInfo_.stepN = BASE_STEP;
    runInfo_.l0cDB = DB_OFF_SIZE;
    runInfo_.mL1 = runInfo_.baseM;
    runInfo_.nL1 = runInfo_.baseN;
    runInfo_.singleCoreK = args_.k;
}

void MatmulTilingEngine::FormulateBasicBlock()
{
    runInfo_.baseM = std::min(TilingUtil::CeilAlign(args_.m, BASIC_BLOCK_SIZE_16), runInfo_.baseM);
    runInfo_.baseN = std::min(TilingUtil::CeilAlign(args_.n, BASIC_BLOCK_SIZE_16), runInfo_.baseN);

    uint64_t mCore = TilingUtil::CeilDivision(args_.m, runInfo_.baseM);
    uint64_t nCore = TilingUtil::CeilDivision(args_.n, runInfo_.baseN);
    
    runInfo_.usedCoreNum = mCore * nCore;
    uint64_t kValueAlign = TilingUtil::CeilAlign(static_cast<uint64_t>(args_.k), BASIC_BLOCK_SIZE_16);
    uint64_t kValueMax = TilingUtil::FloorAlign(compileInfo_.l0ASize / DB_SIZE / args_.dtypeSize /
                                                    std::max(runInfo_.baseM, runInfo_.baseN),
                                                BASIC_BLOCK_SIZE_16);
    runInfo_.baseK = std::min(kValueAlign, kValueMax);
}

void MatmulTilingEngine::CalL1Tiling()
{
    uint64_t depthA1 = compileInfo_.l1Size / NUM_TWO / runInfo_.baseM / runInfo_.baseK / args_.dtypeSize;
    uint64_t depthB1 = compileInfo_.l1Size / NUM_TWO / runInfo_.baseN / runInfo_.baseK / args_.dtypeSize;
    uint64_t stepKa = std::max(depthA1 / DB_SIZE, 1UL);
    uint64_t stepKb = std::max(depthB1 / DB_SIZE, 1UL);
    if (runInfo_.baseM == BASIC_BLOCK_SIZE_256 && runInfo_.baseN == BASIC_BLOCK_SIZE_256 &&
        args_.m % BASIC_BLOCK_SIZE_16 == 0 && args_.n % BASIC_BLOCK_SIZE_16 == 0 &&
        args_.k % BASIC_BLOCK_SIZE_16 == 0 && runInfo_.singleCoreK <= BASIC_BLOCK_SIZE_256) {
        stepKa = std::min(stepKa, NUM_TWO);
        stepKb = std::min(stepKb, NUM_TWO);
    }
    runInfo_.stepK = std::min(std::min(stepKa, stepKb), STEPKA_THERSHOLD);
    runInfo_.kL1 = runInfo_.baseK * runInfo_.stepK;
    runInfo_.mL1 = std::min(TilingUtil::CeilAlign(args_.m, BASIC_BLOCK_SIZE_16), runInfo_.baseM * runInfo_.stepM);
    runInfo_.nL1 = std::min(TilingUtil::CeilAlign(args_.n, BASIC_BLOCK_SIZE_16), runInfo_.baseN * runInfo_.stepN);
    return;
}

void MatmulTilingEngine::PostTiling(MatmulTilingData& tilingData, MatmulTplValue& tplValue)
{
    tilingData.usedCoreNum = runInfo_.usedCoreNum;
    tilingData.m = args_.m;
    tilingData.n = args_.n;
    tilingData.k = args_.k;
    tilingData.mL1 = runInfo_.mL1;
    tilingData.nL1 = runInfo_.nL1;
    tilingData.kL1 = runInfo_.kL1;
    tilingData.baseM = runInfo_.baseM;
    tilingData.baseN = runInfo_.baseN;
    tilingData.baseK = runInfo_.baseK;
    tilingData.skSingleCoreK = runInfo_.singleCoreK;
    tilingData.l1BufferNum = runInfo_.l1BufferNum;
    tilingData.l0cDB = runInfo_.l0cDB;

    tplValue.computeMode = MATMUL_ASWT;
    tplValue.copyOutMode = MATMUL_FIXPIPE_CUBE;
}

void MatmulTilingEngine::GetTiling(uint32_t m, uint32_t k, uint32_t n, bool transA, bool transB,
                                   MatmulTilingData& tilingData, MatmulTplValue& tplValue)
{
    InitCompileInfo();
    InitShapeArgs(m, k, n, transA, transB);
    InitRunInfo();

    FormulateBasicBlock();
    CalL1Tiling();
    PostTiling(tilingData, tplValue);

    PrintTilingData(tilingData);
}

} // namespace matmul
} // namespace x