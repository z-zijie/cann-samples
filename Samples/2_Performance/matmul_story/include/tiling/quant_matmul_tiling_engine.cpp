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
 * \file quant_matmul_tiling_engine.cpp
 * \brief
 */
#include <ATen/Tensor.h>
#include <cstdio>

#include "quant_matmul_tiling_engine.h"

void PrintTilingData(const MatmulTilingData& tilingData)
{
    printf(">>>>>>>>>> QuantMatmulTilingData <<<<<<<<<<\n");
    printf("usedCoreNum: %u, m: %u, n: %u, k: %u, mL1: %u, nL1: %u, kL1: %u, baseM: %u, baseN: %u, baseK: %u, "
           "skSingleCoreK: %u, mTailCnt: %u, nTailCnt: %u, l1BufferNum: %u, dbL0C: %u\n",
           tilingData.usedCoreNum, tilingData.m, tilingData.n, tilingData.k, tilingData.mL1, tilingData.nL1,
           tilingData.kL1, tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.skSingleCoreK,
           tilingData.mTailCnt, tilingData.nTailCnt, tilingData.l1BufferNum, tilingData.dbL0C);
}

void QuantMatmulTilingEngine::InitCompileInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    platformInfo_.aicNum = ascendcPlatform->GetCoreNumAic();
    platformInfo_.aivNum = ascendcPlatform->GetCoreNumAiv();
    platformInfo_.socVersion = ascendcPlatform->GetSocVersion();
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::BT, platformInfo_.btSize);

    printf(">>>>>>>>>> QuantMatmulCompileInfo <<<<<<<<<<\n");
    printf("aicNum: %lu, aivNum: %lu, ubSize: %lu, l1Size: %lu, l0aSize: %lu, l0bSize: %lu, l0cSize: %lu, "
           "l2Size: %lu, btSize: %lu\n",
           platformInfo_.aicNum, platformInfo_.aivNum, platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l0aSize,
           platformInfo_.l0bSize, platformInfo_.l0cSize, platformInfo_.l2Size, platformInfo_.btSize);
}

void QuantMatmulTilingEngine::InitShapeArgs(const uint64_t m, const uint64_t n, const uint64_t k)
{
    args_.m = m;
    args_.n = n;
    args_.k = k;

    printf(">>>>>>>>>> QuantMatmulArgs <<<<<<<<<<\n");
    printf("m: %lu, k: %lu, n: %lu, dtype: %u", args_.m, args_.k, args_.n);
}

void QuantMatmulTilingEngine::DoOpTiling()
{
    CalcBasicBlock();
    OptimizeEdgeBasicBlock();
    IsAFullLoad();
    if (runInfo_.isAFullLoad) {
        CalcTailBasicBlockAfullLoad();
    } else {
        CalcTailBasicBlock();
    }
    CalL1Tiling();
}

void QuantMatmulTilingEngine::CalcBasicBlock()
{
    runInfo_.baseM = Align(std::min(args_.mSize, BASIC_BLOCK_SIZE_256), CUBE_BLOCK);
    runInfo_.baseN = Align(std::min(args_.nSize, BASIC_BLOCK_SIZE_256), CUBE_BLOCK);
    runInfo_.baseK = Align(std::min(args_.kSize, BASIC_BLOCK_SIZE_256), MXFP_DIVISOR_SIZE);

    uint64_t blockNum = CeilDiv(args_.mSize, runInfo_.baseM) * CeilDiv(args_.nSize, runInfo_.baseN);
    if (blockNum < platformInfo_.aicNum) {
        AdjustBasicBlock();
    }
    CHECK_COND(runInfo_.baseM != 0UL || runInfo_.baseN != 0UL || runInfo_.baseK != 0UL,
               "BaseM, baseN and baseK should be greater than 0");

    runInfo_.mBlockCnt = CeilDiv(args_.mSize, runInfo_.baseM);
    runInfo_.nBlockCnt = CeilDiv(args_.nSize, runInfo_.baseN);
    runInfo_.totalBlockCnt = runInfo_.mBlockCnt * runInfo_.nBlockCnt;
    runInfo_.tailBlockCnt = runInfo_.totalBlockCnt % platformInfo_.aicNum;
    runInfo_.calRoundCnt = CeilDiv(runInfo_.totalBlockCnt, platformInfo_.aicNum);
    runInfo_.mTailSize = args_.mSize - (runInfo_.mBlockCnt - 1UL) * runInfo_.baseM;
    runInfo_.nTailSize = args_.nSize - (runInfo_.nBlockCnt - 1UL) * runInfo_.baseN;
}

void QuantMatmulTilingEngine::AdjustBasicBlock()
{
    uint64_t mMaxtile = CeilDiv(args_.mSize, CUBE_BLOCK);
    uint64_t nMaxtile = CeilDiv(args_.nSize, CUBE_BLOCK);
    uint64_t tempBaseM = runInfo_.baseM;
    uint64_t tempBaseN = runInfo_.baseN;

    uint64_t mCnt = CeilDiv(args_.mSize, runInfo_.baseM);
    uint64_t nCnt = CeilDiv(args_.nSize, runInfo_.baseN);

    if (mMaxtile > nMaxtile) {
        tempBaseN = Align(CeilDiv(args_.nSize, nCnt), CUBE_BLOCK);
        nCnt = CeilDiv(args_.nSize, tempBaseN);
        mCnt = platformInfo_.aicNum / nCnt;
        tempBaseM = Align(CeilDiv(args_.mSize, mCnt), CUBE_BLOCK);
    } else {
        tempBaseM = Align(CeilDiv(args_.mSize, mCnt), CUBE_BLOCK);
        mCnt = CeilDiv(args_.mSize, tempBaseM);
        nCnt = platformInfo_.aicNum / mCnt;
        tempBaseN = Align(CeilDiv(args_.nSize, nCnt), CUBE_BLOCK);
    }

    while (tempBaseN > tempBaseM * BASEM_BASEN_RATIO && tempBaseN > CUBE_BLOCK) {
        nCnt = nCnt * NUM_TWO;
        mCnt = platformInfo_.aicNum / nCnt;
        tempBaseM = Align(CeilDiv(args_.mSize, mCnt), CUBE_BLOCK);
        tempBaseN = Align(CeilDiv(args_.nSize, nCnt), CUBE_BLOCK);
        mCnt = CeilDiv(args_.mSize, tempBaseM);
        nCnt = CeilDiv(args_.nSize, tempBaseN);
    }
    while (tempBaseM > tempBaseN * BASEM_BASEN_RATIO && tempBaseM > CUBE_BLOCK) {
        mCnt = mCnt * NUM_TWO;
        nCnt = platformInfo_.aicNum / mCnt;
        tempBaseM = Align(CeilDiv(args_.mSize, mCnt), CUBE_BLOCK);
        tempBaseN = Align(CeilDiv(args_.nSize, nCnt), CUBE_BLOCK);
        mCnt = CeilDiv(args_.mSize, tempBaseM);
        nCnt = CeilDiv(args_.nSize, tempBaseN);
    }
    uint64_t kAlignValue = Align(args_.kSize, BASIC_BLOCK_SIZE_256);
    uint64_t kMaxValue = GetShapeWithDataTypeFP4(platformInfo_.l0aSize / DB_SIZE) / std::max(tempBaseM, tempBaseN);
    kMaxValue = FloorAlign(kMaxValue, BASIC_BLOCK_SIZE_256);
    if (kMaxValue >= BASIC_BLOCK_SIZE_256) {
        runInfo_.baseM = tempBaseM;
        runInfo_.baseN = tempBaseN;
        runInfo_.baseK = std::min(kAlignValue, kMaxValue);
        runInfo_.baseK = runInfo_.baseK > BASEK_LIMIT ? 
                         Align(runInfo_.baseK / NUM_TWO, BASIC_BLOCK_SIZE_256) : runInfo_.baseK;
    }
}

void QuantMatmulTilingEngine::OptimizeEdgeBasicBlock()
{
    if (runInfo_.mBlockCnt == 1UL) {
        return;
    }
    uint64_t mTailSize = args_.mSize % runInfo_.baseM;
    bool isInnerAxisAlign = GetSizeWithDataTypeFP4(args_.kSize) % MTE2_CACHELINE_SIZE == 0UL;
    if (mTailSize > 0UL && isInnerAxisAlign) {
        uint64_t baseTailCntMax = std::min((runInfo_.baseM - mTailSize) / BASIC_BLOCK_SIZE_16, runInfo_.mBlockCnt);
        uint64_t windowSize = std::min(WINDOW_LEN, runInfo_.mBlockCnt);
        uint64_t mainWindowNum = runInfo_.mBlockCnt / windowSize - 1UL;
        uint64_t tailWindowSize = runInfo_.mBlockCnt - mainWindowNum * windowSize;
        uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseM;
        uint64_t mergeWindowNum = 1UL;

        for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
             mergeLen += windowSize, ++mergeWindowNum) {
            uint64_t newTailMain =
                Align(CeilDiv((mergeLen * runInfo_.baseM + mTailSize), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
            uint64_t curPerf = (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseM + mergeWindowNum * newTailMain;
            if (curPerf <= perfRes) {
                perfRes = curPerf;
                runInfo_.mTailMain = newTailMain;
                runInfo_.mBaseTailSplitCnt = mergeLen + 1UL;
            }
        }
    }
}

void QuantMatmulTilingEngine::IsAFullLoad()
{
    uint64_t maxBaseMSize = runInfo_.mBaseTailSplitCnt == 1 ? runInfo_.baseM : runInfo_.mTailMain;
    runInfo_.isAFullLoad =
        runInfo_.mBlockCnt < WINDOW_LEN && platformInfo_.aicNum % runInfo_.mBlockCnt == 0 &&
        GetSizeWithDataTypeFP4(maxBaseMSize * Align(args_.kSize, FP4_C0_SIZE)) <= platformInfo_.l1Size / NUM_TWO &&
        runInfo_.totalBlockCnt > platformInfo_.aicNum;

    if (runInfo_.isAFullLoad) {
        runInfo_.baseM = maxBaseMSize;
    }
}

void QuantMatmulTilingEngine::CalcTailBasicBlockAfullLoad()
{
    runInfo_.mTailTile = 1UL;
    uint64_t nTailTile = 1UL;
    if (runInfo_.tailBlockCnt != 0UL) {
        while (runInfo_.mTailTile * (nTailTile + 1UL) * runInfo_.tailBlockCnt <= platformInfo_.aicNum &&
               runInfo_.baseN / (nTailTile + 1UL) >= BASIC_BLOCK_SIZE_16) {
            nTailTile += 1UL;
        }
    }
    runInfo_.nTailTile = nTailTile;
}

void QuantMatmulTilingEngine::CalcTailBasicBlock()
{
    if (runInfo_.tailBlockCnt == 0UL) {
        return;
    }

    uint64_t mTile = 1UL;
    uint64_t nTile = 1UL;
    uint64_t preSplit = 1UL;
    uint64_t secSplit = 1UL;
    uint64_t& preSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? mTile : nTile;
    uint64_t& secSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? nTile : mTile;
    uint64_t tileMax = platformInfo_.aicNum / runInfo_.tailBlockCnt;
    uint64_t mTileMax = std::min(tileMax, CeilDiv(runInfo_.baseM, CUBE_BLOCK));
    uint64_t nTileMax = std::min(tileMax, CeilDiv(runInfo_.baseN, CUBE_BLOCK));
    uint64_t preSplitMax = runInfo_.mTailSize >= runInfo_.nTailSize ? mTileMax : nTileMax;
    uint64_t secSplitMax = runInfo_.mTailSize >= runInfo_.nTailSize ? nTileMax : mTileMax;
    while ((CalUsedCoreNum(preSplit + 1UL, secSplit) <= platformInfo_.aicNum && preSplit < preSplitMax) ||
           (CalUsedCoreNum(preSplit, secSplit + 1UL) <= platformInfo_.aicNum && secSplit < secSplitMax)) {
        if (CalUsedCoreNum(preSplit + 1UL, secSplit) <= platformInfo_.aicNum && preSplit < preSplitMax) {
            preSplitValid = ++preSplit;
        }
        if (CalUsedCoreNum(preSplit, secSplit + 1UL) <= platformInfo_.aicNum && secSplit < secSplitMax) {
            secSplitValid = ++secSplit;
        }
    }

    runInfo_.mTailTile = mTile;
    runInfo_.nTailTile = nTile;
}

uint64_t QuantMatmulTilingEngine::CalUsedCoreNum(uint64_t mTile, uint64_t nTile)
{
    return mTile * nTile * runInfo_.tailBlockCnt;
}

void QuantMatmulTilingEngine::CalL1Tiling()
{
    runInfo_.usedCoreNum = (runInfo_.totalBlockCnt > 1UL || runInfo_.tailBlockCnt == 0UL) ?
                            platformInfo_.aicNum :
                            runInfo_.tailBlockCnt * runInfo_.mTailTile * runInfo_.nTailTile;
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * DATA_SIZE_L0C * DB_SIZE <= platformInfo_.l0cSize ? DB_SIZE : 1U;
    if (runInfo_.isAFullLoad) {
        CalL1TilingDepthAfullload();
    } else {
        CalL1TilingDepthNotfullload();
    }
}

void QuantMatmulTilingEngine::CalL1TilingDepthAfullload()
{
    runInfo_.stepKa = CeilDiv(args_.kSize, runInfo_.baseK);

    uint64_t aL1Size = GetSizeWithDataTypeFP4(runInfo_.baseM * Align(args_.kSize, FP4_C0_SIZE));
    runInfo_.scaleFactorA = 1U;
    uint64_t leftL1Size = platformInfo_.l1Size - aL1Size;
    uint64_t bL0Size = GetSizeWithDataTypeFP4(runInfo_.baseN * runInfo_.baseK);
    uint64_t scaleAL1Size = runInfo_.baseM * Align(CeilDiv(args_.kSize, MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE);

    leftL1Size -= scaleAL1Size;
    runInfo_.stepKb = GetDepthB1AfullLoad(leftL1Size);
    runInfo_.scaleFactorB = GetScaleFactorBAfullLoad(leftL1Size - runInfo_.stepKb * DB_SIZE * bL0Size);
}

uint64_t QuantMatmulTilingEngine::GetDepthB1AfullLoad(uint64_t leftSize)
{
    uint64_t baseStepK = 1UL;
    uint64_t baseKSize = GetSizeWithDataTypeFP4(runInfo_.baseK);
    if (baseKSize < BASIC_BLOCK_SIZE_128) {
        baseStepK = CeilDiv(BASIC_BLOCK_SIZE_128, baseKSize);
    }

    uint64_t scaleBaseK = Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE);
    uint64_t baseSize = GetSizeWithDataTypeFP4(runInfo_.baseN * (runInfo_.baseK + scaleBaseK) * baseStepK);
    uint64_t stepKBaseScale = 1UL;
    if (leftSize >= MTE2_MIN_LOAD_SIZE * DB_SIZE) {
        stepKBaseScale = CeilDiv(MTE2_MIN_LOAD_SIZE, baseSize);
    } else {
        stepKBaseScale = CeilDiv(leftSize / DB_SIZE, baseSize);
    }
    baseStepK = baseStepK * stepKBaseScale;

    uint64_t refinedStepkb = 2UL;
    if (baseStepK == 1UL && args_.kSize > runInfo_.baseK && leftSize > baseSize * refinedStepkb) {
        baseStepK = refinedStepkb;
    }

    return baseStepK;
}

uint64_t QuantMatmulTilingEngine::GetScaleFactorBAfullLoad(uint64_t leftSize)
{
    uint64_t baseScaleBSize = runInfo_.baseN * Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE);

    uint64_t scaleFactorBBase = 1UL;
    uint64_t scaleBBasekSize = Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE);
    if (scaleBBasekSize < BASIC_BLOCK_SIZE_128) {
        scaleFactorBBase = CeilDiv(BASIC_BLOCK_SIZE_128, scaleBBasekSize);
    }

    uint64_t scaleFactorBMaxFromK = args_.kSize / (runInfo_.stepKb * runInfo_.baseK);
    scaleFactorBMaxFromK = std::min(SCALER_FACTOR_MAX, scaleFactorBMaxFromK);
    scaleFactorBMaxFromK = std::max(SCALER_FACTOR_MIN, scaleFactorBMaxFromK);

    uint64_t scaleFactorB = 1;
    uint64_t scaleFactorBMax =
        std::min(MTE2_MIN_LOAD_SIZE * DB_SIZE, leftSize) / (baseScaleBSize * runInfo_.stepKb * DB_SIZE);
    if (scaleFactorBMax != 0 && scaleFactorBBase != 0) {
        if (scaleFactorBBase <= scaleFactorBMaxFromK && scaleFactorBMax >= scaleFactorBBase) {
            scaleFactorB = std::min(scaleFactorBMax / scaleFactorBBase * scaleFactorBBase, scaleFactorBMaxFromK);
        } else {
            scaleFactorB = std::min(scaleFactorBMax, scaleFactorBMaxFromK);
        }
    }

    return scaleFactorB;
}

void QuantMatmulTilingEngine::CalL1TilingDepthNotfullload()
{
    uint64_t baseASize = GetSizeWithDataTypeFP4(runInfo_.baseM * runInfo_.baseK);
    uint64_t baseBSize = GetSizeWithDataTypeFP4(runInfo_.baseN * runInfo_.baseK);

    uint64_t baseScaleASize = Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE) * runInfo_.baseM;
    uint64_t baseScaleBSize = Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE) * runInfo_.baseN;
    uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
    uint64_t depthInit = GetDepthA1B1(platformInfo_.l1Size, baseL1Size, 1UL);
    uint64_t leftL1SizeByDepthInit = platformInfo_.l1Size - depthInit * (baseL1Size);
    uint64_t depthASec = GetDepthA1B1(leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
    uint64_t depthBSec = GetDepthA1B1(leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
    runInfo_.depthA1 = std::max(depthASec, depthBSec);
    runInfo_.depthB1 = runInfo_.depthA1;
    if (runInfo_.depthA1 * baseL1Size > platformInfo_.l1Size) {
        runInfo_.depthA1 = depthASec >= depthBSec ? depthASec : depthInit;
        runInfo_.depthB1 = depthASec < depthBSec ? depthBSec : depthInit;
    }
    CalStepKs();
    CalScaleFactors(baseASize, baseBSize, baseScaleASize, baseScaleBSize);
}

uint64_t QuantMatmulTilingEngine::GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit)
{
    if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE) {
        return depthInit;
    }
    uint64_t depthScale = leftSize / perDepthSize;
    if (depthInit > 1UL) {
        uint64_t baseKSize = GetSizeWithDataTypeFP4(runInfo_.baseK);
        while ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
               (depthScale * baseKSize) > BASIC_BLOCK_SIZE_512) {
            depthScale -= 1UL;
        }
        if ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
            (depthScale * baseKSize) >= BASIC_BLOCK_SIZE_256) {
            depthScale = BASIC_BLOCK_SIZE_256 / baseKSize;
        }
        depthScale = std::max(depthScale, 1UL);
    } else {
        constexpr uint64_t index = 2; // 2: depth的值是2的幂
        depthScale = 1UL;
        while (depthScale * (perDepthSize) < leftSize) {
            depthScale *= index;
        }
        depthScale = depthScale == 1UL ? depthScale : depthScale / index;
    }
    return depthInit * depthScale;
}

void QuantMatmulTilingEngine::CalStepKs()
{
    runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
    runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;

    if (runInfo_.stepKa * runInfo_.baseK > args_.kSize) {
        runInfo_.stepKa = CeilDiv(args_.kSize, runInfo_.baseK);
    }

    if (runInfo_.stepKb * runInfo_.baseK > args_.kSize) {
        runInfo_.stepKb = CeilDiv(args_.kSize, runInfo_.baseK);
    }

    if (runInfo_.stepKa > runInfo_.stepKb) {
        runInfo_.stepKa = runInfo_.stepKa / runInfo_.stepKb * runInfo_.stepKb;
    }
    if (runInfo_.stepKb > runInfo_.stepKa) {
        runInfo_.stepKb = runInfo_.stepKb / runInfo_.stepKa * runInfo_.stepKa;
    }

    runInfo_.stepKa = std::min(runInfo_.stepKa, static_cast<uint32_t>(4)); // 限制stepKa最大为4, 防止issue queue阻塞
    runInfo_.stepKb = std::min(runInfo_.stepKb, static_cast<uint32_t>(4)); // 限制stepKb最大为4, 防止issue queue阻塞

    runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
    runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;
}

void QuantMatmulTilingEngine::CalScaleFactors(
    uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleASize, uint64_t baseScaleBSize)
{
    // 计算scaleFactorA, scaleFactorB
    // 来自K轴的约束
    uint64_t scaleFactorAMax = std::min(MTE2_MIN_LOAD_SIZE / baseScaleASize, SCALER_FACTOR_MAX);
    uint64_t scaleFactorBMax = std::min(MTE2_MIN_LOAD_SIZE / baseScaleBSize, SCALER_FACTOR_MAX);
    uint64_t scaleFactorA = args_.kSize / (runInfo_.stepKa * runInfo_.baseK);
    uint64_t scaleFactorB = args_.kSize / (runInfo_.stepKb * runInfo_.baseK);
    runInfo_.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
    runInfo_.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
    runInfo_.scaleFactorA = std::min(scaleFactorAMax, runInfo_.scaleFactorA);
    runInfo_.scaleFactorB = std::min(scaleFactorBMax, runInfo_.scaleFactorB);

    // 来自L1 size 的约束
    uint64_t leftL1sie = aicoreParams_.l1Size - (runInfo_.depthA1 * baseASize + runInfo_.depthB1 * baseBSize);
    uint64_t scaleInit = leftL1sie / (runInfo_.depthA1 * baseScaleASize + runInfo_.depthB1 * baseScaleBSize);
    if (runInfo_.scaleFactorA <= scaleInit && runInfo_.scaleFactorB > scaleInit) {
        leftL1sie -= (runInfo_.scaleFactorA * runInfo_.depthA1 * baseScaleASize);
        runInfo_.scaleFactorB = std::min(leftL1sie / (runInfo_.depthB1 * baseScaleBSize), runInfo_.scaleFactorB);
    } else if (runInfo_.scaleFactorB <= scaleInit && runInfo_.scaleFactorA > scaleInit) {
        leftL1sie -= runInfo_.scaleFactorB * runInfo_.depthB1 * baseScaleBSize;
        runInfo_.scaleFactorA = std::min(leftL1sie / (runInfo_.depthA1 * baseScaleASize), runInfo_.scaleFactorA);
    } else if (runInfo_.scaleFactorA > scaleInit && runInfo_.scaleFactorB > scaleInit) {
        leftL1sie -= (scaleInit * runInfo_.depthB1 * baseScaleBSize + scaleInit * runInfo_.depthA1 * baseScaleASize);
        uint64_t scaleASec =
            std::min(leftL1sie / (runInfo_.depthA1 * baseScaleASize), runInfo_.scaleFactorA - scaleInit);
        uint64_t scaleBSec =
            std::min(leftL1sie / (runInfo_.depthB1 * baseScaleBSize), runInfo_.scaleFactorB - scaleInit);
        runInfo_.scaleFactorA = scaleASec >= scaleBSec ? (scaleASec + scaleInit) : scaleInit;
        runInfo_.scaleFactorB = scaleASec < scaleBSec ? (scaleBSec + scaleInit) : scaleInit;
    }
}

void QuantMatmulTilingEngine::SetTiling(QuantMatmulTilingData& tilingData)
{
    tilingData.mTailTile = runInfo_.mTailTile;
    tilingData.nTailTile = runInfo_.nTailTile;
    tilingData.mBaseTailSplitCnt = runInfo_.mBaseTailSplitCnt;
    tilingData.nBaseTailSplitCnt = runInfo_.nBaseTailSplitCnt;
    tilingData.mTailMain = runInfo_.mTailMain;
    tilingData.nTailMain = runInfo_.nTailMain;

    tilingData.m = args_.mSize;
    tilingData.n = args_.nSize;
    tilingData.k = args_.kSize;

    tilingData.baseM = runInfo_.baseM;
    tilingData.baseN = runInfo_.baseN;
    tilingData.baseK = runInfo_.baseK;
    tilingData.dbL0C = runInfo_.dbL0c;

    tilingData.scaleKL1 = std::min(
        runInfo_.scaleFactorA * runInfo_.stepKa * runInfo_.baseK,
        runInfo_.scaleFactorB * runInfo_.stepKb * runInfo_.baseK);
    CalculateNBufferNum4MX();
}

void QuantMatmulTilingEngine::CalculateNBufferNum4MX(QuantMatmulTilingData& tilingData)
{
    tilingData.stepKa = std::min(runInfo_.stepKa, runInfo_.stepKb);
    tilingData.stepKb = tilingData.stepKa;
    uint64_t kL1 = tilingData.stepKa * tilingData.baseK;
    uint64_t usedL1Size = GetSizeWithDataTypeFP4(runInfo_.baseN * kL1) * L1_FOUR_BUFFER;
    usedL1Size += runInfo_.baseN * CeilDiv(tilingData.scaleKL1, MX_GROUP_SIZE) * DB_SIZE;
    if (args_.hasBias) {
        usedL1Size += GetSizeWithDataTypeFP4(runInfo_.baseN, args_.biasDtype) * DB_SIZE;
    }
    if (isAFullLoad_) {
        uint64_t scaleK = CeilDiv(args_.kSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        uint64_t kAligned = CeilAlign(args_.kSize, MXFP_DIVISOR_SIZE);
        usedL1Size += GetSizeWithDataTypeFP4(runInfo_.baseM * kAligned) + runInfo_.baseM * scaleK;
    } else {
        usedL1Size += GetSizeWithDataTypeFP4(runInfo_.baseM * kL1) * L1_FOUR_BUFFER;
        usedL1Size += runInfo_.baseM * CeilDiv(tilingData.scaleKL1, MX_GROUP_SIZE) * DB_SIZE;
    }
    tilingData.nBufferNum = usedL1Size < aicoreParams_.l1Size ? L1_FOUR_BUFFER : DB_SIZE;
}

void QuantMatmulTilingEngine::GetTilingData(
    const uint64_t m, const uint64_t n, const uint64_t k, QuantMatmulTilingData& tilingData)
{
    InitCompileInfo();
    InitShapeArgs(m, n, k);

    DoOpTiling();

    PostTiling(tilingData);
    PrintTilingData(tilingData);
}