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
 * \file quant_matmul_tiling_engine.h
 * \brief
 */

#ifndef QUANT_MATMUL_TILING_ENGINE_H
#define QUANT_MATMUL_TILING_ENGINE_H

#include "quant_matmul_tiling_data.h"
#include "quant_matmul_tiling_common.h"
#include "host_utils/common_utils.h"

class QuantMatmulTilingEngine {
public:
    QuantMatmulTilingEngine() {};
    virtual ~QuantMatmulTilingEngine() {};
    void GetTilingData(const uint64_t m, const uint64_t n, const uint64_t k, QuantMatmulTilingData& tilingData);
    void InitCompileInfo();
    void InitShapeArgs(const uint64_t m, const uint64_t n, const uint64_t k);
    void DoOpTiling();
    void CalcBasicBlock();
    void AdjustBasicBlock();
    void OptimizeEdgeBasicBlock();
    void IsAFullLoad();
    void CalcTailBasicBlockAfullLoad();
    void CalcTailBasicBlock();
    uint64_t CalUsedCoreNum(uint64_t mTile, uint64_t nTile);
    void CalL1Tiling();
    void CalL1TilingDepthAfullload();
    uint64_t GetDepthB1AfullLoad(uint64_t leftSize);
    uint64_t GetScaleFactorBAfullLoad(uint64_t leftSize);
    void CalL1TilingDepthNotfullload();
    uint64_t GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit);
    void CalStepKs();
    void CalScaleFactors(uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleASize, uint64_t baseScaleBSize);
    void SetTiling(QuantMatmulTilingData& tilingData);
    void CalculateNBufferNum4MX(QuantMatmulTilingData& tilingData);

private:
    QuantMatmulArgs args_;
    QuantMatmulPlatformInfo platformInfo_;
    QuantMatmulRunInfo runInfo_;
};

#endif // QUANT_MATMUL_TILING_ENGINE_H