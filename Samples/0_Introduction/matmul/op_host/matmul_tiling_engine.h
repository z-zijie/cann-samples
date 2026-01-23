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
 * \file matmul_tiling_engine.h
 * \brief
 */

#ifndef MATMUL_TILING_ENGINE_H
#define MATMUL_TILING_ENGINE_H

#include "matmul_tiling_common.h"
#include "../op_kernel/utils/matmul_tiling_data.h"
#include "../op_kernel/utils/matmul_tpl_value.h"

namespace x {
namespace matmul {

class MatmulTilingEngine {
public:
    MatmulTilingEngine() {};
    virtual ~MatmulTilingEngine() {};
    void GetTiling(const at::Tensor& input, const at::Tensor& weight, bool transA, bool transB,
                   MatmulTilingData& tilingData, MatmulTplValue& tplValue);
    void InitCompileInfo();
    void InitShapeArgs(const at::Tensor& input, const at::Tensor& weight, bool transA, bool transB);
    void InitRunInfo();
    void FormulateBasicBlock();
    void CalL1Tiling();
    void PostTiling(MatmulTilingData& tilingData, MatmulTplValue& tplValue);

private:
    MatmulArgs args_;
    MatmulCompileInfo compileInfo_;
    MatmulRunInfo runInfo_;
};

} // namespace matmul
} // namespace x

#endif // MATMUL_TILING_ENGINE_H