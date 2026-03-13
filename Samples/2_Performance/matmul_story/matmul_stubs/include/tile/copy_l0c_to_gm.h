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
 * \file copy_l0c_to_gm.h
 * \brief
 */

#ifndef MATMUL_TILE_DATAMOVE_COPY_L0C_TO_GM_H
#define MATMUL_TILE_DATAMOVE_COPY_L0C_TO_GM_H
#include "impl/experimental/tensor_api/atom/cube_datamove/copy_l0c2out.h"

namespace AscendC::Te {

constexpr FixpipeTrait MX_FIXPIPE_TRAIT{
    RoundMode::DEFAULT, // roundMode
    false,              // enableRelu
    false,              // enableChannelSplit
    3,                  // unitFlag
    false               // dualDstCtl
};

struct MxFixpipeTrait {
    using TraitType = FixpipeTrait;
    static constexpr const TraitType value = MX_FIXPIPE_TRAIT;
};

} // namespace AscendC::Te

#endif