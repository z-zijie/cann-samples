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
constexpr FixpipeTrait MX_FIXPIPE_TRAIT_F16{
    QuantMode_t::F322F16, // quantPre
    false,                // enableRelu
    false,                // enableChannelSplit
    3,                    // unitFlag
    false                 // dualDstCtl
};

constexpr FixpipeTrait MX_FIXPIPE_TRAIT_BF16{
    QuantMode_t::F322BF16, // quantPre
    false,                // enableRelu
    false,                // enableChannelSplit
    3,                    // unitFlag
    false                 // dualDstCtl
};

constexpr FixpipeTrait MX_FIXPIPE_TRAIT_F32{
    QuantMode_t::NoQuant, // quantPre
    false,                // enableRelu
    false,                // enableChannelSplit
    3,                    // unitFlag
    false                 // dualDstCtl
};

template<typename T>
struct MxFixpipeTrait {
    using TraitType = FixpipeTrait;
    static constexpr const TraitType value = MX_FIXPIPE_TRAIT_F32;
};

template<>
struct MxFixpipeTrait<bfloat16_t> {
    using TraitType = FixpipeTrait;
    static constexpr const TraitType value = MX_FIXPIPE_TRAIT_BF16;
};

template<>
struct MxFixpipeTrait<half> {
    using TraitType = FixpipeTrait;
    static constexpr const TraitType value = MX_FIXPIPE_TRAIT_F16;
};
} // namespace AscendC::Te

#endif