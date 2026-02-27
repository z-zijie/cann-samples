#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__all__ = ["matmul", "fused_infer_attention_score"]

import torch
from torch import Tensor


def matmul(input: Tensor, weight: Tensor, trans_a: bool = False, trans_b: bool = False) -> Tensor:
    return torch.ops.ascend_ops.matmul(input, weight, trans_a, trans_b)


def fused_infer_attention_score(query: Tensor, key: Tensor, value: Tensor, mask: Tensor,
                                numHeads: int, numKeyValueHeads: int, scaleValue: int, inputLayout: str,
                                sparseMode: int) -> Tensor:
    return torch.ops.ascend_ops.fused_infer_attention_score(query,
                                                            key,
                                                            value,
                                                            mask,
                                                            numHeads,
                                                            numKeyValueHeads,
                                                            scaleValue,
                                                            inputLayout,
                                                            sparseMode)
