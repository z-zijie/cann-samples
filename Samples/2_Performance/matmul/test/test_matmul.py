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

import torch
import torch_npu
import ascend_ops

def cpu_matmul(input_tensor, weight_tensor, trans_a, trans_b):
    output_dtype = input_tensor.dtype
    middle_dtype = torch.float32
    input_tensor = input_tensor.to(dtype=middle_dtype)
    weight_tensor = weight_tensor.to(dtype=middle_dtype)
    if trans_a:
        input_tensor = input_tensor.transpose(0, 1)
    if trans_b:
        weight_tensor = weight_tensor.transpose(0, 1)
    cpu_result = torch.matmul(input_tensor, weight_tensor).to(dtype=output_dtype)
    print(f"CPU Result shape: {cpu_result.shape}")
    print(f"CPU Result: {cpu_result}")
    return cpu_result

def npu_matmul(input_tensor, weight_tensor, trans_a, trans_b):
    npu_result = ascend_ops.ops.matmul(input_tensor.npu(), weight_tensor.npu(), trans_a, trans_b)
    print(f"NPU Result shape: {npu_result.shape}")
    print(f"NPU Result: {npu_result}")
    return npu_result.cpu()

if __name__ == '__main__':
    for data_type in [torch.float16]:
        input_tensor = torch.empty(1024, 2048).uniform_(-1, 1).to(data_type)
        weight_tensor = torch.empty(4096, 2048).uniform_(-1, 1).to(data_type)
        trans_a = False
        trans_b = True

        print(f"Dtype: {data_type}")
        print(f"Self shape: {input_tensor.shape}")
        print(f"Mat2 shape: {weight_tensor.shape}")

        cpu_result = cpu_matmul(input_tensor, weight_tensor, trans_a, trans_b)
        npu_result = npu_matmul(input_tensor, weight_tensor, trans_a, trans_b)

        print(f"compare CPU Result vs NPU Result: "
              f"{torch.allclose(cpu_result, npu_result, rtol=1e-03, atol=1e-03, equal_nan=True)}\n")
