#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import sys
import numpy as np
import en_dtypes
import math
import torch
import torch_npu
import logging



def gen_golden_data_simple(m, k, n):
    M = m
    K = k
    N = n
    cpu_x1 = torch.randint(-10, 10, (M, int(K/2)), dtype=torch.int8)
    cpu_x2 = torch.randint(-10, 10, (N, int(K/2)), dtype=torch.int8)
    scale_x1 = torch.randint(-10, 10, (M, math.ceil(K/64), 2), dtype=torch.int8)
    scale_x2 = torch.randint(-10, 10, (N, math.ceil(K/64), 2), dtype=torch.int8)

    x1_npu = cpu_x1.npu()
    x2_npu = cpu_x2.npu().transpose(-1, -2)
    scale_x1_npu = scale_x1.npu()
    scale_x2_npu = scale_x2.npu().transpose(0, 1)

    #调用npu_quant_matmul函数，指定x1_dtype和x2_dtpe为torch_npu.float4_e2m1fn_x2
    npu_out = torch_npu.npu_quant_matmul(
        x1_npu,
        x2_npu,
        scale_x2_npu,
        pertoken_scale=scale_x1_npu,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        output_dtype=torch.float32,
        group_sizes=[1,1,32],
        x1_dtype=torch_npu.float4_e2m1fn_x2,
        x2_dtype=torch_npu.float4_e2m1fn_x2,
        scale_dtype=torch_npu.float8_e8m0fnu
    ).cpu()
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    cpu_x1.numpy().tofile("./input/input_a.bin")
    cpu_x2.numpy().tofile("./input/input_b.bin")
    scale_x1.numpy().tofile("./input/input_scaleA.bin")
    scale_x2.numpy().tofile("./input/input_scaleB.bin")
    npu_out.numpy().tofile("./output/golden_out.bin")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 gen_data.py m k n")
        sys.exit(1)

    # 获取参数
    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    gen_golden_data_simple(m, k, n)