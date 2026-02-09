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


def gen_golden_data_simple(m, k, n):
    input_shape_x = [m, k]
    input_shape_weight = [n, k]
    dtype = en_dtypes.hifloat8
    y_dtype = np.float32
    input_x = np.random.uniform(-2, 2, input_shape_x).astype(dtype)
    input_weight = np.random.uniform(-2, 2, input_shape_weight).astype(dtype)
    transposed_weight = input_weight.T
    y_golden = np.matmul(input_x, transposed_weight).astype(y_dtype)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_weight.tofile("./input/input_weight.bin")
    y_golden.tofile("./output/golden_y.bin")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python verify_process.py m k n")
        sys.exit(1)
    m, k, n = map(int, sys.argv[1:])
    gen_golden_data_simple(m, k, n)