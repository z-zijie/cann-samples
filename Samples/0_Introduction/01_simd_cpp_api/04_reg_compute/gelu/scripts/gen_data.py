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
import numpy as np


TOTAL_M = 256
TOTAL_N = 32
COEFF_LINEAR = -1.595769
COEFF_CUBIC = -0.071405


def gen_golden_data_simple():
    input_x = np.random.uniform(-10, 10, [TOTAL_M, TOTAL_N]).astype(np.float32)
    exponent = COEFF_LINEAR * input_x + COEFF_CUBIC * (input_x**3)
    golden = input_x / (1 + np.exp(exponent))

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    golden.astype(np.float32).tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
