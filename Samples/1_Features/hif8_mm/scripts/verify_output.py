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

import sys
import subprocess
import numpy as np
import en_dtypes

ERROR_TOL = 1e-3
data_type = np.float32

def verify_result(output, golden):
    # 1ulp对比方式
    output = np.fromfile(output, dtype=data_type).view(data_type)
    golden = np.fromfile(golden, dtype=data_type).view(data_type)
    diff_results = np.abs(np.subtract(output, golden))
    diff_indices = np.where(diff_results > 1)[0]

    npu_nan, golden_nan = np.isnan(output), np.isnan(golden)
    diff_nan = np.logical_and(npu_nan, golden_nan)
    both_nan_idx = np.where(diff_nan)
    diff_indices = np.setdiff1d(diff_indices, both_nan_idx)

    for index in range(len(diff_indices)):
        real_index = diff_indices[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
            abs(output_data - golden_data) / golden_data))
        if index == 100:
            break
    print("golden_data : ", golden)
    print("output : ", output)
    error_ratio = float(diff_indices.size) / golden.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, ERROR_TOL))
    return error_ratio <= ERROR_TOL


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python verify_process.py m k n")
        sys.exit(1)
    m, k, n = map(int, sys.argv[1:])
    print(f"[INFO]: Generating data ...")
    subprocess.run(["python", "./Samples/1_Features/hif8_mm/scripts/gen_data.py", str(m), str(k), str(n)])
    print(f"[INFO]: Kernel Processing ...")
    try:
        result = subprocess.run(["./build/Samples/1_Features/hif8_mm/hif8_mm_demo", str(m), str(k), str(n)], capture_output=True, text=True, check=True)
    except Exception as e:
        print(e)
        sys.exit(1)

    try:
        print(f"[INFO]: Verifying Output ...")
        res = verify_result("./output/output_y.bin", "./output/golden_y.bin")
        if not res:
            raise ValueError("[ERROR] result error")
        else:
            print("test pass")
    except Exception as e:
        print(e)
        sys.exit(1)
