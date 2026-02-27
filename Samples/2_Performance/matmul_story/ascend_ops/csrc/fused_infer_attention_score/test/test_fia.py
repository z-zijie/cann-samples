#!/usr/bin/python3
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import torch
import torch_npu
import ascend_ops
import math
import argparse
from compare import compare


def execute_task(b, n1, n2, s1, s2, d):
    torch.no_grad()
    q = torch.rand((b, n1, s1, d), dtype=torch.bfloat16)
    k = torch.rand((b, n2, s2, d), dtype=torch.bfloat16)
    v = torch.rand((b, n2, s2, d), dtype=torch.bfloat16)
    scaleValue = 1 / math.sqrt(d)
    mask = ~torch.tril(torch.ones(s1, s2)).to(torch.bool)
    enable_gqa = n1 != n2
    cpu_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=~mask, scale=scaleValue,
                                                            enable_gqa=enable_gqa)

    q = q.to('npu')
    k = k.to('npu')
    v = v.to('npu')
    mask = mask.to('npu')
    sparseMode = 1
    numHeads = n1
    numKeyValueHeads = n2
    inputLayout = 'BNSD'
    npu_out = ascend_ops.ops.fused_infer_attention_score(q,
                                                     k,
                                                     v,
                                                     mask,
                                                     numHeads,
                                                     numKeyValueHeads,
                                                     scaleValue,
                                                     inputLayout,
                                                     sparseMode)
    compare(cpu_out.cpu().to(torch.float).numpy().flatten(), npu_out.cpu().to(torch.float).numpy().flatten())
    print(f"compare CPU Result vs NPU Result: "
          f"{torch.allclose(npu_out.to('cpu'), cpu_out, rtol=1e-02, atol=1e-02, equal_nan=True)}\n")


def main():
    parser = argparse.ArgumentParser(description="FIA 测试程序")
    parser.add_argument("--b", type=int, default=1, help="batch size")
    parser.add_argument("--qn", type=int, default=1, help="q numhead")
    parser.add_argument("--kvn", type=int, default=1, help="kv numhead")
    parser.add_argument("--qs", type=int, default=128, help="q length")
    parser.add_argument("--kvs", type=int, default=128, help="kv length")
    parser.add_argument("--d", type=int, default=128, help="hidden dim")
    args = parser.parse_args()
    b, n1, n2, s1, s2, d = (args.b, args.qn, args.kvn, args.qs, args.kvs, args.d)
    execute_task(b, n1, n2, s1, s2, d)


if __name__ == '__main__':
    main()
