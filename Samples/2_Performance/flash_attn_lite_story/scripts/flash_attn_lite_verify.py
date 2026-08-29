#!/usr/bin/env python3
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

"""flash_attn_lite Golden 计算与比对.

默认使用 torch CPU 计算, torch 不可用时回退到 numpy. BF16 读取优先使用
ml_dtypes, 否则使用位运算转换为 FP32.

Q/K/V/O 文件的逻辑 shape 均为 (B,N,S,D)。Golden 计算时将 B、N 展平为
一条独立序列轴，内部 shape 为 (B*N,S,D)，不在不同 Batch 或 Head 之间交换数据。
S 可为任意正整数；输入输出文件都按逻辑长度紧凑保存，不包含尾块填充行。

处理流程:
  1. 读取 q.bin, k.bin, v.bin 和 npuout_o.bin, 并将 BF16 转为 FP32.
  2. 以 FP32 计算 softmax(scale * Q @ Kᵀ) @ V, 并以 FP32 写入 golden_o.bin.
  3. 将 NPU BF16 输出转为 FP32, 直接与 FP32 Golden 逐元素比对.
  4. 可选计算 Torch BF16 低精度基线，检查 NPU 最大绝对误差不超过
     低精度基线最大绝对误差的 2 倍.

环境变量:
  FA_VERIFY_BACKEND = auto|torch|numpy
  FA_VERIFY_THREADS = N
  FA_VERIFY_QUERY_BLOCK = N
  FA_VERIFY_FORCE_NUMPY = 1
  FA_VERIFY_LOW_PRECISION_BASELINE = 1
  FA_CAUSAL_MASK = 1

用法:
    python3 flash_attn_lite_verify.py <data_dir> <B> <N> <S> <D>
退出码:
    0 = 比对通过; 1 = 比对失败或执行出错.
"""

import os
import sys
import time

from thread_limit import configure_python_threads

_VERIFY_THREADS = configure_python_threads()


def _env_enabled(name: str) -> bool:
    """读取 1/true/yes/on 形式的布尔环境变量。"""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


_CAUSAL_MASK = _env_enabled("FA_CAUSAL_MASK")
_USE_LOW_PRECISION_BASELINE = _env_enabled("FA_VERIFY_LOW_PRECISION_BASELINE")

import numpy as np

# torch_npu autoload 失败可能抛出 RuntimeError, 因此捕获 Exception.
_HAS_TORCH = False
try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except Exception as _e:  # pragma: no cover - 环境相关
    _TORCH_IMPORT_ERR = repr(_e)
else:
    _TORCH_IMPORT_ERR = None


def _resolve_backend() -> str:
    """根据环境变量和 torch 可用性选择后端."""
    forced = os.environ.get("FA_VERIFY_FORCE_NUMPY", "").strip()
    backend_env = os.environ.get("FA_VERIFY_BACKEND", "auto").strip().lower()

    # 优先处理向后兼容的旧别名.
    if forced in ("1", "true", "yes", "on"):
        return "numpy"
    if backend_env == "numpy":
        return "numpy"
    if backend_env == "torch":
        if _HAS_TORCH:
            return "torch"
        # 强制 torch 但导入失败时, 警告并回退到 numpy.
        print(
            f"verify: 警告 FA_VERIFY_BACKEND=torch 但 torch 不可导入"
            f"（{_TORCH_IMPORT_ERR}），回退 numpy",
            file=sys.stderr,
        )
        return "numpy"
    return "torch" if _HAS_TORCH else "numpy"


# 模块加载时确定 Golden 后端.
_BACKEND = _resolve_backend()

if _HAS_TORCH and (_BACKEND == "torch" or _USE_LOW_PRECISION_BASELINE):
    torch.set_num_threads(_VERIFY_THREADS)
    torch.set_num_interop_threads(1)


# BF16 读取优先使用 ml_dtypes, 未安装时用位运算转为 FP32.
try:
    import ml_dtypes

    _HAS_ML_DTYPES = True
except Exception:  # pragma: no cover
    _HAS_ML_DTYPES = False


def bf16_to_fp32(arr: np.ndarray) -> np.ndarray:
    """将 BF16 数组转为 FP32 数组."""
    if _HAS_ML_DTYPES and arr.dtype == ml_dtypes.bfloat16:
        return arr.astype(np.float32)
    # 位运算路径将 BF16 放入 FP32 的高 16 位.
    u16 = arr.view(np.uint16)
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


# BF16 混合容差要求全部元素通过.
COMPARE_RTOL = 0.004
COMPARE_ATOL = 0.004
# FlashAttention 风格：Kernel 的最大绝对误差不超过普通 BF16 基线的 2 倍.
LOW_PRECISION_BASELINE_MULTIPLIER = 2.0
# 报告显示前 4 x 4 个元素.
PRINT_ROWS = 4
PRINT_COLS = 4


def read_bf16(path: str, shape, msg_ctx: str = "") -> np.ndarray:
    """读取行主序 raw BF16 文件, 并返回指定 shape 的 FP32 ndarray."""
    with open(path, "rb") as f:
        raw = f.read()
    expect = int(np.prod(shape)) * 2  # bf16 = 2 bytes
    if len(raw) != expect:
        raise ValueError(
            f"读取{msg_ctx} {path} 字节数不符：得到 {len(raw)}，"
            f"期望 {expect}（shape={shape}）"
        )
    if _HAS_ML_DTYPES:
        bf16 = np.frombuffer(raw, dtype=ml_dtypes.bfloat16).reshape(shape)
    else:
        bf16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    return bf16_to_fp32(bf16)


def _query_block_size(seq_len: int) -> int:
    """解析 Query 分块大小；未设置时使用完整序列。"""
    query_block_env = os.environ.get("FA_VERIFY_QUERY_BLOCK", "").strip()
    if not query_block_env:
        return seq_len
    try:
        query_block = int(query_block_env)
    except ValueError as e:
        raise ValueError("FA_VERIFY_QUERY_BLOCK 必须是正整数") from e
    if query_block <= 0:
        raise ValueError("FA_VERIFY_QUERY_BLOCK 必须是正整数")
    return query_block


def compute_golden_torch(qf: np.ndarray, kf: np.ndarray, vf: np.ndarray,
                         scale: float) -> np.ndarray:
    """按展平 shape=(B*N,S,D) 使用 torch FP32 计算 Attention Golden."""
    # from_numpy 与 tensor 共享内存, 后续仅读取输入.
    q = torch.from_numpy(qf)
    k = torch.from_numpy(kf)
    v = torch.from_numpy(vf)

    query_block = _query_block_size(q.shape[1])

    # 仅分块 Q 行，每块仍使用完整 K/V，避免一次性分配 (B*N)*S*S 个 scores。
    o = torch.empty_like(q)
    for row_begin in range(0, q.shape[1], query_block):
        row_end = min(row_begin + query_block, q.shape[1])
        key_end = row_end if _CAUSAL_MASK else q.shape[1]
        scores = (q[:, row_begin:row_end] @ k[:, :key_end].transpose(-2, -1)).mul_(scale)
        if _CAUSAL_MASK:
            query_index = torch.arange(row_begin, row_end).unsqueeze(1)
            key_index = torch.arange(key_end).unsqueeze(0)
            scores.masked_fill_(key_index > query_index, float("-inf"))
        scores.sub_(scores.amax(dim=-1, keepdim=True))
        scores.exp_()
        scores.div_(scores.sum(dim=-1, keepdim=True))
        o[:, row_begin:row_end] = scores @ v[:, :key_end]
    return o.contiguous().numpy()


def compute_low_precision_baseline_torch(qf: np.ndarray, kf: np.ndarray,
                                         vf: np.ndarray,
                                         scale: float) -> np.ndarray:
    """按展平 shape=(B*N,S,D) 使用 Torch BF16 计算低精度 Attention 基线.

    参考 FlashAttention 的低精度基线：保留 BF16 输入和中间结果，
    并在矩阵乘前对 K 缩放，使求值顺序与 FP32 Golden 略有不同.
    """
    if not _HAS_TORCH:  # 调用者应先校验，这里保留防御性检查.
        raise RuntimeError("Torch 不可用，无法计算 BF16 低精度基线")

    q = torch.from_numpy(qf).to(torch.bfloat16)
    k = torch.from_numpy(kf).to(torch.bfloat16)
    v = torch.from_numpy(vf).to(torch.bfloat16)
    k_scaled = k.mul(scale)
    query_block = _query_block_size(q.shape[1])

    o = torch.empty_like(q)
    for row_begin in range(0, q.shape[1], query_block):
        row_end = min(row_begin + query_block, q.shape[1])
        key_end = row_end if _CAUSAL_MASK else q.shape[1]
        scores = q[:, row_begin:row_end] @ k_scaled[:, :key_end].transpose(-2, -1)
        if _CAUSAL_MASK:
            query_index = torch.arange(row_begin, row_end).unsqueeze(1)
            key_index = torch.arange(key_end).unsqueeze(0)
            scores.masked_fill_(key_index > query_index, float("-inf"))
        probability = torch.softmax(scores, dim=-1)
        o[:, row_begin:row_end] = probability @ v[:, :key_end]
    return o.float().contiguous().numpy()


def compute_golden_numpy(qf: np.ndarray, kf: np.ndarray, vf: np.ndarray,
                         scale: float) -> np.ndarray:
    """按展平 shape=(B*N,S,D) 使用 numpy FP32 计算 Attention Golden."""
    query_block = _query_block_size(qf.shape[1])
    o = np.empty_like(qf)
    for row_begin in range(0, qf.shape[1], query_block):
        row_end = min(row_begin + query_block, qf.shape[1])
        key_end = row_end if _CAUSAL_MASK else qf.shape[1]
        scores = scale * (qf[:, row_begin:row_end] @ kf[:, :key_end].transpose(0, 2, 1))
        if _CAUSAL_MASK:
            query_index = np.arange(row_begin, row_end)[:, None]
            key_index = np.arange(key_end)[None, :]
            scores[:, key_index > query_index] = -np.inf
        scores -= scores.max(axis=2, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=2, keepdims=True)
        o[:, row_begin:row_end] = scores @ vf[:, :key_end]
    return np.ascontiguousarray(o)


def compute_golden(qf: np.ndarray, kf: np.ndarray, vf: np.ndarray, scale: float):
    """使用选定的后端计算 Golden."""
    if _BACKEND == "torch":
        return compute_golden_torch(qf, kf, vf, scale)
    return compute_golden_numpy(qf, kf, vf, scale)


def compare_mixed_precision(npu_fp32: np.ndarray, golden_fp32: np.ndarray,
                            head_num: int):
    """逐元素执行 |npu-golden| <= atol + rtol * |golden| 比对."""
    tot = int(npu_fp32.size)
    abs_err = np.abs(npu_fp32 - golden_fp32)
    tol = COMPARE_ATOL + COMPARE_RTOL * np.abs(golden_fp32)
    ok = abs_err <= tol
    fail_count = int(np.count_nonzero(~ok))

    # denom 避免 |golden|=0, 与 C++ max(|golden|, float_min) 一致.
    denom_min = float(np.finfo(np.float32).tiny)  # ~1.18e-38, 等价于 C++ numeric_limits::min.
    denom = np.maximum(np.abs(golden_fp32), denom_min)
    rel_err = abs_err / denom

    max_abs_idx = int(np.argmax(abs_err.reshape(-1)))
    max_abs_err = float(abs_err.reshape(-1)[max_abs_idx])
    max_rel_err = float(np.max(rel_err))
    max_abs_pos = np.unravel_index(max_abs_idx, npu_fp32.shape)  # (bn, i, d)
    max_abs_b = max_abs_pos[0] // head_num
    max_abs_n = max_abs_pos[0] % head_num

    lines = []
    lines.append(
        f"比对：总元素={tot} 失败={fail_count} "
        f"最大绝对误差={max_abs_err:.6e} @idx={max_abs_idx}"
        f"(b={max_abs_b}, n={max_abs_n}, row={max_abs_pos[1]}, col={max_abs_pos[2]}) "
        f"最大相对误差={max_rel_err:.6e}（rtol={COMPARE_RTOL} atol={COMPARE_ATOL}）"
    )

    # 展平维的第 0 项对应 (b=0,n=0).
    _, seq_len, head_dim = npu_fp32.shape
    r = min(PRINT_ROWS, seq_len)
    c = min(PRINT_COLS, head_dim)
    lines.append(f"前 {r}x{c} O 元素（b=0, n=0）：")
    for i in range(r):
        for j in range(c):
            a = float(npu_fp32[0, i, j])
            g = float(golden_fp32[0, i, j])
            tag = "OK" if ok[0, i, j] else "FAIL"
            lines.append(
                f"  [{i:3d},{j:3d}] npu={a:13.6e} golden={g:13.6e} "
                f"abs={abs(a - g):13.6e} {tag}"
            )

    passed = fail_count == 0
    return passed, lines


def compare_low_precision_baseline(npu_fp32: np.ndarray,
                                   golden_fp32: np.ndarray,
                                   baseline_fp32: np.ndarray,
                                   head_num: int):
    """比较 NPU 与 BF16 基线相对 FP32 Golden 的最大绝对误差."""
    finite = (
        np.isfinite(npu_fp32).all()
        and np.isfinite(golden_fp32).all()
        and np.isfinite(baseline_fp32).all()
    )
    if not finite:
        return False, ["比对：NPU、FP32 Golden 或 BF16 低精度基线中存在 NaN/Inf"]

    npu_abs_err = np.abs(npu_fp32 - golden_fp32)
    baseline_abs_err = np.abs(baseline_fp32 - golden_fp32)
    npu_max_idx = int(np.argmax(npu_abs_err.reshape(-1)))
    baseline_max_idx = int(np.argmax(baseline_abs_err.reshape(-1)))
    npu_max = float(npu_abs_err.reshape(-1)[npu_max_idx])
    baseline_max = float(baseline_abs_err.reshape(-1)[baseline_max_idx])
    limit = LOW_PRECISION_BASELINE_MULTIPLIER * baseline_max
    ratio = npu_max / baseline_max if baseline_max > 0.0 else (
        0.0 if npu_max == 0.0 else float("inf"))
    npu_max_pos = np.unravel_index(npu_max_idx, npu_fp32.shape)
    baseline_max_pos = np.unravel_index(baseline_max_idx, baseline_fp32.shape)
    npu_max_b, npu_max_n = divmod(npu_max_pos[0], head_num)
    baseline_max_b, baseline_max_n = divmod(baseline_max_pos[0], head_num)

    lines = [
        f"低精度基线倍率比对：总元素={npu_fp32.size} ",
        f"  NPU 最大绝对误差={npu_max:.6e} @idx={npu_max_idx}"
        f"(b={npu_max_b}, n={npu_max_n}, row={npu_max_pos[1]}, col={npu_max_pos[2]})",
        f"  BF16 基线最大绝对误差={baseline_max:.6e} "
        f"@idx={baseline_max_idx}"
        f"(b={baseline_max_b}, n={baseline_max_n}, row={baseline_max_pos[1]}, col={baseline_max_pos[2]})",
        f"  误差倍率={ratio:.6e}，上限={LOW_PRECISION_BASELINE_MULTIPLIER:g} "
        f"(NPU 误差上限={limit:.6e})",
    ]
    return npu_max <= limit, lines


def main() -> int:
    if len(sys.argv) != 6:
        print(f"用法: {sys.argv[0]} <data_dir> <B> <N> <S> <D>", file=sys.stderr)
        return 1

    data_dir = sys.argv[1]
    b = int(sys.argv[2])
    n = int(sys.argv[3])
    s = int(sys.argv[4])
    d = int(sys.argv[5])
    if b <= 0 or n <= 0 or s <= 0 or d <= 0:
        print(f"错误: B/N/S/D 必须为正整数，得到 B={b} N={n} S={s} D={d}", file=sys.stderr)
        return 1
    if _USE_LOW_PRECISION_BASELINE and not _HAS_TORCH:
        print(
            "错误: FA_VERIFY_LOW_PRECISION_BASELINE=1 需要可用的 Torch 环境"
            f"（Torch 导入失败：{_TORCH_IMPORT_ERR}）",
            file=sys.stderr,
        )
        return 1

    # 记录实际 Golden 后端及 BF16 路径.
    wanted = os.environ.get("FA_VERIFY_BACKEND", "auto").strip().lower()
    forced_np = os.environ.get("FA_VERIFY_FORCE_NUMPY", "").strip() in (
        "1", "true", "yes", "on")
    passive_np = (
        _BACKEND == "numpy" and not _HAS_TORCH
        and wanted != "numpy" and not forced_np
    )
    if _BACKEND == "torch":
        print(f"verify: backend=torch({torch.get_num_threads()} threads)")
    else:
        print("verify: backend=numpy")
    print(f"verify: causal_mask={'true' if _CAUSAL_MASK else 'false'}")
    if _USE_LOW_PRECISION_BASELINE:
        print(
            "verify: compare=low_precision_baseline "
            f"(max_abs_npu <= {LOW_PRECISION_BASELINE_MULTIPLIER:g} * max_abs_bf16_baseline)"
        )
    else:
        print(
            "verify: compare=mixed_fp32_golden "
            f"(rtol={COMPARE_RTOL} atol={COMPARE_ATOL})"
        )
    if passive_np:
        print(
            f"verify: 警告 torch 不可用（{_TORCH_IMPORT_ERR}），用 numpy 回退",
            file=sys.stderr,
        )
    if not _HAS_ML_DTYPES:
        print("verify: bf16 走位运算回退（未装 ml_dtypes）", file=sys.stderr)

    # Kernel 将 B 和 N 合并为独立序列维；原始文件仍按 [B,N,S,D] 连续存放。
    flat_shape = (b * n, s, d)
    try:
        t0 = time.perf_counter()
        qf = read_bf16(os.path.join(data_dir, "q.bin"), flat_shape, "Q")
        kf = read_bf16(os.path.join(data_dir, "k.bin"), flat_shape, "K")
        vf = read_bf16(os.path.join(data_dir, "v.bin"), flat_shape, "V")
        npu_fp32 = read_bf16(os.path.join(data_dir, "npuout_o.bin"), flat_shape, "NPU O")
        t_read = time.perf_counter() - t0
    except (OSError, ValueError) as e:
        print(f"读取输入失败：{e}", file=sys.stderr)
        return 1

    scale = 1.0 / float(d) ** 0.5
    t0 = time.perf_counter()
    o_golden_fp32 = compute_golden(qf, kf, vf, scale)
    t_golden = time.perf_counter() - t0

    # Golden 保持 FP32 落盘，验收时不再引入一次 BF16 量化。
    golden_path = os.path.join(data_dir, "golden_o.bin")
    np.ascontiguousarray(o_golden_fp32, dtype="<f4").tofile(golden_path)

    backend_tag = (
        f"torch×{torch.get_num_threads()}线程" if _BACKEND == "torch" else "numpy"
    )
    print(
        f"verify: 读入 {t_read:.3f}s，Golden(Q@Kᵀ→softmax→P@V, "
        f"causal={'true' if _CAUSAL_MASK else 'false'}, fp32/{backend_tag}) "
        f"{t_golden:.3f}s -> golden_o.bin(FP32, {b*n*s*d*4} bytes，B={b} N={n})"
    )

    if _USE_LOW_PRECISION_BASELINE:
        t0 = time.perf_counter()
        try:
            baseline_fp32 = compute_low_precision_baseline_torch(qf, kf, vf, scale)
        except (RuntimeError, TypeError) as e:
            print(f"低精度基线计算失败：{e}", file=sys.stderr)
            return 1
        t_baseline = time.perf_counter() - t0
        print(f"verify: Torch BF16 低精度基线 {t_baseline:.3f}s")
        passed, report_lines = compare_low_precision_baseline(
            npu_fp32, o_golden_fp32, baseline_fp32, n)
    else:
        passed, report_lines = compare_mixed_precision(npu_fp32, o_golden_fp32, n)
    print("\n".join(report_lines))

    if passed:
        print("比对成功 ✓（npuout 与 golden 在当前标准内）")
        return 0
    # 失败信息使用 stdout, 保持报告顺序稳定.
    print("比对失败 ✗（npuout 超出容差）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
