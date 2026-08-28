# FALite v4：L0 双槽放开 AIC 核内流水

v4 保留 v3 的 CV 双槽分组，再为 L0A/L0B/L0C 增加第二套物理槽。相邻 Cube 阶段可以交替使用两套 L0，AIC 的 MTE1、Mmad 和 Fixpipe 不必反复争用同一个槽。

本版还调整了 C1、C2 的加载顺序，使不同搬运 Pipe 可以在数据依赖允许时并行工作。

本文将全局内存写作 GM，将 AIC（Cube）与 AIV（Vector）之间的数据交接简称为 CV 通路。

## 接口与固定规格

公共 Host 接口为：

```cpp
FlashAttnLiteNPU(Q, K, V, O, B, S, scale, coreNum, stream)
```

本版规格如下：

- `Q`、`K`、`V`、`O` 为 BF16，逻辑形状为 `(B, 1, S, 128)`。
- `Br=Bc=D=128`，要求 `B>0`、`S>0` 且 `S%128==0`。
- `--core-num` 表示 Mix 组数；每组包含 1 个 AIC 和 2 个 AIV。
- Kernel 固定计算同起点、等长度的方阵因果（causal）self-attention。
- Cube 输入为 BF16，并以 FP32 累加；分数、Online Softmax（分块 Softmax 递推）状态、`DeltaO` 和 `OAcc` 使用 FP32；`P` 与最终输出使用 BF16。
- 不支持尾块、不同的 Q/KV 长度、causal offset、滑动窗口、多头、变长序列、dropout 和反向计算。

## 先认识 task、item、group、slot 和四个阶段

一个 **task** 完成一个 128 行 Q tile 的全部计算，Q tile 编号记为 `i`。该 task 每访问一个编号为 `j` 的 128 行 K/V tile，就产生一个 **item**，记为 `(i,j)`。

同一 task 内最多两个连续 item 组成一个 **group**。组内 item 使用 CV slot 0/1；v4 的 L0A/L0B/L0C 也各有 slot 0/1，并随 item 一起切换。

数据布局：DN 指普通二维布局，NZ 指供 Cube 使用的分块布局；NZ 转换中的临时填充会在写入共享 L1 时跳过。

代码名词：`PWork` 是 AIV UB 中暂存并整理 `P` 的工作区；VF（Vector Function）指在 AIV 上执行的一段 Vector 函数。

| 阶段 | 执行核心 | 计算内容 |
| --- | --- | --- |
| `C1[j]` | AIC | `K_j × Q_i^T -> S_j^T` |
| `V1[j]` | AIV0、AIV1 | Online Softmax，更新 `m/l/alpha`，生成 BF16 NZ `P_j` |
| `C2[j]` | AIC | `P_j × V_j -> DeltaO_j` |
| `V2[j]` | AIV0、AIV1 | `OAcc = alpha_j × OAcc + DeltaO_j` |

两路 AIV 各负责 64 个 Query 行。group 尾部只有一个有效 item 时，仅执行一个 slot。

## 从 v3 到 v4

- L0A、L0B、L0C 从单槽改为双槽，C1/C2 可以在两套物理 L0 间交替。
- C1 先发射 Q 的 `L1 -> L0B`，再等待 K 到达并发射 `K L1 -> L0A`，使 Q 的 MTE1 可以和 K 的 MTE2 重叠。
- AIV 的 MTE3 为同一次 P 写入发送 `P_READY` 和 `P_READY_MTE2` 两个通知；C2 的 P MTE1 与 V MTE2 分别在各自 Pipe 上等待，因此可以并行推进。
- CV 双槽分组、Online Softmax、P 的片上布局、causal 语义和 AIV 计算不变。

## 数据路径

```text
C1[s]: K/Q -> AIC L0[s] -> Cube -> S^T[s] -> AIV UB[s]
V1[s]: S^T[s] -> softmax -> PWork[s] -> 共享 P L1[s]
C2[s]: P/V -> AIC L0[s] -> Cube -> DeltaO[s] -> AIV UB[s]
V2[s]: DeltaO[s] -> 更新单份 task 状态 OAcc
Final: OAcc / l -> BF16 O -> GM
```

P 仍只在 AIV UB、共享 L1 与 AIC L0A 之间流动，不经过 GM。

## AIC：L1 双槽与 L0 双槽

C1 路径：

```text
Q: GM -> Q L1 -> L0B[s]
K: GM -> K L1[s] -> L0A[s]
Mmad(K, Q^T) -> L0C[s] -> Fixpipe -> AIV S UB[s]
```

C1 发射 K 的 GM→L1 后，不在 Scalar 侧等待搬运完成。它先把常驻 L1 的 Q 发射到 L0B，再由 K 对应的 MTE1 操作等待 K L1 可读。这样 Q 的 MTE1 可以与 K 的 MTE2 交叠。

C2 路径：

```text
P: AIV -> 共享 P L1[s] -> L0A[s]
V: GM -> V L1[s] -> L0B[s]
Mmad(P, V) -> L0C[s] -> Fixpipe -> AIV DeltaO UB[s]
```

AIC Mutex 如下：

| Mutex ID | 资源 | 所有权交接 |
| ---: | --- | --- |
| 0 / 1 | K L1 slot 0 / 1 | MTE2 写入，MTE1 读取 |
| 2 / 3 | V L1 slot 0 / 1 | MTE2 写入，MTE1 读取 |
| 4 | Q L1 | MTE2 写入，MTE1 在全部有效 C1 中读取 |
| 5 / 6 | L0A/L0B slot 0 / 1 | MTE1 写入，M 流水读取 |
| 7 / 8 | L0C slot 0 / 1 | M 流水写入，Fixpipe 读取 |

L0 双槽形成以下核内错位：

```text
slot 0: Load[0] -> Mmad[0] -> Fix[0]
slot 1:      Load[1] -> Mmad[1] -> Fix[1]
```

每个 Mutex 绑定一个物理槽。写入方释放后，下一条消费 Pipe 才取得所有权；其他槽和其他 Pipe 可以继续推进。

### 为什么 P 需要两个就绪通知

两路 AIV 的 MTE3 写完 P 后，会从同一位置发出两个 CrossCore 通知：

- `P_READY[s]` 交给 AIC MTE1，允许 P 从 L1 搬到 L0A。
- `P_READY_MTE2[s]` 交给 AIC MTE2，允许 V 从 GM 搬到 L1。

CrossCore Wait 绑定到具体硬件 Pipe，一个通知不能同时替代两条 Pipe 上的等待。拆成两个 flag 后，P 的 MTE1 和 V 的 MTE2 可以分别排队，不必由一条流水代替另一条流水等待。

## AIV：沿用 CV 双槽

以下描述一路 AIV：

```text
V1[s]: 等待 S_READY[s]
       -> S UB[s] -> Online Softmax
       -> BF16 NZ PWork[s] -> MTE3 -> 共享 P L1[s]

V2[s]: 等待 O_READY[s]
       -> alpha[s] × OAcc + DeltaO[s]

Final: OAcc / l -> BF16 PWork[0] -> MTE3 -> O GM
```

| 数据或缓冲 | 槽位数 | Mutex | 说明 |
| --- | ---: | --- | --- |
| S、DeltaO | 2 | CrossCore 管理 | Fixpipe 写入，Vector 读取 |
| alpha | 2 | 无独立 Mutex | V1 写入，V2 在同一 Vector 流水中读取 |
| PWork | 2 | 0 / 1 | Vector 写入，MTE3 读取 |
| `m/l/OAcc` | 1 | 无独立 Mutex | 保存整个 task 的递推状态 |

最终输出复用 PWork slot 0。AIV0 与 AIV1 各自使用本地 Mutex 0/1，同号 ID 不表示两颗核心共享一把锁。

## 因果掩码的两层处理

1. Q tile 编号为 `i` 的 task 只生成 `j=0...i` 的 item，未来的完整 K/V tile 不搬运、不计算。
2. 对角 tile `j==i` 内，AIV 在 softmax 的最大值扫描和指数求和/P 扫描中都屏蔽 `keyLocal > queryLocal`。

AIC 和 AIV 使用相同的 `kvTileCount=i+1`。例如三个有效 item 会组成一个满组和一个单 item 尾组，三颗核心都只遍历真实存在的 slot。

整块裁剪适用于同起点方阵和 `Br=Bc`。不同长度、不同起点或尾块需要重新计算 tile 边界。

## CV 调度与 CrossCore 同步

满组的 CV 发射顺序与 v3 相同：

```text
AIC: C1[0] -> C1[1] -> C2[0] -> C2[1]
AIV: V1[0] -> V1[1] -> V2[0] -> V2[1]
```

| flag | ID | Set 所在流水 | Wait 所在流水 | 含义 |
| --- | ---: | --- | --- | --- |
| `S_READY[s]` | 0 / 1 | AIC `PIPE_FIX` | AIV `PIPE_V` | S slot 已写好 |
| `P_READY[s]` | 2 / 3 | AIV `PIPE_MTE3` | AIC `PIPE_MTE1` | P slot 可由 MTE1 读取 |
| `O_READY[s]` | 4 / 5 | AIC `PIPE_FIX` | AIV `PIPE_V` | DeltaO slot 已写好 |
| `P_READY_MTE2[s]` | 6 / 7 | AIV `PIPE_MTE3` | AIC `PIPE_MTE2` | V 的 MTE2 可以开始 |

固定组内顺序与双槽保护复用，不需要 `DONE`。真机 `mode2`（每组 `1 AIC + 2 AIV`）下，AIC 发一次 S/O flag 可通知两路 AIV；AIC 等两路 AIV 都写完各自一半 P 后才消费 P flag。

## 调度伪代码

```python
# AIC
for task in assigned_tasks:
    i = task.q_tile_index
    copy_q_to_l1(i)
    kv_tile_count = i + 1

    for group_begin in range(0, kv_tile_count, 2):
        group_count = min(2, kv_tile_count - group_begin)

        for local in range(group_count):
            j = group_begin + local
            slot = j % 2
            c1_with_l0_slot(j, slot)
            set_aic_to_aiv(S_READY[slot])

        for local in range(group_count):
            j = group_begin + local
            slot = j % 2
            wait_aiv_to_aic(P_READY[slot], pipe=MTE1)
            wait_aiv_to_aic(P_READY_MTE2[slot], pipe=MTE2)
            c2_with_l0_slot(j, slot)
            set_aic_to_aiv(O_READY[slot])
```

```python
# AIV0 和 AIV1
for task in assigned_tasks:
    i = task.q_tile_index
    init_m_l_oacc()
    kv_tile_count = i + 1

    for group_begin in range(0, kv_tile_count, 2):
        group_count = min(2, kv_tile_count - group_begin)

        for local in range(group_count):
            j = group_begin + local
            slot = j % 2
            wait_aic_to_aiv(S_READY[slot])
            v1_and_write_p(j, slot, diagonal_mask=(j == i))
            set_aiv_to_aic(P_READY[slot], P_READY_MTE2[slot])

        for local in range(group_count):
            j = group_begin + local
            slot = j % 2
            wait_aic_to_aiv(O_READY[slot])
            v2(slot)

    normalize_cast_and_store_o()
```

## 流水示意与实测截图

![v4 的 CV 与 L0 双槽流水](../../images/pipeline/falite_v4_pipeline.png)

图中把 CV 槽和 L0 槽分层表示，并画出 `P` 就绪后供 AIC 两条加载路径消费的同步关系。

![v4 的上板 PipeTimeline](../../images/pipe_trace/falite_v4_pipe.png)

截图直接取自完整 PipeTimeline trace 的 `[77.589, 117.589]` μs。与 v3 相同的泳道和 40 μs 时间宽度，便于比较 L0 双槽引入后 AIC MTE1、CUBE 与 FIXP 的交叠变化。

## 本版边界与 v5 的方向

v4 已放开 L0 双槽，但 V 仍在 C2 中从 GM 搬到 L1，并且要等待 P 写入完成后才开始。Q L1 和最终输出也各只有一个 task 级槽，前一个 task 的输出写回可能影响下一个 task 的起步。

v5 将 V 的 GM→L1 前移到 C1，与 K 一起预取；同时为 Q 和 OAcc/输出增加两个 task I/O 槽。

## 构建与精度验证

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target falite_v4 -j
./build/Samples/2_Performance/flash_attn_lite_story/falite_v4 --core-num 1 --size 1 384
```

公共验证脚本以 FP32 计算 causal Golden。NPU 的 BF16 输出转为 FP32 后，直接与 FP32 Golden 逐元素检查：

```text
abs(float(npu_bf16) - golden_fp32) <= 0.004 + 0.004 * abs(golden_fp32)
```

全部元素必须通过，NaN 或 Inf 直接判失败。设置 `FA_VERIFY_LOW_PRECISION_BASELINE=1` 可使用项目统一的低精度基线倍率标准；该模式要求主机能够导入 Torch。

## 性能记录

性能口径为 CANN 9.2.0、Ascend 950PR、真机 mode2、32 个 AIC、`B=1,N=1,S=131072,D=128`、causal。每轮 warm-up 5 次并采集 1 次 Kernel；v4 独立采集 3 轮，表中取 `Task Duration(us)` 中位数。

表中的模型算力利用率（MFU）按因果有效 Cube 工作量计算。

```bash
msopprof --warm-up=5 --launch-count=1 --aic-metrics=BasicInfo --output=<profiling-output> ./build/Samples/2_Performance/flash_attn_lite_story/falite_v4 --dry-run --size 1 131072
```

| 版本 | Task Duration（μs） | 因果有效 Cube MFU | 相对上一版 |
| --- | ---: | ---: | --- |
| v3 | `31164.513672 us` | `32.6677%` | 基线 |
| v4 | `26319.761719 us` | `38.6810%` | 耗时下降 `15.5457%`，加速 `1.1841x` |

分子只统计 causal 有效范围内 `QK^T` 和 `PV` 的 BF16 Cube FLOPs，分母采用 950PR 上 32 个 AIC 的 BF16 Cube 峰值 432 TFLOP/s。

`--dry-run` 仍执行 Kernel、同步和输出回读，只跳过 Golden 与精度比对。
