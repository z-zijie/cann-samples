# FALite v3：用两套 CV 槽错位处理两个 item

v3 为相邻两个 item 准备两套 CV 工作槽。AIC 可以先连续发射两次 C1，AIV 随后连续执行两次 V1；AIC 与两路 AIV 在组内错位处理不同 item，从单槽串行推进变为双槽流水。

P 继续留在片上，Online Softmax（分块 Softmax 递推）、causal 语义和数值路径与 v2 相同。

本文将全局内存写作 GM，将 AIC（Cube）与 AIV（Vector）之间的数据交接简称为 CV 通路。

## 接口与固定规格

公共 Host 接口为：

```cpp
FlashAttnLiteNPU(Q, K, V, O, B, N, S, scale, coreNum, stream)
```

本版规格如下：

- `Q`、`K`、`V`、`O` 为 BF16，逻辑形状为 `(B, N, S, 128)`。
- `Br=Bc=D=128`，要求 `B>0`、`N>0`、`S>0`；`S` 无需按 128 对齐。
- `--core-num` 表示 Mix 组数；每组包含 1 个 AIC 和 2 个 AIV。
- Kernel 固定计算同起点、等长度的方阵因果（causal）self-attention。
- Cube 输入为 BF16，并以 FP32 累加；分数、Online Softmax 状态、`DeltaO` 和 `OAcc` 使用 FP32；`P` 与最终输出使用 BF16。
- 不支持不同的 Q/KV 长度、causal offset、滑动窗口、Q/K/V 的 Head 数不一致（GQA/MQA）、同一批次中每条序列长度不同的 varlen 输入、dropout 和反向计算。

Host 按 `ceil(S/128)` 建立 task。尾块的 `Q/K/V` 只读取有效行并在 L1 补 0；Softmax 不统计补齐的 Key 行，两路 AIV 只写回有效 Query 行。`P` 在 UB/L1 中仍使用固定物理 tile，双槽的编号和归还关系不随有效行数改变。

## 先认识 task、item、group、slot 和四个阶段

一个 **task** 完成某个 `(b,n)` 下一个 128 行 Q tile 的全部计算，Q tile 编号记为 `i`。该 task 每访问一个编号为 `j` 的 128 行 K/V tile，就产生一个 **item**，记为 `(i,j)`。

不同 `(b,n)` 之间没有数据依赖。Host 将 `B×N` 展平为独立序列维，再按 Q tile 分配 task。

v3 把同一 task 内至多两个连续 item 称为一个 **group**。组内 item 使用 slot 0 和 slot 1；最后一组只有一个 item 时，只使用实际存在的 slot。

数据布局：DN 指普通二维布局，NZ 指供 Cube 使用的分块布局；NZ 转换中的临时填充会在写入共享 L1 时跳过。

代码名词：`PWork` 是 AIV UB 中暂存并整理 `P` 的工作区；VF（Vector Function）指在 AIV 上执行的一段 Vector 函数。

| 阶段 | 执行核心 | 计算内容 |
| --- | --- | --- |
| `C1[j]` | AIC | `K_j × Q_i^T -> S_j^T` |
| `V1[j]` | AIV0、AIV1 | Online Softmax，更新 `m/l/alpha`，生成 BF16 NZ `P_j` |
| `C2[j]` | AIC | `P_j × V_j -> DeltaO_j` |
| `V2[j]` | AIV0、AIV1 | `OAcc = alpha_j × OAcc + DeltaO_j` |

两路 AIV 各处理 64 个 Query 行。它们执行相同的组内顺序，但各自维护对应行的 `m/l/OAcc`。

## 从 v2 到 v3

v2 完成一个 item 后才推进下一个 item：

```text
C1[0] -> V1[0] -> C2[0] -> V2[0] -> C1[1] -> ...
```

v3 把两个 item 编成一组：

```text
AIC: C1[g,0] -> C1[g,1] -> C2[g,0] -> C2[g,1]
AIV: V1[g,0] -> V1[g,1] -> V2[g,0] -> V2[g,1]
```

为支撑这个顺序，v3 将 K、V、P L1 和 AIV 的 item 工作区改成双槽，并把 S/P/O 三类 CrossCore flag 各扩为两个 ID。v2 的 `DONE` 被删除。

数学计算、P 的片上布局、causal 裁剪和最终归一化均不改变。

## 数据路径

```text
C1[s]: K/Q -> AIC Cube -> S^T[s] -> AIV UB[s]
V1[s]: S^T[s] -> softmax -> PWork[s] -> 共享 P L1[s]
C2[s]: P L1[s]，V L1[s] -> AIC Cube -> DeltaO[s] -> AIV UB[s]
V2[s]: DeltaO[s] -> 更新单份 task 状态 OAcc
Final: OAcc / l -> BF16 O -> GM
```

`s=j%2`。双槽保存 item 私有中间量，`m/l/OAcc` 仍只有一份，因为它们必须在整个 task 内按 K/V 顺序递推。

## AIC：L1 双槽、L0 单槽

C1：

```text
Q: GM -> Q L1 -> L0B
K: GM -> K L1[s] -> L0A
Mmad(K, Q^T) -> 共用 L0C -> Fixpipe -> AIV S UB[s]
```

C2：

```text
P: AIV -> 共享 P L1[s] -> L0A
V: GM -> V L1[s] -> L0B
Mmad(P, V) -> 共用 L0C -> Fixpipe -> AIV DeltaO UB[s]
```

AIC Mutex 如下：

| Mutex ID | 资源或用途 | 所有权交接 |
| ---: | --- | --- |
| 0 / 1 | K L1 slot 0 / 1 | MTE2 写入，MTE1 读取 |
| 2 / 3 | V L1 slot 0 / 1 | MTE2 写入，MTE1 读取 |
| 4 | Q L1 | MTE2 写入，MTE1 在全部有效 C1 中读取 |
| 5 | 共用 L0A/L0B | MTE1 写入，M 流水读取 |
| 6 | 共用 L0C | M 流水写入，Fixpipe 读取 |
| 7 | C2 加载顺序令牌 | P 完成 L1→L0A 的发射后，MTE2 才搬 V |

K、V、P 已经按 CV slot 分开，但 L0A/L0B/L0C 仍只有一套。因此两次 C1 和两次 C2 在 L0 上仍按以下次序复用：

```text
C1[0] -> C1[1] -> C2[0] -> C2[1]
```

Mutex 等待发生在真正消费该资源的硬件 Pipe 中。命令可以继续分发到其他 Pipe，使用共用 L0 的 MTE1、Mmad 或 Fixpipe 会在各自位置等待所有权。

## AIV：item 双槽与 task 单状态

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

最终输出复用 PWork slot 0 及其 Mutex。AIV0 与 AIV1 是两颗独立核心，同号 Mutex ID 只管理各自本地 UB，不会互相占用。

## 因果掩码的两层处理

1. Q tile 编号为 `i` 的 task 只生成 `j=0...i` 的 item，右侧完整 K/V tile 不发射。
2. 对角 tile `j==i` 内，AIV 在求最大值和求指数和/P 的两次扫描中都屏蔽上三角。

例如 `i=2` 时，task 产生 `j=0,1,2` 三个有效 item。它们被分成 `[0,1]` 和 `[2]` 两组，尾组只使用一个 slot，不发射填充任务。

AIC 与 AIV 使用相同的 `kvTileCount=i+1` 和 `groupCount`，因此短尾组不会等待不存在的 flag。该方法以同起点方阵和 `Br=Bc` 为前提。

## CV 调度与 CrossCore 同步

双槽使 AIC 与 AIV 可以错位处理组内不同 item：

```text
AIC: C1[0]
AIC: C1[1]    || AIV: V1[0]
AIC: C2[0]    || AIV: V1[1]
AIC: C2[1]    || AIV: V2[0]
AIC: next C1  || AIV: V2[1]
```

这只是依赖关系示意，不表示左右阶段耗时相等。

| flag | ID | Set 所在流水 | Wait 所在流水 | 含义 |
| --- | ---: | --- | --- | --- |
| `S_READY[s]` | 0 / 1 | AIC `PIPE_FIX` | AIV `PIPE_V` | S slot 已写好 |
| `P_READY[s]` | 2 / 3 | AIV `PIPE_MTE3` | AIC `PIPE_MTE1` | 两路 AIV 已写好 P slot |
| `O_READY[s]` | 4 / 5 | AIC `PIPE_FIX` | AIV `PIPE_V` | DeltaO slot 已写好 |

v3 不再使用 `DONE`。三类复用关系分别由以下顺序保护：

- P：旧 P 先由 C2 读取；AIV 的 Vector 流水把下一组 V1 排在旧 V2 之后，因此下一次 P 写入不会越过旧 C2；
- DeltaO：C2 发出 `O_READY` 后旧 V2 才能执行，V2 消费完成后该槽才会复用；
- PWork：AIV 本地的 Vector 与 MTE3 通过 Mutex 交接同一 UB 槽。

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
            c1(j, slot)
            set_aic_to_aiv(S_READY[slot])

        for local in range(group_count):
            j = group_begin + local
            slot = j % 2
            wait_aiv_to_aic(P_READY[slot])
            c2(j, slot)
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
            set_aiv_to_aic(P_READY[slot])

        for local in range(group_count):
            j = group_begin + local
            slot = j % 2
            wait_aic_to_aiv(O_READY[slot])
            v2(slot)

    normalize_cast_and_store_o()
```

## 流水示意与实测截图

![v3 的 CV 双槽分组流水](../../images/pipeline/falite_v3_pipeline.png)

图中用 AIC、AIV0、AIV1 三条泳道表示同一 group 中两个 item 的错位执行。虚框是可能发生的跨核等待，红色虚线箭头表示 CrossCore 就绪关系；两路 AIV 都要独立完成同一逻辑事件的 Set，AIC 的聚合 Wait 会等齐二者。

![v3 的上板 PipeTimeline](../../images/pipe_trace/falite_v3_pipe.png)

截图直接取自完整 PipeTimeline trace 的 `[88.796, 128.796]` μs。AIC 的 MTE2、MTE1、CUBE、FIXP 与两路 AIV 的 VECTOR、MTE3 同时展开；与 v2 相比，跨 Pipe 重叠明显增多。

## 流水问题与下一步

v3 的 CV 数据已有两套槽，但所有 C1/C2 仍共用一套 L0A/L0B/L0C。下一次加载、矩阵乘和写回仍被同一套 L0 槽串住，MTE1、CUBE 和 FIXP 不能充分交叠。

v4 保留 CV 双槽分组，将 L0A/L0B/L0C 也改为双槽，并拆开 C2 中 P 的 MTE1 与 V 的 MTE2 等待。

## 构建与精度验证

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target falite_v3 -j
./build/Samples/2_Performance/flash_attn_lite_story/falite_v3 --core-num 1 --size 1 1 385
```

公共验证脚本以 FP32 计算 causal Golden。NPU 的 BF16 输出转为 FP32 后，直接与 FP32 Golden 逐元素检查：

```text
abs(float(npu_bf16) - golden_fp32) <= 0.004 + 0.004 * abs(golden_fp32)
```

全部元素必须通过，NaN 或 Inf 直接判失败。设置 `FA_VERIFY_LOW_PRECISION_BASELINE=1` 可使用项目统一的低精度基线倍率标准；该模式要求主机能够导入 Torch。

## 性能记录

性能口径为 CANN 9.2.0、Ascend 950PR、真机 `mode2`（每组 `1 AIC + 2 AIV`）、32 个 AIC、`B=1,N=1,S=131072,D=128`、causal。每轮 warm-up 5 次并采集 1 次 Kernel；v3 采集 5 轮，表中取 `Task Duration(us)` 中位数。

表中的模型算力利用率（MFU）按因果有效 Cube 工作量计算。

```bash
msopprof --warm-up=5 --launch-count=1 --aic-metrics=BasicInfo --output=<profiling-output> ./build/Samples/2_Performance/flash_attn_lite_story/falite_v3 --dry-run --size 1 1 131072
```

| 版本 | Task Duration（μs） | 因果有效 Cube MFU | 相对上一版 |
| --- | ---: | ---: | --- |
| v2 | `58697.125000 us` | `17.3445%` | 基线 |
| v3 | `31164.513672 us` | `32.6677%` | 耗时下降 `46.91%`，加速 `1.883x` |

分子只统计 causal 有效范围内 `QK^T` 和 `PV` 的 BF16 Cube FLOPs，分母采用 950PR 上 32 个 AIC 的 BF16 Cube 峰值 432 TFLOP/s。

`--dry-run` 仍执行 Kernel、同步和输出回读，只跳过 Golden 与精度比对。
