# FALite v2：P 留在片上，去掉 GM workspace

v2 把 `P` 的交接路径从 `AIV UB -> GM -> AIC L1` 改为 `AIV UB -> 共享 L1`。AIV 完成布局转换后直接写入共享 L1，供 AIC 的第二次矩阵乘读取。

至此，`S`、`P`、`DeltaO` 都在 AIC 与 AIV 的片上通路中交接，Host 不再申请中间 workspace。四阶段仍按单槽顺序执行，便于单独观察 P 通路变化。

本文将 AIC（Cube）与 AIV（Vector）之间的数据交接简称为 CV 通路。

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
- Cube 输入为 BF16，并以 FP32 累加；分数、Online Softmax（分块 Softmax 递推）状态、`DeltaO` 和 `OAcc` 使用 FP32；`P` 与最终输出使用 BF16。
- 不支持不同的 Q/KV 长度、causal offset、滑动窗口、Q/K/V 的 Head 数不一致（GQA/MQA）、同一批次中每条序列长度不同的 varlen 输入、dropout 和反向计算。

Host 按 `ceil(S/128)` 建立 task。尾块的 `Q/K/V` 只读取有效行并在 L1 补 0；Softmax 不统计补齐的 Key 行，两路 AIV 只写回有效 Query 行。`P` 在 UB/L1 中仍使用固定物理 tile，使 C2 和单槽同步流程无需增加特殊分支。

## 先认识 task、item、slot 和四个阶段

一个 **task** 完成某个 `(b,n)` 下一个 128 行 Q tile 的全部计算，Q tile 编号记为 `i`。该 task 每访问一个编号为 `j` 的 128 行 K/V tile，就产生一个 **item**，记为 `(i,j)`。

不同 `(b,n)` 之间没有数据依赖。Host 将 `B×N` 展平为独立序列维，再按 Q tile 分配 task。

**slot** 是一份可以反复使用的片上工作区。v2 只有 slot 0，因此同一 Mix 组中的 item 需要依次复用它。

数据布局：DN 指普通二维布局，NZ 指供 Cube 使用的分块布局；NZ 转换中的临时填充会在写入共享 L1 时跳过。

代码名词：`PWork` 是 AIV UB 中暂存并整理 `P` 的工作区；VF（Vector Function）指在 AIV 上执行的一段 Vector 函数。

| 阶段 | 执行核心 | 计算内容 |
| --- | --- | --- |
| `C1[j]` | AIC | `K_j × Q_i^T -> S_j^T` |
| `V1[j]` | AIV0、AIV1 | Online Softmax，更新 `m/l/alpha`，生成 BF16 NZ `P_j` |
| `C2[j]` | AIC | `P_j × V_j -> DeltaO_j` |
| `V2[j]` | AIV0、AIV1 | `OAcc = alpha_j × OAcc + DeltaO_j` |

两路 AIV 各处理 task 中的 64 个 Query 行。它们共享同一 AIC，但各自保存 task 对应行的 `m/l/OAcc`，不在 AIV 之间交换状态。

## 从 v1 到 v2

| 项目 | v1 | v2 |
| --- | --- | --- |
| P 路径 | `AIV UB -> GM -> AIC L1` | `AIV UB -> 共享 L1` |
| P 布局转换 | AIV 生成普通 BF16 中间结果 | AIV 在 UB 内生成 AIC 可读的 BF16 NZ 数据 |
| `P_READY` 消费 Pipe | AIC MTE2 | AIC MTE1 |
| Host workspace | 32 KiB/task | 0 |
| Host 内部 stream 同步 | 释放 workspace 前需要 | 不需要 |

计算公式、causal 处理和单槽阶段顺序不改变。

## 数据路径

```text
C1: K/Q -> AIC Cube -> S^T(FP32) -> AIV UB
V1: S^T -> AIV softmax -> PWork(BF16 NZ) -> 共享 P L1
C2: P L1，V -> AIC Cube -> DeltaO(FP32) -> AIV UB
V2: DeltaO -> AIV 更新 OAcc
Final: OAcc / l -> BF16 O -> GM
```

V1 的布局转换分三步：

1. Vector 从 FP32 `S^T` 计算 softmax 权重。
2. `FusedDNToNZCastVF` 将权重转为 BF16 NZ 数据，并在每个 16 列分组后保留一个数据块的临时填充；写入共享 L1 时会跳过这些填充。
3. MTE3 写入共享 P L1 时跳过填充，使两路 AIV 各自负责的半块拼成 AIC C2 所需的 P。

这条通路只经过 AIV UB、共享 L1 和 AIC L0A，不读写 GM。

## AIC：单槽矩阵乘

C1 路径：

```text
Q: GM -> Q L1 -> L0B
K: GM -> K L1 -> L0A
Mmad(K, Q^T) -> L0C -> Fixpipe -> AIV UB
```

C2 路径：

```text
P: AIV -> 共享 L1 -> L0A
V: GM -> V L1 -> L0B
Mmad(P, V) -> L0C -> Fixpipe -> AIV UB
```

AIC 使用一套物理工作槽：

| Mutex ID | 资源或用途 | 所有权交接 |
| ---: | --- | --- |
| 0 | K L1 | MTE2 写入，MTE1 读取 |
| 1 | V L1 | MTE2 写入，MTE1 读取 |
| 2 | Q L1 | MTE2 写入，MTE1 在该 task 的全部 C1 中读取 |
| 3 | 共用 L0A/L0B | MTE1 写入，M 流水读取 |
| 4 | 共用 L0C | M 流水写入，Fixpipe 读取 |
| 5 | C2 加载顺序令牌 | P 完成 L1→L0A 的发射后，MTE2 才搬运 V |

Mutex ID 5 不对应物理 Buffer。P 和 V 位于不同 L1 区域，这个令牌只表达 C2 中“先装 P、再搬 V”的跨 Pipe 顺序。

P L1 由两路 AIV 的 MTE3 写入，AIC 通过 `P_READY` 确认数据已就绪，因此不再为 P 设置本地写入 Mutex。

## AIV：在线 softmax 与 P 的片上交付

以下描述一路 AIV：

```text
V1: 等待 S_READY -> S UB -> Online Softmax
    -> BF16 NZ PWork -> MTE3 -> 共享 P L1

V2: 等待 O_READY -> DeltaO UB
    -> alpha × OAcc + DeltaO

Final: OAcc / l -> BF16 PWork -> MTE3 -> O GM
```

`PWork` 与最终输出复用一个 UB 物理槽。`MUTEX_P_WORK_UB=0` 在 Vector 与 MTE3 之间交接所有权，保证 MTE3 读完后 Vector 才覆盖该槽。V1 在等待 `S_READY` 前先让 Vector 取得 PWork 槽，因此等待解除后可以直接计算，不会再与上一轮 MTE3 争用该槽。

S 和 DeltaO 由 Fixpipe 直接写入 AIV UB，其核间可见性由 CrossCore flag 保证。`m/l/OAcc` 是 task 级递推状态，`alpha` 为本 item 的 V2 提供缩放系数。

## 因果掩码的两层处理

1. Q tile 编号为 `i` 的 task 只生成 `j=0...i` 的 item。未来的完整 K/V tile 不产生搬运、Cube 或 Vector 工作。
2. 对角 tile `j==i` 内，AIV0 和 AIV1 按各自负责的 Query 行屏蔽 `keyLocal > queryLocal` 的分数。

mask 同时进入 Online Softmax 的最大值扫描和指数求和扫描。AIC 与 AIV 使用同一 `kvTileCount=i+1`，不会因裁剪造成 CrossCore flag 数量不匹配。

该裁剪方式面向 `Br=Bc` 的同起点方阵。其他 Q/KV 边界需要重新确定可见 tile 和对角位置。

## CV 调度与 CrossCore 同步

v2 仍按单槽顺序处理 item：

```text
C1[j] -> V1[j] -> C2[j] -> V2[j] -> C1[j+1]
```

| flag | ID | Set 所在流水 | Wait 所在流水 | 含义 |
| --- | ---: | --- | --- | --- |
| `S_READY` | 0 | AIC `PIPE_FIX` | AIV `PIPE_V` | S 已写入 AIV UB |
| `O_READY` | 1 | AIC `PIPE_FIX` | AIV `PIPE_V` | DeltaO 已写入 AIV UB |
| `DONE` | 2 | AIV `PIPE_V` | AIC `PIPE_MTE1` | V2 已消费单槽数据 |
| `P_READY` | 4 | AIV `PIPE_MTE3` | AIC `PIPE_MTE1` | 两路 AIV 已把 P 写入共享 L1 |

真机 `mode2`（每组 `1 AIC + 2 AIV`）下，AIC 一次 Set 同时唤醒两路 AIV；AIC 的聚合 Wait 需要两路 AIV 都完成。`DONE` 约束下一次 C1 在 MTE1 及其下游使用单槽资源，独立 MTE2 搬运仍可先进入流水。

## 调度伪代码

```python
# AIC
first_kv_on_this_aic = True
for task in assigned_tasks:
    i = task.q_tile_index
    copy_q_to_l1(i)
    kv_tile_count = i + 1

    for j in range(kv_tile_count):
        if not first_kv_on_this_aic:
            wait_aiv_to_aic(DONE, pipe=MTE1)
        first_kv_on_this_aic = False

        c1_fix_s_to_aiv(j)
        set_aic_to_aiv(S_READY)
        wait_aiv_to_aic(P_READY, pipe=MTE1)
        c2_read_p_from_l1(j)
        set_aic_to_aiv(O_READY)

wait_aiv_to_aic(DONE, pipe=MTE1)
```

```python
# AIV0 和 AIV1
for task in assigned_tasks:
    i = task.q_tile_index
    init_m_l_oacc()
    kv_tile_count = i + 1

    for j in range(kv_tile_count):
        wait_aic_to_aiv(S_READY)
        v1_and_pack_p(j, diagonal_mask=(j == i))
        copy_pwork_to_shared_l1()
        set_aiv_to_aic(P_READY)

        wait_aic_to_aiv(O_READY)
        v2()
        set_aiv_to_aic(DONE)

    normalize_cast_and_store_o()
```

## 流水示意与实测截图

![v2 的单槽片上数据通路](../../images/pipeline/falite_v2_pipeline.png)

图中沿用单槽阶段图，说明三个中间结果都在片上交接。`P` 的 DN→NZ 转换和共享 L1→L0A 路径见前文数据路径。

![v2 的上板 PipeTimeline](../../images/pipe_trace/falite_v2_pipe.png)

截图直接取自完整 PipeTimeline trace 的 `[140.107, 180.107]` μs，保留与其他版本相同的 AIC/AIV 泳道。`P` 已不再往返 GM，但下一个 C1 仍要等待上一 item 的 V2 释放单槽。

## 流水问题与下一步

v2 移除了全部 GM 中间 workspace，但只有一套 CV 工作槽。每轮 V2 结束前，下一轮 C1 的主要消费端不能复用同一槽，AIC 与 AIV 难以同时处理相邻 item。

v3 为 K、V、P 及 AIV 的 item 工作区增加两套 CV 槽，并按两个 item 分组发射 `C1/C2` 与 `V1/V2`，让 AIC 和 AIV 在组内错位运行。

## 构建与精度验证

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target falite_v2 -j
./build/Samples/2_Performance/flash_attn_lite_story/falite_v2 --core-num 1 --size 1 1 385
```

公共验证脚本以 FP32 计算 causal Golden。NPU 的 BF16 输出转为 FP32 后，直接与 FP32 Golden 逐元素检查：

```text
abs(float(npu_bf16) - golden_fp32) <= 0.004 + 0.004 * abs(golden_fp32)
```

全部元素必须通过，NaN 或 Inf 直接判失败。设置 `FA_VERIFY_LOW_PRECISION_BASELINE=1` 可使用项目统一的低精度基线倍率标准；该模式要求主机能够导入 Torch。

## 性能记录

性能口径为 CANN 9.2.0、Ascend 950PR、真机 mode2、32 个 AIC、`B=1,N=1,S=131072,D=128`、causal。每轮 warm-up 5 次并采集 1 次 Kernel，v0/v1/v2 交错采集 5 轮，表中取 `Task Duration(us)` 中位数。

表中的模型算力利用率（MFU）按因果有效 Cube 工作量计算。

```bash
msopprof --warm-up=5 --launch-count=1 --aic-metrics=BasicInfo --output=<profiling-output> ./build/Samples/2_Performance/flash_attn_lite_story/falite_v2 --dry-run --size 1 1 131072
```

| 版本 | Task Duration（μs） | 因果有效 Cube MFU | 相对上一版 |
| --- | ---: | ---: | --- |
| v1 | `62899.636719 us` | `16.1857%` | 基线 |
| v2 | `58697.125000 us` | `17.3445%` | 耗时下降 `6.6813%`，加速 `1.0716x` |

v2 五轮范围为 `58607.460938～58708.312500 us`。分子只统计 causal 有效范围内 `QK^T` 和 `PV` 的 BF16 Cube FLOPs，分母采用 950PR 上 32 个 AIC 的 BF16 Cube 峰值 432 TFLOP/s。

`--dry-run` 仍执行 Kernel、同步和输出回读，只跳过 Golden 与精度比对。
