# FALite v0：S、P、DeltaO 都经 GM 交接

v0 用最直接的数据通路实现固定 causal 的 Flash Attention Lite 前向计算。AIC 完成两次矩阵乘，两路 AIV 完成 Online Softmax（分块 Softmax 递推）和输出累加；阶段之间的 `S`、`P`、`DeltaO` 都先写入 GM（全局内存），再由下一阶段读回。

这条路径容易跟读，适合先认识 1 AIC + 2 AIV 如何协作；代价是三类中间结果都要往返 GM。

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
- 两次 Cube 矩阵乘使用 BF16 输入和 FP32 累加；分数、Online Softmax 状态、`DeltaO` 和输出累加值使用 FP32；`P` 与最终输出使用 BF16。
- 不支持不同的 Q/KV 长度、causal offset、滑动窗口、Q/K/V 的 Head 数不一致（GQA/MQA）、同一批次中每条序列长度不同的 varlen 输入、dropout 和反向计算。

GM 中间缓冲由 Host 申请和释放，不属于对外接口。

Host 按 `ceil(S/128)` 建立 task。尾块的 `Q/K/V` 只从 GM 读取有效行，并在 L1 中把其余行补 0；Softmax 不统计补齐的 Key 行，两路 AIV 最终只写回有效 Query 行。v0 的 `S/P/DeltaO` GM 中间缓冲仍按完整物理 tile 保存，便于保持四阶段交接方式不变。

## 先认识 task、item 和四个阶段

一个 **task** 完成某个 `(b,n)` 下一个 128 行 Q tile 的全部计算，Q tile 编号记为 `i`。该 task 每访问一个 128 行的 K/V tile，就产生一个 **item**；K/V tile 编号记为 `j`，因此一个 item 可记为 `(i,j)`。

不同 `(b,n)` 之间没有数据依赖。Host 将 `B×N` 展平为独立序列维，再按 Q tile 分配 task。

每个 item 经过四个阶段：

| 阶段 | 执行核心 | 计算内容 |
| --- | --- | --- |
| `C1[j]` | AIC | `K_j × Q_i^T -> S_j^T` |
| `V1[j]` | AIV0、AIV1 | 对 `S_j` 做 Online Softmax，更新 `m/l/alpha`，生成 `P_j` |
| `C2[j]` | AIC | `P_j × V_j -> DeltaO_j` |
| `V2[j]` | AIV0、AIV1 | `OAcc = alpha_j × OAcc + DeltaO_j` |

两路 AIV 各处理 task 中的 64 个 Query 行。它们读取不同的 `S` 列、接收不同的 `DeltaO` 行，并独立维护相应行的 `m`、`l` 和 `OAcc`，不交换 Vector 状态。

## 数据路径

v0 的三个阶段结果都经过 GM：

```text
C1: K/Q -> AIC Cube -> S^T(FP32) -> GM
V1: GM -> S^T -> AIV softmax -> P^T(BF16) -> GM
C2: GM -> P，V -> AIC Cube -> DeltaO(FP32) -> GM
V2: GM -> DeltaO -> AIV 更新 OAcc
Final: OAcc / l -> BF16 O -> GM
```

每个 task 需要一份 GM workspace：

表中的 DN 表示普通二维布局。

| 中间量 | 类型与布局 | 大小 |
| --- | --- | ---: |
| `S^T[Bc,Br]` | FP32 DN | 64 KiB |
| `P^T[Bc,Br]` | BF16 DN | 32 KiB |
| `DeltaO[Br,D]` | FP32 DN | 64 KiB |
| 合计 |  | 160 KiB |

Kernel 异步使用这些缓冲。Host 在释放 workspace 前同步 stream，确保所有 GM 访问已经完成。

## AIC：两次矩阵乘和单槽复用

### C1

```text
Q: GM -> Q L1 -> L0B
K: GM -> K L1 -> L0A
Mmad(K, Q^T) -> L0C
Fixpipe: L0C -> S GM
```

Q 在 task 开始时搬入 L1，并由该 task 的全部 C1 复用。Q 的 MTE1 所有权从第一个有效 C1 保持到最后一个有效 C1；最后一次判断使用 causal 裁剪后的实际 `kvTileCount`。

### C2

```text
P: GM -> P L1 -> L0A
V: GM -> V L1 -> L0B
Mmad(P, V) -> L0C
Fixpipe: L0C -> DeltaO GM
```

AIC 只有一套工作槽。核内 Mutex 与物理资源一一对应：

| Mutex ID | 资源 | 所有权交接 |
| ---: | --- | --- |
| 0 | Q L1 | MTE2 写入，MTE1 在全部 C1 中读取 |
| 1 | K L1 | MTE2 写入，MTE1 读取 |
| 2 | V L1 | MTE2 写入，MTE1 读取 |
| 3 | P L1 | MTE2 写入，MTE1 读取 |
| 4 | 共用 L0A/L0B | MTE1 写入，M 流水读取 |
| 5 | 共用 L0C | M 流水写入，Fixpipe 读取 |

C1 和 C2 串行复用同一套 L0A/L0B/L0C。Mutex 管理 AIC 内部异步 Pipe 对物理槽的所有权，不承担 AIC 与 AIV 之间的通知。

## AIV：softmax、输出累加和单槽复用

以下描述一路 AIV；另一路执行相同控制流，只处理另一半 Query 行。

```text
V1: S GM -> S UB -> Online Softmax -> P UB -> P GM
V2: DeltaO GM -> DeltaO UB -> alpha × OAcc + DeltaO
Final: OAcc / l -> BF16 Output UB -> O GM
```

AIV 使用三类 Mutex：

| Mutex ID | 资源 | 所有权交接 |
| ---: | --- | --- |
| 0 | P/最终输出 UB | Vector 写入，MTE3 读取 |
| 1 | S UB | MTE2 写入，Vector 读取 |
| 2 | DeltaO UB | MTE2 写入，Vector 读取 |

`m`、`l`、`alpha` 和 `OAcc` 在 Vector 流水中按顺序更新。`m/l/OAcc` 保存整个 task 的递推状态，处理完全部 item 后才归一化输出。

## 因果掩码的两层处理

causal mask 分两层完成：

1. 对 Q tile 编号为 `i` 的 task，AIC 和 AIV 都只生成 `j=0...i` 的 item。位于右侧的完整 K/V tile 不搬运，也不发射 C1、V1、C2、V2。
2. 当 `j==i` 时，两路 AIV 在对角 tile 内屏蔽 `keyLocal > queryLocal` 的元素。AIV0 的 Query 行从 0 开始，AIV1 的 Query 行从 64 开始。

对角 mask 同时用于 softmax 的最大值扫描和指数求和扫描，未来位置不会进入 `m`、`l` 或 `P`。AIC 与两路 AIV 使用相同的 `GetKvTileCount<true>(i)=i+1`，因此 CrossCore 收发次数保持一致。

这套整块裁剪以同起点方阵和 `Br=Bc` 为前提。不同 Q/KV 长度或不同起点需要重新计算可见 KV tile 范围。

## CV 调度与 CrossCore 同步

单个物理槽使每个 item 按以下顺序推进：

```text
AIC C1 --S_READY--> AIV V1 --P_READY--> AIC C2
AIC C2 --O_READY--> AIV V2 --DONE-----> 下一次 C1 消费单槽资源
```

| flag | ID | Set 所在流水 | Wait 所在流水 | 含义 |
| --- | ---: | --- | --- | --- |
| `S_READY` | 0 | AIC `PIPE_FIX` | AIV `PIPE_MTE2` | `S` 已写入 GM，可以搬入 UB |
| `O_READY` | 1 | AIC `PIPE_FIX` | AIV `PIPE_MTE2` | `DeltaO` 已写入 GM，可以搬入 UB |
| `DONE` | 2 | AIV `PIPE_V` | AIC `PIPE_MTE1` | V2 已消费这一轮的单槽数据 |
| `P_READY` | 4 | AIV `PIPE_MTE3` | AIC `PIPE_MTE2` | `P` 已写入 GM，可以搬入 L1 |

真机 `mode2`（每组 `1 AIC + 2 AIV`）下，AIC 发一次即可唤醒同组两路 AIV；两路 AIV 各发一次后，AIC 的聚合等待才结束。`DONE` 挂在 AIC MTE1 上，它限制下一次 C1 使用 Q/K/L0 等资源的时机，但不妨碍无关 MTE2 指令先进入自己的流水。

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

        c1_write_s_to_gm(j)
        set_aic_to_aiv(S_READY)
        wait_aiv_to_aic(P_READY)
        c2_read_p_and_write_delta_o_to_gm(j)
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
        v1_read_s_from_gm(j, diagonal_mask=(j == i))
        write_p_to_gm()
        set_aiv_to_aic(P_READY)

        wait_aic_to_aiv(O_READY)
        v2_read_delta_o_from_gm()
        set_aiv_to_aic(DONE)

    normalize_cast_and_store_o()
```

## 流水示意与实测截图

![v0 的 AIC/AIV 调度与 GM 数据通路](../../images/pipeline/falite_v0_pipeline.png)

图中同时标出了三个 GM 中间结果、单槽回卷，以及四个阶段之间的 CrossCore 依赖。虚框表示核心可能因跨核数据尚未就绪而等待。

![v0 的上板 PipeTimeline](../../images/pipe_trace/falite_v0_pipe.png)

截图直接取自完整 PipeTimeline trace 的 `[232.419, 272.419]` μs，保留 AIC 的 MTE2、MTE1、CUBE、FIXP，以及两路 AIV 的 VECTOR、MTE3。单槽 item 按次序推进，各 Pipe 之间有明显空隙。

## 流水问题与下一步

v0 的四个阶段按顺序执行，但 `S`、`P`、`DeltaO` 共需 160 KiB/task 的 GM workspace。三次中间结果往返也增加了 MTE 搬运和 Host 的 stream 同步。

v1 保留单槽调度，只把 `S` 和 `DeltaO` 改为 AIC Fixpipe 直接写入 AIV UB，用一处明确改动观察 CV 直连通路的收益。

## 构建与精度验证

在 cann-samples 根目录执行：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target falite_v0 -j
./build/Samples/2_Performance/flash_attn_lite_story/falite_v0 --core-num 1 --size 2 1 385
```

公共验证脚本以 FP32 计算 causal Golden。NPU 的 BF16 输出转为 FP32 后，直接与 FP32 Golden 逐元素检查：

```text
abs(float(npu_bf16) - golden_fp32) <= 0.004 + 0.004 * abs(golden_fp32)
```

全部元素必须通过，NaN 或 Inf 直接判失败。也可设置 `FA_VERIFY_LOW_PRECISION_BASELINE=1`，使用项目统一的低精度基线倍率标准；该模式要求主机能够导入 Torch。

## 性能记录

性能口径统一为 CANN 9.2.0、Ascend 950PR、真机 mode2、32 个 AIC、`B=1,N=1,S=131072,D=128`、causal。每轮先 warm-up 5 次，再采集 1 次 Kernel；表中时间取 5 轮交错采集的 `Task Duration(us)` 中位数。

表中的模型算力利用率（MFU）按因果有效 Cube 工作量计算。

```bash
msopprof --warm-up=5 --launch-count=1 --aic-metrics=BasicInfo --output=<profiling-output> ./build/Samples/2_Performance/flash_attn_lite_story/falite_v0 --dry-run --size 1 1 131072
```

| Task Duration（μs） | 因果有效 Cube MFU |
| ---: | ---: |
| `88377.640625 us` | `11.5196%` |

五轮范围为 `88352.835938～88429.117188 us`。分子只统计 causal 有效范围内 `QK^T` 和 `PV` 的 BF16 Cube FLOPs，分母采用 950PR 上 32 个 AIC 的 BF16 Cube 峰值 432 TFLOP/s。

`--dry-run` 仍执行 Kernel、同步和输出回读，只跳过 Golden 与精度比对。
