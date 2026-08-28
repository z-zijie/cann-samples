# FALite v1：S 和 DeltaO 通过 CV 通路直接交接

v1 保留 v0 的单槽四阶段计算，只缩短两条数据路径：AIC 的 Fixpipe 将 `S` 和 `DeltaO` 直接写入同组 AIV 的 UB，不再经过 GM（全局内存）。`P` 仍从 AIV 写入 GM，再由 AIC 读回。

这一版用最小的结构变化展示 AIC 与 AIV 直接交换数据能省去哪些搬运。

本文将 AIC（Cube）与 AIV（Vector）之间的数据交接简称为 CV 通路。

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
- Cube 使用 BF16 输入和 FP32 累加；分数、Online Softmax（分块 Softmax 递推）状态、`DeltaO` 和 `OAcc` 使用 FP32；`P` 与最终输出使用 BF16。
- 不支持尾块、不同的 Q/KV 长度、causal offset、滑动窗口、多头、变长序列、dropout 和反向计算。

Host 只为 `P` 申请 GM 中间缓冲，每个 task 占 32 KiB。该缓冲不属于对外接口。

## 先认识 task、item 和四个阶段

一个 **task** 完成一个 128 行 Q tile 的全部计算，Q tile 编号记为 `i`。该 task 每访问一个编号为 `j` 的 128 行 K/V tile，就产生一个 **item**，记为 `(i,j)`。

| 阶段 | 执行核心 | 计算内容 |
| --- | --- | --- |
| `C1[j]` | AIC | `K_j × Q_i^T -> S_j^T` |
| `V1[j]` | AIV0、AIV1 | Online Softmax，更新 `m/l/alpha`，生成 `P_j` |
| `C2[j]` | AIC | `P_j × V_j -> DeltaO_j` |
| `V2[j]` | AIV0、AIV1 | `OAcc = alpha_j × OAcc + DeltaO_j` |

两路 AIV 各处理 64 个 Query 行。AIV0 和 AIV1 的控制流相同，Vector 状态彼此独立。

## 从 v0 到 v1

| 阶段结果 | v0 | v1 |
| --- | --- | --- |
| `S^T` | `L0C -> GM -> AIV UB` | `L0C -> Fixpipe -> AIV UB` |
| `P^T` | `AIV UB -> GM -> AIC L1` | 保持不变 |
| `DeltaO` | `L0C -> GM -> AIV UB` | `L0C -> Fixpipe -> AIV UB` |
| GM workspace | 160 KiB/task | 32 KiB/task |

计算公式、causal 语义和单槽调度均不改变。v1 删除的是 `S`、`DeltaO` 的 GM 写回和随后两次 AIV MTE2 读入。

## 数据路径

```text
C1: K/Q -> AIC Cube -> S^T(FP32) -> AIV UB
V1: S^T -> AIV softmax -> P^T(BF16) -> GM
C2: GM -> P，V -> AIC Cube -> DeltaO(FP32) -> AIV UB
V2: DeltaO -> AIV 更新 OAcc
Final: OAcc / l -> BF16 O -> GM
```

Fixpipe 按 AIV 分工直接写入对应 UB：每路 AIV 接收 `S^T[Bc,Br]` 的 64 列和 `DeltaO[Br,D]` 的 64 行。Host 在释放 P workspace 前同步 stream，确保 AIC 不再访问它。

## AIC：单槽 C1 和 C2

C1 路径：

```text
Q: GM -> Q L1 -> L0B
K: GM -> K L1 -> L0A
Mmad(K, Q^T) -> L0C -> Fixpipe -> AIV UB
```

C2 路径：

```text
P: GM -> P L1 -> L0A
V: GM -> V L1 -> L0B
Mmad(P, V) -> L0C -> Fixpipe -> AIV UB
```

AIC 的物理缓冲均为单槽：

| Mutex ID | 资源 | 所有权交接 |
| ---: | --- | --- |
| 0 | K L1 | MTE2 写入，MTE1 读取 |
| 1 | V L1 | MTE2 写入，MTE1 读取 |
| 2 | Q L1 | MTE2 写入，MTE1 在该 task 的全部 C1 中读取 |
| 3 | P L1 | MTE2 写入，MTE1 读取 |
| 4 | 共用 L0A/L0B | MTE1 写入，M 流水读取 |
| 5 | 共用 L0C | M 流水写入，Fixpipe 读取 |

Q 在 task 开始时进入 L1，末次有效 C1 后归还。C1 和 C2 共用 L0 槽，必须按 Mutex 所定义的所有权顺序复用。

## AIV：直接消费 Fixpipe 结果

以下描述一路 AIV：

```text
V1: 等待 S_READY -> 直接读取 S UB -> Online Softmax
    -> P/Output UB -> MTE3 -> P GM

V2: 等待 O_READY -> 直接读取 DeltaO UB
    -> alpha × OAcc + DeltaO

Final: OAcc / l -> BF16 P/Output UB -> MTE3 -> O GM
```

`S` 和 `DeltaO` 的生产者、消费者属于不同核心，所有权由 CrossCore flag 交接，不再使用 AIV MTE2 Mutex。AIV 只保留 `MUTEX_P_UB=0`，在 Vector 与 MTE3 之间管理 P/最终输出复用的 UB 槽。

`m/l/OAcc` 保存整个 task 的递推状态；`alpha` 将 V1 得到的缩放系数传给同一 Vector 流水中稍后发射的 V2。

## 因果掩码的两层处理

1. Q tile 编号为 `i` 的 task 只生成 `j=0...i` 的 item，右侧完整 K/V tile 不发射搬运和计算。
2. 对角 tile `j==i` 内，两路 AIV 屏蔽 `keyLocal > queryLocal` 的元素。

对角 mask 同时用于 softmax 求最大值和求指数和的两次扫描。AIC 与 AIV 都用 `GetKvTileCount<true>(i)=i+1` 缩短循环，保证 CrossCore flag 逐轮配对。

整块裁剪依赖同起点方阵和 `Br=Bc`。不同 Q/KV 长度、不同起点或尾块需要单独计算 causal 边界。

## CV 调度与 CrossCore 同步

v1 仍只有一个工作槽：

```text
AIC C1 --S_READY--> AIV V1 --P_READY--> AIC C2
AIC C2 --O_READY--> AIV V2 --DONE-----> 下一次 C1 消费单槽资源
```

| flag | ID | Set 所在流水 | Wait 所在流水 | 含义 |
| --- | ---: | --- | --- | --- |
| `S_READY` | 0 | AIC `PIPE_FIX` | AIV `PIPE_V` | `S` 已直接写入 AIV UB |
| `O_READY` | 1 | AIC `PIPE_FIX` | AIV `PIPE_V` | `DeltaO` 已直接写入 AIV UB |
| `DONE` | 2 | AIV `PIPE_V` | AIC `PIPE_MTE1` | V2 已消费单槽数据 |
| `P_READY` | 4 | AIV `PIPE_MTE3` | AIC `PIPE_MTE2` | P 已写入 GM |

真机 `mode2`（每组 `1 AIC + 2 AIV`）下，AIC 一次 Set 同时通知两路 AIV；AIC 的聚合 Wait 要等两路 AIV 都完成。`DONE` 约束下一次 C1 的消费端，独立 MTE2 指令仍可先发射到自己的流水。

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
        wait_aiv_to_aic(P_READY)
        c2_read_p_from_gm_and_fix_delta_o_to_aiv(j)
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
        v1(j, diagonal_mask=(j == i))
        write_p_to_gm()
        set_aiv_to_aic(P_READY)

        wait_aic_to_aiv(O_READY)
        v2()
        set_aiv_to_aic(DONE)

    normalize_cast_and_store_o()
```

## 流水示意与实测截图

![v1 的 AIC/AIV 直连数据通路](../../images/pipeline/falite_v1_pipeline.png)

图中突出 `S`、`DeltaO` 的 Fixpipe→AIV UB 直连；`P` 仍按 AIV UB→GM→AIC L1 的路径交接。

![v1 的上板 PipeTimeline](../../images/pipe_trace/falite_v1_pipe.png)

截图直接取自完整 PipeTimeline trace 的 `[152.785, 192.785]` μs，泳道与 v0 保持一致。对照 v0 可以观察 CV 直连减少中间搬运后的流水变化；AIV MTE3 和 AIC MTE2 中仍包含 `P` 经 GM 交接的工作。

## 本版边界与 v2 的方向

v1 已消除 `S` 和 `DeltaO` 的 GM 往返，但 P 仍需执行 `AIV UB -> GM -> AIC L1`。这条路径带来 AIV MTE3、AIC MTE2 和 32 KiB/task 的 Host workspace，Host 也必须在释放 workspace 前同步 stream。

v2 将 P 改为 AIV 直接写入共享 L1，并在片上完成 AIC 所需的 NZ 布局（供 Cube 读取的分块布局），从而移除最后一份 GM workspace。

## 构建与精度验证

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target falite_v1 -j
./build/Samples/2_Performance/flash_attn_lite_story/falite_v1 --core-num 1 --size 2 384
```

公共验证脚本以 FP32 计算 causal Golden。NPU 的 BF16 输出转为 FP32 后，直接与 FP32 Golden 逐元素检查：

```text
abs(float(npu_bf16) - golden_fp32) <= 0.004 + 0.004 * abs(golden_fp32)
```

全部元素必须通过，NaN 或 Inf 直接判失败。设置 `FA_VERIFY_LOW_PRECISION_BASELINE=1` 可改用项目统一的低精度基线倍率标准；该模式要求主机能够导入 Torch。

## 性能记录

性能口径为 CANN 9.2.0、Ascend 950PR、真机 mode2、32 个 AIC、`B=1,N=1,S=131072,D=128`、causal。每轮 warm-up 5 次并采集 1 次 Kernel，v0/v1/v2 交错采集 5 轮，表中取 `Task Duration(us)` 中位数。

表中的模型算力利用率（MFU）按因果有效 Cube 工作量计算。

```bash
msopprof --warm-up=5 --launch-count=1 --aic-metrics=BasicInfo --output=<profiling-output> ./build/Samples/2_Performance/flash_attn_lite_story/falite_v1 --dry-run --size 1 131072
```

| 版本 | Task Duration（μs） | 因果有效 Cube MFU | 相对上一版 |
| --- | ---: | ---: | --- |
| v0 | `88377.640625 us` | `11.5196%` | 基线 |
| v1 | `62899.636719 us` | `16.1857%` | 耗时下降 `28.8286%`，加速 `1.4051x` |

v1 五轮范围为 `62874.187500～62917.863281 us`。分子只统计 causal 有效范围内 `QK^T` 和 `PV` 的 BF16 Cube FLOPs，分母采用 950PR 上 32 个 AIC 的 BF16 Cube 峰值 432 TFLOP/s。

`--dry-run` 仍执行 Kernel、同步和输出回读，只跳过 Golden 与精度比对。
