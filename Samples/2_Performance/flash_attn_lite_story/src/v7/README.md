# FALite v7：把滚动距离增至三代

## 本版内容

FALite v7 把 v6 连续滚动流水中预先保留的 `item` 从两代增加到三代。AIC 在等待较早 `item` 的 `P` 之前可以再发射一次 C1，AIV 在处理较早 `item` 的输出增量之前也可以再发射一次 V1，从而给 Cube 与 Vector 留出更长的错位距离。

本版固定计算因果 Flash Attention 前向，规格如下。

| 项目 | 实现 |
| --- | --- |
| 输入与输出 | BF16 `Q/K/V/O`，物理形状 `[B,S,128]`，逻辑形状 `[B,1,S,128]` |
| 分块 | `Br=Bc=D=128`，`B>0`、`S>0` 且 `S%128==0` |
| 内部精度 | Cube 使用 BF16 输入、FP32 累加；Softmax 状态和输出累加使用 FP32；`P` 为 BF16 |
| 并行方式 | 一个 Mix 组包含 `1 AIC + 2 AIV`；一个 `task` 处理一个 128 行 Query 分块，两路 AIV 各处理 64 行 |
| 因果模式 | Q 与 K/V 等长且同起点，只保留包含对角线的下三角 |
| 未覆盖能力 | 尾块、非方形 Q/KV、多 Head、多种 HeadDim、滑动窗口、KV Cache、dropout 和反向计算 |

v7 的中间结果全部在片上交接，不需要 GM workspace。

## task、item、阶段与 epoch

一个 `task` 表示一个 Query 分块（Q tile）的完整计算。

这个 `task` 每发射一个 Key/Value 分块，就形成一个 `item=(i,j)`。其中 `i` 是 Query 分块编号，`j` 是 Key/Value 分块编号；每个 `item` 依次经过 C1、V1、C2 和 V2。

因果模式只处理 `j=0...i`，所以 `kvTileCount=i+1`。源码变量 `oDelta` 对应本文的 `DeltaO=P_jV_j`，下文统一写作 `DeltaO`。

数据布局：DN 指普通二维布局，NZ 指供 Cube 使用的分块布局；NZ 转换中的临时填充会在写入共享 L1 时跳过。

代码名词：`PWork` 是 AIV UB 中暂存并整理 `P` 的工作区；VF（Vector Function）指在 AIV 上执行的一段 Vector 函数。

每个 `item` 有四个阶段：

| 阶段 | 核心 | 工作 |
| --- | --- | --- |
| `C1(j)` | AIC | 预取 `K_j/V_j`，计算 `K_j × Q_i^T -> S_j^T`，Fixpipe 把 S 分发到两路 AIV UB |
| `V1(j)` | AIV | 缩放、对角块因果掩码和 Online Softmax（分块 Softmax 递推），生成 BF16 NZ `P_j` |
| `C2(j)` | AIC | 等两路 AIV 写好共享 L1 中的 P，计算 `P_j × V_j -> DeltaO_j`，Fixpipe 把结果分发到两路 AIV UB |
| `V2(j)` | AIV | 更新 `OAcc = alpha_j × OAcc + DeltaO_j` |

对每个 Query 行，V1/V2 按下面的 Online Softmax 关系更新状态：

```text
x_j      = causal_mask(scale × Q_i × K_j^T)
m_new    = max(m, rowmax(x_j))
alpha_j  = exp(m - m_new)
P_j      = exp(x_j - m_new)
l_new    = alpha_j × l + rowsum(P_j)
OAcc_new = alpha_j × OAcc + P_j × V_j
```

`P_j` 是尚未除以完整分母的分子贡献，最终输出在全部 `item` 完成后计算为 `O=OAcc/l`。

`epoch` 表示滚动调度循环的一次推进，只描述每颗核心的发射顺序。AIC 与 AIV 独立运行，在真正消费共享数据时通过 CrossCore 信号对齐。

## R=3 的滚动调度

`R` 表示同一个 `item` 的 `C1` 与 `V2` 相隔的 `epoch` 数。对有效 `item` 数不少于 `R` 的 task，它也等于首个 C2 之前已发射的 C1 数量；较短 task 只发射实际存在的 C1。v7 取 `R=3`：

```text
C1(j): epoch = j
V1(j): epoch = j + 1
C2(j): epoch = j + R - 1 = j + 2
V2(j): epoch = j + R     = j + 3
```

前几个 `epoch` 的发射顺序如下。

| `epoch` | AIC 内的发射顺序 | 每路 AIV 内的发射顺序 |
| ---: | --- | --- |
| 0 | `C1(0)` | — |
| 1 | `C1(1)` | `V1(0)` |
| 2 | `C1(2) -> C2(0)` | `V1(1)` |
| 3 | `C1(3) -> C2(1)` | `V1(2) -> V2(0)` |
| 4 | `C1(4) -> C2(2)` | `V1(3) -> V2(1)` |
| `t` | `C1(t) -> C2(t-2)` | `V1(t-1) -> V2(t-3)` |

流水的填充、稳态和排空由同一个 `epoch < kvTileCount + R` 循环完成：

- 填充阶段依次出现 C1、V1、C2，`epoch 3` 才出现第一个 V2。
- 稳态阶段中，AIC 先发射新 C1 再发射旧 C2，AIV 先发射新 V1 再发射旧 V2。
- 最后一个 C1 发射后继续推进 3 个 `epoch`，把剩余 V1、C2 和 V2 排空。

所有阶段都有 `item` 范围判断。即使 `kvTileCount<R`，也不会访问或等待未发射的 `item`。

![v7 三代滚动流水示意图](../../images/pipeline/falite_v7_pipeline.png)

上半图按 `epoch` 展示填充和稳态发射顺序；下半图抽出同一个 `item=j`，分别画出 `S`、`P`、`DeltaO` 的 CrossCore 扇出或聚合。红色虚框表示等待，跨泳道箭头表示 ready 方向；反向 free、Mutex 回卷和尾部排空按图内说明省略。色块宽度不表示真实耗时。

## 因果掩码

因果约束分两层实现。

1. 整块裁剪：第 `i` 个 Query 分块只发射 `j=0...i`，位于它右侧的 K/V 分块不读取、不计算。
2. 对角块内掩码：当 `j==i` 时，AIV 在 `S^T[Bc,Br]` 上屏蔽 `keyLocalIdx > queryLocalIdx` 的元素。

AIV0 处理 Query 局部行 0～63，AIV1 处理 64～127。块内掩码在求最大值和求指数和/P 的两遍扫描中都执行，避免被屏蔽元素影响 Online Softmax 状态。

## AIC 数据路径

### 缓冲与 Mutex

| 缓冲 | 槽数 | 用途 | Mutex ID |
| --- | ---: | --- | --- |
| K L1 | 2 | C1 内的短生命周期双槽 | 0～1 |
| V L1 | 3 | 从 `C1(j)` 保存到 `C2(j)` | 2～4 |
| Q L1 | 2 | `task` 级双槽 | 5～6 |
| P L1 | 3 | 接收两路 AIV 写入的 P | CrossCore `P_READY` |
| L0A/L0B | 各 2 | MTE1 与 Cube 之间交接 | 7～8 |
| L0C | 3 | Cube 与 Fixpipe 之间的结果队列 | 9～11 |

L1 使用 320 KiB，L0A/L0B 各使用 64 KiB，L0C 使用 192 KiB。

### C1 和 C2

```text
C1:
  Q: GM -> L1 -> L0B
  K: GM -> L1 -> L0A
  V: GM -> L1，保存到同一代际槽
  K × Q^T -> L0C -> Fixpipe -> S UB

C2:
  P: AIV UB -> 共享 L1 -> L0A
  V: 代际 L1 槽 -> L0B
  P × V -> L0C -> Fixpipe -> DeltaO UB
```

Q 由首个 C1 取得 MTE1 所有权，在最后一个有效 C1 后归还。K 在单个 C1 内完成 MTE2→MTE1 交接；V 要保留到相隔两个 `epoch` 的 C2，使用 `j%3` 选择代际槽。

C1/C2 共用 L0A、L0B 和 L0C。源码按 `mmadOpIdx` 对全部矩阵乘编号，L0A/L0B 用两槽轮转，L0C 用三槽轮转。Mutex 只保护本核物理槽；S、P 和 DeltaO 的 AIC/AIV 交接仍由 CrossCore 完成。

## AIV 数据路径

两路 AIV 控制流相同，各自保存本 64 行的 Softmax 状态和输出累加。

| 缓冲 | 槽数 | 内容 | Mutex ID |
| --- | ---: | --- | --- |
| S UB | 2 | FP32 `[128,64]` | CrossCore 双向交接 |
| DeltaO UB | 2 | FP32 `[64,128]` | CrossCore 双向交接 |
| PWork UB | 2 | 带 padding 的 BF16 NZ P | 0～1 |
| OAcc/Output UB | 2 | FP32 OAcc 与 BF16 输出共址 | 2～3 |
| alpha UB | 3 | 三代行缩放系数 | Vector 顺序使用 |
| m/l UB | 各 1 | 64 行的最大值和指数和 | Vector 顺序使用 |

单路 AIV 的 UB 使用 225.50 KiB。

V1 采用分步 Vector 路径：`OnlineColwiseSoftmaxVF` 先把 FP32 指数结果写回 S UB，`FusedDNToNZCastVF` 再读出并转换为 BF16 NZ PWork。PWork 通过 Mutex 从 Vector 交给 MTE3，随后写入共享 L1。

`task` 开始时，代码显式初始化 `m/l/OAcc`。首个 `item` 使用 `alpha=1` 更新全零 OAcc，后续 `item` 执行普通的 `alpha × OAcc + DeltaO`。全部 `item` 完成后，`FusedDivCastInplaceVF` 在 OAcc 槽内完成除法、BF16 转换和输出空间复用，再由 MTE3 写回 GM。

## CrossCore 与槽位回卷

真机 `mode2`（每组 `1 AIC + 2 AIV`）的逻辑 flag ID 如下。

| 数据 | flag ID | 方向 | 交接 |
| --- | ---: | --- | --- |
| S | 0～1 | AIC Fixpipe -> AIV Vector | AIV 初始归还 free；AIC 写完发 ready；两路 AIV 读完再归还 free |
| P | 2～4 | AIV MTE3 -> AIC MTE1 | 两路 AIV 各写一半，AIC 等齐后执行 C2 |
| DeltaO | 5～6 | AIC Fixpipe -> AIV Vector | AIV 初始归还 free；AIC 写完发 ready；V2 读完再归还 free |

`SIM_COMPATIBLE` 构建使用 `mode4`，分别同步 AIV0/AIV1，逻辑依赖与 `mode2` 相同。

V、P 和 alpha 按 `j%3` 回卷。首次回卷的时序是：

```text
epoch 2: C2(0) 读取 V(0) 和 P(0)
epoch 3: C1(3) 可以复用 V 槽 0；V2(0) 读取 alpha(0)
epoch 4: V1(3) 可以复用 P/alpha 槽 0
```

因此 P 不需要额外的 free flag。S、DeltaO 和 PWork 的生命周期只跨相邻阶段，继续使用两槽即可。

## 调度伪代码

```python
R = 3
aiv_set_initial_s_and_odelta_free_flags()

for task in tasks_of_this_mix_group:
    q_tile = task % query_tile_count
    kv_tile_count = q_tile + 1
    AIC.load_q(task, io_slot)
    AIV.lock_output_and_init(io_slot)

    for epoch in range(kv_tile_count + R):
        if epoch < kv_tile_count:
            AIC.c1(epoch)

        if epoch >= 1 and epoch - 1 < kv_tile_count:
            j = epoch - 1
            AIV.wait_s_ready(j % 2)
            AIV.v1(j, diagonal_mask=(j == q_tile))
            AIV.copy_p_half_to_l1(j % R)
            AIV.set_p_ready(j % R)

        if epoch >= R - 1 and epoch - R + 1 < kv_tile_count:
            j = epoch - R + 1
            AIC.wait_two_p_ready(j % R)
            AIC.c2(j)

        if epoch >= R and epoch - R < kv_tile_count:
            j = epoch - R
            AIV.wait_odelta_ready(j % 2)
            AIV.v2(j)

    AIV.normalize_cast_and_store(io_slot)
    io_slot ^= 1

AIC.drain_final_s_and_odelta_free_flags()
```

## 从 v6 到 v7

| 项目 | v6 | v7 |
| --- | ---: | ---: |
| `R` | 2 | 3 |
| 首个 `C2` | epoch 1 | epoch 2 |
| 首个 `V2` | epoch 2 | epoch 3 |
| V/P/alpha 代际槽 | 2 | 3 |
| L0C 结果槽 | 2 | 3 |
| L1 占用 | 256 KiB | 320 KiB |
| 单路 AIV UB | 225.25 KiB | 225.50 KiB |

源码除 `CV_PIPELINE_SLOT_NUM: 2→3` 和 `L0C_QUEUE_DEPTH: 2→3` 外完全相同。Host 片上地址、Mutex ID 和 CrossCore flag 范围都由这两个常量推导。v6→v7 的性能差异反映滚动距离与 L0C 队列一同加深后的完整配置。

## 流水证据

![v7 真机核内流水](../../images/pipe_trace/falite_v7_pipe.png)

截图直接取自完整 PipeTimeline trace 的 `[45.915, 85.915]` μs。真实时间线用于观察三代滚动下 AIC 的 MTE1、CUBE、FIXP 与 AIV VECTOR 如何在不同 `item` 上重叠；示意图中的色块只表示顺序，不表示真实时长。

## 局限与下一方向

三代滚动为等待 P 的 AIC 再增加了一代独立 C1，但它没有减少 V1/V2 的 Vector 指令。若三代已经足以覆盖主要等待，继续增加 R 只会增加 V/P/alpha 和 L0C 的片上占用。[v8](../v8/README.md) 把 `R/L0C` 增至 `4/4`，用于判断第四代滚动是否仍能改善端到端耗时。

## 精度与性能

性能数据采用 CANN 9.2.0、Ascend 950PR、mode2、32 个 AIC、1650 MHz 和 `B=1,N=1,S=131072,D=128`，读取 Kernel Task Duration。

模型算力利用率（MFU）按因果有效 Cube 工作量计算。

| 版本 | 配置 | Task Duration（μs） | 因果有效 Cube MFU |
| --- | --- | ---: | ---: |
| v6 | `R=2,L0C=2`，分步 Vector | 19446.865234 | 52.3516% |
| v7 | `R=3,L0C=3`，分步 Vector | 15132.375000 | 67.2779% |

v7 相对 v6 耗时下降 22.19%。该数字表示三代滚动和三槽 L0C 的整体配置收益，不单独归因于其中一个常量。

默认将 NPU 的 BF16 输出转为 FP32，直接与 FP32 causal Golden 逐元素比较：

```text
abs(float(npu_bf16) - golden_fp32) <= 0.004 + 0.004 * abs(golden_fp32)
```

v7 已通过 `--size 1 131072` 回归。构建和小规格验证命令如下：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target falite_v7 -j
./build/Samples/2_Performance/flash_attn_lite_story/falite_v7 --core-num 1 --size 1 512
```

`S=512` 包含 4 个 Query tile，能够覆盖 R=3 的填充、排空和首次代际槽回卷。
