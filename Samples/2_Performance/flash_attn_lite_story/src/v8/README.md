# FALite v8：四代滚动的分步 Vector 版本

## 本版内容

FALite v8 把连续滚动距离从 3 增至 4，使 AIC 在第一次等待 P 之前可以连续发射四个 C1，也使 AIV 在第一次 V2 之前可以连续发射四个 V1。v8 使用分步 Vector 路径；它与同为 `R=4,L0C=4` 的 v10 只在 Vector 写法上不同，可以直接比较两种写法。

本版固定计算因果 Flash Attention 前向。

| 项目 | 实现 |
| --- | --- |
| 输入与输出 | BF16 `Q/K/V/O`，物理形状 `[B,S,128]`，逻辑形状 `[B,1,S,128]` |
| 分块 | `Br=Bc=D=128`，要求 `B>0`、`S>0` 且 `S%128==0` |
| 内部精度 | Cube 使用 BF16 输入和 FP32 累加；Softmax 状态、`alpha` 和 `OAcc` 使用 FP32；`P` 为 BF16 |
| 并行方式 | 一个 Mix 组包含 `1 AIC + 2 AIV`；一个 `task` 处理一个 Query 分块，两路 AIV 各处理 64 行 |
| 因果模式 | Q 与 K/V 等长、同起点，输出包含对角线的标准下三角注意力 |
| 未覆盖能力 | 尾块、非方形 Q/KV、多 Head、多种 HeadDim、滑动窗口、KV Cache、dropout 和反向计算 |

所有中间结果都在片上交接，不使用 GM workspace。

## task、item、阶段与 epoch

一个 `task` 表示一个 Query 分块（Q tile）的完整计算。

这个 `task` 每发射一个 Key/Value 分块，就形成一个 `item=(i,j)`。其中 `i` 是 Query 分块编号，`j` 是 Key/Value 分块编号；每个 `item` 依次经过 C1、V1、C2 和 V2。

因果模式只处理 `j=0...i`。源码变量 `oDelta` 对应本文的 `DeltaO=P_jV_j`，下文统一写作 `DeltaO`。

数据布局：DN 指普通二维布局，NZ 指供 Cube 使用的分块布局；NZ 转换中的临时填充会在写入共享 L1 时跳过。

代码名词：`PWork` 是 AIV UB 中暂存并整理 `P` 的工作区；VF（Vector Function）指在 AIV 上执行的一段 Vector 函数。

| 阶段 | 核心 | 计算与数据路径 |
| --- | --- | --- |
| `C1(j)` | AIC | `K_j × Q_i^T -> S_j^T`，同时把 `V_j` 预取到 L1，Fixpipe 把 S 分发到两路 AIV UB |
| `V1(j)` | AIV | 缩放、对角块因果掩码和 Online Softmax（分块 Softmax 递推），生成 BF16 NZ `P_j` 并写入共享 L1 |
| `C2(j)` | AIC | `P_j × V_j -> DeltaO_j`，Fixpipe 把 DeltaO 分发到两路 AIV UB |
| `V2(j)` | AIV | `OAcc = alpha_j × OAcc + DeltaO_j` |

两路 AIV 分别处理 Query 分块的 64 行。`S^T` 的完整形状为 `[128,128]`，每路 AIV 收到 `[128,64]`；每路的 `DeltaO/OAcc` 形状为 `[64,128]`。

全部 `item` 完成后，AIV 计算 `O=OAcc/l`，原地转为 BF16 并写回 GM。

## Online Softmax

对一个 Query 行，设第 `j` 个分数块为 `x_j`，FALite 按下式更新状态：

```text
x_j      = causal_mask(scale × Q_i × K_j^T)
m_new    = max(m, rowmax(x_j))
alpha_j  = exp(m - m_new)
P_j      = exp(x_j - m_new)
l_new    = alpha_j × l + rowsum(P_j)
OAcc_new = alpha_j × OAcc + P_j × V_j
```

这里的 `P_j` 是 Softmax 分子的分块贡献，还没有除以完整分母 `l_new`。代码在首个 `item` 使用等价的专用初始化：直接写入新的 `m/l`，把 `alpha` 置 1，并用全零 OAcc 执行更新；后续 `item` 再使用上面的普通递推。

## 因果掩码

因果约束分为两层：

1. 整块裁剪：`kvTileCount=i+1`，第 `i` 个 Query 分块只发射 `j=0...i`，右上方整块不读取也不计算。
2. 对角块内掩码：当 `j==i` 时，AIV 屏蔽 `keyLocalIdx > queryLocalIdx` 的位置。

C1 输出转置布局 `S^T[Bc,Br]`，Vector 寄存器通道对应 Query 行，循环变量对应 Key 行。AIV0 的 Query 局部编号从 0 开始，AIV1 从 64 开始。块内掩码同时作用于求最大值和求指数和/P 的两遍扫描。

## R=4 的滚动调度

`epoch` 表示滚动调度循环的一次推进。`R=4` 表示同一个 `item` 的 C1 与 V2 相隔 4 个 `epoch`；对至少有 4 个有效 `item` 的 task，首个 C2 之前会发射 4 个 C1，较短 task 只发射实际存在的 C1。

```text
C1(j): epoch = j
V1(j): epoch = j + 1
C2(j): epoch = j + R - 1 = j + 3
V2(j): epoch = j + R     = j + 4
```

| `epoch` | AIC 内的发射顺序 | 每路 AIV 内的发射顺序 |
| ---: | --- | --- |
| 0 | `C1(0)` | — |
| 1 | `C1(1)` | `V1(0)` |
| 2 | `C1(2)` | `V1(1)` |
| 3 | `C1(3) -> C2(0)` | `V1(2)` |
| 4 | `C1(4) -> C2(1)` | `V1(3) -> V2(0)` |
| `t` | `C1(t) -> C2(t-3)` | `V1(t-1) -> V2(t-4)` |

流水由同一个 `epoch` 循环完成：

- 填充：V1、C2 和 V2 依次进入流水；
- 稳态：C1/V1 处理较新的 `item`，C2/V2 处理四代范围内的较早 `item`；
- 排空：最后一个 C1 后再推进 4 个 `epoch`，完成剩余阶段。

每个阶段都检查 `item` 是否小于 `kvTileCount`，较短的 `task` 不会等待不存在的工作。

![v8 四代滚动流水示意图](../../images/pipeline/falite_v8_pipeline.png)

上半图按 `epoch` 展示填充和稳态发射顺序；下半图抽出同一个 `item=j`，分别画出 `S`、`P`、`DeltaO` 的 CrossCore 扇出或聚合。红色虚框表示等待，跨泳道箭头表示 ready 方向；反向 free、Mutex 回卷和尾部排空按图内说明省略。色块宽度不表示真实耗时。

## AIC 核内流水

### 缓冲与 Mutex

| 缓冲 | 槽数 | 用途 | Mutex ID |
| --- | ---: | --- | --- |
| K L1 | 2 | C1 内短生命周期双槽 | 0～1 |
| V L1 | 4 | 从 `C1(j)` 保存到 `C2(j)` | 2～5 |
| Q L1 | 2 | `task` 级双槽 | 6～7 |
| P L1 | 4 | 接收两路 AIV 合写的 P | CrossCore `P_READY` |
| L0A/L0B | 各 2 | MTE1 与 Cube 之间交接 | 8～9 |
| L0C | 4 | Cube 与 Fixpipe 之间的结果队列 | 10～13 |

L1 使用 384 KiB，L0A/L0B 各使用 64 KiB，L0C 使用 256 KiB。

### 数据路径

```text
C1:
  Q: GM -> L1 -> L0B
  K: GM -> L1 -> L0A
  V: GM -> L1，保存到代际槽
  K × Q^T -> L0C -> Fixpipe -> S UB

C2:
  P: AIV UB -> 共享 L1 -> L0A
  V: 代际 L1 槽 -> L0B
  P × V -> L0C -> Fixpipe -> DeltaO UB
```

首个 C1 取得 Q L1 的 MTE1 所有权，最后一个有效 C1 按 `j+1==kvTileCount` 归还。K 按 `j%2` 复用，V/P 按 `j%4` 保存跨阶段数据。

C1 和 C2 共用 L0A/L0B/L0C。`mmadOpIdx` 对两类矩阵乘统一编号，L0A/L0B 按 2 槽、L0C 按 4 槽轮转。Fixpipe 取得 L0C Mutex 后再等待 AIV 归还目标 UB 槽，保证 L0C 结果不会在写出前被覆盖。

## AIV 核内流水

| 缓冲 | 槽数 | 内容 | Mutex ID |
| --- | ---: | --- | --- |
| S UB | 2 | FP32 `[128,64]` | CrossCore 双向交接 |
| DeltaO UB | 2 | FP32 `[64,128]` | CrossCore 双向交接 |
| PWork UB | 2 | 带 padding 的 BF16 NZ P | 0～1 |
| OAcc/Output UB | 2 | FP32 OAcc 与 BF16 Output 共址 | 2～3 |
| alpha UB | 4 | 四代缩放系数 | Vector 顺序使用 |
| m/l UB | 各 1 | 每行最大值和指数和 | Vector 顺序使用 |

单路 AIV 的 UB 使用 225.75 KiB。

本版使用分步 Vector 路径：

1. `task` 开始时显式初始化 `m/l/OAcc`。
2. `OnlineColwiseSoftmaxVF` 完成缩放、因果掩码、最大值、指数和及 Online Softmax 状态更新，并把 FP32 指数结果写回 S UB。
3. `FusedDNToNZCastVF` 再读取 S UB，把指数结果转换并整理成 BF16 NZ PWork。
4. PWork Mutex 在 Vector 与 MTE3 之间交接 P 槽；MTE3 把两路 AIV 的 P 分片写到共享 L1。
5. `OnlineUpdateVF` 对每个 `item` 执行 `alpha × OAcc + DeltaO`。
6. `FusedDivCastInplaceVF` 计算 `OAcc/l`，原地生成 BF16 输出，再由 MTE3 写回 GM。

分步写法容易对照公式，但 FP32 指数结果会经历一次写回和重读，首个 `item` 也执行了可以省去的 OAcc 乘加。这两项是压缩 Vector 版本的主要优化对象。

## CrossCore 同步

真机 `mode2`（每组 `1 AIC + 2 AIV`）使用下面的逻辑 flag ID。

| 数据 | flag ID | 方向 | 交接过程 |
| --- | ---: | --- | --- |
| S | 0～1 | AIC Fixpipe -> AIV Vector | AIV 初始发 free；AIC 写完发 ready；两路 AIV 读完归还 free |
| P | 2～5 | AIV MTE3 -> AIC MTE1 | 两路 AIV 各写一半，AIC 等齐后执行 C2 |
| DeltaO | 6～7 | AIC Fixpipe -> AIV Vector | AIV 初始发 free；AIC 写完发 ready；V2 读完归还 free |

S 与 DeltaO 使用两槽 ready/free 双向交接，并在 Kernel 结束时由 AIC 消费最后的 free 信号。P 没有 free flag：`C2(j)` 在 `epoch j+3` 读取 P，`V2(j)` 在 `epoch j+4` 读取 alpha，`V1(j+4)` 到 `epoch j+5` 才覆盖同一 P/alpha 代际槽。

`SIM_COMPATIBLE` 构建把两路 AIV 分别映射到 mode4 flag，不改变上述生产消费关系。

## 调度伪代码

```python
R = 4
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

## 从 v7 到 v8

| 项目 | v7 | v8 |
| --- | ---: | ---: |
| `R` | 3 | 4 |
| 首个 C2/V2 | epoch 2/3 | epoch 3/4 |
| V/P/alpha 代际槽 | 3 | 4 |
| L0C 结果槽 | 3 | 4 |
| L1 占用 | 320 KiB | 384 KiB |
| 单路 AIV UB | 225.50 KiB | 225.75 KiB |

两版只有 `CV_PIPELINE_SLOT_NUM` 和 `L0C_QUEUE_DEPTH` 不同，其余 Host、AIC、AIV、causal 与同步代码一致。v7→v8 同时增加滚动距离和 L0C 深度，因此只表示完整配置的变化。

## 两条版本路线

v8 之后，版本路线分成“滚动距离”和“Vector 写法”两条轴：

```text
分步 Vector:  v7(R=3,L0C=3) -> v8(R=4,L0C=4)
压缩 Vector:  v9(R=3,L0C=4) -> v10(R=4,L0C=4) -> v11(R=5,L0C=4)
```

应按下面三组关系解释性能：

- v8→v10 保持 `R=4,L0C=4`，只改变 Vector 写法，可以直接衡量 Vector 压缩。
- v9→v10 保持压缩 Vector 和 `L0C=4`，只改变 `R=3→4`，可以直接衡量滚动距离。
- v10→v11 保持压缩 Vector 和 `L0C=4`，只改变 `R=4→5`。

v8→v9 同时把 R 从 4 降到 3 并替换 Vector，不能用于单项归因；v7→v9 虽然 R 相同，但 L0C 与 Vector 同时变化，也不能用于单项归因。

## 流水证据

![v8 真机核内流水](../../images/pipe_trace/falite_v8_pipe.png)

截图直接取自完整 PipeTimeline trace 的 `[44.312, 84.312]` μs。四代滚动下 AIC 与 AIV 已形成较长的连续忙区；后文与 v10 的对照使用相同的 40 μs 窗口宽度和泳道集合。

## 局限与下一方向

v8 比 v7 多保留一代 C1/V1，但分步 Vector 的计算量没有改变。长序列性能几乎不再随 R 增加，说明继续扩大窗口难以缩短这条 Vector 路径。[v9](../v9/README.md) 在 `R=3` 分支压缩 Vector，[v10](../v10/README.md) 则使用 `R=4` 的相同压缩写法；其中 v8 与 v10 可直接比较 Vector 路径的差别。

## 精度与性能

性能条件为 CANN 9.2.0、Ascend 950PR、mode2、32 个 AIC、1650 MHz 和 `B=1,N=1,S=131072,D=128`。

模型算力利用率（MFU）按因果有效 Cube 工作量计算。

| 版本 | 配置 | Task Duration（μs） | 因果有效 Cube MFU |
| --- | --- | ---: | ---: |
| v7 | `R=3,L0C=3`，分步 Vector | 15132.375000 | 67.2779% |
| v8 | `R=4,L0C=4`，分步 Vector | 15120.490234 | 67.3308% |
| v10 | `R=4,L0C=4`，压缩 Vector | 10618.893555 | 95.8738% |

v7→v8 的完整配置耗时只下降 0.08%。v8→v10 保持滚动和槽位配置不变，压缩 Vector 后耗时下降 29.77%，这组对照用于衡量 Vector 路径的改写收益。

默认将 NPU 的 BF16 输出转为 FP32，直接与 FP32 causal Golden 逐元素比较：

```text
abs(float(npu_bf16) - golden_fp32) <= 0.004 + 0.004 * abs(golden_fp32)
```

v8 已通过 `--size 1 131072` 回归。构建和验证命令如下：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target falite_v8 -j
./build/Samples/2_Performance/flash_attn_lite_story/falite_v8 --core-num 1 --size 1 640
```

`S=640` 包含 5 个 Query tile，可覆盖 R=4 的完整填充、排空和首次代际槽回卷。
