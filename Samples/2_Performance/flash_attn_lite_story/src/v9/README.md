# FALite v9：在 R=3 下缩短 Vector 计算

## 本版内容

FALite v9 采用 `R=3,L0C=4` 的连续滚动配置，并缩短 AIV 上的 Online Softmax（分块 Softmax 递推）、P 布局转换和输出更新路径。v9 与 v10 使用相同的压缩 Vector 和四槽 L0C，二者分别取 `R=3` 和 `R=4`，可以直接比较滚动距离的影响。

本版固定计算因果 Flash Attention 前向。

| 项目 | 实现 |
| --- | --- |
| 输入与输出 | BF16 `Q/K/V/O`，形状 `[B,N,S,128]` |
| 分块 | `Br=Bc=D=128`，`B>0`、`N>0`、`S>0`；`S` 无需按 128 对齐 |
| 内部精度 | Cube 使用 BF16 输入和 FP32 累加；Softmax 状态、`alpha` 和 `OAcc` 使用 FP32；`P` 为 BF16 |
| 并行方式 | 一个 Mix 组包含 `1 AIC + 2 AIV`；一个 `task` 处理一个 128 行 Query 分块，两路 AIV 各处理 64 行 |
| 因果模式 | Q 与 K/V 等长、同起点，固定使用包含对角线的标准下三角掩码 |
| 未覆盖能力 | 非方形 Q/KV、Q/K/V 的 Head 数不一致（GQA/MQA）、同一批次中每条序列长度不同的 varlen 输入、多种 HeadDim、滑动窗口、KV Cache、dropout 和反向计算 |

本版不使用 GM workspace，`S/P/DeltaO` 都在片上完成交接。

Host 按 `ceil(S/128)` 建立 task。尾块的 `Q/K/V` 只读取有效行并在 L1 补 0；Softmax 只遍历有效 Key 行，压缩 Vector 的固定展开段之后再处理不足展开长度的剩余行。两路 AIV 只写回有效 Query 行，`R=3` 的轮换规则不变。

## task、item、阶段与 epoch

一个 `task` 表示某个 `(b,n)` 下一个 Query 分块（Q tile）的完整计算。

不同 `(b,n)` 之间没有数据依赖。Host 将 `B×N` 展平为独立序列维，再按 Q tile 分配 task。

这个 `task` 每发射一个 Key/Value 分块，就形成一个 `item=(i,j)`。其中 `i` 是 Query 分块编号，`j` 是 Key/Value 分块编号；每个 `item` 依次经过 C1、V1、C2 和 V2。

因果模式只处理 `j=0...i`。源码变量 `oDelta` 对应本文的 `DeltaO=P_jV_j`，下文统一写作 `DeltaO`。

数据布局：DN 指普通二维布局，NZ 指供 Cube 使用的分块布局；NZ 转换中的临时填充会在写入共享 L1 时跳过。

代码名词：`PWork` 是 AIV UB 中暂存并整理 `P` 的工作区；VF（Vector Function）指在 AIV 上执行的一段 Vector 函数。

| 阶段 | 核心 | 工作 |
| --- | --- | --- |
| `C1(j)` | AIC | 计算 `K_j × Q_i^T -> S_j^T`，同时预取 `V_j`，Fixpipe 把 S 分发到两路 AIV UB |
| `V1(j)` | AIV | 缩放、对角块因果掩码和 Online Softmax，并直接生成 BF16 NZ `P_j` |
| `C2(j)` | AIC | 等两路 AIV 写好 P，计算 `P_j × V_j -> DeltaO_j`，Fixpipe 把 DeltaO 分发到两路 AIV UB |
| `V2(j)` | AIV | 首个 `item` 用 DeltaO 建立 OAcc，后续执行 `OAcc = alpha_j × OAcc + DeltaO_j` |

对应的 Online Softmax 公式为：

```text
x_j      = causal_mask(scale × Q_i × K_j^T)
m_new    = max(m, rowmax(x_j))
alpha_j  = exp(m - m_new)
P_j      = exp(x_j - m_new)
l_new    = alpha_j × l + rowsum(P_j)
OAcc_new = alpha_j × OAcc + P_j × V_j
```

`P_j` 是 Softmax 分子贡献，全部 `item` 完成后再计算 `O=OAcc/l`。v9 对首个 `item` 使用等价的覆盖写法，省去对零状态的乘加。

`epoch` 表示滚动调度循环的一次推进，只表示每颗核心上的发射顺序。AIC 与 AIV 通过 CrossCore 信号在 S、P 和 DeltaO 的消费点对齐。

## R=3：首个 C2 前发射三个 C1

`R` 表示同一 `item` 的 C1 与 V2 相隔的 `epoch` 数。对有效 `item` 数不少于 `R` 的 task，它也表示首个 C2 之前已经发射的 C1 数量；较短 task 只发射实际存在的 C1。v9 取 `R=3`：

```text
C1(j): epoch = j
V1(j): epoch = j + 1
C2(j): epoch = j + 2
V2(j): epoch = j + 3
```

| `epoch` | AIC 内的发射顺序 | 每路 AIV 内的发射顺序 |
| ---: | --- | --- |
| 0 | `C1(0)` | — |
| 1 | `C1(1)` | `V1(0)` |
| 2 | `C1(2) -> C2(0)` | `V1(1)` |
| 3 | `C1(3) -> C2(1)` | `V1(2) -> V2(0)` |
| `t` | `C1(t) -> C2(t-2)` | `V1(t-1) -> V2(t-3)` |

循环范围是 `epoch=0...kvTileCount+R-1`：

- 填充：C1、V1、C2 和 V2 依次进入流水；
- 稳态：四个阶段处理不同的 `item`；
- 排空：最后一个 C1 后继续推进 3 个 `epoch`，完成剩余阶段。

每个阶段都检查 `item` 范围，较短的 `task` 不会生成不存在的计算或等待。

![v9 三代滚动与压缩 Vector 流水示意图](../../images/pipeline/falite_v9_pipeline.png)

上半图按 `epoch` 展示填充和稳态发射顺序；下半图抽出同一个 `item=j`，分别画出 `S`、`P`、`DeltaO` 的 CrossCore 分发或汇合。红色虚框表示等待，跨泳道箭头表示 ready 方向；反向 free、Mutex 槽位轮换和尾部排空按图内说明省略。色块宽度不表示真实耗时，Vector 改写发生在 V1/V2 内部，耗时变化见后文 PipeTimeline。

## 因果掩码

因果约束分两层实现：

1. `kvTileCount=i+1`，第 `i` 个 Query 分块只发射 K/V 分块 `0...i`。
2. 当 `j==i` 时，AIV 在转置分数布局 `S^T[Bc,Br]` 上屏蔽 `keyLocalIdx > queryLocalIdx` 的元素。

AIV0 处理 Query 局部行 0～63，AIV1 处理 64～127。`OnlineSoftmaxCastPackVF` 扫描分数两遍：第一遍求最大值，第二遍计算指数和并生成 P；对角块因果掩码在两遍中都执行。

## AIC 数据路径与缓冲

v9 的 AIC 计算与 v8 相同，R 的变化只改变 V/P 代际槽和派生编号。

| 缓冲 | 槽数 | 用途 | Mutex ID |
| --- | ---: | --- | --- |
| K L1 | 2 | C1 内短生命周期双槽 | 0～1 |
| V L1 | 3 | 从 `C1(j)` 保存到 `C2(j)` | 2～4 |
| Q L1 | 2 | `task` 级双槽 | 5～6 |
| P L1 | 3 | 接收两路 AIV 合写的 P | CrossCore `P_READY` |
| L0A/L0B | 各 2 | MTE1 与 Cube 之间交接 | 7～8 |
| L0C | 4 | Cube 与 Fixpipe 之间的结果队列 | 9～12 |

L1 使用 320 KiB，L0A/L0B 各使用 64 KiB，L0C 使用 256 KiB。R 与 L0C 深度是两个独立参数，因此 v9 可以使用三代 CV 调度和四槽 L0C。

```text
C1:
  Q: GM -> L1 -> L0B
  K: GM -> L1 -> L0A
  V: GM -> L1，保存到 j%3 代际槽
  K × Q^T -> L0C -> Fixpipe -> S UB

C2:
  P: AIV UB -> 共享 L1 -> L0A
  V: j%3 代际槽 -> L0B
  P × V -> L0C -> Fixpipe -> DeltaO UB
```

Q 由首个 C1 取得、末个有效 C1 归还。K 按 `j%2` 复用，V/P 按 `j%3` 保存。C1/C2 的全部矩阵乘由 `mmadOpIdx` 统一编号，L0A/L0B 按两槽轮转，L0C 按四槽轮转。

## 压缩后的 AIV 路径

### 片上缓冲

| 缓冲 | 槽数 | 内容 | Mutex ID |
| --- | ---: | --- | --- |
| S UB | 2 | FP32 `[128,64]` | CrossCore 双向交接 |
| DeltaO UB | 2 | FP32 `[64,128]` | CrossCore 双向交接 |
| PWork UB | 2 | 带 padding 的 BF16 NZ P | 0～1 |
| OAcc/Output UB | 2 | FP32 OAcc 与 BF16 Output 共址 | 2～3 |
| alpha UB | 3 | 三代缩放系数 | Vector 顺序使用 |
| m/l UB | 各 1 | 64 行最大值与指数和 | Vector 顺序使用 |

单路 AIV 的 UB 使用 225.50 KiB。

### V1：合并 Softmax 与 P 转换

v8 先用一个 VF 把 FP32 指数结果写回 S UB，再由第二个 VF 重读、转 BF16 并整理为 NZ。v9 的 `OnlineSoftmaxCastPackVF` 在第二遍 Softmax 扫描中直接完成 FP32→BF16 转换和 NZ 写入，省去这次 S UB 写回与重读。

最大值和求和扫描每次处理 4 个 Key 位置，用四路寄存器分别累加，再做树状归并。这减少了每条指令连续依赖同一个累加寄存器的情况。首个 `item` 直接写入新的 `m/l`，无需在 `task` 开头初始化这两份状态。

### V2：首轮覆盖与融合乘加

首个 `DeltaO` 直接写入 OAcc，因此无需初始化全零 OAcc，也不执行 `alpha × 0 + DeltaO`。后续 `item` 一次处理两个 Query 行，每行再分成两个 64 元素片段，使用四组寄存器和 `MulDstAdd` 完成更新。

最后的 `FusedDivCastInplaceVF` 保持不变：它计算 `OAcc/l`，把结果转为 BF16，并在 OAcc 槽的前半段原地生成 Output，随后由 MTE3 写回 GM。

## CrossCore 与核内所有权

真机 `mode2`（每组 `1 AIC + 2 AIV`）的逻辑 flag ID 如下。

| 数据 | flag ID | 方向 | 交接 |
| --- | ---: | --- | --- |
| S | 0～1 | AIC Fixpipe -> AIV Vector | AIV 初始发 free；AIC 写完发 ready；两路 AIV 读完归还 free |
| P | 2～4 | AIV MTE3 -> AIC MTE1 | 两路 AIV 各写一半，AIC 等齐后执行 C2 |
| DeltaO | 5～6 | AIC Fixpipe -> AIV Vector | AIV 初始发 free；AIC 写完发 ready；V2 读完归还 free |

S 与 DeltaO 使用两槽 ready/free 双向交接。P 没有 free flag：`C2(j)` 在 `epoch j+2` 读取 P，`V2(j)` 在 `epoch j+3` 读取 alpha，复用同一代际槽的 `V1(j+3)` 到 `epoch j+4` 才发射。

AIC 内的 K/V/Q、L0A/L0B 和 L0C 使用 Mutex；AIV 内的 PWork 与 Output 使用 Mutex。PWork 的 Vector 在覆盖槽位前等待上一轮 MTE3 释放，Output 的 Vector 在复用槽位前等待上一 `task` 的输出 MTE3 完成。

## 调度伪代码

```python
R = 3
aiv_set_initial_s_and_odelta_free_flags()

for task in tasks_of_this_mix_group:
    q_tile = task % query_tile_count
    kv_tile_count = q_tile + 1
    AIC.load_q(task, io_slot)
    AIV.lock_output_slot(io_slot)

    for epoch in range(kv_tile_count + R):
        if epoch < kv_tile_count:
            AIC.c1(epoch)

        if epoch >= 1 and epoch - 1 < kv_tile_count:
            j = epoch - 1
            AIV.wait_s_ready(j % 2)
            AIV.softmax_cast_pack(j, diagonal_mask=(j == q_tile))
            AIV.copy_p_half_to_l1(j % R)
            AIV.set_p_ready(j % R)

        if epoch >= R - 1 and epoch - R + 1 < kv_tile_count:
            j = epoch - R + 1
            AIC.wait_two_p_ready(j % R)
            AIC.c2(j)

        if epoch >= R and epoch - R < kv_tile_count:
            j = epoch - R
            AIV.wait_odelta_ready(j % 2)
            AIV.update_output(first_item=(j == 0))

    AIV.normalize_cast_and_store(io_slot)
    io_slot ^= 1

AIC.drain_final_s_and_odelta_free_flags()
```

## 压缩 Vector 分支怎样比较

| 项目 | v8 | v9 |
| --- | --- | --- |
| `R` | 4 | 3 |
| L0C | 4 槽 | 4 槽 |
| V/P/alpha | 4 代 | 3 代 |
| V1 | Softmax 与 Cast/Pack 分成两个 VF | 第二遍 Softmax 直接 Cast/Pack |
| V2 | 首轮与后续统一乘加 | 首轮直接覆盖，后续使用四组寄存器与 `MulDstAdd` |
| `task` 初始化 | 初始化 `m/l/OAcc` | 首个 V1/V2 直接建立状态 |

v8→v9 同时改变 R 和 Vector，不能把两版性能差全部归因于其中一项。版本关系可按两条路线理解：

```text
分步 Vector:  v7(R=3,L0C=3) -> v8(R=4,L0C=4)
压缩 Vector:  v9(R=3,L0C=4) -> v10(R=4,L0C=4) -> v11(R=5,L0C=4)
```

三组可直接归因的对照是 v8→v10 的 Vector 写法、v9→v10 的 `R=3→4`，以及 v10→v11 的 `R=4→5`。v7→v9 还包含 L0C 从 3 槽增至 4 槽，因此也不作为单项 Vector 对照。

## 流水证据

![v9 真机核内流水](../../images/pipe_trace/falite_v9_pipe.png)

截图直接取自完整 PipeTimeline trace 的 `[41.778, 81.778]` μs。VECTOR 工作块比 v8 更短，但 `R=3` 仍未完全盖住等待；图片与 v10 使用相同的核号、泳道集合和窗口宽度。

## 流水问题与下一步

改写 Vector 后，每个 V1/V2 更短，但 `R=3` 仍未完全盖住等待。v9→v10 的直接对照显示，增至 `R=4` 后耗时下降 19.79%。[v10](../v10/README.md) 保持 Vector 写法和四槽 L0C 不变，只增加一轮提前发射，用来判断这部分收益。

## 精度与性能

统一性能条件为 CANN 9.2.0、Ascend 950PR、mode2、32 个 AIC、1650 MHz 和 `B=1,N=1,S=131072,D=128`。

模型算力利用率（MFU）按因果有效 Cube 工作量计算。

| 版本 | 配置 | Task Duration（μs） | 因果有效 Cube MFU |
| --- | --- | ---: | ---: |
| v9 | `R=3,L0C=4`，压缩 Vector | 13238.221680 | 76.9041% |
| v10 | `R=4,L0C=4`，压缩 Vector | 10618.893555 | 95.8738% |

v9→v10 只改变 R 及其派生代际槽，v10 耗时下降 19.79%。这说明在该固定规格下，压缩 Vector 后仍需要第四代滚动距离。

默认将 NPU 的 BF16 输出转为 FP32，直接与 FP32 causal Golden 逐元素比较：

```text
abs(float(npu_bf16) - golden_fp32) <= 0.004 + 0.004 * abs(golden_fp32)
```

v9 已通过 `--size 1 1 131072`、非整块 `--size 1 1 705` 和三行余数 `--size 1 1 707` 回归。构建和验证命令如下：

```bash
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --target falite_v9 -j
./build/Samples/2_Performance/flash_attn_lite_story/falite_v9 --core-num 1 --size 1 1 387
```

`S=387` 包含 4 个 Query tile，最后一个 tile 有 3 行，可覆盖 R=3 的填充、排空、首次代际槽回卷以及压缩 Vector 的余数循环。
