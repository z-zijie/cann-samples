# Vector Function

向量函数(Vector Function)是 Ascend 950PR/Ascend 950DT 引入的新编程概念，旨在通过显式控制向量寄存器来实现极致计算性能。

在A2/A3编程模型中，计算指令通常在Unified Buffer(UB)和向量寄存器之间频繁搬运数据，而Vector Function(VF)允许开发者直接在向量寄存器中进行多步骤连续计算，大幅减少计算中间步骤数据搬运开销。

本文将深入介绍Vector Function这一编程概念，详细阐述其性能优势原理、编程范式以及在实际应用中的最佳实践。

## 概述

### 什么是 Vector Function
向量函数(Vector Function)是由Main Scalar调用的由vector指令组成的连续指令块。它是一个标量调用、向量执行的编程单元，将一系列向量计算、寄存器数据加载与存储和地址更新等操作封装在一个独立的执行流中。在此函数内部，数据可以直接在向量寄存器间流动和计算，无需将每个中间结果写回UB，从而实现了计算过程的“数据驻留”，极大提升了数据复用效率和计算吞吐。

### 与A2/A3编程模型的对比

A2/A3的向量编程模型采用“单指令-三阶段”的流水线范式，核心特征是基于UB到UB（UB-to-UB） 的数据流。
![A2编程模型](./images/image-1.png)
如上图所示，每一条向量指令（如 `VectorAdd`）都是一个独立的操作单元，其执行过程遵循 **Load（加载）- Compute（计算）- Store（写回）** 三个阶段：

1.  **Load**：从源操作数指定的UB地址，将数据加载至内部的临时向量寄存器。
2.  **Compute**：在向量算术逻辑单元（VALU）中对加载的数据执行指令所定义的计算（如加法）。
3.  **Store**：将计算结果从临时向量寄存器写回目的操作数指定的UB地址。

在这种模型下，**任何计算产生的中间结果都必须写回UB**。当执行一个包含多步运算的复杂函数时（例如 `f(x) = (a*x + b)*x + c`），每一步运算都需要作为一条独立的UB-to-UB指令来执行。因此，执行一个复合运算必须将其分解为多条串行指令，每一步都产生一次完整的存储-加载开销。这种模式导致了计算过程被频繁的存储访问所分割，大量带宽和周期消耗于**UB与向量寄存器之间的数据搬运**而非有效计算。

![数据流](./images/image-2.png)
Vector Function 是VectorCore架构的基本执行单元，每个向量指令（如 VADD、VLD）本身就可以构成一个最简单的 VF。然而，真正的性能优势来自于将多个相关操作融合成更大的 VF，这种融合打破了传统执行模式中对中间结果必须写回存储的限制。

在未融合的朴素执行模式下：
1. 每个向量操作都作为独立的 VF 执行
2. 每个操作都需要从 UB 加载输入到寄存器
3. 计算完成后必须立即将结果写回 UB
4. 下一个操作再从 UB 加载数据，产生大量不必要的存储-加载开销
 
通过人工或编译器优化将多个操作融合成一个复合VF，主要带来三个关键优势：

1. **消除冗余的数据移动**：中间计算结果不再需要写回缓冲区（UB），而是直接驻留在向量寄存器中供后续操作使用。这消除了大量不必要的寄存器存储（ST）和加载（LD）操作，减少了数据搬运开销。
2. **实现计算与数据加载的并行**：在复合VF内部，多个输入数据的加载（LD）指令可以与那些不依赖于这些数据的计算指令穿插执行。这样，数据加载的延迟可以被计算延迟所掩盖，从而同时充分利用UB带宽和VALU计算单元，提升整体吞吐。
3. **减少中间存储占用，增大有效数据载荷**：由于中间结果不再需要写回缓冲区（UB），那么UB也不需要为其预留额外的空间，这样可以让一次加载到UB中的Tile块更大，提高MTE的搬运效率。

这样的优化使得复合VF能够以更高效的方式执行，尤其是在计算密集、数据复用的场景下（如GeLU等激活函数），性能提升尤为显著。

## 编程模型

### Vector Core硬件架构抽象

![VectorCore硬件架构图](./images/image-3.png)
VectorCore中主要包括：存储单元、搬运单元和计算单元；其中计算单元包括MainScalar计算单元和VF计算单元；VF计算单元内又包括VF Scalar、寄存器读单元、寄存器写单元、寄存器计算单元和各类型的寄存器单元。

- 存储单元
  - Global Memory: VectorCore能访问的外部存储，即为Device上的HBM/DDR等内存；
  - Local Memory: VectorCore内部片上高速缓存，即为Unified Buffer（UB），是向量指令直接访问的存储空间；
- 数据搬运单元：负责处理DMA指令（如DataCopy），用于在Global Memory和Local Memory之间搬移数据；
- 计算单元
  - MainScalar计算单元：处理VF函数外的所有Scalar计算，包括各类型的标量数据运算、程序的流程控制（如循环、分支）、以及VF的调用等；
  - VF计算单元:
    - 寄存器:
      - 向量寄存器(V寄存器)：用来存放计算指令操作的目标数据，是VF计算的核心存储单元；
      - 地址寄存器(A寄存器): 用来在硬件循环（Hardware Loop）中自动更新Local Memory上的数据搬运起始地址，实现循环内地址的自动偏移；
      - 对齐寄存器(U寄存器): 用来辅助访问Local Memory上非32B对齐地址的数据，存储基地址的偏移量；
      - 掩码寄存器(P寄存器): 用来控制计算指令操作的数据粒度，按bit位标识V寄存器中的有效数据；1表示有效，0表示无效。常用于处理尾部非完整向量数据；
    - 寄存器读单元: 处理Load指令，用来加载Local Memory的数据到V寄存器；
    - 寄存器写单元: 处理Store指令，用来从V寄存器搬运数据到Local Memroy；
    - 寄存器计算单元：处理计算指令，对V寄存器中的值进行计算，源操作数和目的操作数对象均为V寄存器；
    - VF Scalar单元：处理VF函数内的标量计算，如计算常数、循环控制变量等。

### 编程范式

#### 硬件循环
硬件循环（Hardware Loop）是一种由硬件直接支持的循环机制，通过专用的循环控制寄存器和地址寄存器（A寄存器）自动管理循环迭代和内存地址更新。与软件循环（通过跳转指令实现）相比，硬件循环消除了循环判断和跳转的开销，并能实现地址的自动增量，极大地提升了小粒度、规则循环的性能。另外在乱序执行机制的配合下，硬件有能力根据指令间的依赖关系和寄存器使用情况，自动把后续循环中的指令提前发射出去(比如提前Load数据，掩盖数据加载延迟)。

#### 指令双发
指令有两份硬件单元，可并行执行，充分利用指令双发是提高性能的关键。

#### 乱序执行机制
VF计算单元内部具备乱序执行（Out-of-Order Execution）能力。硬件会自动分析指令间的寄存器依赖关系，将没有数据依赖关系的指令并行调度到空闲的执行单元。在VF编程中，可以通过合理安排计算顺序，让长延迟指令（如Exp）尽早执行，后续安排不依赖其结果的其他计算，从而隐藏指令延迟。


## 实践: 使用Vector Function计算融合加速GeLU计算
GeLU（高斯误差线性单元）是Transformer架构中的核心激活函数，其计算包含多项式、指数、乘除法等多步操作，是典型的计算密集型算子。
### 代码
- [gelu_without_vf](./gelu_without_vf.cpp)
- [gelu_with_vf](./gelu_with_vf.cpp)

注意观察gelu_compute部分的差异

- 不使用Vector Function的传统实现
  ```cpp
  __aicore__ void gelu_compute(const AscendC::LocalTensor<float> &xLocal, const   AscendC::LocalTensor<float> &yLocal,
      const AscendC::LocalTensor<float> &xCube, const AscendC::LocalTensor<float> &tLocal, int64_t n)
  {
      const float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
      const float TANH_APPROX_FACTOR = 1 / 0.044715;
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(xCube, xLocal, xLocal, n);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(xCube, xCube, xLocal, n);
      AscendC::Muls(tLocal, xLocal, TANH_APPROX_FACTOR, n);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(xCube, xCube, tLocal, n);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(xCube, xCube, NEG_SQRT_EIGHT_OVER_PI, n);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp(xCube, xCube, n);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Adds(xCube, xCube, 1.0f, n);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Div(yLocal, xLocal, xCube, n);
      AscendC::PipeBarrier<PIPE_V>();
  }
  ```
  - 关键问题：
    - 9步计算产生8次中间结果写回UB
    - 每步后需要PipeBarrier强制流水线同步
    - 计算单元频繁等待数据搬运
  
- 使用Vector Function的优化实现
  ```cpp
  __aicore__ void gelu_compute(const AscendC::LocalTensor<float> &xLocal, const   AscendC::LocalTensor<float> &yLocal,
      const AscendC::LocalTensor<float> &xCube, const AscendC::LocalTensor<float> &tLocal, int64_t n)
  {
      const float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
      const float TANH_APPROX_FACTOR = 1 / 0.044715;
      uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
      uint32_t loopNum = (n + vectorLength - 1) / vectorLength;
      __VEC_SCOPE__
      {
          __ubuf__ float *xAddr = (__ubuf__ float *)xLocal.GetPhyAddr();
          __ubuf__ float *yAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
          AscendC::MicroAPI::MaskReg pMask;
          AscendC::MicroAPI::RegTensor<float> xReg, yReg, cubeReg, tReg;
          uint32_t count;
          count = static_cast<uint32_t>(n);
          for (uint16_t i = 0; i < loopNum; ++i) {
              pMask = AscendC::MicroAPI::UpdateMask<float>(count);
              AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                  xReg, (__ubuf__ float *)xAddr + i * vectorLength);
              AscendC::MicroAPI::Mul(cubeReg, xReg, xReg, pMask);
              AscendC::MicroAPI::Mul(cubeReg, cubeReg, xReg, pMask);
              AscendC::MicroAPI::Muls(tReg, xReg, TANH_APPROX_FACTOR, pMask);
              AscendC::MicroAPI::Add(cubeReg, cubeReg, tReg, pMask);
              AscendC::MicroAPI::Muls(cubeReg, cubeReg, NEG_SQRT_EIGHT_OVER_PI, pMask);
              AscendC::MicroAPI::Exp(cubeReg, cubeReg, pMask);
              AscendC::MicroAPI::Adds(cubeReg, cubeReg, 1.0f, pMask);
              AscendC::MicroAPI::Div(yReg, xReg, cubeReg, pMask);
              AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                  (__ubuf__ float *)yAddr + i * vectorLength, yReg, pMask);
          }
      }
  }
  ```
  - 优化特点：
    - 中间结果驻留寄存器，无需写回UB
    - 硬件自动管理指令依赖，无需显式同步
    - 支持指令级并行和乱序执行

### 性能
在Ascend 950PR平台上处理409600个float32数据，性能对比如下：
| Function Name | Task Duration | AIV Time(us) | AIV Vec Time(us) | AIV Vec ratio | AIV Scalar Time(us) | AIV Scalar ratio | AIV MTE2 Time(us) | AIV MTE2 ratio | AIV MTE3 Time(us) | AIV MTE3 ratio |
|-------------------------|--------|-------|--------|-------|--------|-------|--------|-------|--------|-------|
| gelu_without_vf_round#1 | 73.363 | 72.08 | 64.471 | 0.895 | 10.546 | 0.147 | 27.328 | 0.380 | 14.361 | 0.200 |
| gelu_without_vf_round#2 | 69.228 | 68.94 | 67.436 | 0.979 | 9.647  | 0.140 | 18.092 | 0.263 | 14.955 | 0.217 |
| gelu_without_vf_round#3 | **69.236** | 68.94 | 67.430 | 0.979 | 9.650  | 0.140 | 18.107 | 0.263 | 14.972 | 0.218 |
| gelu_without_vf_round#4 | **69.204** | 68.90 | 67.429 | 0.979 | 9.615  | 0.140 | 18.098 | 0.263 | 15.001 | 0.218 |
| gelu_without_vf_round#5 | **69.356** | 69.07 | 67.335 | 0.975 | 9.860  | 0.143 | 18.063 | 0.262 | 14.955 | 0.217 |
| gelu_with_vf_round#1    | 31.170 | 30.35 | 22.164 | 0.731 | 7.126  | 0.235 | 23.266 | 0.767 | 14.096 | 0.465 |
| gelu_with_vf_round#2    | 25.241 | 24.97 | 23.692 | 0.949 | 6.277  | 0.252 | 14.196 | 0.569 | 11.393 | 0.457 |
| gelu_with_vf_round#3    | **25.298** | 24.97 | 23.689 | 0.949 | 6.275  | 0.252 | 14.191 | 0.569 | 11.386 | 0.456 |
| gelu_with_vf_round#4    | **25.296** | 24.97 | 23.674 | 0.949 | 6.282  | 0.252 | 14.175 | 0.568 | 11.370 | 0.456 |
| gelu_with_vf_round#5    | **25.496** | 25.23 | 23.493 | 0.932 | 6.674  | 0.265 | 14.071 | 0.768 | 11.290 | 0.448 |

分析：
1. 前两轮是warmup阶段，不作为性能参考
2. 根据数据可以看出，AIV Vec ratio一直处于90%以上，这表明GeLU是一个典型的Compute Bound算子。性能瓶颈在于Vector计算，因此非常适合使用VectorFunction优化。
3. 在优化后，Vector绝对耗时显著降低d(69us --> 25us), 但仍然占整个算子耗时的95%左右。

### 性能优化原理
1. 计算融合消除数据搬运
   传统实现中GeLU的9步计算需要8次中间结果写回UB，每次写回会产生存储延迟。VF实现将这些计算融合在一个硬件循环内，中间变量全部在寄存器中传递，消除了中间数据搬运。
2. 寄存器重用
   输入向量`x`加载一次后，在后续多个计算步骤中被重复使用，显著减少了对UB带宽的需求。
3. 指令级并行优化
   VF的乱序执行机制允许长延迟指令（如指数运算`Exp`）尽早发射，后续不依赖其结果的指令可并行执行。硬件循环的深度流水线特性使得数据加载、计算和存储操作能够充分重叠。
4. 同步开销小
   传统实现中每个`PipeBarrier`强制流水线排空，造成计算单元空闲。
5. 尾块处理优化
   通过掩码寄存器一次性处理所有数据，包括尾部非完整向量，避免了传统模式中特殊的尾部处理逻辑。

## 结论

Vector Function通过寄存器驻留计算和硬件级并行优化，为Ascend芯片提供了接近理论峰值性能的计算能力。对于GeLU这类计算密集型算子，VF实现相比传统方式可获得数倍的性能提升，主要收益来源于数据搬运的消除、寄存器重用的优化以及指令级并行的充分利用。掌握VF编程范式是发挥Ascend芯片极致计算性能的关键技术。
