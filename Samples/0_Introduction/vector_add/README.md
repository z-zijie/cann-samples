# Vector Add

## 描述
本样例展示如何在NPU的VectorCore硬件单元上使用AscendC编程语言实现向量加法操作。

NPU包含多种专用计算单元，其中VectorCore专门负责向量和标量运算。与通用CPU的SIMD指令集不同，VectorCore是专门的硬件单元，具有更高的并行度和计算效率。

## 关键概念

### VectorCore硬件单元

VectorCore是AI Core中的专用向量处理单元，每个AI Core包含多个VectorCore，每个VectorCore能够：
  - 并行处理多个数据元素
  - 支持多种数据类型（int8, int16, int32, float16, float32等）
  - 执行向量加法、乘法、比较等基本运算
  - 通过Local Memory（UB）实现低延迟数据访问

### AI Core架构

NPU的核心计算单元，包含：
  - VectorCore: 负责向量/标量运算
  - CubeCore: 负责矩阵乘法运算

### 核函数（Kernel Function）
使用AscendC编写的设备端函数，通过`__global__ __aicore__`关键字修饰，在AI Core的VectorCore上执行。核函数是使能VectorCore硬件的主要方式。

### Global Memroy 与 Local Memroy
  - Global Memroy: AI Core间共享的内存，容量大但访问延迟高
  - Local  Memroy: 每个VectorCore私有的高速缓存，容量有限但访问速度极快

### 任务切分(Tiling)
根据VectorCore的硬件特性将大规模计算划分为小任务：
  - 基于可用VectorCore数量进行并行化
  - 根据LocalMemory（UB）容量确定分块大小
  - 尽可能的让每个VectorCore负载均衡

## 编译运行

从项目根目录构建, 参考项目根目录README.md

