# HiFloat8 Matmul

## 描述

本样例演示了如何在昇腾AI处理器的CubeCore硬件单元上使用AscendC编程语言实现hifloat8类型的矩阵乘运算。

## 关键特性

- 流水并行：具备DoubleBuffer能力开启流水并行
- 参数可配：支持自定义矩阵维度进行测试
- 精度对比：提供标准的CPU实现作为精度基准
- 特定类型：演示特有数据类型hifloat8的计算

## 支持架构

NPU ARCH 3510

## 参数说明

- m: 矩阵乘中左矩阵的行
- k: 矩阵乘中左矩阵的列/右矩阵的行
- n: 矩阵乘中右矩阵的列

## 编译运行

1. 编译样例

从项目根目录启动构建，参考项目[README.md](../../../README.md)

指定hif8_mm_demo的编译命令：
```shell
cmake --build build --target hif8_mm_demo
```

2. 运行样例

使用`verify_output.py`执行算子用例，需要指定矩阵乘维度，并随机生成输入数据。
```shell
python ./Samples/1_Features/hif8_mm/scripts/verify_output.py 10 4 7
```
打印如下执行结果，证明样例执行成功。
```shell
test pass
```
如果存在精度问题，则会打印错误数据，并显示如下结果。
```shell
[ERROR] result error
```