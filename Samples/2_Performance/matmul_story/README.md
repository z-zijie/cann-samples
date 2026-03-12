# Performance Matmul

## 目录结构

```
matmul_story/
├── common/                                     # 公共代码与 golden 脚本
├── docs/                                       # 性能实践文档目录
├── matmul_stubs/                               # 算子实现与样例
│   ├── include/                                # 头文件 (block, kernel, policy, utils)
│   └── examples/                               # 算子样例总目录
│       ├── quant_matmul_mxfp4/                 # MXFP4 量化 MatMul 样例
|       |   ├── quant_matmul_mxfp4_aswt.cpp     # 算子Aswt模板Host侧调用
|       |   └── README.md                       # 算子样例说明
|       ├── ...
│       └── matmul/                             # 非量化 Matmul 样例
└── matmul_tutorials/                           # step-by-step教程与分步示例
```

## 简介

本章节汇总介绍了Matmul类算子完整的性能优化实践。

- 包含各类Dtype的具体实现样例，如Float16、BFLoat16、MXFP8、MXFP4等。
- 针对每种实现样例，提供了包括性能建模、搬运效率优化、计算效率优化、指令并行度优化等策略。汇总从理论分析到代码实践的完整指南。
- 针对经典场景，提供了`step-by-step`的教程分解示例，介绍从零开始如何完成算子的极致性能优化。

## matmul stubs样例

- [matmul](./matmul_stubs/examples/matmul/README.md)：Matmul算子非量化场景【Float16/Bfloat16/Float32】的优化实践
- [quant_matmul_mxfp4](./matmul_stubs/examples/quant_matmul_mxfp4/README.md)：Matmul算子量化场景【MxFP4】的优化实践

## matmul tutorials样例

待补充