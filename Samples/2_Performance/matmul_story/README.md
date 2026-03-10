# Performance Matmul

以 Atlas A5 训练/推理系列产品为例，介绍了Matmul类算子完整的性能优化实践。

- 包含各类Dtype的具体实现，如Float16、BFLoat16、MXFP8、MXFP4等
- 针对每种实现，介绍了包括性能建模、带宽效率提升、计算效率提升、流水并行度提升等策略。汇总从理论分析到代码实践的完整指南。

下面罗列当前已完成的样例实现：

- [matmul_a16w16](./examples/matmul_a16w16/README.md)：Matmul算子非量化场景【Float16/Bfloat16/Float32】的优化实践
- [quant_matmul_a4w4_mx](./examples/quant_matmul_a4w4_mx/README.md)：Matmul算子量化场景【MxFP4】的优化实践
