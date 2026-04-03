# cann-samples 检索关键词清单（中文 + English）

> 目标：提升 `cann-samples` 在代码平台与搜索引擎中的可检索性，覆盖算子、调优技术、特性能力、问题域与业务术语。

## 1) 典型算子（Operators）

- 矩阵乘 / Matrix Multiplication (MatMul, GEMM)
- 向量加 / Vector Add
- 均方根归一化 / RMSNorm
- 量化矩阵乘 / Quantized MatMul
- 注意力分数计算 / Attention Score
- 融合推理注意力 / Fused Infer Attention
- 分发与合并 / Dispatch and Combine
- 路由初始化 / Routing Init
- 专家混合 / Mixture of Experts (MoE)
- 激活函数向量化 / Vectorized Activation
- 归一化算子 / Normalization Operator
- 高性能核函数 / High-performance Kernel

## 2) 关键技术（Core Techniques）

- 算子性能调优 / Operator Performance Tuning
- 指令级优化 / Instruction-level Optimization
- 访存优化 / Memory Access Optimization
- 系统级优化 / System-level Optimization
- 双缓冲 / Double Buffering
- 多缓冲 / N-Buffering
- 尾块均衡 / Tail Balancing
- SIMD/SIMT 并行 / SIMD/SIMT Parallelism
- 向量函数优化 / Vector Function Optimization
- 数据布局优化 / Data Layout Optimization
- 计算与访存重叠 / Compute-Memory Overlap
- 算子融合 / Operator Fusion
- 全量化推理 / Full Quantized Inference
- 低比特量化 / Low-bit Quantization
- 混合精度计算 / Mixed Precision

## 3) 重要特性/能力（Key Features & Capabilities）

- CANN 算子样例 / CANN Operator Samples
- Ascend NPU 优化 / Ascend NPU Optimization
- 高吞吐低时延 / High Throughput & Low Latency
- 工程化样例 / Production-oriented Samples
- 体系化知识库 / Systematic Knowledge Base
- 可复用调优配方 / Reusable Tuning Recipes
- CMake 构建 / CMake Build
- 多目标编译 / Multi-target Build
- 部分构建 / Selective Build
- 并行编译 / Parallel Build
- 基准与性能故事 / Performance Story
- 教程与实战 / Tutorials and Hands-on

## 4) 常见问题域（Common Problem Domains）

- 大模型推理加速 / LLM Inference Acceleration
- 专家并行通信 / Expert Parallel Dispatch
- 注意力计算瓶颈 / Attention Bottleneck
- 量化精度-性能权衡 / Quantization Accuracy-Performance Tradeoff
- 带宽受限场景 / Bandwidth-bound Workload
- 计算受限场景 / Compute-bound Workload
- 尾块性能退化 / Tail Performance Degradation
- 小批量低时延场景 / Small-batch Low-latency
- 高并发推理 / High-concurrency Inference
- 内存占用优化 / Memory Footprint Reduction

## 5) 业务相关术语（Business / Scenario Terms）

- 生成式 AI / Generative AI
- 大语言模型 / Large Language Model (LLM)
- 推理服务 / Inference Serving
- 云边协同部署 / Cloud-Edge Deployment
- 实时 AI 应用 / Real-time AI Applications
- 智能问答 / AI Question Answering
- 检索增强生成 / Retrieval-Augmented Generation (RAG)
- 多模态推理 / Multimodal Inference
- 企业级算子优化 / Enterprise Operator Optimization
- 端到端性能工程 / End-to-end Performance Engineering

## 6) 仓库结构与主题关键词（Repository Theme Keywords）

- 0_Introduction / 入门样例
- 1_Features / 特性样例
- 2_Performance / 性能样例
- matmul_story / 矩阵乘性能故事
- moe_dispatch_and_combine_story / MoE 分发合并故事
- moe_init_routing_story / MoE 路由初始化故事
- rms_norm_quant_story / RMSNorm 量化故事
- full_quant_fused_infer_attention_score_story / 全量化融合注意力分数故事
- hardware_features / 硬件特性
- instruction_optimization / 指令优化
- memory_optimization / 访存优化
- system_optimization / 系统优化

## 7) 可直接用于仓库标签的短关键词（Tag-ready）

`cann`, `ascend`, `npu`, `operator`, `kernel`, `matmul`, `gemm`, `rmsnorm`, `quantization`, `mxfp4`, `mxfp8`, `moe`, `attention`, `dispatch`, `combine`, `routing`, `optimization`, `performance`, `memory`, `instruction`, `system`, `simt`, `vector`, `inference`, `llm`, `fused-op`

## 8) 中文标签建议（Tag-ready CN）

`算子优化`、`性能调优`、`矩阵乘`、`量化推理`、`全量化`、`MoE`、`注意力`、`访存优化`、`指令优化`、`系统优化`、`硬件特性`、`向量化`、`低时延`、`高吞吐`、`大模型推理`

