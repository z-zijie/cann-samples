# Review

## Summary
- 样例覆盖了较完整的 MatMul pipeline，但复杂度已接近“内核实现文档”而非“入门 sample”；教学焦点被大量模板与同步细节淹没。
- 作为公共样例，结构和叙事仍需重构，当前不满足“高质量可教学样板”标准。

## Strengths
- 展示了较真实的多级存储与分块思路，技术深度足够。
- 使用 CPU golden 对比，有基础正确性闭环。
- README 对算子参数（m/k/n）有基本说明。

## Key Issues
### 1) 单文件体量与职责过重
- Severity: Critical
- Why it matters: 学习路径断裂，读者很难区分“核心思想”与“平台细节”。
- Evidence: `main.cpp` 同时含工具函数、kernel、host runtime、随机数据、精度对比。
- Recommended fix: 分离 `kernel_impl.h`、`host_driver.cpp`、`reference_cpu.cpp`、`tiling_config.h`，并在 README 按层说明阅读顺序。

### 2) 硬编码 tile 参数缺乏可解释策略
- Severity: Important
- Why it matters: 样例应教“如何推导参数”，不是仅给常数；否则读者会机械复制。
- Evidence: `baseM/baseN/baseK/kL1` 等常数内嵌在 kernel。
- Recommended fix: 将参数来源写成配置结构 + 公式说明 + 与硬件约束映射。

### 3) 文档命令可执行性细节不足
- Severity: Minor
- Why it matters: README 示例命令未统一使用可执行相对路径（如 `./matmul`），并且未说明默认输入范围与显存开销。
- Evidence: README 运行段落。
- Recommended fix: 补充“推荐输入规模、最小可跑规模、典型耗时范围、失败诊断”。

## Naming and Structure
- `matmul` 命名直观；但单文件 `main.cpp` 与其内容严重不匹配（实际上是完整 demo framework）。
- 建议用 `matmul_sample_main.cpp` + 子模块文件命名提升意图表达。

## Documentation Review
- 数学描述基本到位，但没有“先看哪里、后看哪里”的学习路线。
- 缺少“为什么这么设计”的解释（如同步事件选择依据）。

## Build-System Review
- 存在对 `third_party/tensor_api` 的相对路径硬编码，目录迁移脆弱。
- 目标级配置尚可，但重复 compile/link 片段应沉淀到公共 CMake 函数。

## Code Quality Review
- 可读性：中低，长函数和高密度平台 API 混杂。
- 架构：职责耦合明显。
- 健壮性：基础流程完整，但可配置性低。
- 可教性：深度够，但“教学分层”失败。
- Code taste：技术能力强，表达方式不克制。

## Action Plan
- Must fix now
  - 拆分文件，建立分层阅读路径。
- Should fix soon
  - 抽离 tiling 参数为可解释配置并写文档。
- Nice to improve later
  - 提供 `small/medium/large` 预设 case。
