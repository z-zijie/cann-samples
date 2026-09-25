# Review

## Summary
- 该样例展示了分步优化思路，但目前更接近“代码集合”而不是“可学习的样例产品”。
- 关键风险是：文档严重不足、代码重复极高、结构未体现差分演进。

## Strengths
- 把多核/双缓冲/带宽利用/SIMT 分阶段拆成独立源码，方向正确。
- CMake 能批量构建多个阶段目标。
- 共享 include 基础较完整。

## Key Issues
### 1) README 几乎为空，教学价值断裂
- Severity: Critical
- Why it matters: 性能故事样例若无实验步骤/指标口径/阶段差异说明，读者无法复现结论。
- Evidence: README 仅标题。
- Recommended fix: 补齐每阶段目标、变更点、性能结果与验证命令。

### 2) 阶段源码高度复制
- Severity: Important
- Why it matters: 五个 `src/*.cpp` 开头与主体大面积重复，维护成本巨大，且差异点难审查。
- Evidence: 各文件中 `ExpertIdxSortOneCore` 大片段相同。
- Recommended fix: 抽象 shared base + stage policy，阶段文件只保留变更。

### 3) 命名与结构缺少演进契约
- Severity: Important
- Why it matters: `1_2_3_4_5` 命名只有顺序没有语义，阅读体验依赖外部口头知识。
- Evidence: 文件名数字前缀但缺少对应 mapping 文档。
- Recommended fix: 改为 `step1_multi_core.cpp` 等，并在 README 建立映射表。

## Naming and Structure
- 目录层级合理，但 stage 语义不自描述。
- 建议增加 `docs/` 存放每步说明与性能图。

## Documentation Review
- 当前不达标（缺失核心内容）。
- 必须补充 quick start、输入规模、预期输出、性能指标定义。

## Build-System Review
- foreach 批量建目标思路好。
- 但目标名过泛（仅数字），安装后可发现性一般。

## Code Quality Review
- 内核代码复杂但缺少模块边界，读者不易定位“本步到底改了什么”。
- 风格上仍有宏/重复/超长文件问题。

## Action Plan
- Must fix now
  - 完整重写 README。
  - 抽取阶段共用代码。
- Should fix soon
  - 改进命名为语义化 step 名称。
  - 补充差异化注释（只解释本步变化）。
- Nice to improve later
  - 增加自动化阶段性能回归基线。
