# Review

## Summary
- 这是特性样例里相对平衡的一项：目标明确（VF 对比），运行闭环完整。
- 但文档与构建仍有可维护性风险，尚未达到“可长期扩展的示范模板”标准。

## Strengths
- 对比式样例设计好（with_vf vs without_vf），教学意图清晰。
- README 提供环境检查步骤，实操友好。
- 明确给出性能分析工具入口（msprof）。

## Key Issues
### 1) README 步骤编号与叙事细节不严谨
- Severity: Minor
- Why it matters: 样例仓库应体现文档质量纪律，避免细节噪声。
- Evidence: “步骤2”后直接出现“步骤6.运行示例”等编号/结构不一致。
- Recommended fix: 统一编号与层级，压缩冗余提示。

### 2) 性能对比结论缺少可复现实验参数表
- Severity: Important
- Why it matters: 没有输入规模、热身轮次、统计方式，读者难以复现结论。
- Evidence: README 描述性能差异，但缺少严格实验设置。
- Recommended fix: 增加 benchmark protocol（shape、repeat、统计口径）。

### 3) 构建逻辑重复
- Severity: Minor
- Why it matters: 两个 target 的 compile/link 配置重复，未来易漂移。
- Evidence: `gelu_with_vf` 与 `gelu_without_vf` 配置段高度重复。
- Recommended fix: 抽函数 `configure_gelu_target(target)`。

## Naming and Structure
- 文件命名直观。
- 建议把 CPU 参考实现放 `reference/`，设备实现放 `kernel/`，强化结构语义。

## Documentation Review
- 快速开始不错，但可复现性和实验严谨度不足。

## Build-System Review
- 基本 target-based，方向正确。
- 建议消除重复并统一安装规则。

## Code Quality Review
- 可读性较好，教学意图清楚。
- 架构轻量，但还可进一步模块化。
- Code taste：在特性样例中相对成熟。

## Action Plan
- Must fix now
  - 补充可复现实验参数与统计方法。
- Should fix soon
  - 抽象重复 CMake 片段。
- Nice to improve later
  - 增加更小输入的 smoke case。
