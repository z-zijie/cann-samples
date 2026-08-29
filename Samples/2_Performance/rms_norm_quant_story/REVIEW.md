# Review

## Summary
- 该样例在“性能故事化”方面做得较好，但代码重复和工程收敛仍明显不足。
- 它有潜力成为仓库标杆，但目前仍未达到“高品味可维护样例”的最终标准。

## Strengths
- README 结构完整，性能建模与优化步骤叙述清晰。
- 分步源码组织有助于展示渐进优化。
- CMake 支持批量构建阶段目标，便于回归。

## Key Issues
### 1) 阶段代码复制过重
- Severity: Important
- Why it matters: 七个阶段文件共享大量样板，修改基础逻辑会产生多点维护和行为漂移。
- Evidence: 各 step 文件均包含大段相同 host 与 kernel 框架。
- Recommended fix: 提取 `common_step_base.hpp` + `step_policy.hpp`，保留差分实现。

### 2) C 风格宏与工具函数泛滥
- Severity: Important
- Why it matters: `CHECK_RET/LOG_PRINT` 宏式模式降低类型安全并模糊控制流。
- Evidence: 各 step 文件顶部重复定义宏。
- Recommended fix: 改为 `inline` 函数/小型错误处理类，减少预处理器依赖。

### 3) 文档与代码映射仍可加强
- Severity: Minor
- Why it matters: README 很强，但未给出“每个 step 对应核心代码文件和关键函数”的精确导航。
- Evidence: 章节很长，读者定位具体代码成本较高。
- Recommended fix: 在每个 step 结尾加入“代码入口索引”。

## Naming and Structure
- `0~6` 编号清楚，但文件注释命名有拼写不一致（如 `1_per_load_gamma` 注释）。
- 建议统一 step 命名模板与注释规范。

## Documentation Review
- 文档是亮点，但可进一步压缩冗余段落，突出可执行摘要。
- 建议补充“不同芯片参数下如何迁移模型”的简表。

## Build-System Review
- 构建逻辑清晰。
- 可考虑把公共编译选项和链接项提炼成 helper function，减少重复。

## Code Quality Review
- 算法实现有深度，但样例工程化尚未到位（重复、宏、长文件）。
- 作为公共模板，仍需更强的结构化与边界清晰度。

## Action Plan
- Must fix now
  - 抽离阶段共用框架，显著减少复制。
- Should fix soon
  - 宏改函数化，统一错误处理。
  - 在 README 添加 step->code 精确索引。
- Nice to improve later
  - 增加自动 benchmark 脚本输出表格。
