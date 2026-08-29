# Review

## Summary
- 样例展示了有效优化思路，但当前存在与 n_buffer 相同的 CMake 结构问题，并且 README 路径与层级细节不一致。
- 作为公开样例，质量尚未达标，需先修正构建与文档严谨性。

## Strengths
- 对比“改造前/后”代码片段，教学方式直观。
- 性能时序图给出可视化证据。
- 主题聚焦在 unit_flag，技术指向明确。

## Key Issues
### 1) 子目录 CMake 修改全局编译器
- Severity: Critical
- Why it matters: 破坏父项目可预测性，不符合现代 CMake 规范。
- Evidence: `set(CMAKE_*_COMPILER ...)` 出现在样例子目录。
- Recommended fix: 统一在 toolchain/顶层设置。

### 2) README 运行路径疑似错误
- Severity: Important
- Why it matters: 新用户按文档执行会失败，直接损伤样例可信度。
- Evidence: README 使用 `build/Samples/1_Features/unit_flag/`，与实际目录层级不一致（缺失 `instruction_optimization`）。
- Recommended fix: 更正路径并增加一条 `find`/`tree` 验证说明。

### 3) 设计边界说明不足
- Severity: Minor
- Why it matters: unit_flag 的收益与副作用高度依赖场景，样例应明确 trade-off。
- Evidence: 文档强调收益，限制条件与反例不足。
- Recommended fix: 增加“何时开启/关闭 unit_flag”决策表。

## Naming and Structure
- 目录命名准确。
- 文档中的路径命名与真实结构未严格对齐。

## Documentation Review
- 内容丰富，但执行可用性细节有缺陷。

## Build-System Review
- 与 n_buffer 同类严重问题，需要仓库级一次性治理。

## Code Quality Review
- 代码技术深度足够。
- 叙事上仍偏“内部经验记录”，欠缺公共样例的边界清晰度。
- Code taste：思路好，工程纪律不足。

## Action Plan
- Must fix now
  - 修正 CMake 全局变量写法与 README 路径错误。
- Should fix soon
  - 补充 unit_flag 开关策略说明。
- Nice to improve later
  - 增加最小复现输入与自动化检查脚本。
