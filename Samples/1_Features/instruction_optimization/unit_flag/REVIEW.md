# Review

## Summary
- 样例能展示 UnitFlag 优化价值，但工程结构与文档精度未达到“世界级 samples 仓库”应有的一致性。
- 主要问题集中在：构建系统污染、路径文档错误、代码重复与契约表达不足。

## Strengths
- 对 UnitFlag 机制的时序解释有实操价值。
- 提供优化前后对比图，易于直观看到收益。
- 示例包含运行、编译、性能解释三个维度。

## Key Issues
### 1) 构建系统污染父工程
- Severity: Critical
- Why it matters: 子目录设置全局编译器/标准会破坏仓库可组合性，是 samples 仓库的高风险反模式。
- Evidence: 子 `CMakeLists.txt` 中存在全局 `set(CMAKE_...)`。
- Recommended fix: 仅保留 target 级配置，统一工具链在顶层处理。

### 2) README 运行路径与实际安装路径不一致
- Severity: Important
- Why it matters: 样例文档必须“拷贝即跑”；路径错误会直接损害可信度。
- Evidence: README 写 `build/Samples/1_Features/unit_flag/`，但 install 目标在 `instruction_optimization/unit_flag`。
- Recommended fix: 统一 README 与 CMake 安装/构建路径，并在 CI 中校验文档命令。

### 3) 与 n_buffer 重复度高
- Severity: Important
- Why it matters: 复制式样例演进会让读者难以提炼核心差异，也增加维护成本。
- Evidence: 主体框架高度相似，仅 UnitFlag 参数与同步点不同。
- Recommended fix: 抽象共同框架，保留“差异 patch”式代码展示。

## Naming and Structure
- `unit_flag` 命名准确。
- 但代码组织未突出“仅展示 UnitFlag 差异”的教学目标，结构可读性一般。

## Documentation Review
- 文档信息量高，但章节之间重复叙述较多。
- 缺少“何时不该启用 UnitFlag”的反例说明，容易被误用。

## Build-System Review
- 与 `n_buffer` 同样存在全局变量污染。
- 依赖直接硬编码，不利于未来切换到统一 `cann_samples::*` 目标体系。

## Code Quality Review
- 逻辑本身清晰，但接口约束（shape、对齐、尾块行为）未显式契约化。
- 教学上应更强调“为什么删掉某同步点仍正确”。

## Action Plan
- Must fix now
  - 修正文档路径错误。
  - 去除 CMake 全局设置。
- Should fix soon
  - 抽离共用框架，突出 UnitFlag 最小差异实现。
  - 增加禁用场景与边界条件文档。
- Nice to improve later
  - 增加自动化性能回归脚本（前后版本基线对比）。
