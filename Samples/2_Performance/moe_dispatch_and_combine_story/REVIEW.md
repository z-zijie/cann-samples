# Review

## Summary
- 样例具备可运行闭环（构建/生成数据/运行/校验），基础质量好于多数性能样例。
- 但仍存在样例仓库常见问题：边界条件文档不足、结构可扩展性一般。

## Strengths
- README 给出完整执行路径，用户能快速跑通。
- 提供数据生成与结果校验脚本，复现链路相对完整。
- 源码与 include 分离，基本结构清晰。

## Key Issues
### 1) 文档缺少参数语义与范围说明
- Severity: Important
- Why it matters: `chip-num-per-server`、`bs` 等参数缺少约束会导致误用。
- Evidence: README 只给示例命令，未给参数边界。
- Recommended fix: 增加参数表（类型、范围、默认值、失败行为）。

### 2) 可观测性不足
- Severity: Minor
- Why it matters: 性能样例应鼓励可测量与可诊断。
- Evidence: README 无 profiling 指南、无关键指标目标值。
- Recommended fix: 增加 msprof 命令与判读模板。

### 3) CMake 可进一步模块化
- Severity: Minor
- Why it matters: 随着子模块增长，单 target 配置会变得拥挤。
- Evidence: 当前目标直接链接/包含多个路径。
- Recommended fix: 引入 `moe_dispatch_common` target 复用公共依赖。

## Naming and Structure
- 目录命名准确，`include/` 与 `src/` 分离合理。
- 建议 `src/dispatch_and_combine_final.cpp` 改为阶段化命名，避免“final”语义漂移。

## Documentation Review
- 可执行性不错，但教学深度与参数解释不足。

## Build-System Review
- 可用但可扩展性一般；建议逐步 target 模块化。

## Code Quality Review
- 结构较清晰。
- 样例可教性中等，需补充性能观测和参数约束信息。
- Code taste：稳健，但尚未“标杆化”。

## Action Plan
- Must fix now
  - 补充参数边界与错误行为说明。
- Should fix soon
  - 增加 profiling 与指标解释。
- Nice to improve later
  - 重命名 `final` 文件并引入阶段对照。
