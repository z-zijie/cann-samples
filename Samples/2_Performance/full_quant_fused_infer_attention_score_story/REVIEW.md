# Review

## Summary
- 样例技术深度高，但在“公开样例可维护性”上存在明显硬伤，当前更像内部移植代码。
- 最突出的问题是 CMake 可维护性和文档可执行闭环不完整。

## Strengths
- 提供了完整 FIA kernel 入口与 host 启动模板。
- 示例注释较多，帮助理解大体流程。
- 包含数据生成与验证脚本安装路径意图。

## Key Issues
### 1) CMake 存在明显语法/质量缺陷
- Severity: Critical
- Why it matters: build 脚本是样例第一入口；`install(FILES ... DESTINATION DESTINATION ...)` 显示基础审查缺失。
- Evidence: `CMakeLists.txt` 的 install 语句重复 `DESTINATION`。
- Recommended fix: 修正 install 语法并加入最小 CMake 配置测试（configure + install dry-run）。

### 2) 依赖 include 路径过度硬编码
- Severity: Important
- Why it matters: 大量 `${ASCEND_HOME}/...` 直写路径使样例高度脆弱，且难迁移到不同环境布局。
- Evidence: `COMMON_INCLUDE_DIRS` 包含多层硬编码路径。
- Recommended fix: 封装为仓库级 imported target 或 find package 模块，样例仅 link target。

### 3) README 偏接口堆叠，缺少“样例执行主线”
- Severity: Important
- Why it matters: 公开样例应先给 quick run，再给参数百科；当前学习路径反向。
- Evidence: README 大量参数表，但缺乏简洁 runbook。
- Recommended fix: 前置最小可运行步骤、输入下载/生成、预期输出与失败排查。

## Naming and Structure
- 目录层次完整但体量巨大，缺少“核心最小路径”标识。
- 建议增加 `docs/quickstart.md` 与 `docs/deep_dive.md` 分层。

## Documentation Review
- 信息全面但不够教学导向。
- 应增加对 shape、dtype、资源占用的明确限制与默认值说明。

## Build-System Review
- 当前是本样例最大短板：可读性差、脆弱、可移植性低。
- 建议重构为 target-based + minimal include interface。

## Code Quality Review
- host 模板注释较丰富。
- 但仍有超长 main 与硬编码 shape/workspace 问题，不利于复用。

## Action Plan
- Must fix now
  - 修复 CMake install 语法缺陷。
  - 清理硬编码 include 路径。
- Should fix soon
  - 提供真正的 quick run 文档。
  - 将 main 拆分为 setup/load/launch/verify 模块。
- Nice to improve later
  - 提供参数化 CLI，替代源码内硬编码 shape。
