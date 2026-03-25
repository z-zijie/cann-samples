# Review

## Summary
- 这是仓库中最有潜力的“体系化样例”，但当前完成度不均衡：框架雄心很强，教程落地却不完整。
- 若定位为公共标杆样例，当前仍未达标。

## Strengths
- 目录分层（common/docs/recipes/tutorials）方向正确。
- CMake 已引入函数化封装（`configure_matmul_recipe_target`），优于多数子样例。
- 覆盖 recipe 与 tutorial 两种学习路径，设计思路先进。

## Key Issues
### 1) 教程链路不完整（文档明确“待补充”）
- Severity: Critical
- Why it matters: 样例承诺的学习路径未闭环，会严重影响可信度。
- Evidence: `README.md` 的“分步教程”仅写“待补充”。
- Recommended fix: 至少补齐每个 tutorial 的目标、输入、预期收益与阅读顺序。

### 2) 子目录中设置编译器与架构硬编码
- Severity: Important
- Why it matters: 降低可移植性，且与仓库级构建职责冲突。
- Evidence: `CMakeLists.txt` 中设置 `CMAKE_*_COMPILER` 与 `--npu-arch=dav-3510`。
- Recommended fix: 改为 toolchain/缓存变量可配置，默认值放文档。

### 3) 命名与章节语义不完全一致
- Severity: Minor
- Why it matters: 读者会困惑 recipe/tutorial 的边界与关系。
- Evidence: 部分子目录仅代码缺说明，部分说明缺对应代码。
- Recommended fix: 建立统一 `README` 模板并要求每目录具备“目的/输入/输出/限制”。

## Naming and Structure
- 顶层结构好，但局部一致性不够。
- 建议统一教程目录编号连续性与命名规则（为何缺 1_*）。

## Documentation Review
- 高层介绍优秀；但落地文档覆盖不完整。

## Build-System Review
- 比其他样例更好，但仍存在硬编码与全局变量问题。

## Code Quality Review
- 代码体现较高工程能力。
- 教学可读性依赖文档补全，否则难形成稳定学习路径。
- Code taste：框架感强，执行细节尚未收束。

## Action Plan
- Must fix now
  - 补齐教程文档闭环。
- Should fix soon
  - 去除编译器/架构硬编码。
- Nice to improve later
  - 建立统一目录 README 模板并自动校验。
