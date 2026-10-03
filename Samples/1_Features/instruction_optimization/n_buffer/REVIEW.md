# Review

## Summary
- 该样例有较强技术深度，但当前工程组织不符合高质量 samples 仓库的“可复制、可维护、可教学”标准。
- 文档非常长，但代码复用策略和构建规范相对薄弱，形成“讲得多、工程约束弱”的反差。

## Strengths
- 对 N-Buffer 原理与时序解释较完整。
- 具备前后版本对照，读者可看到优化意图。
- 示例覆盖 host + kernel + profiling 视角，信息维度丰富。

## Key Issues
### 1) CMake 覆写全局编译器与标准设置
- Severity: Critical
- Why it matters: 样例子目录不应修改全局工具链；这会污染父工程并破坏可组合性。
- Evidence: 子目录 `CMakeLists.txt` 中设置 `CMAKE_C_COMPILER/CMAKE_CXX_COMPILER/CMAKE_LINKER/CMAKE_CXX_STANDARD`。
- Recommended fix: 删除全局 `set(...)`，在顶层统一配置；子目录仅做 target 级声明。

### 2) 重复实现导致演进风险
- Severity: Important
- Why it matters: 与 `unit_flag` 样例存在大量重复 host/kernel 框架，后续修复会出现漂移。
- Evidence: 两个样例均复制完整 Matmul 主体，仅少量参数差异。
- Recommended fix: 提取共享基础层（common host harness + policy headers），示例仅表达差异化优化点。

### 3) 文档可执行性与可验证性不足
- Severity: Important
- Why it matters: 长文档并不等于高教学价值；缺少“最小可复现实验矩阵”和成功判定标准。
- Evidence: README 重点在概念描述，较少给出可自动化验证的步骤。
- Recommended fix: 增加 `quick run`、`expected metrics`、`regression checklist` 三段。

## Naming and Structure
- `n_buffer` 命名合理，但目录中缺少 `common` 导致同类样例复用困难。
- 文件 `main.cpp` 仍然过载，不利于面向主题学习。

## Documentation Review
- 叙述丰富但冗长；缺少章节级“要点总结/不适用场景/失败信号”。
- 建议精简原理图文并增加“读者应该修改哪些参数进行实验”。

## Build-System Review
- 直接链接系统库清单（m/dl/platform/...）可用但不优雅。
- 缺少对依赖来源的封装，用户难以判断哪些依赖是样例必须，哪些是历史遗留。

## Code Quality Review
- 核心实现质量尚可，但架构上“复制粘贴式演进”明显。
- 注释偏解释性，缺少接口契约（输入约束、输出不变量）。

## Action Plan
- Must fix now
  - 移除子目录全局 CMake 覆写。
  - 抽离与 unit_flag 共用代码，避免双份维护。
- Should fix soon
  - README 增加可验证的性能对照模板。
  - 按职责拆分 main.cpp。
- Nice to improve later
  - 提供参数扫描脚本，形成教学闭环。
