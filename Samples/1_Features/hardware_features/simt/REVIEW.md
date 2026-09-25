# Review

## Summary
- 样例主题有价值，但当前实现对新读者不够友好，且错误处理与参数边界表达不达标。
- 该样例更像“技术草图”，尚未达到可示范的公共样例质量。

## Strengths
- README 对 SIMT 概念和硬件背景解释详细。
- 提供了完整 gather 示例，主题聚焦明确。
- CMake 目标简洁，构建入口直接。

## Key Issues
### 1) 错误处理不可靠
- Severity: Important
- Why it matters: `CHECK_ACL` 仅打印不终止，后续继续执行可能放大故障并污染调试信息。
- Evidence: `CHECK_ACL` 函数不返回状态，不中断控制流。
- Recommended fix: 统一返回 `bool`/`expected` 风格，失败立刻清理并退出。

### 2) 关键硬件参数硬编码
- Severity: Important
- Why it matters: 样例会被复制到不同芯片环境；`maxCoreNum=4` 这种硬编码会误导用户。
- Evidence: `main.cpp` 中直接写 “just an example” 的固定核数。
- Recommended fix: 通过平台 API 查询核数，并在 README 解释默认策略。

### 3) 文档与代码契约未打通
- Severity: Minor
- Why it matters: README 讲了很多模型概念，但缺乏与 `main.cpp` 参数（shape/dim/threadNum）的映射。
- Evidence: 文档主要是原理，较少有“对应代码行如何修改”。
- Recommended fix: 增加“最小实验配置表”，将理论点映射到代码参数。

## Naming and Structure
- 目录名合理，`main.cpp` 过于泛化。
- 建议拆为 `gather_kernel.hpp` 与 `host_demo.cpp`。

## Documentation Review
- 讲解质量高，但篇幅很长，操作路径不突出。
- 建议前置 30 秒快速运行段落，再给深度原理。

## Build-System Review
- target 级配置清晰。
- 链接库显式列出虽可用，但缺少解释来源；建议复用统一仓库目标。

## Code Quality Review
- kernel 逻辑直白，但 host 层鲁棒性和资源生命周期管理较弱。
- 类型别名使用 `typedef` 可读性一般，建议使用 `using`。

## Action Plan
- Must fix now
  - 让 ACL 错误处理真正中断流程并做清理。
  - 去掉硬编码 core 数。
- Should fix soon
  - 拆分 host/kernel 文件。
  - README 增加参数到代码映射表。
- Nice to improve later
  - 增加多 shape 示例和边界 case。
