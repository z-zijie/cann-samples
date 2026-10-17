# Review

## Summary
- 该样例具备真实通信场景价值，但代码工程质量不稳定，文档深度不够，尚未达到高标准公共样例水位。
- 主要问题在于：工具层质量、错误恢复策略、多进程实验契约不完整。

## Strengths
- 覆盖 dispatch+combine 端到端流程。
- 提供数据生成与校验脚本，具备基本复现路径。
- CMake 对 shmem 依赖显式声明，构建入口明确。

## Key Issues
### 1) 工具头存在基础 API 使用缺陷
- Severity: Critical
- Why it matters: `WriteFile` 中 `if (!fd)` 判定错误，可能漏报失败；公共样例不应传播此类基础错误。
- Evidence: `src/utils.h`。
- Recommended fix: 改为 `fd < 0`，并在写失败时返回 false 而非仅打印。

### 2) 宏日志与错误宏设计不一致
- Severity: Important
- Why it matters: `ACL_CHECK` 仅打印不终止，而 `ACL_CHECK_WITH_RET` 会返回，语义不一致导致流程脆弱。
- Evidence: `utils.h` 同时定义两类行为不同的检查宏。
- Recommended fix: 统一错误策略（失败即返回/抛错），避免 silent failure。

### 3) README 缺乏关键运行前提
- Severity: Important
- Why it matters: 多 rank 场景对设备数、端口冲突、目录布局高度敏感；文档未明确失败排查。
- Evidence: README 仅给基本命令，缺少前提检查清单。
- Recommended fix: 增加“环境前置检查 + 常见报错 + 结果目录说明”。

## Naming and Structure
- `dispatch_and_combine_final.cpp` 名称透露“final”但缺少阶段上下文。
- 建议拆分为 launcher/dispatch_runner/combine_runner。

## Documentation Review
- 运行说明可用，但教学维度不足。
- 缺少输入输出 tensor 规格表、精度标准、性能指标定义。

## Build-System Review
- target 组织清晰。
- 可进一步引入 sample-level options（如启用 profiling）避免改源码。

## Code Quality Review
- 功能链路完整，但函数体过长（`runDispatchAndCombine`）。
- 资源生命周期管理虽有封装函数，但失败路径仍不够系统。

## Action Plan
- Must fix now
  - 修复 `utils.h` 的 fd 判定与错误返回语义。
  - 统一 ACL 错误处理策略。
- Should fix soon
  - 拆分超长函数，分层组织 dispatch/combine 逻辑。
  - README 补全多进程运行前提与排查手册。
- Nice to improve later
  - 引入可复现性能 profile 脚本。
