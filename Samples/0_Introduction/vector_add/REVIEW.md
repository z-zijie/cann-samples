# Review

## Summary
- 该样例可运行、目标明确，但离“可作为公共样例仓模板”的质量标准仍有明显差距。
- 主要短板是：Host 侧职责过重、错误处理风格不一致、文档与代码之间缺乏可验证参数约束。

## Strengths
- 核函数主体较短，Vector 数据通路（CopyIn/Compute/CopyOut）可读性尚可。
- 给出了 CPU 侧对照验证，具备基本正确性闭环。
- CMake 目标命名与安装路径一致，使用者容易定位可执行文件。

## Key Issues
### 1) Host 侧与 Kernel 配置耦合过深
- Severity: Important
- Why it matters: 教学样例应清晰分离“算法逻辑”和“设备资源配置”；当前 `calc_tiling_params` 将平台内存、核数、切分策略混在一起，后续扩展 dtype 或 shape 时可维护性差。
- Evidence: `main.cpp` 中 `calc_tiling_params` 同时读取 UB、大核数并直接返回 launch 参数。
- Recommended fix: 拆分为 `query_device_caps()` 与 `derive_launch_plan()` 两层，并在 README 明确每层输入/输出含义。

### 2) 错误处理模型过于脆弱
- Severity: Important
- Why it matters: 样例应示范强健实践；`CHECK_ACL` 仅 `return 1`，且资源释放路径依赖调用顺序，容易形成未来代码复制时的泄漏模板。
- Evidence: 宏内直接 `return 1`；多处资源申请后无统一清理路径。
- Recommended fix: 采用 RAII 包装 ACL 资源（stream/device malloc）；保留错误码但避免宏级 `return`。

### 3) README 可执行性细节不足
- Severity: Minor
- Why it matters: 样例仓要“照着跑就通”；README 未说明默认输入规模、误差阈值、失败时定位步骤。
- Evidence: 仅给出成功/失败输出文案，缺少参数边界与误差标准。
- Recommended fix: 增加“默认参数/误差阈值/常见失败原因”三段。

## Naming and Structure
- 目录命名直观，但 `main.cpp` 承担 kernel+host+校验三重职责，作为教学样例过于集中。
- 建议拆分为 `kernel_vector_add.hpp`、`host_runner.cpp`、`reference.cpp`。

## Documentation Review
- 文档结构清楚，但中文术语与代码变量命名对应关系不足（如 tileSize、blockLength 未在 README 定义语义）。
- 建议在“参数说明”加入与代码同名字段映射表。

## Build-System Review
- CMake 用法基本现代化（target 级设置）。
- 但 `-w` 全局静默告警不适合作为公共样例默认；会向用户传达“忽略告警”错误习惯。

## Code Quality Review
- 控制流基本直白，但仍有“宏式错误处理 + 长函数”风格负担。
- 常量定义较分散（kernel 与 host 两侧重复概念），可提炼成统一配置结构。
- 作为教学样例，注释偏“解释 API”而非“解释设计决策”。

## Action Plan
- Must fix now
  - 移除 `-w`，保留至少 `-Wall/-Wextra` 等可见告警策略。
  - 将 ACL 资源管理改成 RAII，消除宏内 `return`。
- Should fix soon
  - 拆分 host/kernel/reference 文件，降低单文件认知负担。
  - README 补齐参数语义、误差阈值、失败排查。
- Nice to improve later
  - 增加“为什么这样切 tile”的设计说明与替代策略。
