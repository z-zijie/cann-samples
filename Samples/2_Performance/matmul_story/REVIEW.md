# Review

## Summary
- 这是仓库中最有潜力的“旗舰样例”，但当前成熟度不均：文档丰富、代码容量大、组织仍有历史包袱。
- 其主要挑战是统一教程/配方/构建三套系统的工程边界。

## Strengths
- 覆盖教程（tutorial）与配方（recipes）双路径，教育价值高。
- 顶层 CMake 已显著现代化，封装了共享函数。
- 提供了性能文档与图示，具备系统学习路径。

## Key Issues
### 1) 教程体系完整性断裂
- Severity: Important
- Why it matters: README 声称分步教程，但 Step 1 仍为占位，影响叙事连续性与可信度。
- Evidence: `matmul_tutorials/README.md` 中 Step 1 标注待补充。
- Recommended fix: 补齐 Step 1 最小可运行实现，或明确标注 roadmap 状态并隐藏未完成章节。

### 2) 子目录 CMake 与顶层 CMake 双轨并存
- Severity: Important
- Why it matters: `matmul_tutorials/0_naive/CMakeLists.txt` 与顶层同名目标并存，易造成重复定义/维护分叉。
- Evidence: 顶层已 add tutorial target，同时 0_naive 下仍保留独立 CMake。
- Recommended fix: 选一种构建入口；推荐仅保留顶层统一构建，子目录改为文档说明。

### 3) Host 样板重复显著
- Severity: Important
- Why it matters: 多个 tutorial cpp 的参数解析、ACL 生命周期、IO 路径处理高度重复，演进成本高。
- Evidence: `0_naive/2_block_swat/3_last_round_tile_balance/4_unit_flag` 的 main 框架近似。
- Recommended fix: 提取通用 runner + step-specific kernel adapter。

## Naming and Structure
- 总体结构较专业。
- 但 `common` 的边界仍模糊（host_utils 与 kernel_utils 并列但无清晰依赖方向说明）。

## Documentation Review
- 文档量大、信息密度高。
- 仍需减少占位内容，增强“本仓当前可执行范围”的真实性表达。

## Build-System Review
- 顶层 CMake 是加分项，函数化较好。
- 仍应避免重复目标定义与潜在多入口漂移。

## Code Quality Review
- 核心代码专业度较高，但样例化封装不足（重复 host harness）。
- 如果目标是公共样例标杆，必须更明显地区分“共性平台代码”与“本步优化差异”。

## Action Plan
- Must fix now
  - 清理双轨 CMake，统一构建入口。
  - 处理 Step 1 占位问题（补齐或下线）。
- Should fix soon
  - 提取教程共享 host harness。
  - 明确 common 模块依赖方向文档。
- Nice to improve later
  - 增加自动化对比报告（各 step 性能/精度表）。
