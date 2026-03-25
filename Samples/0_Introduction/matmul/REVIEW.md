# Review

## Summary
- 该样例技术含量高，但“样例可教学性”明显低于代码复杂度，当前更像内部实验实现而非教学级公开 sample。
- 关键问题是：单文件过载、抽象边界模糊、CMake 和 README 未对复杂度进行降噪。

## Strengths
- 覆盖了较完整的 AscendC Matmul 数据通路，具备真实工程价值。
- 提供 CPU golden 校验，具备基础正确性保障。
- 对尾块处理、tile 组织有较系统实现。

## Key Issues
### 1) 单文件承担过多职责
- Severity: Critical
- Why it matters: 公共样例首要目标是“可学习”；当前 `main.cpp` 混合了 kernel、tiling、数据生成、校验、运行时管理，阅读路径严重拥塞。
- Evidence: 文件同时定义 `tool` 工具、`AscendC::Te` 类型策略、`matmul::MatmulKernel` 与 host 运行逻辑。
- Recommended fix: 至少拆成 `kernel_matmul.hpp`、`tiling_plan.hpp`、`host_driver.cpp`、`golden_check.hpp`。

### 2) “教学注释”与“实现复杂度”失衡
- Severity: Important
- Why it matters: 大量注释在解释 API 行为，但没有解释关键 trade-off（如 baseM/baseN/baseK 的来源），用户无法迁移到新形状。
- Evidence: 常量直接硬编码，缺少“为什么是 256/1024”的建模说明。
- Recommended fix: 在 README 增加性能模型与参数推导；代码内将经验值收敛为 `TilingPolicy`。

### 3) CMake 依赖表达不够收敛
- Severity: Important
- Why it matters: `target_include_directories` 直接引用深层 `third_party/tensor_api` 路径，暴露目录耦合。
- Evidence: `CMakeLists.txt` 使用相对路径跨三级目录引入第三方头。
- Recommended fix: 通过仓库级 INTERFACE target 暴露 include path，样例只链接逻辑目标。

## Naming and Structure
- `main.cpp` 文件名无法反映内容复杂度（实际为“完整 matmul sample implementation”）。
- 目录层次简单，但未体现“kernel/host/doc”分层，阻碍贡献者扩展。

## Documentation Review
- README 能跑通，但缺少“输入约束、误差模型、性能预期区间”。
- 命令示例里可执行文件调用缺 `./`，且未说明默认 dtype 路径与内存需求。

## Build-System Review
- target 级编译选项是优点。
- 但禁用告警 (`-w`) 与硬编码架构参数放在样例层，会降低可移植性和可复用性。

## Code Quality Review
- 控制流可跟踪，但函数边界不干净，局部变量规模过大。
- 可维护性风险主要来自“高度内联 + 配置散落 + 缺少策略对象”。
- 样例气质偏“能跑的复杂实现”，离“可仿写模板”仍有距离。

## Action Plan
- Must fix now
  - 结构化拆分代码文件，降低单文件复杂度。
  - 将 tiling 常量配置化并给出推导依据。
- Should fix soon
  - 引入仓库级 tensor_api 目标，去除相对 include 路径耦合。
  - README 补齐输入限制、误差标准、内存占用说明。
- Nice to improve later
  - 添加“从 naive 到优化版”的对照实现，提升教学价值。
