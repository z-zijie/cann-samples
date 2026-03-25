# Review

## Summary
- 样例主题明确，但 CMake 写法存在明显反模式（子目录重设全局编译器），这在公共样例中不可接受。
- 文档偏“结果展示”，对设计约束和可迁移边界讲解不足。

## Strengths
- 与优化主题（n-buffer）直接相关，性能导向明确。
- 有图示和前后对比材料。
- 包含可执行示例，具备基本验证路径。

## Key Issues
### 1) 子目录 CMake 修改编译器与全局变量
- Severity: Critical
- Why it matters: 会污染上层项目配置，破坏可组合性；这是现代 CMake 明确应避免的做法。
- Evidence: `CMakeLists.txt` 中 `set(CMAKE_C_COMPILER ...)` / `set(CMAKE_CXX_COMPILER ...)` / `set(CMAKE_LINKER ...)`。
- Recommended fix: 在顶层 toolchain 决定编译器；样例内仅做 target 级属性设置。

### 2) 通过深层相对路径引用 third_party
- Severity: Important
- Why it matters: 目录结构一旦调整即失效，样例可移植性差。
- Evidence: `../../../../third_party/tensor_api` 形式。
- Recommended fix: 在顶层导出 `TENSOR_API_ROOT` 或 interface target。

### 3) README 缺少“何时不该用 n-buffer”
- Severity: Minor
- Why it matters: 样例应教授边界条件，而非只给正向案例。
- Evidence: 文档主要讲收益，限制条件不够系统。
- Recommended fix: 增加适用前提/反例章节。

## Naming and Structure
- `n_buffer` 命名可读。
- 建议把性能图、代码、运行脚本分层目录化，避免 README 过载。

## Documentation Review
- 有优化说明，但复现实验细节不够（输入、profiling配置、判定标准）。

## Build-System Review
- 存在严重现代 CMake 反模式，需优先整改。

## Code Quality Review
- 代码具备工程深度，但表达组织较拥挤。
- 样例教学性受构建与文档质量拖累。
- Code taste：技术上强，工程规范上不达标。

## Action Plan
- Must fix now
  - 删除子目录中的全局编译器设置。
- Should fix soon
  - 统一 include 路径来源，移除深相对路径。
- Nice to improve later
  - 增加“适用性边界”章节。
