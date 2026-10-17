# Review

## Summary
- 技术深度高，但样例组织更接近“中型工程代码投放”而非“可学习 sample”。
- 当前不满足公共样例的可读性与教学分层标准。

## Strengths
- 代码覆盖面广，包含 common / include / example 的相对完整工程骨架。
- README 提供了较完整算子语义和参数定义。
- 样例入口代码有注释分段，执行流程可追踪。

## Key Issues
### 1) 复杂度远超样例承载能力，缺少分层教学入口
- Severity: Critical
- Why it matters: 读者会在海量头文件与实现细节中迷失，无法提炼“核心技巧”。
- Evidence: `include/` 下模块众多（block/kernel/memcopy/vf），README 未给“必读最小路径”。
- Recommended fix: 提供 `minimal_path.md`（10个关键文件），并标记可跳过模块。

### 2) 运行依赖与数据约定不够显式
- Severity: Important
- Why it matters: 数据文件名和布局耦合强，稍有偏差即失败。
- Evidence: 示例程序硬编码读取 `./input/input_*.bin`。
- Recommended fix: 提供输入 manifest、shape 校验、错误提示和一键生成脚本。

### 3) CMake 目标配置可维护性一般
- Severity: Important
- Why it matters: 大样例更需要稳定构建抽象，否则扩展成本很高。
- Evidence: include 路径与选项较多，缺少模块化 target 划分。
- Recommended fix: 将 common/include 按模块拆分成 interface/object targets。

## Naming and Structure
- `*_story` 命名表达了叙事意图，但目录内部更像 SDK 内核实现。
- 建议补 `docs/architecture.md` 解释目录职责与依赖方向。

## Documentation Review
- 参数文档充分，但“如何开始阅读代码”几乎缺失。
- 缺少 quickstart（从 clone 到跑通）与失败排查。

## Build-System Review
- 可构建，但可解释性差；需要模块化 target 和公共宏减少重复。

## Code Quality Review
- 技术实现能力强。
- 结构上缺乏样例化裁剪，教学信号被噪声淹没。
- Code taste：工程深度高，sample 呈现不克制。

## Action Plan
- Must fix now
  - 提供最小学习路径和一键可复现输入生成。
- Should fix soon
  - 重构 CMake 为模块化 targets。
- Nice to improve later
  - 增加简化版 micro-kernel 对照样例。
