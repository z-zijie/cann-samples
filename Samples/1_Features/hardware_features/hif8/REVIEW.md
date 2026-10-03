# Review

## Summary
- 样例选题有价值，但当前“工程可验证性”强于“教学可读性”，仍偏内部演示脚本风格。
- 质量处于可用水平，但未达到公共样例应有的结构严谨度。

## Strengths
- README 给出数据生成/执行/校验闭环。
- 引入论文链接，技术来源可追溯。
- 覆盖 hifloat8 数据流（GM/UB/Cast）的最关键接口。

## Key Issues
### 1) 文档与代码叙事层次不一致
- Severity: Important
- Why it matters: README 以概念介绍为主，但对本样例“具体实现边界”描述不足。
- Evidence: README 大段格式介绍，缺少对 `quantize_custom.cpp` 结构图解与关键函数定位。
- Recommended fix: 增加“文件导读 + 执行流程图 + 关键函数索引”。

### 2) CMake 与仓库公共模式重复
- Severity: Minor
- Why it matters: 重复模板会导致后续维护时配置漂移。
- Evidence: 样例内重复 compile/link 选项模式。
- Recommended fix: 统一复用仓库级 `add_ascend_sample()` helper。

### 3) 输入规格硬编码且缺少边界说明
- Severity: Minor
- Why it matters: 样例会被当作参考实现；缺乏范围说明易造成误用。
- Evidence: README 规格固定 `1*2048`，未说明可扩展策略。
- Recommended fix: 标注“当前仅验证固定 shape”的原因与扩展步骤。

## Naming and Structure
- `hif8/` 命名准确。
- 建议将脚本目录与源码目录职责明确拆分（`scripts/`, `src/`, `include/`）。

## Documentation Review
- 资料翔实，但“样例导向”不足：缺少 quick mental model 和失败排查指南。

## Build-System Review
- 基本规范；可进一步减少重复并统一 target 属性管理。

## Code Quality Review
- 可读性：中等。
- 架构：演示目标明确，但工程边界不够清楚。
- 健壮性：依赖外部脚本，运行前置条件较多。
- 可教性：中等偏上。
- Code taste：技术正确，但表达组织尚可打磨。

## Action Plan
- Must fix now
  - README 增加实现结构导读。
- Should fix soon
  - 统一 CMake 复用模式。
- Nice to improve later
  - 增加参数化 shape 示例。
