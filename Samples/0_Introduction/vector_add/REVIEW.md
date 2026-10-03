# Review

## Summary
- 该样例可运行、入口清晰，但作为“公共 samples”仍偏工程草稿风格，尚未达到“可被模仿”的高标准。
- 主要问题在于正确性验证过于脆弱（浮点精确比较）与 host/kernels 逻辑耦合在单文件，教学层次不够清晰。

## Strengths
- 单文件可读性尚可，初学者容易跟踪执行路径。
- 使用 RAII 管理 device memory（`std::unique_ptr` + deleter），避免了显式 free 漏洞。
- README 给出最小编译与运行命令，入门门槛低。

## Key Issues
### 1) 浮点结果校验使用 `==`
- Severity: Important
- Why it matters: 这会向读者传递错误范式；在跨编译器/平台环境下容易误报，样例可信度下降。
- Evidence: `main.cpp` 中逐元素比较 `h_C[i] != h_A[i] + h_B[i]`。
- Recommended fix: 使用相对/绝对误差混合比较，并在 README 明确误差阈值和判定规则。

### 2) 教学边界不清晰：host 管理、tiling、kernel 全挤在一个文件
- Severity: Important
- Why it matters: 作为样例，应该让“算法核心”与“运行时样板”分层，便于迁移和复用。
- Evidence: `main.cpp` 同时承担 ACL 初始化、数据生成、tiling 计算、kernel 实现、验证。
- Recommended fix: 拆分 `host_runner.*`、`kernel.*`、`validation.*`，并在 README 对应映射。

### 3) 错误处理宏控制流隐式
- Severity: Minor
- Why it matters: `CHECK_ACL` 直接 `return 1`，在不同函数上下文中可读性和可维护性较差。
- Evidence: 宏定义携带输出与 return。
- Recommended fix: 用 `Status`/`expected` 风格显式返回，宏仅做日志封装。

## Naming and Structure
- 目录命名准确，但 `main.cpp` 过载，缺少“样例结构化分层”示范价值。
- 目标名 `vector_add` 清晰，建议补充 `vector_add_host` / `vector_add_kernel` 文件级命名。

## Documentation Review
- README 的运行预期清楚，但缺失输入规模、误差容忍、性能预期与限制说明。
- 建议补充“该样例只演示基础数据通路，不代表最优性能实现”。

## Build-System Review
- CMake 基本采用 target 级配置，整体可接受。
- 但编译选项与链接库重复模式在多个样例中反复出现，缺少仓库级复用函数。

## Code Quality Review
- 可读性：中等；局部注释充足但偏“叙述执行步骤”，不是“解释设计决策”。
- 架构：单文件高耦合。
- 生命周期/健壮性：内存释放做得较好，但数值比较策略不专业。
- 样例可教性：入门友好，但会教会不好的数值验证习惯。
- Code taste：可运行但不够“克制与精炼”。

## Action Plan
- Must fix now
  - 将浮点验证改为容差比较并写入 README。
- Should fix soon
  - 分离 host/kernel/validation 文件，降低认知负担。
- Nice to improve later
  - 增加参数化命令行与基准性能输出。
