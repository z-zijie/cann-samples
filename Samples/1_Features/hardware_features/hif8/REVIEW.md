# Review

## Summary
- 样例方向有代表性，但当前代码质量与文档严谨性不足以成为“高标准公开样例”。
- 主要问题是接口安全性、宏风格、输入输出契约和脚本鲁棒性。

## Strengths
- 主题聚焦明确：展示 hifloat8 量化流程。
- 提供了数据生成与结果校验脚本，具备基础闭环。
- Kernel 结构（CopyIn/Compute/CopyOut）清晰。

## Key Issues
### 1) 文件工具实现存在系统调用语义错误风险
- Severity: Critical
- Why it matters: `WriteFile` 中 `open` 失败判定使用 `if (!fd)`，对 `fd==0` 场景错误；这是公开样例不应出现的基础缺陷。
- Evidence: `data_utils.h` 的 `WriteFile`。
- Recommended fix: 使用 `if (fd < 0)`；并改为 RAII 文件封装避免泄漏。

### 2) 宏式日志与无命名空间工具污染
- Severity: Important
- Why it matters: `#define ERROR_LOG` 与头文件内自由函数易产生 ODR/命名冲突。
- Evidence: `data_utils.h` 在头文件中定义宏与非 `inline` 风格工具。
- Recommended fix: 引入 `namespace sample::io` + `inline` 函数；避免可变参数宏日志。

### 3) 文档缺少异常与边界说明
- Severity: Important
- Why it matters: 样例默认 `scale` 可能含零值，README 未说明行为；验证脚本也缺少输入合法性检查。
- Evidence: `gen_data.py` 随机 `[-10,10]` 可能产生接近0分母。
- Recommended fix: 生成数据时约束 `|scale|>eps`，文档说明量化异常处理策略。

## Naming and Structure
- 目录组织简洁。
- 建议把 `data_utils.h` 下沉为 `utils/io.hpp` 并添加单元级复用边界。

## Documentation Review
- README 信息完整，但缺少“预期误差分布”和“失败定位”。
- 建议增加样例输入规格与数值稳定性注意事项。

## Build-System Review
- CMake 基本规范。
- 仍存在 `-w`，且未对 python 脚本依赖（`en_dtypes`）给出版本约束。

## Code Quality Review
- kernel 代码可读性尚可。
- host 端错误处理非常薄弱，工具层实现不够严谨。

## Action Plan
- Must fix now
  - 修复 `WriteFile` 的 fd 判断错误。
  - 为除法输入 scale 引入稳定性约束。
- Should fix soon
  - 清理宏日志与头文件污染。
  - README 加入异常处理和误差说明。
- Nice to improve later
  - 增加更贴近真实模型分布的数据生成模式。
