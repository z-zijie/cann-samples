# Review

## Summary
- 这是当前仓库中教学潜力较高的样例之一，但仍存在样例工程化不足（重复代码、错误处理缺失、性能结论缺乏实验边界）。
- 若要达到“可被外部团队直接仿写”的标准，还需结构化收敛。

## Strengths
- 主题清晰：对比 `with_vf` 与 `without_vf` 的价值直观。
- README 提供了较完整的快速开始和性能对比。
- 代码中 VF 关键路径相对集中，便于定位核心优化点。

## Key Issues
### 1) 大量重复 host 逻辑
- Severity: Important
- Why it matters: 两个可执行几乎复制完整 host 流程，未来修复易出现不一致。
- Evidence: `gelu_with_vf.cpp` 与 `gelu_without_vf.cpp` 从数据生成到 ACL 生命周期基本同构。
- Recommended fix: 提取共享 host harness，仅注入不同 kernel 函数。

### 2) 错误处理缺失
- Severity: Important
- Why it matters: ACL 调用返回值多数未检查，样例会把失败当成功路径继续跑。
- Evidence: `aclInit/aclrtMalloc/...` 直接调用无检查。
- Recommended fix: 统一 `CHECK_COND` 或 RAII + status 返回。

### 3) 文档性能结论缺少实验条件
- Severity: Minor
- Why it matters: 2.8x 结论若无固定 shape、次数、warmup、频率条件，复现实验易漂移。
- Evidence: README 给了数字但缺少统计口径定义。
- Recommended fix: 增加 benchmarking protocol（warmup/iters/统计方法/误差范围）。

## Naming and Structure
- 文件命名清晰。
- 建议新增 `common_runner.*`，减少双文件复制。

## Documentation Review
- 结构完整，教学友好。
- 但偏“宣传式文字”，缺少“何时 VF 可能不占优”的限制说明。

## Build-System Review
- CMake 简洁，target 级配置好。
- 仍使用 `-w`，不符合公开样例示范标准。

## Code Quality Review
- kernel 逻辑整洁。
- host 工程质量偏弱（异常路径、资源释放、可测试性）。

## Action Plan
- Must fix now
  - 补齐 ACL 错误检查。
  - 去除 `-w`。
- Should fix soon
  - 提取共享 host 运行框架。
  - README 增加性能测试协议。
- Nice to improve later
  - 增加 VF 不适用场景示例。
