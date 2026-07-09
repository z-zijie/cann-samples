# cann-samples vs cuda-samples · 开源生态深度对比报告

对 **华为 cann-samples**（分支 `sync/upstream/master`）与 **NVIDIA/cuda-samples**（`master`，Release v13.3）
的目录级、定位级、覆盖级与开源开放生态级全方位对比。数据快照 **2026-07-09**。

> 本目录为独立技术对比分析，非任何厂商官方文档；商标与许可条款归各自权利人所有。

## 交付物

| 文件 | 说明 |
| --- | --- |
| [`index.html`](./index.html) | **交互式对比仪表盘**（自包含，浏览器直接打开）。带旋钮：主题（自动/浅/深）、图表视图（并排/叠加）、主指标选择、线性/对数刻度、密度、领域筛选、以及七轴**加权契合度**滑杆 + 身份预设。 |
| [`cann-vs-cuda-samples-comparison.pdf`](./cann-vs-cuda-samples-comparison.pdf) | **专业 PDF 报告**（16 页 A4）：封面 · 执行摘要 · 七轴画像 · 目录映射 · 许可证逐条对照 · 覆盖热力矩阵 · 开发者样例 · 生态治理 · 选型建议 · 数据附录与对抗式核验日志。 |
| [`print.html`](./print.html) | PDF 的打印源（静态、内联 SVG 图表）。 |
| [`src/`](./src) | 可复现构建脚本与数据。 |

## 核心结论

- **定位**：cuda-samples = 面向全体开发者的「CUDA 工具链特性全景陈列馆」（广度）；cann-samples = 面向昇腾算子工程师的「实战调优知识库」（深度）。
- **规模**：cuda 14 类目 / 238 叶子样例 / 2,022 文件 / 189 个 `.cu`；cann 4 类目 / 19 样例组（展开约 62 叶）/ 782 文件 / 116 个 `.asc`。
- **开源定位**：cuda = **BSD-3-Clause**（OSI 认证、宽松、任意用途）；cann = **CANN OSL v2.0**（限华为 AI 处理器用途、可撤销、含专利反制、**非 OSI**）。
- **覆盖互补**：cuda 独占数值库 / 图形互操作 / 经典 HPC / 跨平台；cann 独深 LLM 推理算子调优（MoE、融合注意力、KV-RMSNorm-RoPE、MX 量化）。
- **教学范式**：cuda「一样例一特性」的百科式；cann「0_naive→7_fullload」的逐步教程式（README 单篇均值 ~10.6KB 长文 vs cuda ~2.3KB 卡片）。

所有计数均由对两仓工作树的 `find`/`ls` 直接枚举得到，关键论断经独立子代理**对抗式核验**（详见 PDF 附录 C）。

## 复现构建

```bash
cd analysis/report/src
# 生成 print.html（内联静态图表）
node make_print.mjs
# 渲染 PDF（需 Playwright + Chromium）
NODE_PATH=$(npm root -g) node make_pdf.mjs ../print.html ../cann-vs-cuda-samples-comparison.pdf
```

交互式 `index.html` 由 `src/index.template.html`（含 `__DATA__` 占位符）注入 `src/data.js` 生成：

```bash
node -e 'const fs=require("fs"),D=require("./src/data.js");
fs.writeFileSync("index.html",fs.readFileSync("src/index.template.html","utf8").replace("__DATA__",JSON.stringify(D)))'
```

`src/data.js` 是唯一数据源（头部指标、类目计数、文件构成、覆盖矩阵、七轴评分、许可记分卡、独占能力列表），修改它即可同时更新两份报告。
