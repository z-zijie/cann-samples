// ============================================================================
// Canonical dataset — cann-samples (sync/upstream/master) vs NVIDIA/cuda-samples (master, v13.3)
// Every count grounded in direct `find`/`ls` inspection of both working trees on 2026-07-09.
// This object is inlined verbatim into both the interactive HTML and the print/PDF report.
// ============================================================================
const DATA = {
  meta: {
    generated: "2026-07-09",
    cann: {
      key: "cann",
      name: "cann-samples",
      vendor: "华为 Huawei",
      platform: "Ascend NPU（昇腾）",
      ref: "分支 sync/upstream/master",
      home: "GitCode（gitcode.com/cann/cann-samples）",
      githubRole: "GitHub 为只读镜像（sync/upstream/master）",
      version: "对齐 CANN Toolkit 9.1.0（社区版）",
      license: "CANN Open Software License Agreement v2.0",
      licenseClass: "受限 / 源码可得（source-available）",
      osi: false,
      kernelLang: "Ascend C（.asc）",
      totalFiles: 782,
      catRoot: "Samples/",
      color: "cann",
    },
    cuda: {
      key: "cuda",
      name: "cuda-samples",
      vendor: "英伟达 NVIDIA",
      platform: "CUDA GPU",
      ref: "分支 master（Release v13.3）",
      home: "GitHub（github.com/NVIDIA/cuda-samples）",
      githubRole: "GitHub 原生仓库（PR 协作，已至 #435）",
      version: "对齐 CUDA Toolkit 13.3",
      license: "BSD 3-Clause",
      licenseClass: "OSI 认证 · 宽松许可（permissive）",
      osi: true,
      kernelLang: "CUDA C++（.cu / .cuh）",
      totalFiles: 2022,
      catRoot: "cpp/ + python/",
      color: "cuda",
    },
  },

  // Top-line quantitative comparison — used in the metric-selector bar chart.
  headline: [
    { label: "顶层样例类目 (categories)",        cann: 4,   cuda: 14, note: "cann: 4 (Samples/0-3)；cuda: 10 (cpp) + 4 (python)" },
    { label: "叶子样例 (leaf samples)",           cann: 62,  cuda: 238, note: "cann 展开 recipes/tutorials/variants 计数；cuda 205 cpp + 33 python" },
    { label: "顶层样例组 (sample groups)",        cann: 19,  cuda: 238, note: "cann 顶层可见样例目录数 19，深藏于 story/recipes" },
    { label: "仓库总文件数 (files)",              cann: 782, cuda: 2022 },
    { label: "内核源文件 (kernel src)",           cann: 116, cuda: 239, note: "cann .asc 116；cuda .cu 189 + .cuh 50" },
    { label: "文档文件 README/.md",               cann: 67,  cuda: 258 },
    { label: "Python 文件",                       cann: 79,  cuda: 37,  note: "cann: 数据生成/校验脚本；cuda: 可运行 CUDA-Python 样例" },
    { label: "图片/图示资产",                     cann: 210, cuda: 200, note: "cann .png184+.svg26=210；cuda png20+jpg28+ppm54+pgm21+bmp12+gif6+…" },
  ],

  // Per-category sample counts (grouped bar / small multiples).
  cudaCpp: [
    { cat: "0_Introduction", n: 46 },
    { cat: "1_Utilities", n: 3 },
    { cat: "2_Concepts_and_Techniques", n: 32 },
    { cat: "3_CUDA_Features", n: 24 },
    { cat: "4_CUDA_Libraries", n: 39 },
    { cat: "5_Domain_Specific", n: 36 },
    { cat: "6_Performance", n: 5 },
    { cat: "7_libNVVM", n: 9 },
    { cat: "8_Platform_Specific", n: 1 },
    { cat: "9_CUDA_Tile", n: 10 },
  ],
  cudaPython: [
    { cat: "1_GettingStarted", n: 8 },
    { cat: "2_CoreConcepts", n: 20 },
    { cat: "3_FrameworkInterop", n: 2 },
    { cat: "4_DistributedComputing", n: 3 },
  ],
  cannCat: [
    { cat: "0_Introduction", n: 5, groups: ["matmul", "npu_execution", "vector_add", "vector_add_c_api", "vector_function_getting_started"] },
    { cat: "1_Features", n: 15, groups: ["hardware_features (hif8, simt, vector_function, simd_vf_constraints)", "instruction_optimization (mte2_preload, n_buffer, unit_flag, weightnz)", "memory_optimization (full_load, l1_bank_conflict, reg_data_movement, scale_cache, slide_window_adaptive_template)", "system_optimization (streamk, tail_rebalance)"] },
    { cat: "2_Performance", n: 41, groups: ["matmul_story (7 recipes + 8 tutorials)", "grouped_matmul_story (4 recipes)", "moe_dispatch_and_combine_story", "moe_init_routing_story", "full_quant_fused_infer_attention_score_story", "kv_rms_norm_rope_cache_story (membase, regbase)", "rms_norm_quant_story", "simd_vf_story (broadcast, elemwise, reduce)", "scalar_story"] },
    { cat: "3_Utilities", n: 1, groups: ["simulation-based-vf-profiling"] },
  ],

  // File-type composition (stacked/diverging).
  fileTypes: {
    cann: [
      { ext: ".h 头文件", n: 235 }, { ext: ".png 图片", n: 184 }, { ext: ".asc 内核源", n: 116 },
      { ext: ".py 脚本", n: 79 }, { ext: ".md 文档", n: 67 }, { ext: ".txt", n: 51 },
      { ext: ".svg", n: 26 }, { ext: ".sh", n: 9 }, { ext: "其它", n: 15 },
    ],
    cuda: [
      { ext: ".json 测试参数", n: 379 }, { ext: ".txt", n: 265 }, { ext: ".md 文档", n: 258 },
      { ext: ".h 头文件", n: 219 }, { ext: ".cu 内核源", n: 189 }, { ext: ".cpp", n: 189 },
      { ext: ".hpp", n: 106 }, { ext: ".cuh 内核头", n: 50 }, { ext: ".py 样例", n: 37 },
      { ext: "图片/数据", n: 168 }, { ext: "其它", n: 162 },
    ],
  },

  // Domain coverage matrix. depth: 0 none, 1 basic, 2 covered, 3 deep.
  coverage: [
    { domain: "入门原语 (vector add / hello / template)", cann: 3, cuda: 2, tag: "入门" },
    { domain: "设备/拓扑查询 (device / topology query)", cann: 1, cuda: 2, tag: "入门" },
    { domain: "并行算法原语 (reduction / scan / sort / histogram)", cann: 1, cuda: 3, tag: "通算" },
    { domain: "矩阵乘 / GEMM", cann: 3, cuda: 2, tag: "计算" },
    { domain: "注意力 / LLM 推理算子 (FIA / MoE / RoPE / RMSNorm)", cann: 3, cuda: 1, tag: "LLM" },
    { domain: "量化 (MX FP4/FP8 · HiF8 · INT8)", cann: 3, cuda: 1, tag: "LLM" },
    { domain: "数值库 (BLAS/FFT/SOLVER/RAND/CUB/Thrust/CCCL)", cann: 0, cuda: 3, tag: "库" },
    { domain: "图像/信号处理 (NPP · JPEG · DCT · 滤波)", cann: 0, cuda: 3, tag: "库" },
    { domain: "图形互操作 (D3D11/12 · Vulkan · OpenGL)", cann: 0, cuda: 3, tag: "图形" },
    { domain: "金融/科学 HPC (BlackScholes · nbody · FDTD · Monte Carlo)", cann: 0, cuda: 3, tag: "HPC" },
    { domain: "执行模型 (Graphs · Streams · CDP · IPC · P2P)", cann: 1, cuda: 3, tag: "系统" },
    { domain: "性能调优方法论 (逐步优化实践)", cann: 3, cuda: 1, tag: "性能" },
    { domain: "硬件微架构特性 (SIMD/SIMT · bank conflict · 多缓冲)", cann: 3, cuda: 1, tag: "硬件" },
    { domain: "多卡/分布式 (multi-GPU · MPI · shmem)", cann: 1, cuda: 2, tag: "系统" },
    { domain: "Python / 框架互操作 (PyTorch / TF / cuda-python)", cann: 0, cuda: 2, tag: "生态" },
    { domain: "仿真 / Profiling 工具", cann: 2, cuda: 1, tag: "工具" },
    { domain: "跨平台/嵌入式 (Windows · Tegra · DriveOS)", cann: 0, cuda: 3, tag: "平台" },
  ],

  // Radar / weighted-lens axes. Scores are an explicit editorial analytic model (0-100),
  // deliberately exposed so the reader can re-weight them via the sliders.
  axes: [
    { key: "breadth",   label: "覆盖广度 Breadth",        cann: 35, cuda: 95 },
    { key: "depth",     label: "单点深度 Depth",          cann: 88, cuda: 60 },
    { key: "openness",  label: "开放度 Openness (许可)",   cann: 32, cuda: 92 },
    { key: "portable",  label: "跨平台 Portability",       cann: 22, cuda: 90 },
    { key: "pedagogy",  label: "教学/文档 Pedagogy",       cann: 90, cuda: 58 },
    { key: "maturity",  label: "成熟度/社区 Maturity",      cann: 45, cuda: 95 },
    { key: "llmops",    label: "LLM 算子聚焦 AI-ops",      cann: 95, cuda: 28 },
  ],
  // Slider presets ("旋钮" persona buttons).
  presets: [
    { id: "balanced", label: "均衡视角", w: { breadth:1, depth:1, openness:1, portable:1, pedagogy:1, maturity:1, llmops:1 } },
    { id: "breadth",  label: "我要广度学 CUDA 生态", w: { breadth:3, depth:1, openness:2, portable:2, pedagogy:1, maturity:2, llmops:0.3 } },
    { id: "llmtune",  label: "我要深挖昇腾 LLM 算子调优", w: { breadth:0.5, depth:3, openness:0.5, portable:0.3, pedagogy:2.5, maturity:1, llmops:3 } },
    { id: "compliance", label: "我关心开源合规/可分发", w: { breadth:1, depth:1, openness:3, portable:2, pedagogy:0.5, maturity:2, llmops:0.5 } },
    { id: "teaching", label: "我要教学/自学素材", w: { breadth:1.5, depth:1.5, openness:1, portable:1, pedagogy:3, maturity:1, llmops:1 } },
  ],

  // Open-source / ecosystem scorecard (openness gauge + table).
  openness: [
    { crit: "许可证类型", cann: "CANN OSL v2.0（受限）", cuda: "BSD-3-Clause（宽松）", edge: "cuda" },
    { crit: "OSI 认证开源", cann: "否", cuda: "是", edge: "cuda" },
    { crit: "使用范围限制", cann: "仅限为华为 AI 处理器开发/分发", cuda: "无用途限制", edge: "cuda" },
    { crit: "许可可撤销 / 专利反制条款", cann: "可撤销 + 专利诉讼即终止", cuda: "无（永久授权）", edge: "cuda" },
    { crit: "衍生作品分发", cann: "须随附本协议、保留声明、限昇腾用途", cuda: "保留版权声明即可，任意用途", edge: "cuda" },
    { crit: "主托管平台", cann: "GitCode（中国）", cuda: "GitHub（全球）", edge: "cuda" },
    { crit: "GitHub 角色", cann: "只读镜像 sync/upstream/master", cuda: "原生开发主仓", edge: "cuda" },
    { crit: "贡献流程", cann: "SIG + Committer/Maintainer 门禁，620 行 CONTRIBUTING", cuda: "Fork + PR，pre-commit.ci", edge: "tie" },
    { crit: "功能性 CI 门禁", cann: "run_ci_functional.py（真机功能测试）", cuda: "run_tests.py + test_args.json + pre-commit", edge: "tie" },
    { crit: "文档语言 / i18n", cann: "中文优先", cuda: "英文 / 全球", edge: "cuda" },
    { crit: "硬件可得性", cann: "需 Ascend 910B/C 或 950 + 指定 Toolkit", cuda: "任意 NVIDIA GPU", edge: "cuda" },
    { crit: "领域文档/调优知识库深度", cann: "逐步教程 + 性能建模 + profiling 图", cuda: "以可运行样例为主，文档较简", edge: "cann" },
  ],

  // "Only-in-X" — what each repo uniquely offers.
  onlyCann: [
    "逐步性能调优教程：matmul_tutorials 0_naive → 7_fullload（Ping-Pong / SWAT / UnitFlag / Half-L1 / 全载）",
    "LLM 推理算子专题 story：MoE dispatch/combine、MoE init routing、Fused-Infer-Attention-Score、KV-RMSNorm-RoPE-Cache",
    "前沿低比特量化 recipes：MXFP4 / MXFP8 / HiFloat8 / INT8 及 Weight-NZ",
    "昇腾微架构特性样例：HiF8、SIMT、Vector Function、L1 bank conflict、N-buffer、MTE2 preload、Unit Flag",
    "面向 NPU 的访存/指令/系统三层优化方法（StreamK、尾轮均衡、滑窗自适应模板）",
    "基于仿真的 VF Profiling 工具链（simulation-based-vf-profiling）",
    "结构化社区治理：SIG ops-basic、Committer/Maintainer 分级评审、真机 CI 功能门禁",
  ],
  onlyCuda: [
    "完整数值库样例：cuBLAS / cuFFT / cuSOLVER / cuRAND / NPP / nvJPEG / CUB / Thrust / CCCL 3.3",
    "图形互操作：Direct3D 11/12、Vulkan、OpenGL 全套渲染 + CUDA 互操作",
    "经典 HPC / 金融：BlackScholes、二项式期权、Monte Carlo、nbody、FDTD3d、Mandelbrot",
    "并行算法教科书样例：reduction、scan、radix/merge sort、histogram、卷积、协作组",
    "CUDA 执行模型全景：CUDA Graphs、条件节点、Streams、动态并行(CDP)、IPC、P2P、Tensor Core GEMM",
    "跨平台：Windows / Linux / Tegra 交叉编译 / DriveOS 汽车平台",
    "CUDA-Python 可运行样例 + PyTorch / TensorFlow 自定义算子互操作 + 新 CUDA Tile 编程模型",
    "自动化测试：run_tests.py 驱动全样例回归，test_args.json 逐样例参数化",
  ],
};
if (typeof module !== "undefined") module.exports = DATA;
