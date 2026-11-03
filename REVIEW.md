# Professional Code Review: cann-samples Repository

> **Reviewer perspective**: Open-source developer and beginner user unfamiliar with Huawei's Ascend/CANN stack
> **Branch reviewed**: `claude/review-cann-samples-eKWHu`
> **Date**: 2026-03-29
> **Repository purpose**: Learning resource for NPU operator programming on Ascend chips via the CANN software stack

---

## Executive Summary

The cann-samples repository has a clear and well-intentioned three-tier structure (Introduction → Features → Performance). Several of its samples — especially in the `1_Features` section — are genuinely well-crafted teaching resources with strong documentation, architectural diagrams, and step-by-step progressions. The build system is consistent, and the project's contribution and security guidelines are thorough.

However, there are several areas that need significant attention before this repository can be considered a high-quality open-source learning resource:

1. **Accessibility barrier**: All 30+ documentation files are in Simplified Chinese only. This is the single largest obstacle to international adoption.
2. **Commercial code pasted wholesale**: Several performance samples appear to have been copied directly from production codebases with minimal adaptation, defeating the purpose of a focused learning resource.
3. **Introduction samples are not truly introductory**: The entry-level examples assume significant prior knowledge and include complexity that obscures the core concepts being taught.
4. **Boilerplate copy-paste has silently introduced real bugs** in the tutorial series.
5. **Directory naming inconsistencies and typos** create confusion and erode credibility.

---

## What Works Well

Before detailing problems, it is worth acknowledging what this repository does well:

- **Clear macro-level structure**: The `0_Introduction → 1_Features → 2_Performance` progression is a sound pedagogical model.
- **`1_Features` quality**: Samples like `simt/`, `vector_function/`, `hif8/`, `n_buffer/`, and `unit_flag/` are genuinely well-constructed — each focuses on a single concept, includes architectural diagrams, and provides both the "with" and "without" comparison where relevant. This is exactly the right approach.
- **Performance stories with step-by-step progression**: The `rms_norm_quant_story` (0_naive → 6_binary_sum, 7693μs → 49μs, 157× speedup) is an excellent demonstration of incremental optimization. The `matmul_tutorials` progression follows the same sound model.
- **Consistent build system**: The CMake hierarchy is clean and consistent. All 21 individual `CMakeLists.txt` files follow the same pattern. The `cmake/ascend.cmake` integration is well-structured.
- **CONTRIBUTING.md and SECURITY.md**: Detailed, professional contribution guidelines at 25KB.
- **`common/` directory in matmul_story**: The existence of `common/host_utils/io_utils.h` shows awareness of the need to avoid pure boilerplate repetition.

---

## Issue 1: Documentation Is Chinese-Only [Severity: Critical]

**Finding**: Every single README file in the repository — including the main `README.md` — is written entirely in Simplified Chinese. No English content exists beyond code identifiers and build commands.

This is the most significant barrier to international open-source adoption. The Ascend/CANN ecosystem is actively trying to expand its developer community outside China. A repository that cannot be read by developers who do not read Chinese will simply be ignored.

Specific observations:

- `README.md` (root): Section headers use Chinese emoji decorations (`🔥Latest News`, `🚀概述`, `📝环境部署`, `⚡️快速入门`). The "Latest News" section mentions the project renamed from "ops-samples" to "cann-samples" in March 2026, but this information is only visible to Chinese readers.
- `Samples/0_Introduction/vector_add/README.md`: A beginner's first point of contact — entirely in Chinese.
- Technical READMEs in `1_Features/` contain deep mathematical content (LaTeX formulas, memory diagrams) that would be highly valuable to international researchers — if they could read them.
- Inline code comments across source files are predominantly in Chinese (e.g., `// 获取可执行文件所在目录`, `// 固定写法，初始化`, `// 申请device内存`).

**Recommendation**:
- At minimum, provide English translations of the root `README.md` and all `0_Introduction/` READMEs.
- For `1_Features/` and `2_Performance/`, add an English summary section at the top of each README covering: what the sample demonstrates, prerequisites, and how to build and run.
- Replace inline Chinese comments in source files with English equivalents. These comments are often the most important teaching content ("this is the fixed pattern for initialization"), yet they are inaccessible to non-Chinese readers.
- Consider a bilingual README model: Chinese and English side-by-side, or separate `README.md` (Chinese) and `README_en.md` (English) files.

---

## Issue 2: Commercial Code Pasted Into Sample Files [Severity: High]

**Finding**: Several samples in `2_Performance/` appear to have been copied from production/commercial codebases with minimal adaptation. This contradicts the repository's stated purpose as a focused learning resource.

**`full_quant_fused_infer_attention_score_story/`**:
This sample contains 44+ header files implementing a full flash-attention production stack: tensor management utilities, tiling infrastructure, memory layout wrappers, quantization blocks, softmax blocks, and FIXPIPE dispatch logic. A learner interested in understanding "per-block full-quantization for a fused attention score operator" is immediately confronted with thousands of lines of infrastructure whose purpose is not explained.

In contrast to the `1_Features` samples — which are tight, focused, and introduce exactly one concept — this performance story dumps an entire production operator implementation into the repository. There is no pedagogical progression, no "here is the simplest possible version," and no isolation of which code is essential vs. infrastructure.

**`moe_dispatch_and_combine_story/` and `moe_init_routing_story/`**:
Similar pattern: these contain distributed dispatch/combine logic, quantization variants, multi-core sorting, and SIMT routing components — all co-mingled in a set of headers that are clearly derived from a commercial MoE operator implementation.

**What a good performance sample looks like** (the repo already has one): The `rms_norm_quant_story/` follows the right model — start from the simplest correct implementation (0_naive.cpp), apply one optimization at a time, and document the effect at each step. The entire teaching chain is visible and traceable.

**Recommendation**:
- For each complex performance story, add a `0_naive` baseline that implements the operator with no performance optimizations. This baseline should be readable in isolation.
- Clearly separate "teaching code" (the kernel implementation and optimization logic) from "support infrastructure" (memory management utilities, tiling wrappers). Do not require the learner to understand the full infrastructure to follow the main optimization thread.
- If the full production implementation must be included as a reference, put it in a `reference/` subdirectory and clearly label it as a complete implementation rather than a teaching artifact.
- Consider whether some of these samples are ready to be published at all. If the infrastructure headers are not yet at a quality level appropriate for public consumption, it may be better to wait until they are, rather than publishing partially-adapted commercial code.

---

## Issue 3: Introduction Samples Are Not Introductory [Severity: High]

**Finding**: The `0_Introduction/` section, which should be a beginner's first experience with CANN programming, contains samples that are more complex than necessary and assume significant background knowledge.

### `vector_add/main.cpp` (223 lines)

The kernel implementation itself is appropriate in length, but it introduces several advanced concepts simultaneously with no explanation:

- **Double-buffer pipelining**: The kernel uses `PIPELINE_DEPTH = 2` and a tile loop with explicit `TQue` management. A first-time reader sees `inQueueX.AllocTensor`, `inQueueX.EnQue`, `inQueueX.DeQue`, `inQueueX.FreeTensor` — four queue operations per tensor — before understanding why.
- **Duplicated tile body**: Lines 87–113 (main tile loop body) and lines 116–143 (tail tile body) are nearly identical — 27 lines repeated with only the element count changed. A beginner cannot immediately see why this is necessary or what differs. A comment explaining tail-tile handling would help significantly.
- **Tiling parameter calculation**: The `calc_tiling_params()` function queries the platform API, divides work across cores, and calculates UB buffer sizes. This is legitimate, but it is presented without explanation as a black box.

A true introductory vector add should, at first, show the simplest possible version: no tiling, no double buffering, just "copy to UB, add, copy back." Then a second version can introduce tiling and pipelining as explicit improvements. The current single file conflates the introductory concept with production-quality concerns.

### `matmul/main.cpp` (480 lines)

This is labeled "Introduction to Matrix Multiplication" but requires the reader to already understand:

- Ascend's multi-level memory hierarchy (GM → L1 → L0A/L0B/L0C)
- NZ, ZN, and ND layout formats and their conversion
- Hardware synchronization primitives (`WaitFlag`, `SetFlag`)
- K-dimension tiling and loop-carried state
- The `MatmulApiTiling` and related structures

480 lines of matrix multiply with explicit layout management and multi-level synchronization is not an introduction — it is an intermediate sample. Placing it in `0_Introduction/` misleads learners about the difficulty of getting started with CANN.

**Recommendation**:
- For `vector_add`: Add a "minimal" version (50–70 lines) that performs the add without tiling or double-buffering. Label the existing file as "optimized version." Add a comment at the top of the tail-tile block explaining its purpose.
- For `matmul`: Move it to `1_Features/` or `2_Performance/` where its complexity is appropriate. Replace it with a genuinely simple matmul introduction — ideally one that uses the high-level CANN MatMul API rather than manual layout management.
- Both introduction samples lack a "What you need to know before reading this" prerequisites section in their READMEs.

---

## Issue 4: Boilerplate Copy-Paste Has Introduced Real Bugs [Severity: High]

**Finding**: Tutorial series files are copy-pasted from one step to the next. While keeping each sample self-contained is the correct approach for a samples repository, the current implementation has two specific problems.

### A. Divergent copies have introduced bugs

Comparing `matmul_tutorials/0_naive/matmul_tutorial_mxfp4_base.cpp` and `matmul_tutorials/1_pingpong/matmul_tutorial_mxfp4_openpingpong.cpp`:

**Bug 1 — Hardcoded path in `1_pingpong`** (line 81):
```cpp
// 0_naive: correct — resolves relative to executable path
char exePath[PATH_MAX];
ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
// ... computes baseDir from executable location

// 1_pingpong: BUG — hardcoded path relative to working directory
const std::string goldenDir = "Samples/2_Performance/matmul_story/matmul_tutorials/golden";
```
The `1_pingpong` version will fail to find input data unless the user runs it from the repository root. No other tutorial step should depend on the working directory at runtime.

**Bug 2 — Missing timing in `1_pingpong`**: `0_naive` includes event-based kernel timing (`aclrtCreateEvent`, `aclrtRecordEvent`, `aclrtEventElapsedTime`) and prints the elapsed microseconds. `1_pingpong` silently drops this — making it impossible to observe the performance effect of the ping-pong optimization, which is the entire point of the step.

**Bug 3 — Include path typo propagated**: `1_pingpong` includes `"include/blcok/block_mmad_mx_base.h"` — the directory name typo `blcok` (see Issue 5) was copy-pasted into the include paths of steps 1, 5, 6, and 7.

### B. Mechanical helpers repeated verbatim across 6 files

In `rms_norm_quant_story/src/`, the following code is byte-for-byte identical across all 6 files (`0_naive.cpp` through `6_binary_sum.cpp`):

- `getExeDir()` function (~10 lines) — resolves executable path
- `getDataFromBin<T>()` function (~55 lines) — reads binary data files
- `segmentProduct()` function (~12 lines) — computes shape products
- `CHECK_ACL()` function (~3 lines)
- `Init()` function (~10 lines) — ACL device initialization
- `CHECK_RET` and `LOG_PRINT` macros

These ~90 lines of mechanical I/O infrastructure have no relationship to the RMS normalization optimization being demonstrated. A reader studying step `3_vf.cpp` to understand Vector Function integration gains nothing from reading the same `getDataFromBin` implementation for the third time.

**Recommendation**:
- Fix the `1_pingpong` hardcoded path and missing timing immediately — these are functional bugs.
- For `rms_norm_quant_story/`, create a local `host_utils.h` within the story directory (not a global utility) containing the mechanical helpers. Each `.cpp` file includes it with one line. The type aliases, constants, tiling structs, kernel class, and `main()` body all remain in the individual files — preserving self-containedness at the algorithmic level.
- Similarly, `printUsage()` and `parseArguments()` in `matmul_tutorials` should move into `common/host_utils/` where `io_utils.h` already lives.
- Add a `// Changes from previous step:` comment block at the top of each tutorial file identifying exactly what was modified. This makes the delta immediately visible without requiring a diff tool.

---

## Issue 5: Directory Structure Problems [Severity: Medium]

### Typos in directory names

Four directories in `matmul_tutorials` have a consistent misspelling — `blcok` instead of `block`:

```
matmul_tutorials/1_pingpong/include/blcok/          ← typo
matmul_tutorials/3_block_swat/include/block/        ← correct
matmul_tutorials/5_halfl1_ping_halfl1_pong/include/blcok/   ← typo
matmul_tutorials/5_unit_flag/include/blcok/         ← typo (same step number)
matmul_tutorials/6_scale_memory_access_coalescing/include/blcok/  ← typo
matmul_tutorials/7_fullload/include/blcok/          ← typo
```

This misspelling also propagates into `#include` paths in the corresponding source files (e.g., `#include "include/blcok/block_mmad_mx_base.h"`).

### Inconsistent step numbering in `matmul_tutorials`

```
0_naive/
1_pingpong/
                      ← step 2 is missing
3_block_swat/
4_last_round_tile_balance/
5_halfl1_ping_halfl1_pong/
5_unit_flag/          ← two directories with the same step number "5"
6_scale_memory_access_coalescing/
7_fullload/
```

There is a gap at step 2 and two directories with the number 5. A reader following the tutorial sequence cannot determine the correct order.

### Empty placeholder directories

`Samples/1_Features/memory_optimization/` and `Samples/1_Features/system_optimization/` exist as directories containing only a stub README ("currently reserved for future content"). Empty placeholder directories with no content create a misleading impression of the repository's scope. The main `README.md` also lists these as available categories, which is inaccurate.

**Recommendation**:
- Rename all `blcok` directories to `block` and update all include paths accordingly.
- Renumber the `matmul_tutorials` steps to be sequential with no gaps or duplicates. If `5_unit_flag` and `5_halfl1_ping_halfl1_pong` represent divergent optimization paths (not sequential steps), this should be clearly explained in the README and the naming convention should reflect it (e.g., `5a_` and `5b_`).
- Either populate the placeholder directories or remove them entirely. Do not commit empty directories with stub READMEs to the main branch.

---

## Issue 6: Missing Repository-Level Information [Severity: Medium]

### No hardware/software prerequisites documented

A developer finding this repository for the first time has no way to determine:
- Which Ascend hardware model(s) are required
- Which CANN toolkit version to install
- Which operating system is supported
- What compiler and driver versions are needed

The root `README.md` links to a download page for CANN toolkit but provides no version requirements. There is no compatibility matrix anywhere in the repository.

### No per-sample compatibility table

Different samples appear to target different hardware architectures. `6_binary_sum.cpp` includes `platform/platform_ascendc.h` and `basic_api/reg_compute/kernel_reg_compute_utils.h` that earlier files do not. There is no documentation of which samples work on which NPU architecture variants.

### No CHANGELOG or version history

The repository has no `CHANGELOG.md`. The "Latest News" section in the main README is a narrative paragraph in Chinese. There is no structured record of what changed between versions or what samples were added when.

### Inconsistent build target naming

Some build targets use short descriptive names (`vector_add`, `matmul`), while others use full file names (`matmul_tutorial_mxfp4_base`, `quantize_hif8_demo`, `gather_simt_demo`). There is no naming convention document.

**Recommendation**:
- Add a "Prerequisites" section to `README.md` listing hardware model, CANN version range, OS, compiler, and driver requirements.
- Add a compatibility table mapping each sample to the NPU architecture(s) it requires.
- Establish and document a build target naming convention.
- Add a `CHANGELOG.md` with at minimum a record of the initial samples and their descriptions.

---

## Issue 7: Minor Code Quality Issues [Severity: Low]

### `third_party/shmem/` is empty

The `third_party/shmem/` directory exists but contains no files. Either the shared memory headers are not yet ready, or they were accidentally omitted. This should be resolved before publication.

### `requirements.txt` is nearly empty

The root `requirements.txt` contains only `numpy` (27 bytes). Several samples use additional Python libraries (e.g., for data generation scripts). The dependencies are not comprehensively listed.

### Spelling error in `1_preload_gamma.cpp` docblock

The file `Samples/2_Performance/rms_norm_quant_story/src/1_preload_gamma.cpp` has its file docblock comment as `\file 1_per_load_gamma.cpp` — the file is named `1_preload_gamma.cpp` but the docblock says `1_per_load_gamma.cpp`.

### Inline function definition in `rms_norm_quant_story`

In `0_naive.cpp` (and all subsequent steps), `AlignBytes()` and `Align()` are defined with the `__aicore__` annotation — an NPU-specific attribute — yet they are called from host code in the tiling setup. These functions serve both host and device purposes, but the annotation implies device-only usage. This is either an annotation error or an undocumented dual-use pattern that should be explained.

---

## Summary Table

| # | Issue | Severity | Affected Files |
|---|-------|----------|----------------|
| 1 | All documentation is Chinese-only | Critical | All 30+ READMEs, inline comments |
| 2 | Commercial code pasted wholesale without adaptation | High | `full_quant_fused_infer_attention_score_story/`, `moe_dispatch_and_combine_story/` |
| 3 | Introduction samples too complex; not beginner-friendly | High | `0_Introduction/matmul/`, `0_Introduction/vector_add/` |
| 4a | Hardcoded path bug in `1_pingpong` host code | High | `matmul_tutorials/1_pingpong/*.cpp` |
| 4b | Missing timing in `1_pingpong` (copy-paste omission) | High | `matmul_tutorials/1_pingpong/*.cpp` |
| 4c | Mechanical helpers repeated 6× in `rms_norm_quant_story` | Medium | `rms_norm_quant_story/src/*.cpp` |
| 5a | `blcok` directory name typo in 4+ locations | Medium | `matmul_tutorials/1_pingpong/`, `5_*/`, `6_*/`, `7_*/` |
| 5b | Gaps and duplicates in `matmul_tutorials` step numbering | Medium | `matmul_tutorials/` |
| 5c | Empty placeholder directories committed | Low | `1_Features/memory_optimization/`, `1_Features/system_optimization/` |
| 6a | No hardware/software prerequisites documented | Medium | `README.md` |
| 6b | No per-sample CANN version compatibility table | Medium | Repository-level |
| 7a | `third_party/shmem/` is empty | Low | `third_party/shmem/` |
| 7b | `requirements.txt` incomplete | Low | `requirements.txt` |
| 7c | Docblock filename mismatch in `1_preload_gamma.cpp` | Low | `rms_norm_quant_story/src/1_preload_gamma.cpp` |

---

## Recommended Priority Order

1. **Fix the `1_pingpong` path bug** — this is a functional defect that makes the sample non-runnable for most users.
2. **Add English README translations** for `0_Introduction/` samples — these are the first thing a new user reads.
3. **Fix directory name typos** (`blcok` → `block`) and update include paths.
4. **Fix step numbering** in `matmul_tutorials/`.
5. **Add prerequisite documentation** to the root README.
6. **Simplify or reposition `0_Introduction/matmul/`** — either simplify it or move it to a more appropriate section.
7. **Evaluate `2_Performance/` commercial samples** — decide whether to refactor them into focused teaching samples or label them clearly as reference implementations.
8. **Extract mechanical host helpers** in `rms_norm_quant_story/` into a local `host_utils.h`.
9. **Add "Changes from previous step" comments** to all tutorial series files.
10. **Add English translations** for `1_Features/` and `2_Performance/` READMEs.
