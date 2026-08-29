# Review

## Summary
- This is one of the strongest and most complete sample bundles in the repository.
- It is close to high-quality public-sample standard, but still has correctness and curation gaps that should be fixed to become exemplary.

## Strengths
- Strong architectural layering (`common`, `recipes`, `tutorials`, `docs`).
- CMake uses helper functions to reduce repetition and improve scalability.
- Documentation depth is substantial and performance-oriented.
- Includes tutorial progression and recipe-style executable artifacts.
- Includes generator/verification scripts for reproducible validation.

## Key Issues
### 1) README over-claims and under-signals completion status
- Severity: Important
- Why it matters: Public sample docs must clearly distinguish available vs planned content.
- Evidence: top-level README advertises broad coverage while “分步教程 待补充” remains unresolved.
- Recommended fix: explicitly mark shipped components, pending sections, and versioned roadmap.

### 2) Broken tutorial link in `matmul_tutorials/README.md`
- Severity: Important
- Why it matters: Broken links degrade trust and user flow.
- Evidence: link to `1_double_buffer/` points to nonexistent directory.
- Recommended fix: remove/replace with existing section or add the missing tutorial.

### 3) CMake sets global language/tool settings inside subdirectory
- Severity: Minor
- Why it matters: Local sample CMake should avoid re-stating global policies unless intentionally scoped.
- Evidence: sample CMake resets standard/PIC/export flags and conditionally overrides compilers via `BISHENG`.
- Recommended fix: move toolchain/standard policy to root and keep sample CMake target-focused.

## Naming and Structure
- Very good structure and naming clarity overall.
- “story” suffix is understandable internally but may be opaque externally; consider documenting naming rationale repository-wide.
- Recipes vs tutorials split is excellent pedagogically.

## Documentation Review
- Rich technical material and strong explanatory depth.
- Need stronger quick-start and completion signaling at top-level README.
- Ensure all links are valid and all referenced assets exist.

## Build-System Review
- Function-based CMake abstraction is a positive example for the repo.
- Fallback branch for standalone linkage is pragmatic but should be documented with explicit prerequisites.
- Some global-state settings in this subdirectory should be removed for cleaner composition.

## Code Quality Review
- Architecture appears deliberate and modular compared with other samples.
- Good separation between recipe and tutorial intent.
- Main improvement area is not core code quality but curation/consistency polish.

## Action Plan
- Must fix now
  - Fix broken tutorial link.
  - Clarify shipped vs planned content in README.
- Should fix soon
  - Move toolchain and global compile policy out of sample subdirectory.
- Nice to improve later
  - Add concise “start here” path for new users (recipe-first or tutorial-first).
