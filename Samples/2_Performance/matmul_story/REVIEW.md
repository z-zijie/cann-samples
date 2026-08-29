# Review

## Summary
- This is the most architecturally ambitious sample bundle and shows promising structure, but still has policy inconsistencies and discoverability gaps.
- It is close to high quality, but not yet uniformly exemplary across build/doc/code surfaces.

## Strengths
- Strong hierarchical layout (`common`, `recipes`, `tutorials`, `docs`).
- CMake introduces reusable helper functions and avoids brute-force duplication.
- Code is split into many focused headers/modules compared to other samples.
- Installation layout is thoughtfully curated, including compatibility copies.

## Key Issues
### 1) Sample-level CMake still mutates global compiler state
- Severity: Critical
- Why it matters: Even guarded mutation (`if(BISHENG)`) from subdirectory is the wrong layering and risks side effects.
- Evidence: sets `CMAKE_C_COMPILER/CMAKE_CXX_COMPILER/CMAKE_LINKER` inside sample CMake.
- Recommended fix: move compiler selection to top-level toolchain handling only.

### 2) Documentation claims broad tutorial progression but README is incomplete
- Severity: Important
- Why it matters: A large sample bundle must provide a clear learning path; “待补充” leaves onboarding unfinished.
- Evidence: README “分步教程” section states pending content.
- Recommended fix: add guided progression table mapping each tutorial stage, command, and expected outcome.

### 3) Compile policy still suppresses warnings
- Severity: Important
- Why it matters: Repository-wide sample quality suffers when warnings are hidden.
- Evidence: shared options include `-w`.
- Recommended fix: replace with explicit warning levels.

## Naming and Structure
- Overall naming is strong and intentionally layered.
- Long nested paths are justified by scope, but entry points should be highlighted more prominently in docs.

## Documentation Review
- Good high-level overview and directory map.
- Missing runnable quickstart and tutorial completion details.
- Needs explicit “start here” for first-time users.

## Build-System Review
- Better than most samples in this repo: helper functions and shared options improve maintainability.
- Still violates top-level policy separation by compiler override in subdirectory.

## Code Quality Review
- Appears modular and reusable with clear block/kernel/tiling boundaries.
- Better code taste than average in repository.
- Remaining gaps are more about consistency and pedagogical packaging than core implementation depth.

## Action Plan
- Must fix now
  - Remove subdirectory compiler overrides.
  - Remove `-w`.
- Should fix soon
  - Complete tutorial documentation with concrete commands and expected outcomes.
- Nice to improve later
  - Add “minimal path” walkthrough for newcomers.
