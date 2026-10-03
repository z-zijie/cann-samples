# Review

## Summary
- The sample is runnable and focused, but it does **not** yet meet a world-class public-sample bar because robustness, numerical validation, and build hygiene are under-specified.
- It is a decent internal demo; it is not yet a polished teaching artifact for external users.

## Strengths
- Single-file flow makes the kernel lifecycle easy to follow end-to-end.
- Uses RAII for device memory via `std::unique_ptr` deleters.
- Introduces tiling concepts early (`calc_tiling_params`) instead of hardcoding one launch shape.
- Has a CPU-side correctness check path.

## Key Issues
### 1) Exact float equality for validation
- Severity: Important
- Why it matters: Samples should teach numerically robust verification; exact `==` on floating-point results teaches bad practice.
- Evidence: Validation compares `h_C[i] != h_A[i] + h_B[i]` directly.
- Recommended fix: Use tolerance-based comparison (`abs(diff) <= atol + rtol * abs(ref)`) and print first mismatch details.

### 2) Error handling style is macro-heavy and non-composable
- Severity: Important
- Why it matters: `CHECK_ACL` returns from current function, which obscures control flow and complicates future refactoring.
- Evidence: Macro-based early returns are used throughout host code.
- Recommended fix: Replace with small checked helper functions returning `aclError` or `expected`-style status.

### 3) Warning suppression in sample build
- Severity: Important
- Why it matters: `-w` hides quality regressions and sends the wrong pedagogy signal in a samples repository.
- Evidence: CMake adds `-w` in target compile options.
- Recommended fix: Remove `-w`; use explicit warning levels and fix warnings.

### 4) Documentation does not define expected output quality threshold
- Severity: Minor
- Why it matters: Users cannot tell what correctness/precision criteria should pass.
- Evidence: README has run steps but no explicit expected numeric behavior.
- Recommended fix: Add a short "Expected output" section with pass/fail criteria.

## Naming and Structure
- Directory name is clear and appropriate for an introductory sample.
- Single source file is acceptable at this complexity, but `host/runtime` and `kernel` sections should be separated (or clearly sectioned) to improve teachability.

## Documentation Review
- README explains purpose and basic run flow.
- Missing expected output examples and failure troubleshooting.
- Does not describe precision expectations for float verification.

## Build-System Review
- Target-scoped commands are used (good).
- Uses architecture and optimization flags directly in sample CMake, which duplicates policy across many samples.
- `-w` is not acceptable for educational code quality.

## Code Quality Review
- Readability is generally acceptable but function responsibilities are broad (`run_vector_add` does setup, launch, transfer, and validation).
- Ownership/lifetime on device buffers is better than average due to deleter wrappers.
- Validation robustness is below sample-grade due to strict float equality.
- Overall code taste: practical, but not yet "reference-quality".

## Action Plan
- Must fix now
  - Replace exact float equality with tolerance-based checks and diagnostic mismatch output.
  - Remove `-w` from compile options.
- Should fix soon
  - Refactor macro-based error handling into explicit helpers.
  - Split host orchestration into smaller named functions.
- Nice to improve later
  - Add CLI parameters for vector size and seed.
  - Add a short performance note (throughput baseline).
