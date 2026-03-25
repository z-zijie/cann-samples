# Review

## Summary
- The sample is practically runnable and has basic scripts, but code volume and header complexity are under-documented for a public teaching artifact.
- It meets baseline utility, not elite sample clarity.

## Strengths
- Includes end-to-end flow with data generation and verification scripts.
- README provides direct command sequence.
- CMake target wiring is concise and readable.

## Key Issues
### 1) Very large header-level implementations obscure boundaries
- Severity: Important
- Why it matters: 500–1000+ line headers are hard to reason about and review, especially in samples.
- Evidence: `moe_distribute_dispatch.h` is ~1099 lines; combine/quant headers are similarly large.
- Recommended fix: split into smaller modules by stage (routing, packing, communication, merge) with explicit interfaces.

### 2) Warning suppression and manual runtime linkage
- Severity: Important
- Why it matters: Weakens quality signal and portability.
- Evidence: uses `-w`; links `ascendcl runtime stdc++` directly.
- Recommended fix: remove `-w`; prefer shared interface target encapsulating runtime deps.

### 3) Documentation lacks expected metrics/output examples
- Severity: Minor
- Why it matters: Users cannot quickly validate success quality beyond script completion.
- Evidence: README lists commands but no expected key output snippets/thresholds.
- Recommended fix: include expected pass message and minimal performance baseline.

## Naming and Structure
- Directory naming is clear and accurate.
- `include`/`src`/`scripts` layout is good.
- Internal header granularity needs refinement.

## Documentation Review
- Better operationally than many siblings.
- Needs stronger explanation of inputs, assumptions, and interpretation of verify results.

## Build-System Review
- Mostly target-local and simple.
- Could improve through dependency abstraction and warning policy cleanup.

## Code Quality Review
- Technical depth appears high.
- Teachability suffers due to very large translation units and limited narrative mapping.
- Code taste: strong kernel intent, weak packaging discipline.

## Action Plan
- Must fix now
  - Remove `-w`.
- Should fix soon
  - Split oversized headers into composable modules.
  - Document verification output expectations.
- Nice to improve later
  - Add architecture diagram for dispatch/combine dataflow.
