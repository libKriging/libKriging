# libKriging test suite

All tests live in `tests/`. Build targets follow the same name as the source file
(minus `.cpp`). Most use [Catch2 v2](https://github.com/catchorg/Catch2/tree/v2.x);
a few older ones use plain assertions.

## Building tests

```bash
# Build the library and ALL test binaries
cmake --build build --target all_test_binaries

# Or build a single test binary
cmake --build build --target <TestName>
```

## Running tests

```bash
# Run all registered tests via CTest (from the build directory)
cd build && ctest --output-on-failure

# Run a specific test binary directly (verbose)
export LD_LIBRARY_PATH=build/installed/lib:$LD_LIBRARY_PATH
build/tests/<TestName>

# Filter by Catch2 tag
build/tests/<TestName> [<tag>]

# List test cases without running
build/tests/<TestName> --list-tests
```

Tests tagged `[intensive]` are skipped in CI (`GITHUB_ACTIONS=true`) because they
take too long. Run them locally by omitting the tag filter.

---

## Test files

### Infrastructure

| File | Description |
|------|-------------|
| `catch2_unit_test.cpp` | Sanity-check that the Catch2 framework itself works |
| `KSTestValidation.cpp` | Validates the KS-test helper used in simulate tests |
| `ks_test.hpp` | Two-sample Kolmogorov–Smirnov test utility (header-only) |
| `GetVersion.cpp` | Checks that the compiled version string is retrievable |

### LinearAlgebra

| File | Description |
|------|-------------|
| `LinearAlgebraTest.cpp` | Unit tests for `LinearAlgebra` helpers (solve, chol, inv, …) |

### Kriging (standard, no nugget/noise)

| File | Tags | Description |
|------|------|-------------|
| `KrigingTest.cpp` | — | Basic fit / predict smoke test |
| `KrigingFitTest.cpp` | `[fit]` | Correctness of log-likelihood and parameter bounds |
| `KrigingLogLikTest.cpp` | `[loglik]` | Gradient correctness (vs finite differences) |
| `KrigingPredictTest.cpp` | `[predict]` | Mean and variance against reference values |
| `KrigingSimulateTest.cpp` | `[simulate]` | Simulate output statistics vs predict |
| `KrigingUpdateTest.cpp` | `[update]` | Update data point by point |
| `KrigingUpdateSimulateTest.cpp` | `[update_simulate]` | `update_simulate` ≡ update-then-simulate |
| `KrigingWarmRestartTest.cpp` | `[warm_restart]` | Warm restart (re-use previous optimum) |
| `KrigingLeaveOneOutTest.cpp` | `[loo]` | Leave-one-out cross-validation |
| `unstableLLTest.cpp` | — | Numerical stability of log-likelihood near singularity |

### NuggetKriging

| File | Tags | Description |
|------|------|-------------|
| `NuggetKrigingTest.cpp` | — | Smoke tests |
| `NuggetKrigingFitTest.cpp` | `[fit]` | Log-likelihood and gradient |
| `NuggetKrigingLogLikTest.cpp` | `[loglik]` | Gradient vs FD |
| `NuggetKrigingPredictTest.cpp` | `[predict]` | Predict correctness |
| `NuggetKrigingSimulateTest.cpp` | `[simulate]` | Simulate distribution |
| `NuggetKrigingUpdateTest.cpp` | `[update]` | Data update |
| `NuggetKrigingUpdateSimulateTest.cpp` | `[update_simulate]` | `update_simulate` correctness |

### NoiseKriging

| File | Tags | Description |
|------|------|-------------|
| `NoiseKrigingTest.cpp` | — | Smoke tests |
| `NoiseKrigingFitTest.cpp` | `[fit]` | Fit correctness |
| `NoiseKrigingLogLikTest.cpp` | `[loglik]` | Gradient vs FD |
| `NoiseKrigingPredictTest.cpp` | `[predict]` | Predict correctness |
| `NoiseKrigingSimulateTest.cpp` | `[simulate]` | Simulate distribution |
| `NoiseKrigingUpdateTest.cpp` | `[update]` | Data update |
| `NoiseKrigingUpdateSimulateTest.cpp` | `[update_simulate]` | `update_simulate` correctness |

### WarpKriging

| File | Tags | Description |
|------|------|-------------|
| `WarpKrigingTest.cpp` | — | Smoke tests (fit, predict, simulate, update) for all warp types |
| `WarpKrigingUpdateSimulateTest.cpp` | `[update_simulate][warpkriging]` | `update_simulate` vs update-then-simulate for `none` and `affine` warpings |
| `WarpKrigingPerWarpTest.cpp` | `[predict][simulate][derivative][fd][loglik][gradient][update][update_simulate][warpkriging]` | Per-warping 1-D correctness tests (see below) |

#### WarpKrigingPerWarpTest

Exercises **7 warp types** (`none`, `affine`, `boxcox`, `kumaraswamy`, `knots(3)`,
`neural_mono`, `mlp(8,1,selu)`) on a 1-D sin function with 15 equispaced training
points. Four test cases are run for every warp type:

1. **predict vs simulate** `[predict][simulate]`  
   Fits the model and draws 2 000 conditional simulations. Checks that the empirical
   mean and standard deviation at 6 test points match the analytical `predict()`
   values to within 4 standard errors (mean) and 25 % relative tolerance (stdev).

2. **predict derivative vs finite differences** `[predict][derivative][fd]`  
   Verifies `predict(..., return_deriv=true)` mean and stdev gradients against
   central finite differences (h = 1 × 10⁻⁵) at 5 test points, using a mixed
   tolerance of `max(0.02, 0.10 × |value|)`.

3. **update+simulate vs update_simulate** `[update][simulate][update_simulate]`  
   Compares two code paths that must produce statistically identical results:
   - **Path A**: `fit` → `simulate(will_update=true)` → `update_simulate(y_new, X_new)`
   - **Path B**: `fit` → `update(y_new, X_new, refit=false)` → `simulate(same seed)`  
   A two-sample KS test is applied at each of 8 simulation points (α = 10⁻⁶).
   At most 1 marginal failure out of 8 is allowed.

4. **log-likelihood gradient vs finite differences** `[loglik][gradient][fd]`  
   Evaluates `logLikelihoodFun(theta, return_grad=true)` at the fitted theta and
   checks every component of the analytical gradient against central finite
   differences (h = 1 × 10⁻⁵), using a mixed tolerance of
   `max(0.001, 0.01 × |value|)` (1 % relative).

### MLPKriging

| File | Tags | Description |
|------|------|-------------|
| `MLPKrigingTest.cpp` | — | Smoke tests for MLP-kernel Kriging |
| `MLPKrigingUpdateSimulateTest.cpp` | `[update_simulate]` | `update_simulate` correctness |
