# Intermittent Windows R segfault in `test-estimnone.R` — investigation prep

**Status: OPEN, not yet reproduced natively or root-caused.** This document
was written from a Linux sandbox (no Windows access) after a single
observed CI failure, to hand off to a session with real Windows access. It
records everything confirmed so far, everything already ruled out, and a
concrete plan for continuing. See
[`WindowsPythonHangDiagnostic354.md`](WindowsPythonHangDiagnostic354.md) for
the writing style this aims to follow once evidence exists, and
[`win_debug.md`](win_debug.md) for the general Windows debugging
environment/tooling setup (VS Build Tools, WinDbg/`cdb.exe`, Application
Verifier, py-spy-equivalent native inspection) — this doc assumes that
environment and doesn't repeat its setup instructions.

- No GitHub issue opened yet — open one and link it here once this is
  picked back up (or ask the user first; not done automatically per
  standing instructions against creating tracking issues without asking).
- Branch: `rebase/cg-predict-on-v1.2.2` (the canonical branch for the
  current `feature/cg-predict` work — see `feature/cg-predict`/
  `feature/iterative-convergence` for now-superseded history if this
  branch has since moved).
- Failing run: https://github.com/libKriging/libKriging/actions/runs/36435180775/job/108971136040
  (commit `31b1671e`).
- Immediate rerun of the *same* job, *same* commit, no code change:
  **passed** (`run_attempt: 2`, conclusion `success`). This is the single
  strongest piece of evidence so far: the bug is intermittent, not a
  deterministic logic error.

## Symptom

CI job "Build and tests / build (R Windows, windows-latest, Release, on)"
fails after ~20 minutes with:

```
make: *** [Makefile:31: test] Segmentation fault
##[error]Process completed with exit code 2.
```

This is a **native crash** (the OS killing the process with a segfault),
not an R-level error — `testthat`'s error handling never gets a chance to
catch or report it, so there's no R traceback, no `testthat` summary, no
`Error:` message anywhere in the log. `grep`-ing the log for
`error|Error|ERROR|FAILED` only turns up one unrelated, benign line (see
"False lead already ruled out" below) — the actual crash only shows up as
this `Segmentation fault` line from `make` itself.

## Precise localization

The R script is `Rscript testthat.R` → `testthat::test_check('rlibkriging')`,
which runs every file under `bindings/R/rlibkriging/tests/testthat/` in
order. Matching the log's last-printed model summaries (n=5, theta=0.1,
sigma2=0.01, beta=0.123, then nugget=0.0456, then heterogeneous noise
`5x[0.05,0.05]`) against the test sources pins the crash to
**`bindings/R/rlibkriging/tests/testthat/test-estimnone.R`**, specifically
this last block:

```r
# lines 42-51
context("Kriging with noise")

rno_noestim <- Kriging(y, X, "gauss", noise=rep(0.05,nrow(X)), optim="none",
                        parameters=list(theta=matrix(0.1), sigma2=0.01, beta=matrix(0.123)))
print(rno_noestim)
test_that(desc="theta noestim", expect_equal(rno_noestim$theta()[1], 0.1, tol=1E-10))
test_that(desc="sigma2 noestim", expect_equal(rno_noestim$sigma2(), 0.01, tol=1E-10))
test_that(desc="beta noestim", expect_equal(rno_noestim$beta()[1], 0.123, tol=1E-10))
```

`n=5`, `X`/`y` from a fixed seed (`set.seed(123)`), all three preceding
`Kriging(...)` calls in the same file (no noise, then `noise="nugget"`)
print fine and don't crash — only the **third**, with a per-point
heterogeneous noise vector (`noise=rep(0.05, 5)`), is followed by the
crash. `print(rno_noestim)` itself completes and its full output is in the
log — so the crash happens **after** printing succeeds: either in one of
the three `test_that`/`expect_equal` calls right after, or (more likely,
given how trivial those are) at R's garbage collection of `rno_noestim` or
one of the two earlier model objects, or when `testthat` moves on to the
next test file.

## What's already been checked (from a Linux sandbox — no native repro yet)

These were checked by static inspection and log analysis only; none of
this required Windows, so a Windows session can skip straight past them:

- **The R binding's `Rcpp::XPtr<Kriging>` construction pattern
  (`bindings/R/rlibkriging/src/kriging_binding.cpp`, `new_KrigingFit`,
  lines ~129-150) for the `Heterogeneous` noise path is structurally
  identical** to the noise-free and `Nugget` paths that never crash — same
  `new Kriging(...)` → optional `fit()` → `Rcpp::XPtr<Kriging> impl_ptr(ok)`
  → wrap in an R `list` with `class = "Kriging"`. Nothing jumps out at the
  C++ level as heterogeneous-noise-specific.
- **`lkalloc`/`ARMA_ALIEN_MEM_*` allocator routing is present and
  consistent for the R binding** (`Makevars.win`: `-include
  "libKriging/utils/lkalloc.hpp" -DARMA_ALIEN_MEM_ALLOC_FUNCTION=lkalloc::malloc
  -DARMA_ALIEN_MEM_FREE_FUNCTION=lkalloc::free`, applied to every R source
  file via `-include`). This is genuinely different from the #354 root
  cause (Python's version of this exact wiring was silently disabled) — R
  never calls `lkalloc::set_allocation_functions` at all, so it always
  uses `lkalloc`'s default fallback (`_aligned_malloc`/`_aligned_free`),
  self-consistent within one allocator. **#354's specific mechanism
  (two independent allocators in one process) doesn't obviously transfer
  to R** — but this deserves a second look with an actual debugger before
  fully ruling it out, since "doesn't obviously transfer" is not the same
  as "confirmed absent."

### False lead already ruled out

The log's only other `Error`/`Error 1` line, right before the real crash,
is:

```
export R_LIBS="..." && R CMD REMOVE rlibkriging
Removing from library 'D:/a/_temp/Library'
Error in find.package(pkgs, lib) :
  there is no package called 'rlibkriging'
make: *** [Makefile:40: uninstall] Error 1
```

This is the Makefile's `uninstall` target trying to remove a package that
was never installed yet (expected on every fresh runner) — the driving
shell script tolerates this (`+ true` immediately follows in the log) and
proceeds to `make clean` / `make` / `make test` normally. **Not related to
the actual segfault**, just noisy pre-cleanup — don't waste time on it.

### A compiler-flag oddity, checked and ruled out

The compile log shows, for every `.cpp` in the R package:

```
g++ -std=gnu++20 -I"C:/R/include" ... -std=c++17 -fopenmp ... -c MLPKriging_binding.cpp -o MLPKriging_binding.o
```

Two `-std=` flags: `-std=gnu++20` (from R 4.6.1's own `Makeconf`
defaults on this Rtools/gcc14 toolchain) followed by `-std=c++17` (from
this package's own `Makevars.win`, `PKG_CXXFLAGS=-std=c++17 -fopenmp`).
**Checked directly** (not Windows-specific, just GCC flag precedence):
GCC takes the *last* `-std=` flag on the command line, confirmed with
`g++ -std=gnu++20 -std=c++17 -dM -E -xc++ /dev/null | grep __cplusplus`
→ `201703L` (C++17). So the effective standard is C++17 as intended,
matching how the core `libKriging.a` this package links against is
built. **Ruled out** as an ABI-mismatch lead — but if the core library's
own `CMAKE_CXX_STANDARD` were ever bumped past 17 without updating
`Makevars.win` too, this redundant-flag setup would silently keep
compiling the R side at 17, which *would* be a real (if different) bug
worth a lint/CI check some day. Not implicated in this crash.

## Hypotheses not yet tested (start here)

In rough priority order, informed by this repo's prior Windows
investigations (`WindowsPythonHangDiagnostic354.md`, and the OpenMP
thread-pool-churn fixes for Octave/Python Windows referenced there):

1. **GC/finalizer timing on `Rcpp::XPtr<Kriging>` destruction for the
   heterogeneous-noise object specifically.** `Rcpp::XPtr`'s default
   finalizer calls `delete` on the wrapped pointer when R's GC collects
   the R-level wrapper object — R's GC timing is not deterministic
   (triggered by allocation pressure, not immediately at
   end-of-statement), so if a use-after-free or double-free exists
   somewhere in `Kriging`'s destructor chain specific to the
   `Heterogeneous` noise model (e.g. in how the per-point noise
   `arma::vec` is stored/freed vs. the scalar `nugget` or no-noise cases),
   it would manifest exactly like this: intermittent, timing-dependent,
   no crash until sometime after the object should logically still be
   alive. **First thing to try**: force GC deterministically right after
   line 45 (`gc(); gc()` in a scratch copy of the test, or run under `R
   -d gdb`/`R -d "windbg"` equivalent and single-step) and see if the
   crash becomes reproducible on demand instead of intermittent.
2. **OpenMP thread-pool churn** — the exact mechanism already fixed twice
   in this repo for Windows (Octave, then Python — see the history table
   in `WindowsPythonHangDiagnostic354.md`). Less likely here specifically
   because `optim="none"` means no BFGS repeated-parallel-region churn is
   happening in *this* test, but worth a quick check: is
   `OMP_NUM_THREADS` forced to `1` for the R Windows CI job anywhere
   (`tools/r-windows/*.sh`, `.github/workflows/main.yml`)? If not (unlike
   the Python/Octave Windows jobs, which explicitly force it), that's a
   quick, cheap thing to try.
3. **Something specific to how a per-point noise `arma::vec` moves
   through the R↔C++ boundary** — `Rcpp::as<arma::vec>(noise)` followed by
   `std::move(noise_vec)` into `Kriging::fit(...)`
   (`kriging_binding.cpp:132-140`). Worth checking whether `Kriging::fit`
   (or whatever it delegates to for the `Heterogeneous` case,
   `src/lib/Kriging.cpp`) stores this vector in a way that could outlive
   or alias memory `Rcpp::as` doesn't actually own long-term — `Rcpp::as`
   for a `NumericVector` → `arma::vec` typically copies, but worth
   confirming there isn't a view/alias path taken here for small vectors
   specifically.
4. **Re-check the `lkalloc` allocator story with an actual debugger**,
   even though static inspection above didn't find an obvious analog to
   #354 — that investigation only found the real mechanism after eight
   other hypotheses were disproven with hard evidence, so "looks fine by
   inspection" is weak evidence on its own for this class of bug.

## Suggested reproduction plan

Adapting `WindowsPythonHangDiagnostic354.md`'s approach (see that doc and
`win_debug.md` for full environment setup — VS Build Tools, CMake ≥ 4.2,
Rtools, WinDbg, Application Verifier):

### Build (mirrors `tools/r-windows/build.sh` / CI's `BUILD_NAME=r-windows`)

```bash
cd /path/to/libKriging
export MODE=Release   # the failing CI job uses Release, not Debug -- match it first
export ENABLE_R_BINDING=on
export ENABLE_PYTHON_BINDING=off
export ENABLE_OCTAVE_BINDING=off
export ENABLE_MATLAB_BINDING=off
export ENABLE_JULIA_BINDING=off
export BUILD_DIR=build_win_r
tools/r-windows/build.sh
```

### Isolate just the crashing block

Create a scratch copy of the relevant lines (don't edit the real test file
in place while debugging — copy it, e.g. to
`bindings/R/repro_estimnone_noise.R`, matching CI's `set.seed(123)`/`n=5`
exactly):

```r
library(rlibkriging)
f <- function(x) 1 - 1/2*(sin(12*x)/(1+x) + 2*cos(7*x)*x^5 + 0.7)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y <- f(X)

rno_noestim <- Kriging(y, X, "gauss", noise = rep(0.05, nrow(X)), optim = "none",
                        parameters = list(theta = matrix(0.1), sigma2 = 0.01, beta = matrix(0.123)))
print(rno_noestim)
rm(rno_noestim)
gc(); gc()   # force collection immediately, instead of waiting on GC timing
cat("survived one iteration\n")
```

Since this is *intermittent* under normal GC timing, wrap the whole thing
in a loop (50-100 iterations, fresh object each time) to raise the odds of
hitting it in a single debugging session, and to get a rough failure rate:

```r
for (i in 1:100) {
  m <- Kriging(y, X, "gauss", noise = rep(0.05, nrow(X)), optim = "none",
               parameters = list(theta = matrix(0.1), sigma2 = 0.01, beta = matrix(0.123)))
  invisible(capture.output(print(m)))
  rm(m)
  gc()
  cat("iter", i, "ok\n")
}
```

Run under `R -d "cdb -c 'sxe av; g; .exr -1; kv; q'"` (see `win_debug.md`'s
`cdb.exe` section — same pattern used for #354, just launching `R.exe`/
`Rscript.exe` instead of `python.exe`) to catch the access violation with a
native stack the moment it happens, rather than relying on R's own crash
handler (which may not produce a useful native stack for an `Rcpp::XPtr`
finalizer running during GC).

If Application Verifier's Page Heap (`appverif -enable Heaps -for
Rscript.exe -with Heaps.full=true` / `R.exe`, whichever actually hosts
the crash — check both, since `R CMD` scripts sometimes fork into a
sub-`Rscript.exe`) doesn't immediately reproduce, increase the loop count
before concluding it's not memory corruption — #354 needed a specific,
deterministic matrix before Page Heap product a hard fault; this bug's
intermittency means an equivalent "first bad iteration" may take more than
one run to hit.

**Remember to disable Page Heap when done** (systemwide per-image-name
setting — see `win_debug.md`).

## Cleanup notes

Nothing has been committed or changed in the working tree for this
investigation — this document is the only artifact. If a future session
adds scratch repro files (`bindings/R/repro_estimnone_noise.R`, any dumped
crash matrices, throwaway builds), list them here and remove before
merging anything, matching the cleanup discipline in
`WindowsPythonHangDiagnostic354.md`.
