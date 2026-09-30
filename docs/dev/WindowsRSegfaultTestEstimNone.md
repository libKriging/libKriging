# Intermittent Windows native crashes — investigation prep

**Status: OPEN, not yet reproduced natively or root-caused.** Two distinct
symptom clusters, both Windows-only, both intermittent, found on the same
branch within a few days of each other — plausibly the same underlying
memory-corruption class of bug (cf. the historical
[`WindowsPythonHangDiagnostic354.md`](WindowsPythonHangDiagnostic354.md)
case, also Windows-only heap corruption, also initially looking scattered
before its real mechanism was found), but **not confirmed to share a root
cause** — treat as two open questions until evidence says otherwise:

1. [§A — R Windows: `test-estimnone.R`](#a-r-windows-test-estimnoner-heterogeneous-noise-kriging) —
   segfault in the R binding's test suite, always at the same test.
2. [§B — Julia Windows: core `ctest` segfault cluster](#b-julia-windows--core-ctest-segfault-cluster) —
   14 different C++ Catch2 tests, spanning Nystrom/Vecchia/NestedKriging/
   LinearAlgebra, segfaulting in the same CI run.

This document was written from a Linux sandbox (no Windows access) after
observed CI failures, to hand off to a session with real Windows access.
It records everything confirmed so far, everything already ruled out, a
CI-assisted plan to collect real crash dumps *before* anyone needs to sit
in front of a Windows VM (§C), and the native-debugging fallback plan
(§D) adapted from `WindowsPythonHangDiagnostic354.md`/
[`win_debug.md`](win_debug.md) if the CI dumps aren't enough — this doc
assumes that environment and doesn't repeat its setup instructions.

- No GitHub issue opened yet — open one and link it here once this is
  picked back up (or ask the user first; not done automatically per
  standing instructions against creating tracking issues without asking).
- Branch: `rebase/cg-predict-on-v1.2.2` (the canonical branch for the
  current `feature/cg-predict` work — see `feature/cg-predict`/
  `feature/iterative-convergence` for now-superseded history if this
  branch has since moved).

## A. R Windows: `test-estimnone.R` heterogeneous-noise Kriging

- First failing run: https://github.com/libKriging/libKriging/actions/runs/36435180775/job/108971136040
  (commit `31b1671e`). Immediate rerun, same commit, no code change:
  **passed**.
- Second occurrence, different commit, unrelated changes (bench refresh +
  a CI timeout tweak): https://github.com/libKriging/libKriging/actions/runs/36691416263/job/109809264142
  (commit `e4f076a6`). Rerun: **passed** again.
- **Three passes, two failures, same exact crash point both times** — see
  "Precise localization" below. This is the strongest evidence so far:
  intermittent, not a deterministic logic error, and recurring across
  unrelated commits (so not tied to one specific change).

### A.1 Symptom

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

### A.2 Precise localization

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

### A.3 What's already been checked (from a Linux sandbox — no native repro yet)

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

### A.4 Hypotheses not yet tested (start here)

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

### A.5 Suggested reproduction plan

Start with §C (CI-assisted dump collection) — cheap and requires no
Windows access. If that doesn't yield enough, §D below has the full
native-VM reproduction plan for both §A and §B.

## B. Julia Windows — core `ctest` segfault cluster

### B.1 Symptom

CI job "Build and tests / build (Julia Windows, windows-latest, Release,
on)" — despite the name, this job runs the **entire core C++ `ctest`
suite** (`tools/windows/test.sh`: `ctest -C "${MODE}" ...`, no test
filter), Julia binding enabled alongside; it isn't Julia-specific. One
run failed with **14 different Catch2 tests** all marked `(SEGFAULT)` by
ctest's own reporting, spanning several unrelated features:

```
	 53 - LLIterative(m,precond_rank): the Nystrom preconditioner is applied to the SLQ log-determinant (SEGFAULT)
	101 - LinearAlgebra - rapid fire varying sizes (SEGFAULT)
	134 - NestedKriging is close to full Kriging on moderate n (SEGFAULT)
	147 - LLVecchia(20) estimation is consistent with the exact MLE (SEGFAULT)
	150 - predictVecchia matches exact predict (SEGFAULT)
	152 - light Vecchia fit skips the exact factorization (SEGFAULT)
	158 - LLNystrom analytic gradient matches finite differences (SEGFAULT)
	159 - LLNystrom fit is a permanent light fit: predict routes to predictNystrom (SEGFAULT)
	161 - LLNystrom(k) estimation is consistent with the exact MLE (SEGFAULT)
	162 - predictNystrom matches exact predict after an LLNystrom fit close to full rank (SEGFAULT)
	163 - LLNystrom update(refit=false) extends data at fixed theta/landmarks (SEGFAULT)
	164 - LLNystrom update(refit=true) warm-restarts theta over the same landmarks (SEGFAULT)
	166 - simulateNystrom mean/marginal-variance are consistent with predictNystrom (SEGFAULT)
	169 - LLNystrom honors optim=none identically to optim=BFGS (SEGFAULT)
```

Tests *between* these (e.g. #175/#176, MLPKriging-related) printed and
passed normally in the same run — so this is not "everything after some
point in the run crashes" (which would suggest one corruption cascading
through the rest of the process); each failure looks self-contained.

- Run: https://github.com/libKriging/libKriging/actions/runs/36691416263/job/109809263774
  (commit `e4f076a6`). Two rerun attempts afterward (`run_attempt: 2` and
  `3`) both **passed**, running the identical full suite with no code
  change in between.
- Neither commit in `e4f076a6` (a GPU-bench result refresh, and a CI-only
  change tagging one unrelated test `[intensive]` + raising Coverage
  mode's timeout) touches anything in `src/lib/`, `LinearAlgebra.cpp`, or
  any of Nystrom/Vecchia/NestedKriging — nothing in that diff plausibly
  causes this. Checked all commits on this branch going back through this
  session's own `predictIterative`/CG-solver work: **every one of them
  passed this same job before `e4f076a6`** — so if this is related to
  that work rather than a pre-existing latent bug, it was already present
  and just hadn't been hit yet, not introduced by `e4f076a6` itself.

### B.2 What's already been checked

- `#354`'s fix is still active: `CMakeLists.txt` still has the three
  `ARMA_ALIEN_MEM_*`/`CARMA_DO_NOT_EXPORT_ALIEN_MEM_FUNCTIONS` defines
  (lines ~327-329) — not a regression of that specific fix.
- Not obviously tied to any single recent source change (see run
  history above) — this needs a bisection or a repeatable native repro
  to make progress, which is exactly what §C is for.

### B.3 Hypotheses not yet tested

The breadth (Nystrom + Vecchia + NestedKriging + a generic
"LinearAlgebra rapid fire varying sizes" stress test, i.e. not one
feature) suggests either (a) a shared low-level mechanism many features
happen to call into (Cholesky/`rcond`/CG solves in `LinearAlgebra.cpp`,
matching the *mechanism* — though not necessarily the same exact bug —
behind #354), or (b) a test-process-level issue unrelated to any single
feature's logic (stack size exhaustion under deep/recursive calls,
OpenMP thread-pool churn again, or a shared test fixture/RNG state
corrupting something that only some tests are sensitive to). Both are
worth checking; §C's repeated-run + crash-dump approach should surface a
consistent (or inconsistent) faulting stack across occurrences, which
would discriminate between them immediately.

## C. CI-assisted dump collection (do this first)

Before reaching for a live Windows VM (§D, adapted from
`WindowsPythonHangDiagnostic354.md`), this branch's CI has been
**temporarily** set up to collect real crash dumps automatically, for
free, without anyone needing to sit in front of Windows:

- `.github/workflows/main.yml`: the `build` job's matrix is trimmed to
  just `R Windows` and `Julia Windows` (every other config commented out
  of scope for this investigation, not deleted — `git log` for the
  commit that did this trim to see/restore the full matrix).
- A new step, Windows-only, right before `test`, enables **Windows Error
  Reporting LocalDumps** (`HKCU:\...\Windows Error
  Reporting\LocalDumps`, `DumpType=2` = full memory dump) pointed at
  `<workspace>/crash_dumps/`. This is passive and process-name-agnostic —
  *any* process that crashes while the job runs gets a `.dmp` written,
  without needing to predict in advance whether it's `Rscript.exe`, a
  forked sub-`Rscript.exe`, or one of the `ctest`-launched Catch2 test
  `.exe`s.
- `tools/r-windows/test.sh` reruns `make test` up to `R_TEST_REPEAT`
  times (set to 8 for the `R Windows` matrix entry), stopping at the
  first failure.
- `tools/windows/test.sh` passes `--repeat until-fail:${CTEST_REPEAT}`
  (set to 8) and `${CTEST_EXTRA_ARGS}` (set to `-R
  Nystrom|Vecchia|NestedKriging|LinearAlgebra`, i.e. just the 14
  previously-crashing tests' features) to `ctest`, so each of the ~14
  selected tests gets up to 8 fresh-process attempts within the job's
  time budget instead of one.
- The existing "upload artifacts on test failure" step now also picks up
  `**/*.dmp`.

Both env-var knobs (`R_TEST_REPEAT`, `CTEST_REPEAT`/`CTEST_EXTRA_ARGS`)
default to a no-op (repeat once, no filter) when unset, so these script
changes are themselves safe to keep permanently if useful — only the
*matrix trim* in `main.yml` is meant to be temporary.

### C.1 Using this

**Update after first use**: the `R_TEST_REPEAT`/`CTEST_REPEAT` in-job
repeat knobs (8 each) ran clean **16/16 times total across two separate
full job-runs** (8 R + 8 Julia-suite repeats, twice) — zero crashes,
right after the organic failures in §A/§B's "First/Second occurrence"
entries. That's a meaningfully large sample to come back clean if the
per-attempt crash probability were anywhere near the ~40% the raw
job-level pass/fail count suggested (P(16/16 clean) ≈ 2.8% at p=0.2,
≈0.0028% at p=0.4) — **the leading interpretation is that repeating
*within* one job (same runner VM, same boot) doesn't resample whatever
actually varies between occurrences.** If the trigger depends on
something tied to the VM instance itself (heap layout/ASLR seed fixed
per boot, or some other per-VM state established once at boot and then
constant for every process launched in that job), many in-job repeats
would systematically undersample it while two independent *job*
launches (fresh VM each) already caught it twice. **Prefer many
separate job/run launches over high in-job repeat counts** going
forward — e.g. `gh run rerun <run-id>` (whole run, fresh VMs) in a loop
across several distinct runs, rather than cranking `R_TEST_REPEAT`/
`CTEST_REPEAT` higher on a single run. The repeat knobs aren't useless
(still cheap insurance, and would help if the true mechanism turns out
to be process-level after all) — just don't rely on them alone.

1. Push to this branch (or `gh workflow run` / re-push an empty commit)
   to trigger the trimmed matrix. Given ~3 passes for every ~2 failures
   observed at the *job* level so far (§C.1's update above), budget
   several separate pushes/reruns, not just one — `gh run rerun <run-id>`
   (whole run) is the cheap way to retry without a new commit
   (`--job <id>` alone sometimes errors "cannot be rerun" when another
   attempt is already in flight; `--failed` only works once nothing in
   the run is still in progress — plain `gh run rerun <run-id>` re-runs
   the whole thing unconditionally and sidesteps both issues).
2. Once a run fails with a genuine segfault (not a build error — check
   the log first), download the `artifacts-Windows` artifact
   (`gh run download <run-id>`) and look for `crash_dumps/*.dmp`.
3. If a `.dmp` was captured: open it with WinDbg (`File > Open Dump
   File`) or `cdb.exe -z path\to\file.dmp`, then `!analyze -v` for an
   automated first pass (faulting instruction, module, basic stack), and
   `kv`/`.ecxr; kv` for the full stack with arguments. This needs the
   matching PDBs — if the CI build doesn't already produce/upload them,
   the fastest path is usually rebuilding the exact same commit locally
   in a Windows VM (§D's "Build" section) so local PDBs line up, rather
   than trying to get symbol servers to resolve a CI-only build.
4. If no `.dmp` appears despite a failed job: LocalDumps may need `HKLM`
   instead of `HKCU` on GH-hosted runners (untested — flip it if `HKCU`
   turns out not to trigger; GH-hosted Windows runners run elevated by
   default, so `HKLM` should also be writable), or the crash may be a
   `SIGABRT`/CRT-detected error Windows doesn't route through WER the
   same way as a hard access violation — check the job log for which
   flavor it actually was before assuming the dump mechanism itself
   failed.
5. Once a dump's stack is in hand, update §A or §B above with it, and
   follow the elimination-trail style of `WindowsPythonHangDiagnostic354.md`
   for whatever hypothesis it points to.

### C.2 Reverting the temporary CI changes

Once this investigation concludes (root cause found, or deliberately
parked), restore the full matrix: `git revert` the commit that
introduced the trim (identify it via `git log --oneline -- .github/workflows/main.yml`,
the one whose message starts with the matrix-trim description — based on
`e4f076a6` at the time this was written). The `tools/r-windows/test.sh`/
`tools/windows/test.sh` repeat-knob changes are safe to keep (no-op by
default) or revert together with the matrix trim — either is fine.

## D. Native Windows VM debugging (fallback, if §C's dumps aren't enough)

Adapting `WindowsPythonHangDiagnostic354.md`'s approach (see that doc and
`win_debug.md` for full environment setup — VS Build Tools, CMake ≥ 4.2,
Rtools, WinDbg, Application Verifier):

### D.1 Build (mirrors `tools/r-windows/build.sh` / CI's `BUILD_NAME=r-windows`)

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

For §B (Julia Windows / core ctest), swap `ENABLE_R_BINDING=off`,
`ENABLE_JULIA_BINDING=on`, and build via `tools/windows/build.sh`
instead — same pattern, see that script for the exact flags.

### D.2 Isolate just the crashing block (§A)

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

### D.3 Isolate just the crashing tests (§B)

Build per D.1's Julia/core variant, then loop the 14 known-crashing tests
directly instead of the whole suite:

```bash
cd build_win_r  # or wherever BUILD_DIR pointed
for i in $(seq 1 100); do
  echo "=== attempt $i ==="
  ctest -C Release -R "Nystrom|Vecchia|NestedKriging|LinearAlgebra" --output-on-failure || break
done
```

Same `cdb.exe -c 'sxe av; g; .exr -1; kv; q'` / Application Verifier Page
Heap approach as D.2 applies here too, attached to whichever Catch2 test
`.exe` ctest launches for the specific test that ends up crashing (check
`ctest -N` or the build directory's `tests/` output for the exact `.exe`
name per test).

## Cleanup notes

Nothing has been committed to a scratch/throwaway state for this
investigation beyond the **intentional, documented-as-temporary** CI
changes in §C (matrix trim in `main.yml`, repeat-knob support in
`tools/r-windows/test.sh`/`tools/windows/test.sh`) — see §C.2 for how to
revert those once done. If a future session adds scratch repro files
(`bindings/R/repro_estimnone_noise.R`, dumped crash matrices/dumps,
throwaway Windows VM builds), list them here and remove before merging
anything, matching the cleanup discipline in
`WindowsPythonHangDiagnostic354.md`.
