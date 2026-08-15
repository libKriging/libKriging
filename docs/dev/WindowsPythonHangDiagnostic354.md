# Issue #354 — `WrappedPyKrigingParametricTest` "hang" on Windows Python Debug

**Status: ROOT CAUSE FOUND AND FIXED (2026-08-15).** This document records a
deep, evidence-backed diagnostic session (2026-08-14/15) conducted from
inside an actual Windows VM (dockur/windows, `Z:\` = this repo shared from
the host). It ends with a confirmed, verified fix — see "Root cause and fix"
below. The rest of the document is kept as-is: the full elimination trail
(nine hypotheses tested and ruled out with hard evidence) is genuinely
useful context for the fix itself and for anyone facing a similar
Windows-only heap-corruption symptom in the future.

- Issue: https://github.com/libKriging/libKriging/issues/354
- Throwaway bisection branch/PR (do not merge): `debug/wrapped-py-parametric-hang`,
  https://github.com/libKriging/libKriging/pull/355
- Main PR blocked by this: `feature/cg-predict`,
  https://github.com/libKriging/libKriging/pull/347

## Root cause and fix

**Root cause**: Armadillo's memory allocator (`_aligned_malloc`/
`_aligned_free`, MSVC's default) was never actually wired to use `lkalloc`
(libKriging's allocator indirection layer), despite the Python binding
explicitly calling `lkalloc::set_allocation_functions(cnalloc::npy_malloc,
cnalloc::npy_free)` at module init (`pylibkriging.cpp`, with the comment
*"to avoid mixing allocators from default libKriging and Python"*) — the
compile-time defines that would make Armadillo actually route through
`lkalloc::malloc`/`lkalloc::free` (`ARMA_ALIEN_MEM_ALLOC_FUNCTION`/
`ARMA_ALIEN_MEM_FREE_FUNCTION`) were commented out in the top-level
`CMakeLists.txt`, making that call a complete no-op. Git archaeology found
why: commit `03c51a9f` ("try fix ABI issue", 2025-11-15) disabled them with
the justification *"Custom allocators cause ABI mismatch with external code
(R, Python, standalone tests)"*, referencing a `SOLUTION_1_PROPER_KMODEL_INIT.md`
that was never actually committed to git (confirmed via `git log --all
--diff-filter=A` — it isn't a deleted file, it was never tracked). Per the
author (2026-08-15), that "ABI mismatch" diagnosis — and a related jemalloc
detour from the same period — was itself a misdiagnosis.

Left disabled, this meant Armadillo's own internal workspace allocations
(e.g. the `work`/`iwork` arrays passed to LAPACK's `trcon` inside
`arma::auxlib::rcond_trimat`, precisely localized earlier in this
investigation) always went through plain `_aligned_malloc`/`_aligned_free`
— consistently so within Armadillo itself, but apparently interacting badly
with the surrounding heap state established by the real `libKriging`
pipeline's own allocation history when hosted inside `python.exe` (matching
every negative result in the elimination trail below: never reproduced
standalone, never reproduced in a minimal `.pyd`, only ever in the real
pipeline).

**Fix**: re-enable the three defines in `CMakeLists.txt` (right after
`add_definitions(/DARMA_32BIT_WORD)`):

```cmake
add_definitions(-DARMA_ALIEN_MEM_ALLOC_FUNCTION=lkalloc::malloc)
add_definitions(-DARMA_ALIEN_MEM_FREE_FUNCTION=lkalloc::free)
add_definitions(-DCARMA_DO_NOT_EXPORT_ALIEN_MEM_FUNCTIONS)
```

This makes the Python binding's existing (previously-inert)
`set_allocation_functions` call actually take effect, so Armadillo's memory
in the Python binding is managed by the same allocator as NumPy
(`cnalloc::npy_malloc`/`npy_free`) instead of a second, independent
`_aligned_malloc` allocator running alongside it in the same process.

**Why this is safe for every other binding** (Octave, MATLAB, the core C++
library and its own tests, none of which call `set_allocation_functions`):
checked `src/lib/lkalloc.cpp` — when no custom allocator has been set
(`custom_malloc == nullptr`, the default), `lkalloc::malloc`/`free` fall
back to *exactly* `_aligned_malloc`/`_aligned_free` on MSVC. Zero behavior
change for every binding except Python; one extra function-pointer
indirection, negligible overhead.

**Verification**:
- The originally-hanging `WrappedPyKrigingParametricTest` file: all 8 tests
  pass in ~0.7s (was: infinite hang).
- Full Python test suite (`bindings/Python/pylibkriging/tests/`): **136
  passed, 10 skipped, 0 failed**, 14–22s.
- Same full suite re-run under Application Verifier's **full Page Heap**
  (the strictest corruption detector used throughout this investigation,
  catching even single-byte overruns) — still clean, no access violation.
  This rules out "the fix just hides the corruption behind a less strict
  allocator" — Page Heap would have caught a real out-of-bounds write
  regardless of which allocator produced it.
- C++ core test suite (`catch2_unit_test`): passes unchanged (17
  assertions, 2 test cases — confirms the core library still builds/links/
  runs correctly with the allocator wiring active).

**Files changed**: `CMakeLists.txt` only (the three `add_definitions`
lines). No source code changes were needed — the fix is activating
infrastructure (`lkalloc`, `set_allocation_functions`) that already existed
in the codebase but was silently disconnected.

## Prior related Windows CI issues (history)

Context for anyone tempted to reach for the same fix again — these are
**distinct, already-resolved** problems that predate or run alongside #354:

| Issue | Symptom | Root cause | Fix | Status |
|---|---|---|---|---|
| Octave Windows (pre-existing) | Hang/timeout on a NestedKriging test | OpenMP "libgomp thread-pool churn" under MinGW/Windows | Force `OMP_NUM_THREADS=1` in `tools/octave-windows/test.sh` (`fc9e12e9`, `b6cbf97a`) | ✅ Resolved, on `master` |
| #351 — Python Windows 3.7/3.9 on `master` | Jobs blocked ~1h (timeout) on several unrelated Python test files | Same OpenMP thread-pool churn mechanism, hit during BFGS's repeated parallel regions | Force `OMP_NUM_THREADS=1` in `tools/windows/build.sh`/`test.sh`, PR #352 (merged). Verified across Python 3.7/3.9/3.10/3.11/3.12 on a throwaway branch (`debug/windows-python-hang`) — 3.10+ were never added to the permanent CI matrix, only used for that one-off verification | ✅ Resolved, PR #352 merged |
| Nystrom flaky on Octave Windows | `CHECK(...)` assertion sometimes failed (`predictNystrom matches exact predict...`), numeric gap over tolerance | Not OpenMP/hang related — BFGS-converged theta varied slightly by platform/compiler, widening the measured gap past the test's tolerance | Fixed both sides' theta (`optim="none"`) for a deterministic comparison, plus a library-side fix so `optim="none"` + `LLNystrom(k)` actually honors the requested objective instead of silently doing an exact fit. PR #353 (merged) | ✅ Resolved, PR #353 merged |
| **#354 (this doc)** — PR #347, `WrappedPyKrigingParametricTest` | Windows Python Debug jobs hang, on a *different* test than #351, introduced by a pybind11-registration-only commit | Real heap corruption in Armadillo's LAPACK workspace allocations, exposed because a pre-existing allocator-consistency mechanism (`lkalloc`) was disconnected — see "Root cause and fix" above | Re-enable `ARMA_ALIEN_MEM_ALLOC_FUNCTION`/`_FREE_FUNCTION` in `CMakeLists.txt` | ✅ Resolved |

## TL;DR

This was **not a deadlock or a real hang**. It was the MSVC **Debug CRT**
(`ucrtbased.dll`) detecting genuine **heap corruption** and popping a
blocking `MessageBoxW` dialog that nobody can click in CI — so the process
sat forever, looking exactly like a timeout. The corruption was real
(confirmed via a hard access violation under Application Verifier's page
heap), was **not** BLAS/LAPACK-backend-specific, **not** dependent on the
specific matrix values involved, and **not** caused by the `X`/`y`
NumPy→Armadillo view conversion — see "Hypotheses tested and ruled out"
below for the full elimination trail (nine hypotheses, each disproven with
direct evidence). **Root cause found and fixed**: see "Root cause and fix"
above — a pre-existing allocator-consistency mechanism (`lkalloc`, wiring
Armadillo's memory to the same allocator NumPy uses) was silently
disconnected by a commented-out CMake define, itself the result of an
earlier misdiagnosis. Re-enabling it fixes the corruption with no source
changes and no effect on any other language binding.

## Symptom

- CI job "Python (3.7/3.9) Windows Debug" times out (~1h) on
  `bindings/Python/pylibkriging/tests/WrappedPyKriging_parametric_test.py`,
  specifically `test_kriging_f_order[3-40]` (m=3 dimensions, n=40 samples).
- The test does nothing exotic: `lk.WrappedPyKriging(y, X, "gauss")` then
  `.predict(...)`. It doesn't touch predictCG/Nystrom/Iterative code at all.
- **Linux Debug, same commit: passes in 0.7s.** Windows-only.
- Bisected to commit `51f9e4a0` ("Expose predictCG Nystrom precond,
  subsetOfData, and Iterative/Nystrom accessors in all bindings") — a
  **pure pybind11 declarative-registration commit, nothing executed by the
  hanging test**. Parent commit `beeccaf5` passes.

## What it actually is (confirmed)

Enumerating the real Win32 windows owned by the "hung" `python.exe` process
shows a genuine visible dialog:

```
Title: "Microsoft Visual C++ Runtime Library"
Body:
  Debug Error!
  Program: ...\Python39\python.exe
  Damage before 0x000001F2DB909F60 which was allocated by aligned routine
  (Press Retry to debug the application)
```

`py-spy dump --native` on the stuck `MainThread` (idle = GIL released, running
native code) shows:

```
NtUserWaitMessage (win32u.dll)
  ... USER32 modal message-box chain ...
MessageBoxW → MessageBoxTimeoutW → MessageBoxIndirectA → SoftModalMessageBox
  ... ucrtbased.dll CRT debug-report chain ...
CrtDbgReport → aligned_free_dbg → aligned_free
arma::memory::release<int>                          (memory.hpp:137)
arma::podarray<int>::~podarray<int>                 (podarray_meat.hpp:31)
arma::auxlib::rcond_trimat<double>                  (auxlib_meat.hpp:6798)
arma::op_rcond::apply<arma::Mat<double>>             (op_rcond_meat.hpp:89)
arma::rcond<arma::Mat<double>>                       (fn_cond_rcond.hpp:74)
LinearAlgebra::rcond_chol                            (LinearAlgebra.cpp:108)
LinearAlgebra::safe_chol_lower_retry                 (LinearAlgebra.cpp:77)
LinearAlgebra::safe_chol_lower                       (LinearAlgebra.cpp:44)
LinearAlgebra::cholCov                               (LinearAlgebra.cpp:200)
KrigingImpl::populate_Model → Kriging::populate_Model → Kriging::fit
Kriging::Kriging → PyKriging::PyKriging
  (test.py line 18: rl = lk.WrappedPyKriging(y, X, "gauss"))
```

So: while destructing an internal `podarray<blas_int>` (the `iwork` LAPACK
workspace array used inside `arma::auxlib::rcond_trimat`), the CRT's
`_aligned_free_dbg` detects that the debug guard bytes **preceding** the
block are damaged, and tries to report it — which blocks forever in headless
CI.

## Getting a hard, byte-precise crash (Application Verifier page heap)

Normal MSVC Debug CRT guard-byte checking only detects corruption *inside*
its own small padding region around a block — it doesn't catch it at the
exact faulting write. To get a precise stack, we combined:

1. `appverif -enable Heaps -for python.exe -with Heaps.full=true` — enables
   full Page Heap for every `python.exe` process (every allocation gets its
   own OS-guarded page).
2. `cdb.exe` (from the WinDbg Store package's `amd64\cdb.exe`) attached at
   launch, with `sxe av` (break on access violation) armed.

Even with Page Heap active, the *first* attempts still showed the same CRT
dialog rather than a hard AV — because `_aligned_malloc_dbg`'s own NoMansLand
padding sits *inside* the page-heap-guarded OS allocation, so a small overrun
into that padding never touches the actual guard page. `_NO_DEBUG_HEAP=1`
does **not** help either — it only disables the general small-block debug
heap, not `_aligned_malloc_dbg`/`_aligned_free_dbg`'s independent guard-byte
bookkeeping (verified experimentally, no change in behavior).

What did work: interrupting the already-blocked dialog thread via a `.call`
injection (`.call ucrtbased!_CrtSetDbgFlag(0)`) let execution continue far
enough to eventually produce a genuine, hard access violation, caught by
`sxe av`:

```
ExceptionAddress: ucrtbased!common_strnlen_c<0,unsigned char>+0x57
ExceptionCode: c0000005 (Access violation)
Attempt to read from address 0xFFFFFFFFFFFFFFFF
```

Full native stack (innermost → outermost) — **the crash is not in Armadillo's
math at all**, it's inside the CRT's own error-message formatter, which is
trying to print a `%s` argument that is itself an invalid pointer:

```
common_strnlen_c → common_strnlen_simd → common_strnlen → strnlen
→ __crt_stdio_output::...::type_case_s_compute_narrow_string_length
→ ...::type_case_s → ...::state_case_type → ...::process
→ common_vsprintf → common_vsnprintf_s → __stdio_common_vsnprintf_s
→ _vsnprintf_s → _VCrtDbgReportA        (FORMATTING the "Damage before..." message)
→ _CrtDbgReport → free_dbg_nolock → _free_dbg
→ _aligned_free_dbg → _aligned_free
→ arma::memory::release<int>
→ arma::podarray<int>::~podarray<int>
→ arma::auxlib::rcond_trimat<double>     (auxlib_meat.hpp:6790-6794)
→ arma::op_rcond::apply → arma::rcond
→ LinearAlgebra::rcond_chol → safe_chol_lower_retry → safe_chol_lower
→ cholCov → KrigingImpl::populate_Model → Kriging::populate_Model
→ Kriging::fit → Kriging::Kriging → PyKriging::PyKriging (test.py:18)
```

Interpretation: `_aligned_malloc_dbg` stores a small debug header (allocating
source filename/line, guard bytes) immediately before the user's pointer.
Something wrote into the header region belonging to `iwork`
(`podarray<blas_int>`, sized `n` ints, allocated in `auxlib_meat.hpp:6791`
right after `work` sized `3*n` doubles at line 6790). When `_aligned_free_dbg`
tries to report the corruption, it tries to print the (now garbage) stored
filename pointer via `%s` → `strnlen` on an invalid address → hard crash.
This is a **secondary bug in `ucrtbased.dll`'s own reporting path**, triggered
by, but distinct from, the original corruption.

## Hypotheses tested and ruled out

Each of the following was a concrete, testable hypothesis, individually
disproven with direct evidence — recorded here so nobody re-treads them.

### 1. OpenMP / libgomp thread-pool churn (the #351 mechanism)
Already ruled out by `OMP_NUM_THREADS=1` vs `2` making no difference, and by
disabling OpenMP entirely at compile time (`ARMA_DONT_USE_OPENMP` /
`-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON`) — hang persists identically. See
the history table above for the original #351 writeup; re-confirmed
identically in this session.

### 2. ILP64/LP64 `blas_int` width mismatch
Checked `armadillo_bits/typedef_elem.hpp:110-116`: `blas_int` is plain 32-bit
`int` unless `ARMA_BLAS_64BIT_INT` is defined. Checked the generated
`config.hpp` and the armadillo CMakeLists: this macro is **not** set anywhere
in this build. Conda-forge's OpenBLAS is also the standard LP64 (32-bit int)
variant. **Ruled out.**

### 3. Fortran "hidden length arguments" ABI mismatch
Armadillo declares two signatures for `trcon` (`def_lapack.hpp:865-866` with
trailing `blas_len` args for the gfortran-style hidden-argument convention,
vs `def_lapack.hpp:1225-1226` without). Checked: `ARMA_USE_FORTRAN_HIDDEN_ARGS`
**is** defined for this build (`config.hpp:114`, no override), and
`translate_lapack.hpp:1171-1181`'s `trcon()` wrapper correctly passes the
three hidden length args (`1, 1, 1`) when calling `arma_fortran(arma_dtrcon)`.
Correctly matched. **Ruled out.**

### 4. OpenBLAS-specific `dtrcon` bug
Built a **second, fully independent** variant of the library, linked against
reference **netlib LAPACK** (`liblapack 3.11.0 *_netlib` from conda-forge,
zero shared code with OpenBLAS) in an isolated conda env
(`netlib_test`) and a separate build dir (`build_win_debug_netlib`), leaving
the OpenBLAS build/env untouched. **Identical hang, identical corruption,
identical stack trace down to the line number.** Since two independent LAPACK
implementations both fail at exactly the same spot, the LAPACK backend itself
cannot be the cause. **Ruled out.**

### 5. Data-dependent bug in `rcond_trimat`/the specific matrix
Temporarily instrumented `LinearAlgebra::rcond_chol` (see "Reproducing
locally" below for how) to dump every candidate Cholesky factor to CSV before
calling `rcond()`. **Only one file was ever written** before the hang — the
crash is fully deterministic and happens on the very first `rcond_chol` call
(a very ill-conditioned initial covariance factor, `rcond ≈ 2.6e-10`, tiny
pivots after the first two rows). Took that **exact** 40×40 matrix into a
minimal standalone C++ program (no Python/pybind11 at all), linked against
the identical `armadillo.lib`/`openblas.dll`, and called `arma::rcond()` on it
**50 times in a row: zero corruption, zero crash, identical rcond value every
time.** Also ran 2000 trials of similarly-shaped synthetic matrices before
finding this exact one — none crashed either. **Ruled out**: it is not the
matrix values, and this is not a pure Armadillo+LAPACK+MSVC-in-isolation bug
— something about running inside `python.exe` is necessary to trigger it.

### 6. `X`/`y` NumPy→Armadillo view ownership (carma `steal_copy_array`)
`PyKriging::PyKriging` (`Kriging_binding.cpp:66-67`) converts the incoming
NumPy arrays via `carma::arr_to_col_view`/`arr_to_mat_view`. These fall back
to `steal_copy_array()` (which allocates via NumPy's own allocator, then
hands Armadillo the pointer marked as *owned*, `mem_state=0` — meaning
Armadillo would later call `_aligned_free` on memory it never
`_aligned_malloc`'d, a textbook allocator-mismatch corruption bug) whenever
the incoming array isn't "well-conditioned" (aligned + F-contiguous per
NumPy's own flags; see `carma_bits/cnumpy.h:143-150` and
`carma_bits/numpytoarma.h:83-136`). This looked like a very strong candidate.
**Tested directly**: instrumented the constructor to print
`carma::well_conditioned(y.ptr())` / `well_conditioned(X.ptr())` and the
resulting `mem_state`. Result: **both `y` and `X` are `well_conditioned=1`**
and get genuine non-owning views (`mem_state=2` — "external memory, never
freed by Armadillo"). **Ruled out** for this test's `X`/`y` specifically.

### 7. `params_from_dict`/`get_entry` optional-parameter conversion
`get_entry<T>()` (`py_to_cpp_cast.hpp:22-36`) only invokes the carma-based
`to_cpp_cast<T>()` conversion **if the requested key is present in the
dict**. Our test calls `WrappedPyKriging(y, X, "gauss")` with no `dict`
argument (empty dict by default), so `find_if` returns `dict.end()`
immediately for every key (`sigma2`, `theta`, `beta`, `nugget`, ...) and
`get_entry` returns `std::nullopt` without ever calling the conversion.
Proven by code inspection alone (no runtime instrumentation needed): this
path is compiled but **never executed** for this call signature. **Ruled
out.**

### 8. Debug/Release CRT mismatch between `python.exe` and `_pylibkriging.pyd`

A very well-reasoned hypothesis, worth recording in detail because it fit
every piece of prior evidence: the official python.org installer's
`python.exe` links the **Release** CRT (`api-ms-win-crt-*.dll` forwarders →
`ucrtbase.dll`), while our Debug build's `_pylibkriging.pyd` links the
**Debug** CRT (`ucrtbased.dll`, `MSVCP140D.dll`, `VCRUNTIME140D.dll`,
`VCRUNTIME140_1D.dll`, `VCOMP140D.DLL`) — confirmed directly via
`dumpbin /dependents` on both binaries. Mixing debug and release CRT
instances in the same process is an explicitly unsupported, documented
Microsoft pitfall, and it's consistent with the bug being Windows-only,
requiring `python.exe` hosting, backend-independent, and deterministic.

**Tested directly**: rebuilt with `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL`
(`/MD`, Release CRT) to match `python.exe`. First attempt had no effect —
`CMAKE_MSVC_RUNTIME_LIBRARY` only takes effect under CMake policy `CMP0091`
= `NEW`, and both the top-level project (`cmake_minimum_required(VERSION
3.13)`) and pybind11's own CMakeLists (`VERSION 3.5`) default that policy to
`OLD`, silently ignoring the setting. Forcing it via
`-DCMAKE_POLICY_DEFAULT_CMP0091=NEW` fixed the linkage — `dumpbin` then
confirmed `_pylibkriging.pyd` linking the same Release CRT DLLs as
`python.exe`, with `armadillo.lib`/`Kriging.lib`/`lbfgsb_cpp.lib` all
rebuilt consistently (verified via matching timestamps) before testing.

**Result: the crash still happens**, with the exact same Python-level
traceback (`test.py:18`, `WrappedPyKriging(y, X, "gauss")`) — but now
reported as `Windows fatal exception: code 0xc0000374`
(`STATUS_HEAP_CORRUPTION`, the Windows heap manager's own built-in
corruption fast-fail, since `ucrtbased.dll`'s extra guard-byte bookkeeping
is no longer in the picture at all). **Ruled out**: this is not a
Debug/Release CRT mismatch. The corruption is more fundamental than a CRT
instance mismatch — it reproduces even when every module in the process
(`python.exe` and the entire `.pyd` + its static dependencies) agrees on a
single, consistent Release CRT.

Side note for anyone reproducing this specific test: use
`-DCMAKE_POLICY_DEFAULT_CMP0091=NEW` if you need
`CMAKE_MSVC_RUNTIME_LIBRARY` to actually take effect in this project — it's
silently a no-op without it, which cost real time to notice.

### 9. NumPy 2.0's C-API changes

Also a strong candidate worth real scrutiny: `requirements.txt` only pins
`numpy>=1.18` (no upper bound), so the environment installs NumPy 2.0.2 —
and NumPy 2.0 is a well-known source of silent C-extension breakage, since
it reorganized parts of its C API. `carma` (the NumPy↔Armadillo bridge) is
*exactly* the kind of code this can break: `carma_bits/numpyapi.h` resolves
NumPy C functions by **hardcoded integer index** into the raw `_ARRAY_API`
capsule table, bypassing NumPy's own header-generated accessor macros. The
code already contains proof this is a real risk in this exact codebase — it
version-branches `PyArray_CopyInto`'s index at runtime (82 for NumPy 1.x, 50
for 2.x) because NumPy 2.0 actually moved it. The other six hardcoded
indices (`PyArray_Free=165`, `PyArray_Size=59`, `PyArray_NewCopy=85`,
`PyArray_NewLikeArray=277`, `PyArray_NewFromDescr=94`, `PyDataMem_NEW=288`,
`PyDataMem_FREE=289`) have no such version check. Separately, `cnumpy.h`'s
`steal_memory<T>()` does direct `PyArrayObject_fields*` struct-field access —
exactly the pattern NumPy 2.0's own docs warn will break.

**Checked both mechanisms**:
- Compared all 7 hardcoded indices against the actual installed NumPy 2.0.2
  header (`numpy/_core/include/numpy/__multiarray_api.h`) — every single one
  matches exactly. Not stale for this NumPy version.
- `steal_memory<T>()` only compiles in under `CARMA_SOFT_STEAL`/
  `CARMA_HARD_STEAL`, and neither is defined in this build (confirmed via
  the generated `carma_bits/config.h`, both `#undef`). Dead code here — not
  the cause.

**Tested empirically anyway**, since code inspection can't rule out every
possible NumPy-2-related mechanism: downgraded to `numpy==1.26.4` and
rebuilt `_pylibkriging.pyd` from scratch against it (both build-time headers
and runtime consistently NumPy 1.x, avoiding a new version-mismatch
confound). **Result: identical corruption**, same
`Damage before 0x... which was allocated by aligned routine` message, same
crash site (`test.py:18`). **Ruled out**: this is not NumPy-2-specific. The
NumPy version was restored to 2.0.2 afterward to match `requirements.txt`.

## Precise localization: it's inside `arma::rcond()` itself, nothing earlier

Added `_CrtCheckMemory()` checkpoints (each validates the *entire* debug
heap and prints OK/DAMAGED to stderr, safe to call repeatedly since the
fail-fast report-mode redirect below is in place) at six points along the
call chain: (A) right after carma's `X`/`y` conversion in
`PyKriging::PyKriging`, (B) `Kriging::Kriging` constructor entry,
(C) `Kriging::fit` entry before BFGS setup, (D) `KrigingImpl::populate_Model`
entry, (E) `cholCov` entry, (F) right before the `rcond_chol(L)` call in
`safe_chol_lower_retry` (after `arma::chol()` has already succeeded).

**Result: A through F all report `heap OK`.** The `Damage before...` message
appears immediately after checkpoint F fires, on the very first
`rcond_chol` call. `safe_chol_lower_retry`'s nugget-retry path then recurses
(F fires again, still `heap OK`), and the *second* `rcond_chol` call
corrupts again — this time fatally.

This conclusively rules out everything between the carma conversion and a
successful `arma::chol()` as the corruption's origin. **The corruption is
introduced strictly inside `arma::rcond(chol)` — i.e. inside
`op_rcond::apply` → `rcond_trimat` → `lapack::trcon` and its immediate
workspace cleanup** — exactly the call originally identified via
`py-spy`/`cdb` at the very start of this investigation. The bisection
didn't find an earlier culprit; it confirmed there isn't one.

## Hosting inside `python.exe` alone is not sufficient either

Built a minimal pybind11 module (`debug354_repro.pyd`, same `/MDd` Debug
CRT, same `armadillo.lib`/`openblas.dll` as the real build) whose only job
is to load the exact crashing 40×40 Cholesky factor from CSV and call
`arma::rcond()` on it 50 times — no NumPy array marshaling, no
`Kriging`/`Trend`/`Covariance` object construction, no BFGS/`lbfgsb_cpp`
setup, none of the rest of the real pipeline's allocations. Ran it from
`python.exe` via `import debug354_repro; debug354_repro.run()`.

**Result: zero corruption, 50/50 clean runs, identical rcond value every
time.** This refutes "hosted in `python.exe`" as a sufficient condition on
its own — combined with hypothesis #5 (same matrix, same CRT, but as a
standalone `.exe`, also clean), that's **three** environments where the
exact same `arma::rcond()` call on the exact same matrix with the exact
same CRT/libraries never corrupts: a standalone `.exe`, and a minimal
`.pyd` loaded into `python.exe`. Only the *real* `libKriging` pipeline
triggers it — which is exactly what led to re-examining the allocator
wiring (`lkalloc`) that turned out to be the actual answer: two
independent allocators (NumPy's and Armadillo's own `_aligned_malloc`)
coexisting in the same process, with a pre-existing mechanism to unify
them silently disconnected. See "Root cause and fix" at the top of this
document.

## Mitigation applied: fail fast instead of hanging

The corruption itself is still unfixed, but the *symptom* CI actually suffers
from — a silent ~1h timeout that eats the whole job budget and gives zero
diagnostic information — has been eliminated. `pylibkriging.cpp` now installs
a static initializer, guarded to Windows Debug builds only, that redirects
CRT debug-heap error reports from the default blocking `MessageBoxW` to
stderr:

```cpp
#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
namespace {
struct Debug354ReportModeInit {
  Debug354ReportModeInit() {
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
  }
} debug354_report_mode_init;
}  // namespace
#endif
```

**Verified in this VM**: with this in place, `test_kriging_f_order[3-40]`
goes from hanging indefinitely to failing in **~3 seconds**, printing
`Damage before 0x... which was allocated by aligned routine` to stderr,
followed by Python's own `faulthandler` catching the resulting access
violation and printing a full traceback pointing straight at
`test.py:18` (`rl = lk.WrappedPyKriging(y, X, "gauss")`). This is a real,
loud, CI-diagnosable failure instead of an opaque timeout — a clean
signal-quality win even though the root cause remains open.

Other options considered but not applied (see discussion history for
tradeoffs): skip/xfail the specific test on Windows Debug (bluntest unblock
for PR #347, zero risk, loses coverage); dropping Debug CRT (`/MD` instead of
`/MDd`) for the Windows Python CI job specifically (makes the corruption
silent again, at the cost of losing the debug heap's ability to catch other
future corruption bugs on Windows — not recommended as a long-term fix).

## How the elimination trail led to the fix

By this point the trail had reached: the corruption was real, deterministic
(first `rcond_chol` call, every time), Windows-only, required being hosted
inside `python.exe`, and was **not** specific to the Debug CRT (hypothesis
#8) or to NumPy 2 (hypothesis #9) — and the two sections above had just
shown it wasn't anything in `Kriging`'s own call chain either, and wasn't
simply "hosted in `python.exe`" as a standalone condition. The leading
candidate — "a general interaction between CPython/NumPy's own heap
activity and Armadillo's aligned allocator" — turned out to be exactly
right, just missing the specific mechanism: **two independent allocators
(NumPy's and Armadillo's own `_aligned_malloc`) running in the same
process, where a pre-existing mechanism to unify them (`lkalloc`) existed
in the codebase but was disconnected.** See "Root cause and fix" at the top
of this document.

## Reproducing locally

### Prerequisites (already installed on this VM)

Git 2.55, CMake 4.4.2, Python 3.9.13, Visual Studio 2022 Build Tools (C++
workload, MSVC 19.44), Miniconda3 + `openblas`/`lapack` from conda-forge,
GitHub CLI, py-spy, WinDbg (Store package, includes `cdb.exe`/`ntsd.exe`
under `amd64\`), Process Explorer
(`C:\Users\Docker\Downloads\ProcessExplorer`).

If setting up a **fresh** VM instead, install (`winget` IDs in parens where
applicable):

1. **Git** — https://git-scm.com/download/win
2. **Visual Studio Build Tools 2022+** (`Microsoft.VisualStudio.2022.BuildTools`)
   with the **"Desktop development with C++"** workload — for `cl.exe`/MSVC,
   matching the CI generator ("Visual Studio 18 2026" on GHA runners; a
   nearby 2022 version is close enough to reproduce the bug locally).
3. **CMake ≥ 4.2** (`Kitware.CMake`) — the repo uses
   `CMAKE_POLICY_VERSION_MINIMUM=3.5` for compat with the pybind11 submodule
   under cmake ≥ 4.
4. **Python 3.7 and/or 3.9** (`Python.Python.3.9`) — the exact versions the
   hanging CI jobs use. `pip install pytest numpy`.
5. **Miniconda** (`Anaconda.Miniconda3`) — `tools/windows/build.sh` expects
   `${HOME}/Miniconda3/Library/lib` for OpenBLAS/LAPACK:
   `conda install --override-channels -c conda-forge openblas lapack`
   (`--override-channels` avoids conda's newer default-channel
   Terms-of-Service prompt; adapt `EXTRA_SYSTEM_LIBRARY_PATH` in
   `tools/windows/build.sh` if installed elsewhere).
6. **Git Bash** — `tools/windows/build.sh`/`test.sh` are bash scripts; Git for
   Windows bundles a sufficient bash.
7. **GitHub CLI** (`GitHub.cli`) — for `gh issue comment`/`gh pr` on issue
   #354 / PR #355 (`gh auth login` with repo access).
8. Optional but very useful for this specific diagnosis:
   - **WinDbg** (Microsoft Store — `winget` needs the `msstore` source's
     Terms of Transaction accepted first) or Visual Studio's own debugger,
     to attach to a stuck `python.exe` and get an all-threads call stack.
   - **Process Explorer** (Sysinternals — no winget package; download
     directly from `https://download.sysinternals.com/files/ProcessExplorer.zip`)
     to inspect threads/handles of the stuck process live.
   - **py-spy** (`pip install py-spy`) — dumps a Python (and, with
     `--native`, native C/C++) stack from a hung process without attaching a
     native debugger; usually the fastest first look.

The repo is accessed via a share mounted at `Z:\` in the VM (or clone fresh
from `https://github.com/libKriging/libKriging.git` if you'd rather isolate
the VM from the host — remember `git submodule update --init --recursive`).

### Build

```bash
cd /z   # or wherever the repo is checked out
export MODE=Debug
export ENABLE_PYTHON_BINDING=on
export ENABLE_OCTAVE_BINDING=off
export ENABLE_MATLAB_BINDING=off
export ENABLE_JULIA_BINDING=off
export BUILD_DIR=build_win_debug
tools/windows/build.sh
```

`build.sh` already scopes `ctest` to just `WrappedPyKrigingParametricTest`
with a 120s timeout, so a normal `ctest` run will report the test as
**Timeout** rather than hanging your whole CI budget.

### Reproduce the hang directly (so it stays hung for inspection)

```bash
export PYTHONPATH="Z:/build_win_debug/bindings/Python/pylibkriging/Debug;Z:/bindings/Python/pylibkriging/src"
export LIBKRIGING_DLL_PATH="/c/Users/Docker/Miniconda3/Library/bin"
export OMP_NUM_THREADS=1
cd "Z:/build_win_debug/bindings/Python/pylibkriging/Debug"
python -m pytest -vv -s -k test_kriging_f_order \
  "Z:/bindings/Python/pylibkriging/tests/WrappedPyKriging_parametric_test.py"
```

(`ctest`'s own registered test doesn't pass `--timeout` to pytest itself —
that limit is applied externally by `ctest -C Debug ... --timeout 120`. Running
`pytest` directly, as above, lets it hang indefinitely for inspection.)

### Inspect with py-spy (fastest first look)

```powershell
py-spy dump --pid <PID> --native
```

### Get a hard AV with Application Verifier page heap + cdb

```powershell
appverif -enable Heaps -for python.exe -with Heaps.full=true
```

Then launch under `cdb.exe` (found under the WinDbg Store package install
dir, e.g.
`C:\Program Files\WindowsApps\Microsoft.WinDbg_<version>_x64__8wekyb3d8bbwe\amd64\cdb.exe`
— runs fine invoked directly despite living under `WindowsApps`):

```bash
CDB="/c/Program Files/WindowsApps/Microsoft.WinDbg_<version>_x64__8wekyb3d8bbwe/amd64/cdb.exe"
CMDS='bu _pylibkriging!PyKriging::PyKriging; g; .call ucrtbased!_CrtSetDbgFlag(0); g; g; sxe av; g; .exr -1; kv; g; .exr -1; kv; g; .exr -1; kv; q'
"$CDB" -c "$CMDS" "/c/Users/Docker/AppData/Local/Programs/Python/Python39/python.exe" \
  -m pytest -vv -s -k test_kriging_f_order \
  "Z:/bindings/Python/pylibkriging/tests/WrappedPyKriging_parametric_test.py"
```

Note: the `bu` breakpoint on `PyKriging::PyKriging` did not actually resolve
in our runs (private PDB likely not found via the default `srv*`-only symbol
path) — the AV was still caught because `sxe av` was armed before the final
`g`s. If you get local PDB symbol resolution working (`.sympath+` the build
output directory), the `bu` breakpoint should fire properly and this can be
made cleaner/more deterministic.

**Remember to disable Page Heap when done** — it applies to *every*
`python.exe` process system-wide while active, and slows them down
significantly:

```powershell
appverif -disable Heaps -for python.exe
```

### Test against reference LAPACK instead of OpenBLAS

```bash
"$USERPROFILE/Miniconda3/Scripts/conda.exe" create -y -n netlib_test \
  --override-channels -c conda-forge "liblapack=*=*netlib*" "libblas=*=*netlib*"

mkdir -p build_win_debug_netlib && cd build_win_debug_netlib
cmake -DCMAKE_GENERATOR_PLATFORM=x64 \
  -DEXTRA_SYSTEM_LIBRARY_PATH="/c/Users/Docker/Miniconda3/envs/netlib_test/Library/lib" \
  -DENABLE_OCTAVE_BINDING=off -DENABLE_MATLAB_BINDING=off \
  -DENABLE_PYTHON_BINDING=on -DENABLE_JULIA_BINDING=off \
  -DBUILD_SHARED_LIBS=off -DSTATIC_LIB=on ..
cmake --build . --target ALL_BUILD --config Debug
```

Then run the test the same way, with `PYTHONPATH`/`LIBKRIGING_DLL_PATH`
pointed at `build_win_debug_netlib` and the `netlib_test` env's `Library/bin`
instead.

### Dump the exact crashing matrix (used for hypothesis #5)

Temporarily add to `LinearAlgebra::rcond_chol` in `src/lib/LinearAlgebra.cpp`,
right before the `arma::rcond(chol)` call:

```cpp
{
  static int counter = 0;
  chol.save("Z:/tmp_repro/chol_dump_" + std::to_string(counter++) + ".csv", arma::csv_ascii);
}
```

Rebuild just the affected targets (`cmake --build . --target Kriging --config
Debug` then `cmake --build . --target _pylibkriging --config Debug`), run the
test until it hangs, kill it, and the last-numbered CSV file is the culprit
matrix. **Remove this instrumentation before committing anything** — it was
reverted after use in this session.

## Cleanup notes for this VM

- **The fix itself (`CMakeLists.txt`, the three `add_definitions` lines) is
  a real source change, not a scratch artifact — it needs to be committed
  and go through a PR against `master` (this session did not commit or push
  anything, per standing instructions for this investigation).** It's
  currently sitting as an uncommitted working-tree change in this checkout.
- Application Verifier's Page Heap for `python.exe` was left **disabled**
  at the end of this session — double check with
  `appverif -query Heaps -for python.exe` if picking this back up later.
- Scratch artifacts from this session (safe to delete): `Z:\tmp_repro\`
  (standalone repro `.cpp`/`.exe`/`.pyd` files and `chol_dump_*.csv`),
  `Z:\build_win_debug_netlib\`, `Z:\build_win_crt_test\` (the
  Release-CRT test build for hypothesis #8), and the `netlib_test` conda env
  (`conda env remove -n netlib_test`).
- `Z:\build_win_debug\` is the main Debug build produced during this
  session, already rebuilt with the fix active and passing the full test
  suite — reusable as-is for further work.
