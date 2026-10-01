# Windows native crashes (R binding + core ctest cluster) — investigation

**Status: ROOT-CAUSED (2026-10-01), fix applied, awaiting confirmation
on an AVX-512 Zen 4 VM.** It's an **OpenBLAS 0.3.34 regression**, not a
libKriging bug: see [§0](#0-root-cause). Sections §1-§4 keep the
investigation trail that led there. Where they disagree with §0, §0 wins
(in particular, §1.1's "OpenBLAS did not change" was wrong).

## 0. Root cause

**OpenBLAS 0.3.34 overflows the stack inside `dgemm_kernel_ZEN` /
`dgemm_kernel_HASWELL` on AMD Zen 4/5 hosts that expose AVX-512**, in
`DYNAMIC_ARCH` + `NO_AVX512` clang-cl Windows builds, which is exactly
conda-forge's `libopenblas`. Upstream:
[OpenMathLib/OpenBLAS#6013](https://github.com/OpenMathLib/OpenBLAS/issues/6013)
(0.3.33 fine) and
[#6021](https://github.com/OpenMathLib/OpenBLAS/issues/6021). The cause
is a Zen 4 GEMM P/Q blocking override in `kernel/setparam-ref.c`
`init_parameter()`. It fires when the CPU is AuthenticAMD, L2 = 1024 KB,
L3 % 32 MB == 0 and AVX512F is in CPUID, and it also applies to forced
coretypes.

Evidence, from the first instrumented run (run 36841355421, Julia job
110300961652, a bad VM on the first try):

- CPU **AMD EPYC 9V74 (Zen 4, Family 25 Model 17)**, L2 1024 KB/core
  (WMI reports 2048 for 2 cores), L3 32 MB. OpenBLAS core `Zen`.
  `OPENBLAS_CORETYPE=SkylakeX` falls back to `Haswell`, consistent with a
  `NO_AVX512` build.
- `cdb`: first-chance access violation at the **`ret` of
  `openblas!dgemm_kernel_ZEN+0x6ea6`**, return address non-canonical, the
  whole visible stack overwritten with doubles 0.97-0.999 (covariance
  values). A stack smash inside OpenBLAS; no libKriging frame involved.
- Probe matrix on that VM:

  | Variant | T1 LLNystrom gradient (n=150) | T2 rapid fire (50 threads) |
  |---|---|---|
  | baseline (Zen) | **crash** | **crash** |
  | `OPENBLAS_NUM_THREADS=1` | **crash** | pass |
  | `OMP_NUM_THREADS=1` | **crash** | pass |
  | `OPENBLAS_CORETYPE=Haswell` | **crash** | **crash** |
  | `OPENBLAS_CORETYPE=SkylakeX` (→Haswell) | **crash** | **crash** |
  | `OPENBLAS_CORETYPE=Prescott` | pass | pass |

  The override doesn't apply to Prescott kernels, and a CORETYPE pin
  doesn't escape it (#6021). That also explains §1.6's
  threaded/unthreaded split.
- **Timing**: CI logs show `openblas-0.3.33` until 2026-07-11 and
  `openblas-0.3.34` from 2026-07-18, the day of the very first crash.
  §1.1's statement that the version was constant was wrong: the earlier
  scan only covered August onward.
- Good VMs seen: AMD EPYC 7763 (Zen 3), unaffected per #6013. Also one
  **EPYC 9V74 whose R job passed**. #6013 says Zen 4 VMs that do *not*
  expose AVX-512 are spared. That VM's AVX-512 exposure wasn't logged
  then; it is now (`PROBE AVX512F supported`).
- The ~60 s delay on one crashing test per bad run (§1.7) comes from WER:
  `HKLM\...\Windows Error Reporting\ServiceTimeout = 60000`, with `WerSvc`
  stopped and manual-start. That's also why LocalDumps never produced a
  `.dmp`.

**Fix (commit after this doc revision):**

1. `tools/windows/install.sh` and `tools/octave-windows/install.sh` pin
   `openblas=0.3.33 libopenblas=0.3.33`. `tools/r-windows/install.sh`
   reuses the first. This DLL **ships with the Python wheel**
   (`bindings/Python/pylibkriging/setup.py`) and **the R release**
   (`tools/release/r-release.sh`), so Windows binaries built between
   2026-07-18 and this fix embed 0.3.34 and can crash for users on
   AVX-512 Zen 4/5 machines. Consider rebuilding them.
2. `OPENBLAS_L2_SIZE=2048` exported in the three Windows `loadenv.sh` as
   a belt-and-braces runtime guard: it disables the override's
   `l2 == 1024` condition (#6021).
3. Lift the pin once a fixed OpenBLAS reaches conda-forge.

Still open:
- Confirm on a VM that reports `AVX512F supported: True` and now passes.
- Why the R crash only appeared from 2026-09 (0/63 R jobs crashed before),
  while 0.3.34 was there since July. It's either the AVX-512 exposure
  share in the pool or the R path (n=1000 `dpotrf`/`dsyrk`) needing a
  specific condition; probe R1/R2 never ran on a bad VM.
- Then revert the TEMPORARY CI changes (§6).

## 1. Facts established from the CI history

Method: every `main.yml` run on `master`, `release/*` and this branch from
2026-07-01 to 2026-10-01, Windows jobs only, logs fetched via
`gh api repos/libKriging/libKriging/actions/jobs/<id>/logs` and grepped for
ctest's `Exception: SegFault` and make's `test] Segmentation fault`.

### 1.1 Three symptom families, one pattern

| Job | Build | Crashing set | First seen |
|---|---|---|---|
| Julia Windows (runs the **whole core ctest**, not Julia-specific) | MSVC Release, multi-threaded | 6 → 14 Catch2 tests: `LinearAlgebra - rapid fire varying sizes`, NestedKriging moderate n (n=400), LLVecchia(20)/predictVecchia/light Vecchia (n=300-400), the LLNystrom family (n≥150), LLIterative Nystrom-preconditioned SLQ (n=150) | master, 2026-08-03 |
| Python 3.7/3.9 Windows Debug | MSVC Debug, `OMP_NUM_THREADS=1` since 2026-08-13 | before 08-13: `rapid fire` + NestedKriging moderate n; after: 9-11 tests, LLNystrom family + LLIterative (n=150/160) + NestedKriging/LLVecchia (n=600), **no more `rapid fire`** | docs-only branch, **2026-07-18** |
| R Windows | Rtools gcc package linking the MSVC-built core | `make test` dies with `Segmentation fault` at the exact same log position, 6/6 times | master, **2026-09-19** |

All three link the **same conda-forge OpenBLAS 0.3.34 (pthreads,
DYNAMIC_ARCH)**, installed unpinned by `tools/windows/install.sh`. ~~That
version did not change over the whole period, so an OpenBLAS *update* is
not the trigger.~~ **Wrong, see §0:** 0.3.33 until 2026-07-11, 0.3.34
from 2026-07-18, the day of the first crash.

### 1.2 Deterministic per VM, not intermittent per process

- The 4 recent Julia failures crashed **the exact same 14 tests**, with the
  same test numbers, every time. The earlier master ones (08-03, 08-13,
  09-19) are the same set minus the tests that didn't exist yet. Random heap
  corruption doesn't produce this.
- With `--repeat until-fail:8`, a bad VM crashes each of those tests on
  **attempt 1**. Good VMs: **0 crashes in 16/16 repeats** (§C.1 of the old
  doc). The R job with `R_TEST_REPEAT=8` crashed on **attempt 1/8**.
- So the outcome is fixed by the VM, not by chance inside a process: about
  1 VM in 4 to 5 is "bad", and a bad VM fails 100% of the time. This is
  why in-job repeats were useless and whole-run reruns are not.

### 1.3 Not caused by a code change

- **First crash ever: 2026-07-18, branch `docs/agents-known-pitfalls`**,
  which only touches docs. Before that, 65 successful Julia/Python Windows
  jobs from 07-04 to 07-14 ran `rapid fire` with 0 crashes.
- R: **the same commit `63da4f37` passed on 2026-09-12 and crashed on
  2026-09-19** (master, runs 34679455425 and 35428409221).
- Present on `master`, `release/1.2.2` and this branch alike. The
  `cg-predict`/`predictIterative` work is not involved.

### 1.4 Onsets coincide with runner-image epochs (time proxy)

| Family | Runner image `win25-vs2026/...` | Result |
|---|---|---|
| core (Julia + Python) | 20260628.158 | 0 crashes (65 successful jobs) |
| core | 20260714.173 and later | crashes start (07-18) |
| R | ≤ 20260824.214 | **0/63 R jobs crash** |
| R | 20260907.229, .246, .250 | 6 crashes |

Images roll out over time, so this cannot separate "the image changed
something" from "the Azure hardware pool behind `windows-latest` changed
around the same dates". R itself was 4.6.1 throughout.

### 1.5 Size dependence

- LLNystrom tests: **every crashing test uses n ≥ 150, every passing one
  n ≤ 120** (60, 60, 100, 80, 100, 120 pass; 150, 150, 150, 150, 150, 150,
  200, 300 crash). The n=3000 smoke test is `[intensive]`, so skipped.
- Vecchia/Nested: crashes at n = 300-600, but some n=150/200 tests pass:
  the size involved is that of the matrix that actually reaches BLAS/LAPACK,
  not the raw n.
- `rapid fire` (50 `std::thread`s, sizes up to 100, `crossprod` of 200×100
  operands) crashes; the 20-thread tests with sizes ≤ 100 that only call
  `chol`/`solve` pass.
- R: see §2.1. The crash is most likely in the **first sizeable fit of the
  whole R suite (n=1000)**.

### 1.6 Threading dependence (partial)

Python jobs run with `OMP_NUM_THREADS=1`. OpenBLAS also honors that
variable as a fallback for its own thread count. Since that change
(2026-08-13), `rapid fire` **no longer crashes** there, but the LLNystrom
family **still does**. So there are two components:

- one that needs threads (`rapid fire`, and maybe NestedKriging moderate n
  and the Vecchia trio, which only crash in the multi-threaded Julia job);
- one that doesn't (LLNystrom family, LLIterative Nystrom-preconditioned).

### 1.7 Other observations

- On every bad-VM Julia run, **one** crashing test takes ~60 s instead of
  <1 s before reporting SegFault (`#152` 60.08 s, `#162` 60.10 s, `#161`
  60.09 s...), always a different test. Unexplained (WER timeout? OpenBLAS
  thread-server timeout?).
- Catch2 reports `SIGSEGV - Segmentation violation signal` at the
  `TEST_CASE` line, so there is no finer localization from its output.

## 2. Corrections to the first version of this doc

### 2.1 The R crash is (very likely) not in `test-estimnone.R`

testthat runs files in this order:
`test-AllKrigingConcistency.R`, `test-binding-consistency.R`,
`test-estimnone.R`, **`test-KrigingCholCrash.R`**, ... In a passing log, the
next output after estimnone's last `print()` (the heterogeneous-noise model)
comes **from `test-KrigingCholCrash.R`**: its first section (n=10 fit) prints
nothing, and the first output only comes from its second section, a
**n=1000, d=3, gauss, BFGS fit** with `linalg_check_chol_rcond(FALSE)`.
A crash anywhere in KrigingCholCrash before that point leaves exactly the
log we see. All R tests before it use n=5, so it is also the **first
sizeable linear algebra of the whole R suite**, which matches §1.5. The
"GC/finalizer of the heterogeneous-noise `Rcpp::XPtr`" theory was based on
the wrong localization and is dropped. To be confirmed by probe R1/R2
(§4).

### 2.2 "Intermittent" → deterministic per VM

See §1.2. The old ~40% "per-attempt" probability was really a ~20-25%
bad-VM draw.

### 2.3 WER LocalDumps was not really tested 3 times

The two §B reproductions that "produced no dump" ran on commit `d5c882ac`,
**before** the Catch2-SEH disable (`1bfdb4e1`). With Catch2's handler
active, no dump was expected. The SEH-disabled build never landed on a bad
VM (Julia passed 4/4 on `1bfdb4e1`). Only the R crash really tested WER.
The most likely reason for no dump at all is that WER is disabled on
GitHub's runner image. This is checked by the setup step in §4.
Dumps are no longer the plan anyway: §4 uses `cdb`, which sees the
first-chance access violation before any handler (Catch2's, R's) does.

## 3. Hypotheses, ranked

**H1 — CPU-dependent OpenBLAS kernel (lead).** Some host CPU model in the
`windows-latest` pool makes OpenBLAS's DYNAMIC_ARCH pick a kernel, or a
cache-size-derived blocking (GEMM_P/Q buffers), that faults for operands
above some size. This explains: per-VM determinism, identical test sets,
size threshold, all three jobs (same `openblas.dll`), onset without any
code change, and Linux/macOS being unaffected (different BLAS and pool).
The threaded/unthreaded split (§1.6) fits too: the threaded GEMM driver
and the single-thread kernel are separate code paths.
*Predictions:* crash ⟺ specific CPU model; `OPENBLAS_CORETYPE=Haswell`
or `Prescott` removes the crash on a bad VM; cdb shows the fault inside
`openblas.dll` (e.g. `dgemm_kernel_*`, `dsyrk_*`, `dtrsm_*`).

**H2 — latent libKriging memory bug, triggered by CPU-dependent
rounding.** A different kernel (FMA/AVX-512) changes the last bits, so a
different branch is taken (e.g. pivot choice / `k_eff` in
`LinearAlgebra::nystromFactor`, a NaN reaching an index) and an
out-of-bounds access in *our* code shows up only there.
*Predictions:* `OPENBLAS_CORETYPE` may **also** hide it (rounding
changes), so that variant alone can't separate H1 from H2. **cdb's
faulting module does**: `KrigingNystromTest.exe`/`Kriging.dll`/R package
DLL → H2, `openblas.dll` with sane arguments → H1.

**H3 — OpenBLAS pthreads server with many concurrent callers (Windows
`blas_server_win32.c`).** Explains `rapid fire` and maybe the Vecchia/
Nested ones, but not the single-threaded LLNystrom crashes, so at most a
secondary component. *Prediction:* `OPENBLAS_NUM_THREADS=1` removes T2
(`rapid fire`) but not T1 (LLNystrom).

Dropped: R GC/finalizer timing (wrong localization, §2.1); #354-style
two-allocator heap mismatch (doesn't fit per-VM determinism); in-process
randomness, ASLR or GC timing (bad VMs fail 100%).

## 4. The discriminating experiment (currently wired in CI)

TEMPORARY, on branch `rebase/cg-predict-on-v1.2.2` (trimmed matrix: R
Windows + Julia Windows only, see §6 to revert):

1. **Workflow step "Windows segfault investigation setup"**, on every VM:
   CPU model, cores, logical processors, L2/L3 cache size, the OpenBLAS
   core actually selected (`openblas_get_corename()` via Python `ctypes`),
   OpenBLAS config/threads, the WER policy keys and service state, and
   whether `cdb.exe` is present. Comparing good and bad VMs is H1's first
   test, and it costs nothing even on runs that don't crash.
2. **`tools/windows/segfault-probe.sh`**, called on the faulting VM right
   after a failing test (from `tools/windows/build.sh`'s embedded ctest,
   `tools/windows/test.sh` and `tools/r-windows/test.sh`). It reruns:
   - core: T1 = `KrigingNystromTest.exe "LLNystrom analytic gradient
     matches finite differences"` (single-thread crasher) and
     T2 = `LinearAlgebraTest.exe "LinearAlgebra - rapid fire varying
     sizes"` (threaded crasher), each under: baseline
     (`OPENBLAS_VERBOSE=2`, prints the core), `OPENBLAS_NUM_THREADS=1`,
     `OMP_NUM_THREADS=1`, `OPENBLAS_CORETYPE=Haswell|SkylakeX|Prescott`;
     then T1 under `cdb -g -G -o` (break on first-chance AV, then
     `.lastevent; .exr -1; r; kb 60; lm`);
   - R: R1 = `test-KrigingCholCrash.R` alone (confirms §2.1), R2 = its
     n=1000 fit alone, R3 = same at n=150, R2 under the same env variants,
     then R2 under `cdb -o` (Rscript re-launches Rterm).
   Grep the job log for `PROBE ` to get the summary.
3. In-job repeats set back to 1 (useless, §1.2), and Catch2's SEH handler
   is back on (cdb sees the exception first anyway). So bad-VM crash
   signatures stay comparable with the history.

**Reading the outcome:**

| Observation on a bad VM | Conclusion |
|---|---|
| CPU model differs between good and bad VMs, consistently | per-VM factor confirmed = hardware |
| cdb: fault in `openblas.dll` + `CORETYPE=Haswell` fixes it | H1 → pin a CORETYPE or swap BLAS build in CI, report upstream with the CPU/kernel name |
| cdb: fault in our code | H2 → fix our bug (the stack points at it) |
| `OPENBLAS_NUM_THREADS=1` fixes T2 only | H3 is real, as a secondary component |
| R1 does not crash but `make test` does | §2.1 is wrong, go back to estimnone/GC |

Then trigger runs (`gh run rerun <run-id>` on the whole run = fresh VMs)
until a bad VM is drawn, i.e. ~1 in 4 jobs. Each rerun lasts ~25 min.

## 5. Native Windows fallback

Only if §4 is not enough. Same recipes as
[`WindowsPythonHangDiagnostic354.md`](WindowsPythonHangDiagnostic354.md) and
[`win_debug.md`](win_debug.md) (VS Build Tools, Rtools, WinDbg/cdb,
Application Verifier page heap). One caveat: per §1.2, a local VM only
reproduces if **its CPU matches a bad runner's**. Get the CPU model from
§4 first, otherwise a clean local run proves nothing.

- core: build with `tools/windows/build.sh` (`ENABLE_JULIA_BINDING=on` or
  plain core, `MODE=Release`), then
  `cdb -g -G -c ".lastevent; .exr -1; kb 60; q" build\bin\KrigingNystromTest.exe "LLNystrom analytic gradient matches finite differences"`.
- R: `tools/r-windows/build.sh`, then the probe's `probe_fit.R` under
  `cdb -o`.
- To test H1 without the specific hardware: `OPENBLAS_CORETYPE=<the bad
  VM's core>` forces the same kernel on any CPU that supports its ISA
  (if the CPU lacks it, the result is an illegal instruction, not an AV).

## 6. Temporary CI changes to revert

All on this branch, all marked `TEMPORARY`:

- `.github/workflows/main.yml`: matrix trimmed to R/Julia Windows (commit
  `16cf4da2`), `R_TEST_REPEAT`/`CTEST_REPEAT`/`CTEST_EXTRA_ARGS`/
  `DISABLE_CATCH_WINDOWS_SEH` env plumbing, the setup step,
  `continue-on-error` on `script`/`test`, `**/*.dmp` in the artifact glob.
- `tools/windows/segfault-probe.sh` and the 3 call sites
  (`tools/windows/build.sh`, `tools/windows/test.sh`,
  `tools/r-windows/test.sh`).
- `tests/CMakeLists.txt` `DISABLE_CATCH_WINDOWS_SEH` option (harmless, off
  by default).

`git log --oneline -- .github/workflows/main.yml tools/windows tools/r-windows tests/CMakeLists.txt`
lists the commits. Revert them all before merging this branch.
