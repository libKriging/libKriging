---
description: Configure and build libKriging (core C++ and/or a language binding) with the correct CMake options, avoiding the known pitfalls.
---

Build libKriging from the current checkout. Arguments (optional): $ARGUMENTS
— e.g. a binding name (`python`, `r`, `octave`, `matlab`, `julia`), `core`,
or `test` to also run the test suite.

Steps:

1. **Submodules first.** `dependencies/` (armadillo, pybind11, Catch2,
   lbfgsb_cpp, carma) are git submodules. If any is empty, run
   `git submodule update --init --recursive` before configuring — a missing
   submodule shows up as confusing missing-header errors, not a clear message.

2. **Check CMake ≥ 3.13** (the Octave binding needs a newer CMake than the
   rest). On old distros upgrade via the Kitware apt repo — see
   `tools/linux-macos/install.sh` and `docs/dev/DevTips.md`.

3. **Configure** into `build/` with the binding requested in $ARGUMENTS:
   - `ENABLE_PYTHON_BINDING`, `ENABLE_R_BINDING`, `ENABLE_OCTAVE_BINDING`,
     `ENABLE_MATLAB_BINDING`, `ENABLE_JULIA_BINDING` as appropriate.
   - `ENABLE_OCTAVE_BINDING` and `ENABLE_MATLAB_BINDING` are **mutually
     exclusive** — never both `ON`.
   - `ENABLE_JULIA_BINDING` defaults `OFF` and needs Julia ≥ 1.10; it must be
     set explicitly (no `AUTO` detection).
   - On Windows pass `CMAKE_GENERATOR_PLATFORM=x64`.
   - Keep `ARMA_32BIT_WORD` at its in-tree default — do not override it for a
     single binding, it corrupts objects across the Rcpp boundary.
   - Full option table: `docs/dev/AllCMakeOptions.md`.

4. **Build** with the generator's build tool (`cmake --build build -j`).

5. If $ARGUMENTS contains `test`, run `ctest --test-dir build --output-on-failure`.

6. Report what was configured, what built, and any test results (with output
   on failure) — do not claim success if a step failed or was skipped.

Do not run `git push`.
