# Agent Instructions

## API usage guidance

Before writing or reviewing code that fits/predicts/simulates a Kriging
model (any of `Kriging`, `WarpKriging`, `MLPKriging`, `NestedKriging`, in
C++ or any binding), read `skills/libkriging/SKILL.md` for guidance on
which class and options to use, and the matching file under
`skills/libkriging/references/` for exact call syntax in that language.

## Allowed tools

The following tools are pre-approved and may be used freely without asking for user authorization:

- **Build/compile:** `cmake`, `ctest`, `ninja`, `make`, `gcc`, `g++`, `cc`, `c++`
- **Python:** `python`, `python3`, `pip`, `pip3`
- **R:** `R`, `Rscript`
- **Shell:** `cd`, `export`, `LD_LIBRARY_PATH`
- **File inspection:** `ls`, `find`, `cat`, `head`, `tail`, `grep`, `rg`
- **Text processing:** `sed`, `awk`, `echo`, `printf`
- **File operations:** `mkdir`, `cp`, `mv`, `rm`, `touch`, `chmod`
- **Version control:** `git` (except `git push` — see below), `gh`

## Git usage

Never run `git push` or any variant that pushes commits to a remote (e.g. `git push --force`, `git push origin`). If a task requires pushing, stop and ask the user to do it manually.

## Known pitfalls

### Build

- **Clone with submodules.** `dependencies/` (armadillo, pybind11, Catch2,
  lbfgsb_cpp, carma) are git submodules. A plain `git clone` leaves them
  empty and the build fails with confusing missing-header errors, not an
  obvious "submodule" message. Use
  `git clone --recurse-submodules https://github.com/libKriging/libKriging.git`,
  or if already cloned, `git submodule update --init --recursive`.
- **CMake ≥ 3.13**, and the Octave binding specifically needs a newer CMake
  than the rest of the build. On old distros (e.g. Ubuntu 18's default
  3.10.2) upgrade via the Kitware apt repo — see `tools/linux-macos/install.sh`
  and `docs/dev/DevTips.md` for the exact commands.
- **`ENABLE_OCTAVE_BINDING` and `ENABLE_MATLAB_BINDING` are mutually
  exclusive** — don't set both to `ON` in the same build.
- **`ENABLE_JULIA_BINDING` defaults to `OFF`** and requires Julia ≥ 1.10;
  it must be explicitly enabled, it won't turn on via `AUTO` detection like
  the other bindings. See `docs/dev/AllCMakeOptions.md` for the full option
  table.
- **On Windows**, set `CMAKE_GENERATOR_PLATFORM=x64` to build a 64-bit
  target.
- **On macOS**, a `fatal error: 'math.h' file not found` after an OS
  upgrade usually means outdated Xcode Command Line Tools or an outdated
  `llvm` from `brew` — see `docs/dev/DevTips.md` for the fix, not a libKriging
  bug.
- **Don't mix ABI across compilers** on already-built dependencies (e.g.
  switching the system compiler without rebuilding armadillo/lbfgsb_cpp) —
  this tends to fail at link time or, worse, silently corrupt data at
  runtime rather than give a clear error.
- **`ARMA_32BIT_WORD` must stay consistent** between the C++ core and the R
  binding: a mismatch changes `sizeof(arma::mat)` between the two sides
  (160 vs 176 bytes) and corrupts objects passed across the Rcpp boundary.
  This is now the default everywhere in-tree — don't override it in only
  one binding.

### Matrix / vector formats

- **`X` is always `n × d`: rows are observations, columns are input
  variables** — this convention is identical across C++ and every binding.
  Never transpose to a `d × n` layout, even if a downstream tool (e.g. a
  DOE/optimization library) uses the opposite convention.
- **`y` is a plain length-`n` vector, not an `n × 1` matrix.** In R this
  usually means calling `as.numeric(y)` on a single-column `data.frame`
  or `matrix` column before passing it in; in Julia it means a
  `Vector{Float64}`, not a `Matrix{Float64}` with one column.
- **R:** if `X` comes from a `data.frame` (e.g. from a design-of-experiments
  package), wrap it explicitly with `as.matrix(X)` first — passing a
  `data.frame` directly tends to fail or silently coerce in unexpected ways.
- **Julia:** the `ccall`-based FFI expects `Matrix{Float64}`/`Vector{Float64}`
  exactly — there is no automatic conversion from `Matrix{Int}`,
  `Vector{Any}`, or mixed-type columns. Convert explicitly
  (`Float64.(X)`) rather than relying on Julia's usual numeric promotion.
- **Octave/MATLAB:** integer-typed arguments such as `nsim`/`seed` in
  `simulate(...)` must be passed as `int32(...)`, not a plain `double` —
  the mex layer does not coerce them, and a type mismatch raises a
  low-level error far from the actual mistake.
- **Python:** NumPy arrays should be `float64`. The pybind11/carma layer
  converts NumPy's default row-major (`C`-order) layout to Armadillo's
  column-major layout for you, so this is not a correctness issue — but if
  profiling shows an unexpected copy on very large arrays, laying `X` out
  as Fortran-order (`numpy.asfortranarray(X)`) up front avoids it.

For which class/options to use in the first place (as opposed to how to
call them correctly), see `skills/libkriging/SKILL.md` if present in this
checkout.
