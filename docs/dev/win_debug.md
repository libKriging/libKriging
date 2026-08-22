# Setting up a Windows debugging VM for libKriging

General-purpose reference for getting a Windows environment ready to build,
test, and debug libKriging natively — as opposed to from a Linux sandbox,
which can't reproduce Windows-only bugs (see
[`WindowsPythonHangDiagnostic354.md`](WindowsPythonHangDiagnostic354.md) for
a full case study built entirely from a setup like this one: a Windows-only
heap corruption that needed native debuggers to diagnose). Use this doc
whenever a CI failure only reproduces on a Windows runner and needs a live
debugger, `dumpbin`, or Application Verifier attached.

## Accessing the repo

If working from a VM (e.g. dockur/windows) with the repo shared from the
host, it typically shows up as a mounted drive (`Z:\` in the sessions this
doc was written from). Otherwise, clone fresh and remember submodules:

```bash
git clone https://github.com/libKriging/libKriging.git
cd libKriging
git submodule update --init --recursive
```

**Known gotcha — "dubious ownership" on network shares**: git refuses to
operate on a repo mounted from a network path (`//host.lan/Data/`,
`\\host.lan\Data`, etc.) with `fatal: detected dubious ownership in
repository`. Fix (only if you actually want to run git in this checkout —
it edits global git config, so don't do this without the user's OK):

```bash
git config --global --add safe.directory '%(prefix)///host.lan/Data/'
```

## Prerequisites

Install via `winget` where an ID is given; the `winget` **source** for
regular packages needs no special handling, but the Microsoft **Store**
source (`msstore`, used for WinDbg) requires accepting its Terms of
Transaction on first use — don't do that without checking with the user
first, since it also sends the machine's region to Microsoft.

1. **Git** — https://git-scm.com/download/win (or `winget install
   Git.Git`). Includes **Git Bash**, needed because this repo's Windows
   build scripts (`tools/windows/*.sh`) are bash, not PowerShell/batch.
2. **Visual Studio Build Tools 2022+** (`Microsoft.VisualStudio.2022.BuildTools`)
   with the **"Desktop development with C++"** workload — provides
   `cl.exe`/MSVC and `dumpbin.exe`. A 2022 toolset is close enough to
   whatever generator CI uses to reproduce most Windows-only bugs locally.
   Install with the workload attached, e.g.:
   ```powershell
   winget install --id Microsoft.VisualStudio.2022.BuildTools --source winget --accept-package-agreements --accept-source-agreements --silent --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```
3. **CMake ≥ 4.2** (`Kitware.CMake`) — the repo pins
   `CMAKE_POLICY_VERSION_MINIMUM=3.5` for compat with the pybind11 submodule
   under cmake ≥ 4.
4. **Python** (`Python.Python.3.9`, or whichever version(s) the target CI
   job(s) use) — `pip install pytest numpy`.
5. **Miniconda** (`Anaconda.Miniconda3`) — `tools/windows/build.sh` expects
   `${HOME}/Miniconda3/Library/lib` for OpenBLAS/LAPACK:
   ```bash
   conda install --override-channels -c conda-forge openblas lapack
   ```
   `--override-channels` avoids conda's newer default-channel Terms-of-Service
   prompt (`defaults`/`pkgs/main` etc. now require accepting ToS before any
   install; restricting to `conda-forge` sidesteps it entirely). Adapt
   `EXTRA_SYSTEM_LIBRARY_PATH` in `tools/windows/build.sh` if Miniconda is
   installed somewhere else.
6. **GitHub CLI** (`GitHub.cli`) — `gh auth login` for `gh issue`/`gh pr`
   commands against issues/PRs.
7. Optional, but very useful for native debugging sessions:
   - **WinDbg** (Microsoft Store package) or Visual Studio's own debugger —
     to attach to a stuck/crashing process and get an all-threads call
     stack. The Store package's install directory also contains
     `cdb.exe`/`ntsd.exe` (command-line debuggers) under `amd64\`, which run
     fine invoked directly despite living under `WindowsApps` (a folder
     normally locked down against direct execution).
   - **Process Explorer** (Sysinternals — no winget package; download
     directly from
     `https://download.sysinternals.com/files/ProcessExplorer.zip`) — live
     thread/handle inspection of a running or stuck process.
   - **py-spy** (`pip install py-spy`) — dumps a Python (and, with
     `--native`, native C/C++) call stack from a running or hung process
     without attaching a native debugger. Usually the fastest first look at
     a Python-hosted hang, and works even when a native debugger's symbol
     resolution is being difficult.

## Known environment gotchas

Things that cost real time to figure out in practice, worth checking first:

- **`winget`-installed tools aren't on `PATH` in already-open shells.**
  `winget` updates the registry's `PATH`, but a shell session (or a tool
  that spawns subprocesses, like this session's Bash/PowerShell tools) that
  was already running won't pick that up automatically. Refresh it
  explicitly instead of opening a new shell:
  ```powershell
  $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
  ```
  In Git Bash, either do the equivalent or just prefix commands with the
  known install directories (e.g. `/c/Program Files/CMake/bin`) until a
  fresh shell is available.
- **`dumpbin.exe` needs its environment set up first.** Running it directly
  fails silently/weirdly unless launched after `vcvars64.bat`, and
  `vcvars64.bat` itself calls `vswhere.exe`, which needs `%ProgramFiles(x86)%\Microsoft
  Visual Studio\Installer` on `PATH`. From PowerShell:
  ```powershell
  $env:Path += ";C:\Program Files (x86)\Microsoft Visual Studio\Installer"
  cmd /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && dumpbin.exe /dependents "path\to\file.exe"'
  ```
- **`CMAKE_MSVC_RUNTIME_LIBRARY` can be silently ignored.** It only takes
  effect under CMake policy `CMP0091 = NEW`, which itself depends on every
  `cmake_minimum_required()` in the whole project tree (including
  submodules!) declaring a version ≥ 3.15 — this repo's top-level
  `CMakeLists.txt` and the pybind11 submodule both declare older versions,
  so the policy silently defaults to `OLD` and the runtime-library setting
  does nothing. Force it from the command line regardless of those
  declared minimums:
  ```
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW
  ```
- **`tools/windows/build.sh` overwrites `EXTRA_CMAKE_OPTIONS`, it doesn't
  append to it.** The script does
  `EXTRA_CMAKE_OPTIONS="-DBUILD_SHARED_LIBS=... -DSTATIC_LIB=..."`
  unconditionally partway through, discarding anything passed in via the
  environment. To pass extra CMake flags for a one-off build, invoke
  `cmake` directly instead of through `build.sh` (mirror its other flags —
  see the "Build" section below for the exact invocation), or edit the
  script temporarily.
- **Python 3.8+'s "secure DLL loading" breaks naive extension-module
  testing.** A compiled `.pyd` that depends on non-standard DLLs (e.g.
  `openblas.dll`) will fail to import with a generic `DLL load failed`
  error unless those directories are registered first — `PATH` alone is no
  longer enough:
  ```python
  import os
  os.add_dll_directory(r"C:\Users\<user>\Miniconda3\Library\bin")
  import your_module
  ```
- **A file/folder named after a Windows reserved device or a leading
  `/` in a Bash command can get silently mangled by MSYS's path
  translation** when calling native `.exe` tools from Git Bash (e.g. a
  bare `/dependents` flag can get reinterpreted as a POSIX path). If a
  native tool's output looks like it swallowed an argument, prefer running
  it from PowerShell instead of chasing MSYS quoting.

## Build

Mirror of what `tools/windows/build.sh` does, useful to invoke directly
when passing custom CMake flags (see the `EXTRA_CMAKE_OPTIONS` gotcha
above) or building into a second, parallel build directory to compare
configurations:

```bash
cd /path/to/libKriging
export MODE=Debug   # or Release
export ENABLE_PYTHON_BINDING=on
export ENABLE_OCTAVE_BINDING=off
export ENABLE_MATLAB_BINDING=off
export ENABLE_JULIA_BINDING=off
export BUILD_DIR=build_win_debug
tools/windows/build.sh
```

Or directly, to add extra flags:

```bash
mkdir -p build_win_debug && cd build_win_debug
cmake -DCMAKE_GENERATOR_PLATFORM=x64 \
  -DEXTRA_SYSTEM_LIBRARY_PATH="$HOME/Miniconda3/Library/lib" \
  -DENABLE_OCTAVE_BINDING=off -DENABLE_MATLAB_BINDING=off \
  -DENABLE_PYTHON_BINDING=on -DENABLE_JULIA_BINDING=off \
  -DBUILD_SHARED_LIBS=off -DSTATIC_LIB=on \
  ..   # + any extra -D flags here
cmake --build . --target ALL_BUILD --config Debug
```

To rebuild just one target after a source change (much faster than a full
rebuild): `cmake --build . --target <target> --config Debug` — CMake/MSBuild
correctly picks up and relinks dependent static libs automatically.

## Debugging a running or hung Windows process

**First look — py-spy** (fastest, no setup):
```powershell
py-spy dump --pid <PID>            # Python-level stack
py-spy dump --pid <PID> --native   # + native C/C++ frames, needs symbols nearby (PDB next to the DLL/EXE)
```

**Live process inspection — Process Explorer**: GUI, good for seeing
threads/handles/loaded modules of a stuck process without attaching a
debugger that might disturb its state.

**Full native debugging — `cdb.exe` (WinDbg's command-line front end)**:
useful for scripted, reproducible attach-and-inspect sessions. Example:
break on any access violation and dump the stack when it hits:

```bash
CDB="/c/Program Files/WindowsApps/Microsoft.WinDbg_<version>_x64__8wekyb3d8bbwe/amd64/cdb.exe"
"$CDB" -c 'sxe av; g; .exr -1; kv; q' \
  "/c/path/to/python.exe" -m pytest -vv -s path/to/test.py
```

`sxe av` arms a break on `STATUS_ACCESS_VIOLATION`; `g` runs; `.exr -1`
prints the exception record; `kv` prints the call stack with arguments;
`q` quits after. Private PDB symbols may need an explicit `.sympath+`
pointed at the build output directory if `srv*` (the default,
Microsoft-public-symbols-only path) doesn't resolve local module symbols.

**Catching memory corruption at the exact faulting write — Application
Verifier's Page Heap**: normal heap corruption detection (e.g. MSVC's
Debug CRT) only notices damage the next time a block's own bookkeeping is
checked, often long after the actual bad write — Page Heap instead puts
every allocation on its own OS-guarded page, so even a single-byte overrun
faults immediately at the culprit instruction:

```powershell
appverif -enable Heaps -for <image.exe> -with Heaps.full=true
# ... reproduce, ideally under cdb with `sxe av` armed (see above) ...
appverif -disable Heaps -for <image.exe>   # always disable when done!
```

**This is a systemwide, per-image-name setting** — it applies to *every*
process with that exact executable name (e.g. every `python.exe`) while
enabled, and slows them down significantly. Always disable it again when
finished (`appverif -query Heaps -for <image.exe>` to check current state).

Caveat: MSVC's `_aligned_malloc_dbg`/`_aligned_free_dbg` (used
automatically in Debug CRT builds) add their own small guard-byte padding
*inside* the page-heap-guarded allocation — a small overrun into that
padding gets caught by the CRT's own bookkeeping (at free time, not at the
faulting write) rather than by Page Heap's guard page. If Page Heap isn't
catching something you're confident is a real overrun, that's why; a
hard AV needs either a large-enough overrun to reach the actual guard page,
or removing the CRT's own padding layer from the equation (e.g. testing a
Release-CRT build instead of Debug).
