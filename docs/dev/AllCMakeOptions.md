List of CMake options to configure libKriging build.

They should be used as `-D<option>=<value>` in `cmake` command line.

| Standard CMake option         |  Default value   | Allowed values                                     | Comment                                                  |
|:------------------------------|:----------------:|:---------------------------------------------------|:---------------------------------------------------------|
| `CMAKE_BUILD_TYPE`            | `RelWithDebInfo` | `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel` |                                                          |
| `CMAKE_INSTALL_PREFIX`        |  `./installed`   |                                                    | path to install libs                                     |
| `CMAKE_GENERATOR_PLATFORM`    |  &lt;empty&gt;   | empty or `x64`                                     | should be set to `x64` on Windows to build 64bits target |
| `CMAKE_CXX_COMPILER_LAUNCHER` |  &lt;empty&gt;   | a compiler cache like `ccache`                     | to optimize recompilation                                | 

| libKriging CMake option      |        Default value        | Allowed values                                      | Comment                                           |
|:-----------------------------|:---------------------------:|:----------------------------------------------------|:--------------------------------------------------|
| `EXTRA_SYSTEM_LIBRARY_PATH`  |        &lt;empty&gt;        | &lt;path&gt;                                        | add extra path for finding required libs          |
| `LIBKRIGING_BENCHMARK_TESTS` |            `OFF`            | `ON`, `OFF`                                         |                                                   |
| `ENABLE_COVERAGE`            |            `OFF`            | `ON`, `OFF`                                         |                                                   |
| `ENABLE_MEMCHECK`            |            `OFF`            | `ON`, `OFF`                                         |                                                   |
| `ENABLE_STATIC_ANALYSIS`     |           `AUTO`            | `ON`, `OFF`, `AUTO` (if available and `Debug` mode) |                                                   |
| `ENABLE_OCTAVE_BINDING`      |           `AUTO`            | `ON`, `OFF`, `AUTO` (if available)                  | Exclusive with `ENABLE_MATLAB_BINDING=on`         |
| `ENABLE_MATLAB_BINDING`      |           `AUTO`            | `ON`, `OFF`, `AUTO` (if available)                  | Exclusive with `ENABLE_OCTAVE_BINDING=on`         |
| `ENABLE_PYTHON_BINDING`      |           `AUTO`            | `ON`, `OFF`, `AUTO` (if available)                  |                                                   |
| `ENABLE_JULIA_BINDING`       |            `OFF`            | `ON`, `OFF`                                         | Requires Julia ≥ 1.10                             |
| `ENABLE_GPU_ITERATIVE`       |        &lt;empty&gt;        | `AUTO`, `OFF`, empty                                | Overrides the four options below at once: `AUTO` detects every GPU backend, `OFF` = CPU only; see below |
| `ENABLE_CUDA_ITERATIVE`      |           `AUTO`            | `ON`, `OFF`, `AUTO` (if a CUDA toolkit is found)    | GPU (CUDA) backend for `LLIterative`/`predictIterative`, see below |
| `ENABLE_HIP_ITERATIVE`       |            `OFF`            | `ON`, `OFF`, `AUTO` (if ROCm and an AMD GPU are found) | GPU (AMD ROCm/HIP) backend; CMake ≥ 3.21 |
| `ENABLE_SYCL_ITERATIVE`      |            `OFF`            | `ON`, `OFF`, `AUTO` (if the compiler accepts `-fsycl`) | GPU (Intel oneAPI/SYCL) backend, unverified |
| `ENABLE_METAL_ITERATIVE`     |            `OFF`            | `ON`, `OFF`, `AUTO` (macOS with metal-cpp headers)  | GPU (Apple Metal) backend, unverified; `METAL_CPP_DIR` |
| `USE_COMPILER_CACHE`         |        &lt;empty&gt;        | &lt;string&gt;                                      | name of a compiler cache program                  |
| `BUILD_SHARED_LIBS`          |            `ON`             | `ON`, `OFF`                                         |                                                   |
| `PYTHON_PREFIX_PATH`         |        &lt;empty&gt;        | &lt;string&gt;                                      | overrides default python path detection           |
| `Matlab_ROOT_DIR`            |        &lt;empty&gt;        | &lt;string&gt;                                      | locate Matlab root directory to help CMake finder |
| `SANITIZE`                   |            `OFF`            | `OFF`, `THREAD`, `ADDRESS`, `LEAK`                  | Enable sanitize feature (is available)            |
| `LBFGSB_SHOW_BUILD`          |            `OFF`            | `ON`, `OFF`                                         | Show details of `lbfgsb_cpp` sub-build            |
| `USE_JEMALLOC`               |            `OFF`            | `ON`, `OFF`                                         | Download and build jemalloc 5.3.0 as the default allocator (ignored on Windows). Off by default: it was disabled while debugging a TLS error in the Python bindings |

## GPU backends

Only `objective="LLIterative(...)"` fits and `predictIterative` use the GPU;
every other method runs on the CPU.

### Published variants

| Variant | What it is | How to get it |
|:--|:--|:--|
| default (CPU) | the binary packages: PyPI wheels `pylibkriging`, CRAN `rlibkriging`, release archives (C++, R, Octave, MATLAB) | `pip install pylibkriging`, `install.packages("rlibkriging")`, release assets |
| `-gpu` | a source package, compiled on the installing machine with every GPU backend found there (`ENABLE_GPU_ITERATIVE=AUTO`); a plain CPU build when there is none | Python: `pip install pylibkriging-gpu` (sdist, needs a C++ compiler, BLAS/LAPACK and the GPU toolkit). Other bindings: the `libKriging-gpu_<version>_src.tar.gz` release asset (repository + submodules), then `tools/install-gpu.sh [python] [r] [octave\|matlab] [julia]` |

Both Python variants install the same `pylibkriging` module: install one or
the other, not both. The release workflows export `ENABLE_GPU_ITERATIVE=OFF`
for every binary package; `tools/release/python-sdist-gpu.sh` and
`tools/release/gpu-source-archive.sh` produce the two `-gpu` assets. Julia
users install through [JLibKriging.jl](https://github.com/libKriging/JLibKriging.jl),
which compiles libKriging itself: it gets the GPU variant by passing
`-DENABLE_GPU_ITERATIVE=AUTO` (or `OFF`) to that build.

### Options

- `ENABLE_GPU_ITERATIVE`: empty (default) lets each `ENABLE_<BACKEND>_ITERATIVE`
  decide; `AUTO` sets all four to `AUTO`; `OFF` sets all four to `OFF`.
- `ENABLE_<BACKEND>_ITERATIVE` = `ON` (required, configuring fails without
  it), `OFF`, or `AUTO` (enabled iff detected; never fails). The configure
  log prints one line per backend, e.g.
  `CUDA iterative backend: AUTO -> ON (CUDA 12.0.140, architectures: all-major)`.
  Defaults: CUDA `AUTO`, HIP/SYCL/Metal `OFF`.
- Detection under `AUTO`:
  - CUDA: a working CUDA compiler and toolkit, CMake ≥ 3.23.
    `CMAKE_CUDA_ARCHITECTURES` defaults to `all-major` (portable, no GPU
    needed at build time; `native` under `ON`). If `nvcc` rejects the
    default host compiler, set `CMAKE_CUDA_HOST_COMPILER` (e.g. `g++-12`).
  - HIP: a working HIP compiler, `hip-config.cmake`
    (`CMAKE_PREFIX_PATH`) and at least one AMD GPU reported by
    `rocm_agent_enumerator`, whose architectures are then compiled for;
    set `CMAKE_HIP_ARCHITECTURES` to build without a GPU. CMake ≥ 3.21.
  - SYCL: the C++ compiler compiles a `-fsycl` test program (icpx,
    oneAPI clang++). **Unverified backend**: never built in this project's CI.
  - Metal: macOS and the metal-cpp headers (`METAL_CPP_DIR`, environment
    variable of the same name, or `.deps/metal-cpp`). **Unverified backend**.
- A CUDA-enabled build has **no load-time dependency on the CUDA toolkit**:
  the CUDA runtime is linked statically and cuBLAS is loaded on first use
  (`LK_CUBLAS_LIBRARY` may name the library explicitly). On a machine
  without GPU, driver or cuBLAS, the library still loads and runs the CPU
  path (a message is printed once if a GPU is present but cuBLAS is not).
  HIP, SYCL and Metal builds do depend on their runtime at load time,
  which is fine for the `-gpu` variant (built where it runs).
- Not covered: the R package on Windows (Rtools/MinGW is not a supported
  `nvcc` host compiler; `tools/r-windows/build.sh` defaults to `OFF`), and
  the R package with HIP/SYCL/Metal (its `Makevars` only adds the CUDA
  runtime). CMake consumers of an installed *static* CUDA-enabled
  `libKriging` must `find_package(CUDAToolkit)` before including
  `lib/cmake/libKriging.cmake` (it references `CUDA::cudart_static`).
- Existing build directories keep their cached value: an old
  `ENABLE_CUDA_ITERATIVE:BOOL=OFF` stays `OFF` until reconfigured.

The tool scripts (`tools/*/build.sh`) forward the `ENABLE_GPU_ITERATIVE` and
`ENABLE_CUDA_ITERATIVE` environment variables; `setup.py` forwards
`ENABLE_GPU_ITERATIVE` and every `ENABLE_<BACKEND>_ITERATIVE`, and appends
`LIBKRIGING_CMAKE_ARGS` (e.g. `LIBKRIGING_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90"`).

### Runtime

At runtime, the backend is on as soon as a usable device is found, unless
the environment variable `LK_ITERATIVE_GPU=0` is set. Every binding exposes
the same switch:

| Binding       | Query                                                         | Switch                     |
|:--------------|:--------------------------------------------------------------|:---------------------------|
| C++           | `libKriging::gpu::available()`, `active_backend()`, `enabled()` (`libKriging/Gpu.hpp`) | `libKriging::gpu::set_enabled(bool)` |
| Python        | `gpu_available()`, `gpu_backend()`, `gpu_enabled()`, `gpu_compiled_backends()` | `set_gpu_enabled(bool)` |
| R             | `gpu_available()`, `gpu_backend()`, `gpu_enabled()`, `gpu_compiled_backends()` | `set_gpu_enabled(value)` |
| Julia         | `gpu_available()`, `gpu_backend()`, `gpu_enabled()`, `gpu_compiled_backends()` | `set_gpu_enabled(value)` |
| Octave/MATLAB | `Gpu.available()`, `Gpu.backend()`, `Gpu.enabled()`, `Gpu.compiled_backends()` | `Gpu.set_enabled(value)` |
