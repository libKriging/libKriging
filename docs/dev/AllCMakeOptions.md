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
| `ENABLE_CUDA_ITERATIVE`      |            `OFF`            | `ON`, `OFF`, `AUTO` (if a CUDA toolkit is found)    | GPU (CUDA) backend of method `"iterative(...)"`, see below |
| `ENABLE_HIP_ITERATIVE`       |            `OFF`            | `ON`, `OFF`, `AUTO` (if ROCm and an AMD GPU are found) | GPU (AMD ROCm/HIP) backend; CMake ≥ 3.21 |
| `ENABLE_SYCL_ITERATIVE`      |            `OFF`            | `ON`, `OFF`, `AUTO` (if the compiler accepts `-fsycl`) | GPU (Intel oneAPI/SYCL) backend, unverified |
| `ENABLE_METAL_ITERATIVE`     |            `OFF`            | `ON`, `OFF`, `AUTO` (macOS with metal-cpp headers)  | GPU (Apple Metal) backend; `METAL_CPP_DIR` |
| `USE_COMPILER_CACHE`         |        &lt;empty&gt;        | &lt;string&gt;                                      | name of a compiler cache program                  |
| `BUILD_SHARED_LIBS`          |            `ON`             | `ON`, `OFF`                                         |                                                   |
| `PYTHON_PREFIX_PATH`         |        &lt;empty&gt;        | &lt;string&gt;                                      | overrides default python path detection           |
| `Matlab_ROOT_DIR`            |        &lt;empty&gt;        | &lt;string&gt;                                      | locate Matlab root directory to help CMake finder |
| `SANITIZE`                   |            `OFF`            | `OFF`, `THREAD`, `ADDRESS`, `LEAK`                  | Enable sanitize feature (is available)            |
| `LBFGSB_SHOW_BUILD`          |            `OFF`            | `ON`, `OFF`                                         | Show details of `lbfgsb_cpp` sub-build            |
| `USE_JEMALLOC`               |            `OFF`            | `ON`, `OFF`                                         | Download and build jemalloc 5.3.0 as the default allocator (ignored on Windows). Off by default: it was disabled while debugging a TLS error in the Python bindings |

## GPU backends

Only the iterative method (`method = "iterative(...)"` at `fit`, `predict`,
`update`) uses the GPU; every other method runs on the CPU.

### Variants

The default build and every published binary package are **CPU only** (the
four backend options default to `OFF`). The GPU variant is a separate
`-gpu` package, built from source on the installing machine with
`ENABLE_GPU_ITERATIVE=AUTO` (every backend whose toolchain is found there;
a plain CPU build when there is none).

### Options

- `ENABLE_GPU_ITERATIVE`: empty (default) lets each `ENABLE_<BACKEND>_ITERATIVE`
  decide; `AUTO` sets all four to `AUTO`; `OFF` sets all four to `OFF`.
- `ENABLE_<BACKEND>_ITERATIVE` = `ON` (required, configuring fails without
  it), `OFF` (default), or `AUTO` (enabled iff detected; never fails). The
  configure log prints one line per backend, e.g.
  `CUDA iterative backend: AUTO -> ON (CUDA 12.0.140, architectures: all-major)`.
- Detection under `AUTO`:
  - CUDA: a working CUDA compiler and toolkit, CMake ≥ 3.23.
    `CMAKE_CUDA_ARCHITECTURES` defaults to `all-major` (portable, no GPU
    needed at build time; `native` under `ON`). If `nvcc` rejects the
    default host compiler, set `CMAKE_CUDA_HOST_COMPILER` (e.g. `g++-12`).
  - HIP: a working HIP compiler, `hip-config.cmake` (`CMAKE_PREFIX_PATH`)
    and at least one AMD GPU reported by `rocm_agent_enumerator`, whose
    architectures are then compiled for; set `CMAKE_HIP_ARCHITECTURES` to
    build without a GPU. CMake ≥ 3.21.
  - SYCL: the C++ compiler compiles a `-fsycl` test program (icpx, oneAPI
    clang++). **Unverified backend**: never built in this project's CI.
  - Metal: macOS and the metal-cpp headers (`METAL_CPP_DIR`, environment
    variable of the same name, or `.deps/metal-cpp`).
- A CUDA build has **no load-time dependency on the CUDA toolkit**: the CUDA
  runtime is linked statically and cuBLAS is loaded on first use
  (`LK_CUBLAS_LIBRARY` may name the library). Without GPU, driver or cuBLAS,
  the library loads and runs the CPU path. HIP, SYCL and Metal builds depend
  on their runtime at load time.
- `find_package(libKriging CONFIG)` (`cmake/libKrigingConfig.cmake.in`) also
  finds the CUDA toolkit referenced by a *static* CUDA-enabled `libKriging.a`
  (`CUDA::cudart_static`) and the HIP runtime of a HIP build.
  `<prefix>/share/libKriging/libKriging_gpu_link_libs.txt` lists the GPU link
  items for non-CMake consumers of a static library (empty without GPU).

### Runtime settings (`libKriging/Gpu.hpp`)

Process-wide, set before the calls, as the `Parallel` settings; each reads
its default from an environment variable.

| Setting | Default | Environment |
|:--|:--|:--|
| `Gpu::set_enabled(bool)` / `get_enabled()` | on (when a device is available) | `LK_ITERATIVE_GPU=0` turns it off |
| `Gpu::set_dense_max_mb(int)` / `get_dense_max_mb()` | 4096 MiB of device memory for the dense correlation matrix; 0 = matrix-free | `LK_GPU_DENSE_MAX_MB` |
| `Gpu::set_mixed_precision(bool)` / `get_mixed_precision()` | off (CUDA only) | `LK_GPU_MIXED_PRECISION` |
| `Gpu::set_verbose(bool)` / `get_verbose()` | off | `LK_GPU_VERBOSE` |

Read only: `Gpu::compiled_backends()`, `Gpu::available()`,
`Gpu::active_backend()` (`"none"` on the CPU path). The CPU counterpart of
the dense budget is `LinearAlgebra::set_iterative_dense_max_mb` (host memory,
default 6144 MiB, `LK_ITERATIVE_DENSE_MAX_MB`).
