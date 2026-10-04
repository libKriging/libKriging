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
| `ENABLE_CUDA_ITERATIVE`      |           `AUTO`            | `ON`, `OFF`, `AUTO` (if a CUDA toolkit is found)    | GPU (CUDA) backend for `LLIterative`/`predictIterative`, see below |
| `ENABLE_HIP_ITERATIVE`       |            `OFF`            | `ON`, `OFF`                                         | GPU (AMD ROCm/HIP) backend, explicit opt-in; CMake ≥ 3.21 |
| `ENABLE_SYCL_ITERATIVE`      |            `OFF`            | `ON`, `OFF`                                         | GPU (Intel oneAPI/SYCL) backend, explicit opt-in, unverified |
| `ENABLE_METAL_ITERATIVE`     |            `OFF`            | `ON`, `OFF`                                         | GPU (Apple Metal) backend, explicit opt-in, unverified; needs `METAL_CPP_DIR` |
| `USE_COMPILER_CACHE`         |        &lt;empty&gt;        | &lt;string&gt;                                      | name of a compiler cache program                  |
| `BUILD_SHARED_LIBS`          |            `ON`             | `ON`, `OFF`                                         |                                                   |
| `PYTHON_PREFIX_PATH`         |        &lt;empty&gt;        | &lt;string&gt;                                      | overrides default python path detection           |
| `Matlab_ROOT_DIR`            |        &lt;empty&gt;        | &lt;string&gt;                                      | locate Matlab root directory to help CMake finder |
| `SANITIZE`                   |            `OFF`            | `OFF`, `THREAD`, `ADDRESS`, `LEAK`                  | Enable sanitize feature (is available)            |
| `LBFGSB_SHOW_BUILD`          |            `OFF`            | `ON`, `OFF`                                         | Show details of `lbfgsb_cpp` sub-build            |
| `USE_JEMALLOC`               |            `OFF`            | `ON`, `OFF`                                         | Download and build jemalloc 5.3.0 as the default allocator (ignored on Windows). Off by default: it was disabled while debugging a TLS error in the Python bindings |

## GPU backend (`ENABLE_CUDA_ITERATIVE`)

Only `objective="LLIterative(...)"` fits and `predictIterative` use the GPU;
every other method runs on the CPU.

- `AUTO` (default): CUDA is enabled iff CMake (≥ 3.23) finds a working CUDA
  compiler and toolkit; the configure log prints
  `CUDA iterative backend: AUTO -> ON|OFF (reason)`. Without a toolkit the
  build is the usual CPU-only one. If `nvcc` rejects the default host
  compiler, set `CMAKE_CUDA_HOST_COMPILER` (e.g. `g++-12`).
- `ON`: CUDA is required; configuring fails without it.
- `OFF`: never. The release workflows (`.github/workflows/release-*.yml`)
  and `tools/release/python-release*.sh` pass `OFF`, so published packages
  (PyPI wheels, CRAN/Octave/MATLAB/Julia archives) are CPU-only.
- `CMAKE_CUDA_ARCHITECTURES`: `all-major` under `AUTO` (portable, no GPU
  needed at build time), `native` under `ON`, unless set explicitly.
- A CUDA-enabled build has **no load-time dependency on the CUDA toolkit**:
  the CUDA runtime is linked statically and cuBLAS is loaded on first use
  (`LK_CUBLAS_LIBRARY` may name the library explicitly). On a machine
  without GPU, driver or cuBLAS, the library still loads and runs the CPU
  path (a message is printed once if a GPU is present but cuBLAS is not).
- Not covered: the R package on Windows (Rtools/MinGW is not a supported
  `nvcc` host compiler; `tools/r-windows/build.sh` defaults to `OFF`), and
  HIP/SYCL/Metal, which stay explicit opt-ins with a load-time dependency
  on their runtime. CMake consumers of an installed *static* CUDA-enabled
  `libKriging` must `find_package(CUDAToolkit)` before including
  `lib/cmake/libKriging.cmake` (it references `CUDA::cudart_static`).
- Existing build directories keep their cached value: an old
  `ENABLE_CUDA_ITERATIVE:BOOL=OFF` stays `OFF` until reconfigured with
  `-DENABLE_CUDA_ITERATIVE=AUTO`.

The tool scripts (`tools/*/build.sh`) forward the `ENABLE_CUDA_ITERATIVE`
environment variable (default `AUTO`); `pip install` from source
(`setup.py`) does the same and also appends `LIBKRIGING_CMAKE_ARGS`
(e.g. `LIBKRIGING_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90"`).

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
