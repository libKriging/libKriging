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
| `USE_COMPILER_CACHE`         |        &lt;empty&gt;        | &lt;string&gt;                                      | name of a compiler cache program                  |
| `BUILD_SHARED_LIBS`          |            `ON`             | `ON`, `OFF`                                         |                                                   |
| `PYTHON_PREFIX_PATH`         |        &lt;empty&gt;        | &lt;string&gt;                                      | overrides default python path detection           |
| `Matlab_ROOT_DIR`            |        &lt;empty&gt;        | &lt;string&gt;                                      | locate Matlab root directory to help CMake finder |
| `SANITIZE`                   |            `OFF`            | `OFF`, `THREAD`, `ADDRESS`, `LEAK`                  | Enable sanitize feature (is available)            |
| `CMAKE_Fortran_COMPILER`     |         `gfortran`          | &lt;path&gt;                                        | Path to a fortran compiler                        |
| `Fortran_LINK_FLAGS`         | `-lgfortran -lquadmath -lm` | &lt;string&gt;                                      | Flags for additional libs and search path         |
| `LBFGSB_SHOW_BUILD`          |            `OFF`            | `ON`, `OFF`                                         | Show details of `lbfgsb_cpp` sub-build            |

