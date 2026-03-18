# jlibkriging — Julia binding for libKriging

## Prerequisites

- Julia ≥ 1.10
- C++ compiler with C++17 support
- CMake ≥ 3.13
- Linear algebra library (BLAS/LAPACK, OpenBLAS, or MKL)

## Build from source

### Clone

```shell
git clone --recurse-submodules https://github.com/libKriging/libKriging.git
cd libKriging
```

### Linux / macOS

```shell
ENABLE_JULIA_BINDING=on tools/linux-macos/install.sh
ENABLE_JULIA_BINDING=on tools/linux-macos/build.sh
ENABLE_JULIA_BINDING=on tools/linux-macos/test.sh
```

Or build directly with CMake:

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_JULIA_BINDING=on .
cmake --build build
```

### Windows

```shell
ENABLE_JULIA_BINDING=on tools/windows/install.sh
ENABLE_JULIA_BINDING=on tools/windows/build.sh
```

Or build directly with CMake:

```shell
cmake -B build -DCMAKE_GENERATOR_PLATFORM=x64 \
  -DENABLE_JULIA_BINDING=on .
cmake --build build --config Release
```

### Install the Julia package

After building, register the package in your Julia environment:

```shell
julia -e 'using Pkg; Pkg.develop(path="bindings/Julia/jlibkriging")'
```

The shared library `libkriging_c` is auto-detected from the `build/` directory.

## Test

Via CMake/CTest (runs all tests including Julia):

```shell
cd build
ctest --output-on-failure -L Julia
```

Or run individual test files:

```shell
julia --project=bindings/Julia/jlibkriging bindings/Julia/jlibkriging/tests/jlibkriging_demo.jl
```

## Usage

```julia
using jlibkriging

X = reshape([0.0, 0.25, 0.5, 0.75, 1.0], :, 1)
f(x) = 1 - 1/2 * (sin(12*x) / (1+x) + 2*cos(7*x) * x^5 + 0.7)
y = f.(X[:, 1])

k = Kriging(y, X, "gauss")
println(jlibkriging.summary(k))

x = reshape(collect(0:0.01:1), :, 1)
p = predict(k, x; stdev=true, cov=false)

s = simulate(k, 10, 123, x)
```

Full demo: [jlibkriging/tests/jlibkriging_demo.jl](jlibkriging/tests/jlibkriging_demo.jl)

## CI

Tested in GitHub Actions (`main.yml`):

| Job name       | OS           |
|:---------------|:-------------|
| Julia Linux    | Ubuntu 22.04 |
| Julia macOS    | macOS latest |
| Julia Windows  | Windows      |
