# mlibkriging — Octave/MATLAB binding for libKriging

## Prerequisites

- Octave ≥ 6.0 or MATLAB ≥ R2021
- C++ compiler with C++17 support
- CMake ≥ 3.13
- Linear algebra library (BLAS/LAPACK, OpenBLAS, or MKL)

## Install from pre-built packages

Download the archive from [libKriging releases](https://github.com/libKriging/libKriging/releases):

```shell
curl -LO https://github.com/libKriging/libKriging/releases/download/v0.9.0/mLibKriging_0.9.0_Linux-x86_64.tgz
tar xzf mLibKriging_0.9.0_Linux-x86_64.tgz
octave --path /path/to/mLibKriging
```

## Build from source

### Clone

```shell
git clone --recurse-submodules https://github.com/libKriging/libKriging.git
cd libKriging
```

### Linux / macOS

```shell
ENABLE_OCTAVE_BINDING=on ENABLE_PYTHON_BINDING=off tools/linux-macos/install.sh
ENABLE_OCTAVE_BINDING=on ENABLE_PYTHON_BINDING=off tools/linux-macos/build.sh
ENABLE_OCTAVE_BINDING=on tools/linux-macos/test.sh
```

Or build directly with CMake:

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OCTAVE_BINDING=on -DENABLE_PYTHON_BINDING=off .
cmake --build build --target install
```

### Windows

```shell
ENABLE_OCTAVE_BINDING=on tools/octave-windows/install.sh
ENABLE_OCTAVE_BINDING=on tools/octave-windows/build.sh
ENABLE_OCTAVE_BINDING=on tools/octave-windows/test.sh
```

### MATLAB (Linux only in CI)

```shell
ENABLE_MATLAB_BINDING=on ENABLE_PYTHON_BINDING=off tools/linux-macos/install.sh
ENABLE_MATLAB_BINDING=on ENABLE_PYTHON_BINDING=off tools/linux-macos/build.sh
ENABLE_MATLAB_BINDING=on tools/linux-macos/test.sh
```

## Test

```shell
cd build
ctest --output-on-failure
```

## Usage

Inside Octave or MATLAB:

```matlab
addpath("/path/to/build/installed/bindings/Octave")

X = [0.0;0.25;0.5;0.75;1.0];
f = @(x) 1-1/2.*(sin(12*x)./(1+x)+2*cos(7.*x).*x.^5+0.7);
y = f(X);

k_m = Kriging(y, X, "gauss");
disp(k_m.summary());

x = reshape(0:(1/99):1,100,1);
[p_mean, p_stdev] = k_m.predict(x, true, false);

s = k_m.simulate(int32(10), int32(123), x);
```

Full demo: [mlibkriging/tests/mLibKriging_demo.m](mlibkriging/tests/mLibKriging_demo.m)

## CI

Tested in GitHub Actions (`main.yml`):

| Job name        | OS           |
|:----------------|:-------------|
| Octave Linux    | Ubuntu 22.04 |
| Octave macOS    | macOS latest |
| Octave Windows  | Windows      |
| Matlab Linux    | Ubuntu 22.04 |
