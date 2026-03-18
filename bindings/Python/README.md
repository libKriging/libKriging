# pylibkriging — Python binding for libKriging

## Prerequisites

- Python ≥ 3.7 with pip
- C++ compiler with C++17 support
- CMake ≥ 3.13
- Linear algebra library (BLAS/LAPACK, OpenBLAS, or MKL)

## Quick install from PyPI

```shell
pip3 install pylibkriging
```

## Build from source

### Clone

```shell
git clone --recurse-submodules https://github.com/libKriging/libKriging.git
cd libKriging
```

### Linux / macOS

```shell
ENABLE_PYTHON_BINDING=on tools/linux-macos/install.sh
ENABLE_PYTHON_BINDING=on tools/linux-macos/build.sh
ENABLE_PYTHON_BINDING=on tools/linux-macos/test.sh
```

Or build directly with CMake:

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON_BINDING=on .
cmake --build build --target install
```

### Windows

```shell
ENABLE_PYTHON_BINDING=on tools/windows/install.sh
ENABLE_PYTHON_BINDING=on tools/windows/build.sh
```

### Using `pip install` from source

This will download, compile, and install pylibkriging in one step:

```shell
pip3 install .
```

Or directly from GitHub:

```shell
pip3 install "git+https://github.com/libKriging/libKriging.git"
```

## Test

```shell
cd build
ctest --output-on-failure
```

Or with pytest:

```shell
cd bindings/Python/pylibkriging
python3 -m pytest tests/
```

## Usage

```python
import numpy as np
import pylibkriging as lk

X = [0.0, 0.25, 0.5, 0.75, 1.0]
f = lambda x: (1 - 1 / 2 * (np.sin(12 * x) / (1 + x) + 2 * np.cos(7 * x) * x ** 5 + 0.7))
y = [f(xi) for xi in X]

k = lk.Kriging(y, X, "gauss")
print(k.summary())

x = np.arange(0, 1, 1 / 99)
p = k.predict(x, True, False)
```

Full demo: [tests/pylibkriging_demo.py](pylibkriging/tests/pylibkriging_demo.py)

## CI

Tested in GitHub Actions (`main.yml`):

| Job name               | OS           |
|:-----------------------|:-------------|
| Linux Debug            | Ubuntu 22.04 |
| macOS Debug            | macOS latest |
| Python (3.7) Windows   | Windows      |
| Python (3.9) Windows   | Windows      |
