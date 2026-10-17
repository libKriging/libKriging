# rlibkriging — R binding for libKriging

## Prerequisites

- R ≥ 4.0
- C++ compiler with C++17 support (R toolchain is used)
- CMake ≥ 3.13
- Linear algebra library (BLAS/LAPACK, OpenBLAS, or MKL)
- On Windows: [Rtools](https://cran.r-project.org/bin/windows/Rtools/)

## Quick install from CRAN

```r
install.packages('rlibkriging')
```

## Build from source

### Clone

```shell
git clone --recurse-submodules https://github.com/libKriging/libKriging.git
cd libKriging
```

### Linux / macOS

The R binding uses R's own compiler toolchain. The helper scripts handle this automatically:

```shell
tools/r-linux-macos/install.sh
tools/r-linux-macos/build.sh
tools/r-linux-macos/test.sh
```

Or build manually:

```shell
# Build the C++ library using R's compiler
CC=$(R CMD config CC) CXX=$(R CMD config CXX) \
  cmake -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release .
cmake --build build --target install

# Build and install the R package
export LIBKRIGING_PATH=$PWD/build/installed
cd bindings/R
make
```

### Windows

```shell
tools/r-windows/install.sh
tools/r-windows/build.sh
tools/r-windows/test.sh
```

## Test

```shell
cd bindings/R
export LIBKRIGING_PATH=$PWD/../../build/installed
make test
```

## Usage

```r
library(rlibkriging)

X <- as.matrix(c(0.0, 0.25, 0.5, 0.75, 1.0))
f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
y <- f(X)

k_R <- Kriging(y, X, "gauss")
print(k_R)

x <- as.matrix(seq(0, 1, , 100))
p <- predict(k_R, x, TRUE, FALSE)

s <- simulate(k_R, nsim = 10, seed = 123, x = x)
```

Full demo: [rlibkriging/tests/testthat/test-rlibkriging-demo.R](rlibkriging/tests/testthat/test-rlibkriging-demo.R)

## Parallel R code (`mclapply`, forked workers)

libKriging uses OpenMP (covariance loops), threads (parallel `WarpKriging` /
`MLPKriging` multistart) and the system BLAS, which may be multithreaded. A process
forked while such a thread pool is alive (`parallel::mclapply`, `parallel::mcparallel`,
`future::multicore`) can deadlock in the child. libKriging quiesces its own OpenMP and OpenBLAS threads before every
`fork()` (`pthread_atfork`), which covers the usual OpenBLAS builds. If forked workers
still hang (e.g. with an OpenMP build of OpenBLAS, `libopenblaso`, selected as the
system BLAS, or another threaded BLAS), make the parent single-threaded before
starting R:

```shell
export OMP_THREAD_LIMIT=1 GOMP_SPINCOUNT=0 OPENBLAS_NUM_THREADS=1
```

or use socket workers (`parallel::makeCluster(type = "PSOCK")`, `future::multisession`),
which do not fork. Models are shipped to them by R serialization (rlibkriging versions
including [#327](https://github.com/libKriging/libKriging/issues/327)) or with `$save()` /
`load.Kriging()`.

## CI

Tested in GitHub Actions (`main.yml`):

| Job name               | OS           |
|:-----------------------|:-------------|
| R Linux                | Ubuntu 22.04 |
| R Linux without OpenMP | Ubuntu 22.04 |
| R macOS                | macOS latest |
| R Windows              | Windows      |
