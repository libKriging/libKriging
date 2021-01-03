<!-- [![Build Status](https://travis-ci.com/libKriging/libKriging.svg?branch=master)](https://travis-ci.com/libKriging/libKriging) -->
[![Github CI](https://github.com/libKriging/libKriging/workflows/Github%20CI/badge.svg)](https://github.com/libKriging/libKriging/actions?query=workflow%3A%22Github+CI%22)
[![Coverage Status](https://coveralls.io/repos/github/libKriging/libKriging/badge.svg?branch=master)](https://coveralls.io/github/libKriging/libKriging?branch=master)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Table of contents

1. [Installation](#installation)
    1. [Requirements](#requirements)
    1. [Compilation and unit tests](#compilation-and-unit-tests)
    1. [Deployment](#deployment)

# Get the code

Just clone it:
```
git clone https://github.com/libKriging/libKriging.git
```

Dependencies are plugged in as [subrepo](https://github.com/ingydotnet/git-subrepo) components.
If you want to manage then, you need to install [subrepo](https://github.com/ingydotnet/git-subrepo) (for normal usage, this is not required).

If you want to contribute read [Contribution guide](CONTRIBUTING.md).

# Installation

## Requirements
* CMake ≥ 3.11

  You can install it by hand (cf [cmake](https://cmake.org/download/)) or using automated method from `.travis-ci/<your-environment>/install.sh` script.

  NB: On Windows, `choco` package manager requires admin rights, so that it could be simpler to do it by hand.
  
* C++ Compiler with C++11 support;
  
  Tested with:
     
  |   Compiler/OS    | Linux        | macOS                    | Windows       |
  |:----------------:|:-------------|:-------------------------|:--------------|
  |       GCC        | **5.4**, 8.3 |                          | **4.9.3** (R) |
  |      Clang       | 7.1          | AppleClang **9.1**<br>AppleClang 10.0 (Mojave)<br>AppleClang 11.0 (Catalina)<br>Clang 9.0|               |
  | MS Visual Studio |              |                          | **15 (2017)** |
  
  (bold values represent configurations used in Travis CI pipelines)
  
  NB: Ensure C++ compiler is available. On macOS systems you will need to install Xcode
     with additional Command Line Tools using:  
     ```
     xcode-select --install
     sudo xcodebuild -license accept
     ```
     (should be done after macOS upgrade) 
     
  NB: On Windows with R environment (R + Rtools), you can use R's recommanded compiler. See compilation woth R toolchain below.
  
* Linear algebra packages providing blas and lapack functions.
  
  You can use standard blas and lapack, OpenBlas, MKL.
  
  On Windows, the simplest method is to use Anaconda (cf `install.sh` scripts in `.travis-ci` or [Readme_Windows.md](.travis-ci/Readme_Windows.md)).
  
### Optional tools
     
* [lcov](http://ltp.sourceforge.net/coverage/lcov.php) is required for test coverage (with `genhtml` for pretty test coverage html reporting)
* clang-format for automated code formatting
* clang-tidy for static analysis
* Doxygen for doc generation

## Integrated scripts for CI

Note: calling these scripts "by hand" should produce the same results than following "Compilation and unit tests" instructions (and it should be also easier).

### Integration for Linux and MacOS

With standard cmake & system libs:
```shell
cd libKriging
.travis-ci/linux-macos/build.sh
```

With R specific cmake & system libs (needed for rlibkriging):
```shell
cd libKriging
.travis-ci/r-linux-macos/build.sh
```

### Integration for Windows

With standard cmake & system libs:
```shell
cd libKriging
.travis-ci/windows/build.sh
```

With R specific cmake & system libs (needed for rlibkriging):
```shell
cd libKriging
.travis-ci/r-windows/build.sh
```

## Compilation and unit tests

### Preamble

We assume that:
  * [libKriging](https://github.com/libKriging/libKriging.git) code is available locally in directory *`${LIBKRIGING}`*
    (could be a relative path like `..`)
  * you have built a fresh new directory *`${BUILD}`* 
    (should be an absolute path)
  * following commands are executed in *`${BUILD}`* directory 
  
PS: *`${NAME}`* represents a word or an absolute path of your choice

Select your compilation *`${MODE}`* between: 
  * `Release` : produce an optimized code
  * `Debug` (default) : produce a debug code
  * `Coverage` : for code coverage analysis (not yet tested with Windows)

Following commands are made for Unix shell. To use them with Windows use [Mingw](http://www.mingw.org) or [git-bash](https://gitforwindows.org) environment.

### Compilation for Linux and MacOS
  
  * Configure
      ```shell
      cmake -DCMAKE_BUILD_TYPE=${MODE} ${LIBKRIGING}
      ```
  * Build
      ```shell
      cmake --build .
      ```
      aka with classical makefiles
      ```shell
      make  
      ```
  * Run tests
      ```shell
      ctest
      ```
      aka with classical makefiles
      ```shell
      make test  
      ```
  * Buidl documentation (requires doxygen)
      ```shell
      cmake --build . --target doc
      ```
      aka with classical makefiles
      ```shell
      make doc
      ```
  * if you have selected `MODE=Coverage` mode, you can generate code coverage analysis over all tests using
      ```shell
      cmake --build . --target coverage --config Coverage
      ```
      aka with classical makefiles
      ```shell
      make coverage
      ```
      or 
      ```shell
      cmake --build . --target coverage-report --config Coverage
      ```
      aka with classical makefiles
      ```shell
      make coverage-report
      ```
      to produce an html report located in `${BUILD}/coverage/index.html`
   
### Compilation for Windows 64bits with Visual Studio
  * Configure
      ```shell
      cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DEXTRA_SYSTEM_LIBRARY_PATH=${EXTRA_SYSTEM_LIBRARY_PATH} ${LIBKRIGING}
      ```
      where `EXTRA_SYSTEM_LIBRARY_PATH` is an extra path where libraries (e.g. OpenBLAS) can be found.
  * Build
      ```shell
      cmake --build . --target ALL_BUILD --config ${MODE}
      ```
  * Run tests
      ```shell
      export PATH=${BUILD}/src/lib/${MODE}:$PATH
      ctest -C ${MODE}
      ```
    
### Compilation for Linux/Mac/Windows using R toolchain

  With this method, you need [R](https://cran.r-project.org) (and [R-tools](https://cran.r-project.org/bin/windows/Rtools/) if you are on Windows).
  
  We assume you have previous requirements and also `make` command available in your `PATH`.
  
  * Configure
      ```shell
      CC=$(R CMD config CC) CXX=$(R CMD config CXX) cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=${MODE} ${LIBKRIGING}
      ```
  * Build
      ```shell
      cmake --build .
      ```
  * Run tests
      ```shell
      ctest
      ```
       
## Deployment

To deploy libKriging as an installed library, you have to add `-DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX}` option to 
first `cmake` configuration command.

If `CMAKE_INSTALL_PREFIX` variable is not set with CMake, default installation directoty is `${BUILD}/installed`.

### For Linux and MacOS

e.g.:
```shell
cmake -DCMAKE_BUILD_TYPE=${MODE} -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX} ${LIBKRIGING}
```
and then 
```shell
cmake --build . --target install
```
aka with classical makefiles
```shell
make install
```

### For Windows 64bits with Visual Studio

e.g.:
```shell
cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DEXTRA_SYSTEM_LIBRARY_PATH=${EXTRA_SYSTEM_LIBRARY_PATH} -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX} ${LIBKRIGING} 
```
and then 
```shell
cmake --build . --target install --config ${MODE}
```
