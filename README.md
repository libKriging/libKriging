[![Build Status](https://travis-ci.org/MASCOTNUM/libKriging.svg?branch=master)](https://travis-ci.org/MASCOTNUM/libKriging)
[![Coverage Status](https://coveralls.io/repos/github/MASCOTNUM/libKriging/badge.svg?branch=master)](https://coveralls.io/github/MASCOTNUM/libKriging?branch=master)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Table of contents

1. [Installation](#installation)
    1. [Requirements](#requirements)
    1. [Compilation and unit tests](#compilation-and-unit-tests)
    1. [Deployment](#deployment)

# Get the code

Just clone it:
```
git clone https://github.com/MASCOTNUM/libKriging.git
```

Dependencies are plugged in as [subrepo](https://github.com/ingydotnet/git-subrepo) components.
If you want to manage then, you need to install [subrepo](https://github.com/ingydotnet/git-subrepo) (for normal usage, this is not required).

# Installation

## Requirements
* CMake ≥ 3.11
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
     
* [lcov](http://ltp.sourceforge.net/coverage/lcov.php) is required for test coverage (with `genhtml` for pretty test coverage html reporting) 

## Compilation and unit tests

### Preamble

We assume that:
  * [libKriging](https://github.com/MASCOTNUM/libKriging.git) code is available locally in directory *`${LIBKRIGING}`*
    (could be a relative path like `..`)
  * you have built a fresh new directory *`${BUILD}`* 
    (should be an absolute path)
  * following commands are executed in *`${BUILD}`* directory 
  
PS: *`${NAME}`* represents a word or an absolute path of your choice

 Select your compilation *`${MODE}`* between: 
  * `Release` : produce an optimized code
  * `Debug` (default) : produce a debug code
  * `Coverage` : for code coverage analysis (not yet tested with Windows)

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
