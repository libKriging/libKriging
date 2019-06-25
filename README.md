[![Build Status](https://travis-ci.org/MASCOTNUM/libKriging.svg?branch=master)](https://travis-ci.org/MASCOTNUM/libKriging)
[![Coverage Status](https://coveralls.io/repos/github/MASCOTNUM/libKriging/badge.svg?branch=master)](https://coveralls.io/github/MASCOTNUM/libKriging?branch=master)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](https://opensource.org/licenses/Apache-2.0)


# Installation

## Requirements
* CMake ≥ 3.11
* C++ Compiler with C++11 support;
  
  Tested with:
     
  |   Compiler/OS    | Linux        | MacOS                    | Windows       |
  |:----------------:|:-------------|:-------------------------|:--------------|
  |       GCC        | **5.4**, 8.3 |                          |               |
  |      Clang       | 7.1          | AppleClang **9.1**, 10.0 |               |
  | MS Visual Studio |              |                          | **15 (2017)** |
  
  (bold values represent configurations used in Travis CI pipelines)
  
* [lcov](http://ltp.sourceforge.net/coverage/lcov.php) is required for test coverage (with `genhtml` for pretty test coverage html reporting) 

## Compilation and unit tests

### Preamble

We assume that:
  * [libKriging](https://github.com/MASCOTNUM/libKriging.git) code is available locally in directory *`$LIBKRIGING`*  
  * you have built a fresh new directory *`$BUILD`*
  * following commands are executed in *`$BUILD`* directory 
  
PS: *`$NAME`* represents an absolute path of your choice

 Select your compilation *`$MODE`* between: 
  * `Release` : produce an optimized code
  * `Debug` (default) : produce a debug code
  * `Coverage` : for code coverage analysis (not yet tested with Windows)

### Compilation for Linux and MacOS
  * Configure
  ```shell
  cmake -DCMAKE_BUILD_TYPE=$MODE $LIBKRIGING
  ```
  * Build
  ```shell
  cmake --build .
  ```
  * Run tests
  ```shell
  ctest
  ```
  
  * if you have selected `Coverage` mode, you can generate code coverage analysis over all tests using
  ```shell
  make coverage
   ```
  or 
  ```shell
  make coverage-report
   ```
  to produce an html report located in `$BUILD/coverage/index.html`
   
### Compilation for Windows
  * Configure
  ```shell
  cmake $LIBKRIGING
  ```
  * Build
  ```shell
  cmake --build . --target ALL_BUILD --config $MODE
  ```
  * Run tests
  ```shell
  export PATH=$BUILD/src/lib/$MODE:$PATH
  ctest -C $MODE
  ```
   
