# Requirements

* [CMake](https://cmake.org/download/) ≥ 3.13 
  
  because we use `target_link_options` feature.

* C++ Compiler with C++17 support

  Tested with (***not up-to-date***):

  |   Compiler/OS    | Linux        | macOS                    | Windows       |
  |:----------------:|:-------------|:-------------------------|:--------------|
  |       GCC        | **5.4**, 8.3 |                          | **4.9.3** (R) |
  |      Clang       | 7.1          | AppleClang **9.1**<br>AppleClang 10.0 (Mojave)<br>AppleClang 11.0 (Catalina)<br>Clang 9.0|               |
  | MS Visual Studio |              |                          | **15 (2017)** |

  (bold values represent configurations used in Travis CI pipelines)

* Linear algebra packages providing blas and lapack functions.

  You can use standard blas and lapack, OpenBlas, MKL.

* Python ≥ 3.6 (optional)

* Octave ≥ 4.2 (optional)

* R ≥ 3.6 (optional)

## [Linux setup](Readme_Linux.md)
## [macOS setup](Readme_macOS.md)
## [Windows setup](Readme_Windows.md)

## Optional useful tools
* [lcov](http://ltp.sourceforge.net/coverage/lcov.php) is required for test coverage (with `genhtml` for pretty test coverage html reporting)
* clang-format for automated code formatting
* clang-tidy for static analysis
* Doxygen for doc generation
