# Using C++ libKriging inside RStudio #

We describe here how to embed `libKriging` C++ library in your own package built with RStudio.

## 1. Build and install `libKriging` C++ library

Follow steps writen in main [README.md](../../README.md) in sections
[Compilation for Linux/Mac/Windows using R toolchain](../../README.md#compilation-for-linuxmacwindows-using-r-toolchain) and [Deployment](../../README.md#deployment).

## 2. Starting with RStudio

If you have already clone libKriging, start a new package project using `File > New Project... > New Directory > R Package using Rcpp Armadillo`. You can also start a new package project using `File > New Project... > Version control > git` and fill "Repository URL" with 'https://github.com/MASCOTNUM/libKriging'.

## 3. Update configuration

1. Replace content of `Makevars` (or `Makevars.win` depending on your target architecture) using files from those from `libKriging/bindings/R/rlibkriging/src`
2. Add a newline at the beginning with:
```
LIBKRIGING_PATH=/path/to/libKriging/build/installed
```
3. Replace the last line
```
include ../../check.mk
```
by
```
.check:
```
(this will bypass specific checks from rlibkriging compilation).

## 3. At this point your new package should compile

You can use this method to prepare new R bindings. In that case, you should use `libKriging/bindings/R/rlibkriging` to do so.

(DO NOT COMMIT previsouly modified `Makevars` or `Makevars.win`).
