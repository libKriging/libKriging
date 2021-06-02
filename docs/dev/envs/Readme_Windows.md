# Windows minimal setup

There are many ways to provide the required tools in Windows environment.

The way explained here is driven by the ability to be scriptable (as it is done in CI pipelines).

First of all, we need [Git Bash](https://gitforwindows.org) to be able to run Shell script on Windows (we do not recommend to use Cygwin).

## Package Managers

We use two different package managers: [Chocolatey](https://chocolatey.org) and [Anaconda](https://docs.conda.io/en/latest/miniconda.html).

### [Chocolatey](https://chocolatey.org) package manager

`choco` is the default package in Travis-CI and in GitHub Actions.

However `choco` has only few packages available about scientific computing.

Package list is available here: [https://chocolatey.org/search](https://chocolatey.org/search)

NB: On Windows, `choco` package manager requires admin rights, so that it could be simpler to do it by hand.

### Anaconda package manager

`conda` is an interesting alternative but not available by default in Travis-CI.

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) is one of its client.

Package list is available here: [https://anaconda.org/search](https://anaconda.org/search).

To install it, you can use [`install.sh`](../../../.travis-ci/windows/install.sh) script for more information.

This script will also install all the other required dependencies.

## Compiler

We recommend to use [Visual Studio Community](https://visualstudio.microsoft.com/fr/vs/community/)

Nevertheless, if you want to build libKriging for R, we recommend using a complete R environment (R + Rtools) and to use the associated compiler (See R toolchain).

# [Development for Windows without native Windows](Readme_Windows_Advanced.md)
