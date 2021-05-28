## [Dependency details](docs/dev/Dependencies.md)
libKriging requires external dependencies:
* [Armadillo](https://gitlab.com/conradsnicta/armadillo-code.git) for linear algebra
* [OptimLib](https://github.com/kthohr/optim.git) as an optimization library
* [Catch2](https://github.com/catchorg/Catch2.git) for unit tests
* [Pybind11](https://github.com/pybind/pybind11.git) for Python binding of C++ objects
* [Carma](https://github.com/libKriging/carma.git) for Python binding of armadillo objects (fork of [RUrlus/carma](https://github.com/RUrlus/carma.git))

The legacy dependencies have been integrated using [subrepo](https://github.com/ingydotnet/git-subrepo) components.
If you want to manage then, you need to install [subrepo](https://github.com/ingydotnet/git-subrepo) (for normal usage, this is not required).

Today, we include new dependencies using submodule technique.
