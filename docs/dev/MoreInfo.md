# Dependency details

libKriging relies on the following external dependencies, vendored as git
submodules under `dependencies/` (see `.gitmodules`):

* [Armadillo](https://github.com/libKriging/armadillo-code) (fork of
  [conradsnicta/armadillo-code](https://gitlab.com/conradsnicta/armadillo-code))
  — C++ linear algebra library (Apache-2.0), by Conrad Sanderson et al.
* [lbfgsb_cpp](https://github.com/libKriging/lbfgsb_cpp) — C++ port by Pascal
  Havé of the `lbfgsb` bound-constrained optimizer (BSD-3), by Ciyou Zhu,
  Richard Byrd, Jorge Nocedal and Jose Luis Morales. Used for hyperparameter
  optimization.
* [Catch2](https://github.com/catchorg/Catch2) (v2.x) — unit-test framework
  (BSL-1.0).
* [pybind11](https://github.com/pybind/pybind11) — C++/Python interop for the
  Python binding (BSD-3).
* [carma](https://github.com/libKriging/carma) (fork of
  [RUrlus/carma](https://github.com/RUrlus/carma)) — Armadillo <-> NumPy
  bridge for the Python binding (Apache-2.0).

All dependencies are included as git **submodules**. Clone the repository with
`--recurse-submodules` (see the top-level [README](../../README.md#get-the-code)).
