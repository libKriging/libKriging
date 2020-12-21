#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC_RANDOM_GENERATOR_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC_RANDOM_GENERATOR_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>

#include <carma/carma.h>

void set_random_seed(unsigned int seed);

py::array_t<double> generate_uniform_random_array(const unsigned int nrows, const unsigned int ncols);

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_RANDOM_GENERATOR_HPP
