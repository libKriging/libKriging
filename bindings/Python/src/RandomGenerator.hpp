#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC_RANDOMGENERATOR_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC_RANDOMGENERATOR_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <carma>

#include <armadillo>
#include <random>

class RandomGenerator {
 public:
  RandomGenerator(unsigned int seed);
  py::array_t<double> uniform(const unsigned int nrows, const unsigned int ncols);

 private:
  std::mt19937 m_engine;
};

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_RANDOMGENERATOR_HPP
