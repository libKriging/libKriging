//
// Created by Pascal Havé on 27/06/2020.
//

#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC_ORDINARYKRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC_ORDINARYKRIGING_BINDING_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <libKriging/Kriging.hpp>
#include <string>
#include <tuple>

namespace py = pybind11;

class PyKriging {
 public:
  PyKriging(const std::string& kernel);
  ~PyKriging();

  void fit(const py::array_t<double>& y, const py::array_t<double>& X);

  // TODO The result should be a namedtuple
  // see
  // - https://docs.python.org/3/library/collections.html#namedtuple-factory-function-for-tuples-with-named-fields
  // - https://github.com/pybind/pybind11/issues/1244
  std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> predict(const py::array_t<double>& X,
                                                                                    bool withStd,
                                                                                    bool withCov);

 private:
  std::unique_ptr<Kriging> m_internal;
};

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_ORDINARYKRIGING_BINDING_HPP
