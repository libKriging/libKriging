#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_DICTTEST_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_DICTTEST_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11;

bool check_dict_entry(const py::dict& dict, std::string_view name, std::string_view type_name);

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_DICTTEST_HPP
