#include "py_to_cpp_cast.hpp"

#include <carma>
#include <iostream>

template <>
std::optional<bool> to_cpp_cast<bool>(py::handle obj) {
  const auto py_type_name = py::str(obj.get_type()).cast<std::string>();
  if (py_type_name == "<class 'bool'>") {
    return std::make_optional<bool>(obj.cast<bool>());
  } else {
    return std::nullopt;
  }
}

template <>
std::optional<int> to_cpp_cast<int>(py::handle obj) {
  const auto py_type_name = py::str(obj.get_type()).cast<std::string>();
  if (py_type_name == "<class 'int'>") {
    return std::make_optional<int>(obj.cast<int>());
  } else {
    return std::nullopt;
  }
}

template <>
std::optional<double> to_cpp_cast<double>(py::handle obj) {
  const auto py_type_name = py::str(obj.get_type()).cast<std::string>();
  if (py_type_name == "<class 'int'>") {
    return std::make_optional<double>(obj.cast<int>());
  } else if (py_type_name == "<class 'float'>") {
    return std::make_optional<double>(obj.cast<double>());
  } else if (py_type_name == "<class 'numpy.float64'>") {
    return std::make_optional<double>(obj.cast<double>());
  } else {
    std::cout << "unknown py_type_name: " << py_type_name << std::endl;
    return std::nullopt;
  }
}

template <>
std::optional<std::string> to_cpp_cast<std::string>(py::handle obj) {
  const auto py_type_name = py::str(obj.get_type()).cast<std::string>();
  if (py_type_name == "<class 'str'>") {
    return std::make_optional<std::string>(obj.cast<std::string>());
  } else {
    return std::nullopt;
  }
}

template <>
std::optional<arma::mat> to_cpp_cast<arma::mat>(py::handle obj) {
  const auto py_type_name = py::str(obj.get_type()).cast<std::string>();
  if (py_type_name == "<class 'numpy.ndarray'>") {
    py::array_t<double> arr = obj.cast<py::array_t<double>>();
    return std::make_optional<arma::mat>(carma::arr_to_mat(arr));
  } else {
    return std::nullopt;
  }
}

template <>
std::optional<arma::colvec> to_cpp_cast<arma::vec>(py::handle obj) {
  auto opt_mat = to_cpp_cast<arma::mat>(obj);
  if (opt_mat.has_value() && opt_mat.value().n_cols > 1) {
    return std::nullopt;
  }
  return opt_mat;
}

template <>
std::optional<arma::rowvec> to_cpp_cast<arma::rowvec>(py::handle obj) {
  auto opt_mat = to_cpp_cast<arma::mat>(obj);
  if (opt_mat.has_value() && opt_mat.value().n_rows > 1) {
    return std::nullopt;
  }
  return opt_mat;
}
