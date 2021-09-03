#include "BindingTest.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>
#include <libKriging/demo/DemoArmadilloClass.hpp>

py::array_t<double> direct_binding(py::array_t<double>& input1, py::array_t<double>& input2) {
  auto buf1 = input1.request(), buf2 = input2.request();

  if (buf1.ndim != 1 || buf2.ndim != 1)
    throw std::runtime_error("Number of dimensions must be one");

  if (buf1.shape[0] != buf2.shape[0])
    throw std::runtime_error("Input shapes must match");

  auto result = py::array(py::buffer_info(nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
                                          sizeof(double), /* Size of one item */
                                          py::format_descriptor<double>::value, /* Buffer format */
                                          buf1.ndim,                            /* How many dimensions? */
                                          {buf1.shape[0]}, /* Number of elements for each dimension */
                                          {sizeof(double)} /* Strides for each dimension */
                                          ));

  auto buf3 = result.request();

  double *ptr1 = (double*)buf1.ptr, *ptr2 = (double*)buf2.ptr, *ptr3 = (double*)buf3.ptr;

  for (size_t idx = 0; idx < buf1.shape[0]; idx++)
    ptr3[idx] = ptr1[idx] + ptr2[idx];

  return result;
}

py::array_t<double> one_side_carma_binding(py::array_t<double>& input1, py::array_t<double>& input2) {
  if (input1.request().ndim != 1 || input2.request().ndim != 1)
    throw std::runtime_error("Number of dimensions must be one");

  arma::rowvec a = carma::arr_to_row<double>(input1);
  arma::rowvec b = carma::arr_to_row<double>(input2);

  if (a.n_rows != 1 || b.n_rows != 1)
    throw std::runtime_error("Vector should be rowvec");

  if (a.n_cols != b.n_cols)
    throw std::runtime_error("Input shapes must match");

  arma::rowvec result = a + b;

  return carma::row_to_arr(result, true);
}

py::array_t<double> two_side_carma_binding(py::array_t<double>& input1, py::array_t<double>& input2) {
  if (input1.request().ndim != 1 || input2.request().ndim != 1)
    throw std::runtime_error("Number of dimensions must be one");

  arma::rowvec a = carma::arr_to_row(input1);
  arma::rowvec b = carma::arr_to_row(input2);

  if (a.n_rows != 1 || b.n_rows != 1)
    throw std::runtime_error("Vector should be rowvec");

  if (a.n_cols != b.n_cols)
    throw std::runtime_error("Input shapes must match");

  DemoArmadilloClass cl{a};
  arma::rowvec result = cl.apply(b);

  return carma::row_to_arr(result, true);
}