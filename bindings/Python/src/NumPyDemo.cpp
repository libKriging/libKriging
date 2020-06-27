//
// Created by Pascal Havé on 27/06/2020.
//

#include "NumPyDemo.hpp"

py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
  auto buf1 = input1.request(), buf2 = input2.request();

  if (buf1.ndim != 1 || buf2.ndim != 1)
    throw std::runtime_error("Number of dimensions must be one");

  if (buf1.shape[0] != buf2.shape[0])
    throw std::runtime_error("Input shapes must match");

  auto result = py::array(py::buffer_info(
      nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double),     /* Size of one item */
      py::format_descriptor<double>::value, /* Buffer format */
      buf1.ndim,          /* How many dimensions? */
      { buf1.shape[0] },  /* Number of elements for each dimension */
      { sizeof(double) }  /* Strides for each dimension */
  ));

  auto buf3 = result.request();

  double *ptr1 = (double *) buf1.ptr,
      *ptr2 = (double *) buf2.ptr,
      *ptr3 = (double *) buf3.ptr;

  for (size_t idx = 0; idx < buf1.shape[0]; idx++)
    ptr3[idx] = ptr1[idx] + ptr2[idx];

  return result;
}
