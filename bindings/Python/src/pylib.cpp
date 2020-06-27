#include <pybind11/pybind11.h>
#include <iostream>

#include "AddDemo.hpp"
#include "NumPyDemo.hpp"
#ifndef DISABLE_KRIGING
#include "LinearRegression_binding.hpp"
#endif

namespace py = pybind11;

PYBIND11_MODULE(pylibkriging, m) {
  
    load_test();
    
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: pylibkriging

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("add_arrays", &add_arrays, "Add two NumPy arrays");

    m.attr("__version__") = VERSION_INFO;
    
}