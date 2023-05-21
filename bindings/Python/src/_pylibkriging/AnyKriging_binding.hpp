#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_ANYKRIGING_BINDING_H
#define LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_ANYKRIGING_BINDING_H

#include <pybind11/pybind11.h>
#include <libKriging/KrigingLoader.hpp>

namespace py = pybind11;

namespace AnyKriging {
    void describe(const py::dict &dict, std::string_view name, std::string_view type_name);
}

#endif //LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_ANYKRIGING_BINDING_H
