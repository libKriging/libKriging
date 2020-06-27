cmake_minimum_required(VERSION 3.1)
project(pylibkriging)

set(LIBKRIGING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(LIBKRIGING_PYTHON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(LIBKRIGING_PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(pybind11)

set(pylibkriging_SRCS src/pylib.cpp)
list(APPEND pylibkriging_SRCS 
        src/AddDemo.cpp src/AddDemo.hpp
        src/NumPyDemo.cpp src/NumPyDemo.hpp
        )
if (NOT DEFINED DISABLE_KRIGING)
    list(APPEND pylibkriging_SRCS 
            src/LinearRegression_binding.cpp src/LinearRegression_binding.hpp
            )
endif()

pybind11_add_module(pylibkriging ${pylibkriging_SRCS})
if (NOT DEFINED DISABLE_KRIGING)
    target_link_libraries(pylibkriging PRIVATE Kriging)
else()
    target_compile_definitions(pylibkriging PUBLIC DISABLE_KRIGING)    
endif()
target_compile_definitions(pylibkriging PUBLIC VERSION_INFO=\"${KRIGING_VERSION}\")

enable_testing()
add_subdirectory(tests)
