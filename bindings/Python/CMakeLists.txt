cmake_minimum_required(VERSION 3.1)
project(pylibkriging)

set(CMAKE_CXX_STANDARD 17)

set(LIBKRIGING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(LIBKRIGING_PYTHON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(LIBKRIGING_PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

include(${LIBKRIGING_SOURCE_DIR}/cmake/version.cmake)


# force pybind11 to use the right python interpreter, it could be useful 
# to define CMAKE_PREFIX_PATH with PYTHON_PREFIX_PATH (cf carma CMake config)
# pybind11 is loading manually to be always available even if conditionally not loaded by carma
add_subdirectory(${LIBKRIGING_SOURCE_DIR}/dependencies/pybind11 pybind11)
set(CARMA_DEV_TARGET false)
# following CACHE is required (cf https://cmake.org/cmake/help/git-stage/policy/CMP0077.html)
set(BUILD_TESTS on CACHE BOOL "Enable test in carma") 
add_subdirectory(${LIBKRIGING_SOURCE_DIR}/dependencies/carma carma)

set(pylibkriging_SRCS src/pylibkirging.cpp)
list(APPEND pylibkriging_SRCS 
        src/AddDemo.cpp src/AddDemo.hpp
        src/NumPyDemo.cpp src/NumPyDemo.hpp
        )
if (NOT DISABLE_KRIGING)
    list(APPEND pylibkriging_SRCS 
            src/LinearRegression_binding.cpp src/LinearRegression_binding.hpp
            )
endif()

pybind11_add_module(pylibkriging ${pylibkriging_SRCS})
if (NOT DISABLE_KRIGING)
    target_link_libraries(pylibkriging PRIVATE Kriging carma)
else()
    target_link_libraries(pylibkriging PRIVATE carma)
    target_compile_definitions(pylibkriging PUBLIC DISABLE_KRIGING)    
endif()
target_compile_definitions(pylibkriging PUBLIC VERSION_INFO=\"${KRIGING_VERSION}\")

enable_testing()
add_subdirectory(tests)
