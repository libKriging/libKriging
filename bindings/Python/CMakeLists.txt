cmake_minimum_required(VERSION 3.1)
project(pylibkriging)

set(CMAKE_CXX_STANDARD 17)

set(LIBKRIGING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
if (WIN32)
    # $<CONFIG> is preferred since CMake 3.0
    set(LIBKRIGING_PYTHON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
else()
    set(LIBKRIGING_PYTHON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()
set(LIBKRIGING_PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

include(${LIBKRIGING_SOURCE_DIR}/cmake/version.cmake)

set(CARMA_DEV_TARGET false)
# following CACHE is required (cf https://cmake.org/cmake/help/git-stage/policy/CMP0077.html)
# set(BUILD_TESTS on CACHE BOOL "Enable test in carma") 
add_subdirectory(${LIBKRIGING_SOURCE_DIR}/dependencies/carma carma)

set(pylibkriging_SRCS src/pylibkirging.cpp)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND pylibkriging_SRCS 
            src/NumPyDemo.cpp src/NumPyDemo.hpp
            )
endif()
list(APPEND pylibkriging_SRCS 
        src/LinearRegression_binding.cpp src/LinearRegression_binding.hpp
        src/Kriging_binding.cpp src/Kriging_binding.hpp
        src/RandomGenerator.cpp src/RandomGenerator.hpp
    )

pybind11_add_module(pylibkriging ${pylibkriging_SRCS})
target_link_libraries(pylibkriging PRIVATE Kriging carma)
target_compile_definitions(pylibkriging PUBLIC KRIGING_VERSION_INFO=\"${KRIGING_VERSION}\") # renamed to avoid collision with VERSION_INFO from carma
target_compile_definitions(pylibkriging PUBLIC BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\")

include(CTest)
add_subdirectory(tests)

install(TARGETS pylibkriging
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bindings/Python)