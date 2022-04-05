cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

## Forbids in-source builds (placed before PROJECT keyword)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if (NOT OCTAVE_BINDING_MODE)
    message(FATAL_ERROR "OCTAVE_BINDING_MODE should be defined")
endif ()

project(libKriging-${OCTAVE_BINDING_MODE}
        DESCRIPTION "LibKriging's ${OCTAVE_BINDING_MODE} binding")

set(LIBKRIGING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(LIBKRIGING_OCTAVE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(LIBKRIGING_OCTAVE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

include(OctaveMode.cmake)

add_subdirectory(tools)

add_mex_function(NAME mLibKriging
        SOURCES mLibKriging.cpp
        LinearRegression_binding.cpp LinearRegression_binding.hpp
        Kriging_binding.cpp Kriging_binding.hpp
        NuggetKriging_binding.cpp NuggetKriging_binding.hpp
        LINK_LIBRARIES Kriging OctaveShared)

install(TARGETS mLibKriging
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bindings/${OCTAVE_BINDING_MODE}
        )

install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/LinearRegression.m
        ${CMAKE_CURRENT_SOURCE_DIR}/Kriging.m
        ${CMAKE_CURRENT_SOURCE_DIR}/NuggetKriging.m
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bindings/${OCTAVE_BINDING_MODE}
        )

include(CTest)
add_subdirectory(tests)