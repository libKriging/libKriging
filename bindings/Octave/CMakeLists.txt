cmake_minimum_required(VERSION 3.10)

## Forbids in-source builds (placed before PROJECT keyword)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(libKriging-Octave
        DESCRIPTION "LibKriging's Octave binding")

set(LIBKRIGING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

include(${LIBKRIGING_SOURCE_DIR}/cmake/configureOctave.cmake)

add_subdirectory(tools)

add_mex_function(NAME mLibKriging
        SOURCES mLibKriging.cpp LinearRegression_binding.cpp LinearRegression_binding.hpp
        LINK_LIBRARIES Kriging OctaveShared)

install(TARGETS mLibKriging
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bindings/Octave
        )

install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/LinearRegression.m
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bindings/Octave
        )
