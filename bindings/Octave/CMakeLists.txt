cmake_minimum_required(VERSION 3.10)

## Forbids in-source builds (placed before PROJECT keyword)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(libKriging-Octave
        DESCRIPTION "LibKriging's Octave binding")

set(LIBKRIGING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

include(${LIBKRIGING_SOURCE_DIR}/cmake/configureOctave.cmake)

add_subdirectory(tools)
add_subdirectory(toys)

add_mex_function(NAME mLibKriging SOURCES mLibKriging.cpp LINK_LIBRARIES Kriging OctaveShared OctaveToys)
add_mex_function(NAME mypow2 SOURCES mypow2.cpp)
add_mex_function(NAME mkStructure SOURCES mkStructure.cpp)
add_mex_function(NAME demoStruct SOURCES demoStruct.cpp)
add_mex_function(NAME mycell SOURCES mycell.cpp)