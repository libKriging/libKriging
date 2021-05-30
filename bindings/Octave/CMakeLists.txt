cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

## Forbids in-source builds (placed before PROJECT keyword)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(libKriging-Octave
        DESCRIPTION "LibKriging's Octave binding")

set(LIBKRIGING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(LIBKRIGING_OCTAVE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(LIBKRIGING_OCTAVE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

include(${LIBKRIGING_SOURCE_DIR}/cmake/configureOctave.cmake)

add_subdirectory(tools)

add_mex_function(NAME mLibKriging
        SOURCES mLibKriging.cpp 
        LinearRegression_binding.cpp LinearRegression_binding.hpp
        Kriging_binding.cpp Kriging_binding.hpp
        LINK_LIBRARIES Kriging OctaveShared)

# https://cmake.org/cmake/help/v3.11/module/GenerateExportHeader.html
include (GenerateExportHeader)
GENERATE_EXPORT_HEADER(mLibKriging # generates the export header `lib`_EXPORTS.h automatically
        BASE_NAME MLIBKRIGING
        EXPORT_FILE_NAME mlibKriging_exports.h)
target_include_directories(mLibKriging PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS mLibKriging
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bindings/Octave
        )

install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/LinearRegression.m
        ${CMAKE_CURRENT_SOURCE_DIR}/Kriging.m
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bindings/Octave
        )

include(CTest)
add_subdirectory(tests)
