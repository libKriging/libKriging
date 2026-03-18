cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(jlibkriging VERSION ${KRIGING_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(LIBKRIGING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

include(${LIBKRIGING_SOURCE_DIR}/cmake/version.cmake)

add_library(libkriging_c SHARED
        csrc/libkriging_c.cpp
        csrc/libkriging_c.h)

target_link_libraries(libkriging_c PRIVATE Kriging)

target_include_directories(libkriging_c PRIVATE
        ${LIBKRIGING_SOURCE_DIR}/src/lib/include)

target_compile_definitions(libkriging_c PRIVATE
        KRIGING_VERSION_INFO="${KRIGING_VERSION}")

set_target_properties(libkriging_c PROPERTIES
        OUTPUT_NAME "libkriging_c"
        PREFIX ""
        POSITION_INDEPENDENT_CODE ON)

install(TARGETS libkriging_c
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bindings/Julia)
install(FILES csrc/libkriging_c.h
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bindings/Julia/include)

# Julia tests (only if Julia is found)
find_program(JULIA_EXECUTABLE julia)
if(JULIA_EXECUTABLE)
    include(CTest)
    add_subdirectory(tests)
endif()
