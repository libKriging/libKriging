cmake_minimum_required(VERSION 3.10)

# CMake utils
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

## Forbids in-source builds (placed before PROJECT keyword)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
# Partially disabled since Armadillo do not preserve its source directory when
# this is due to line: configure_file(${PROJECT_SOURCE_DIR}/examples/Makefile.cmake ${PROJECT_SOURCE_DIR}/examples/Makefile)
# set(CMAKE_DISABLE_SOURCE_CHANGES ON)

# VERSION must follow pattern : <major>.<minor>.<patch>
set(KRIGING_VERSION_MAJOR 0)
set(KRIGING_VERSION_MINOR 0)
set(KRIGING_VERSION_PATCH 0)
set(KRIGING_VERSION "${KRIGING_VERSION_MAJOR}.${KRIGING_VERSION_MINOR}.${KRIGING_VERSION_PATCH}")

project(libKriging
        VERSION ${KRIGING_VERSION}
        DESCRIPTION "LibKriging")
# PROJECT_VERSION now contains also KRIGING_VERSION

#------------------------------------------------------

# Prevent from root system installation
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/installed" CACHE PATH "default install path" FORCE)
    # Force update for sub-libraries (to follow current installation directive)
    set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE CACHE BOOL "Installation prefix has been set" FORCE)
endif ()

#------------------------------------------------------

set(LIBKRIGING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LIBKRIGING_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
include(${LIBKRIGING_SOURCE_DIR}/cmake/configuration.cmake)

#------------------------------------------------------

#option(BUILD_SHARED_LIBS "Build the shared library" ON) # BUILD_SHARED_LIBS config should be after armadillo config
#option(BUILD_STATIC_LIBS "Build the static library" ON) # TODO not yet implemented; today, equivalent to BUILD_SHARED_LIBS=off 
option(LIBKRIGING_BENCHMARK_TESTS "Enable benchmark tests while testing" OFF)

#------------------------------------------------------
# Check options

# Default build type is RelWIthDebInfo
if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
            "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel Coverage"
            FORCE)
else ()
    string(REGEX MATCH "^(Debug|Release|RelWithDebInfo|MinSizeRel|Coverage)$" VALID_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
    if (VALID_BUILD_TYPE STREQUAL "")
        logFatalError("Invalid CMAKE_BUILD_TYPE: '${CMAKE_BUILD_TYPE}'")
    endif ()
endif ()

# ENABLE_STATIC_ANALYSIS default is AUTO
string(TOUPPER "${ENABLE_STATIC_ANALYSIS}" ENABLE_STATIC_ANALYSIS)
if (NOT DEFINED ENABLE_STATIC_ANALYSIS OR ENABLE_STATIC_ANALYSIS STREQUAL "")
    set(ENABLE_STATIC_ANALYSIS "AUTO" CACHE STRING
            "Enable static analysis; choose between ON, OFF and AUTO (if available and Debug mode)"
            FORCE)
else ()
    string(REGEX MATCH "^(ON|OFF|AUTO)$" VALID_STATIC_ANALYSIS "${ENABLE_STATIC_ANALYSIS}")
    if (VALID_STATIC_ANALYSIS STREQUAL "") # /!\ IF(VALID_STATIC_ANALYSIS) is false when ENABLE_STATIC_ANALYSIS is OFF
        logFatalError("Invalid ENABLE_STATIC_ANALYSIS option '${ENABLE_STATIC_ANALYSIS}'; choose between ON, OFF and AUTO.")
    endif ()
endif ()

# ENABLE_OCTAVE_BINDING default is AUTO
string(TOUPPER "${ENABLE_OCTAVE_BINDING}" ENABLE_OCTAVE_BINDING)
if (NOT DEFINED ENABLE_OCTAVE_BINDING OR ENABLE_OCTAVE_BINDING STREQUAL "")
    set(ENABLE_OCTAVE_BINDING "AUTO" CACHE STRING
            "Enable static analysis; choose between ON, OFF and AUTO (if available and Debug mode)"
            FORCE)
else ()
    string(REGEX MATCH "^(ON|OFF|AUTO)$" VALID_OCTAVE_BINDING "${ENABLE_OCTAVE_BINDING}")
    if (VALID_OCTAVE_BINDING STREQUAL "") # /!\ IF(VALID_OCTAVE_BINDING) is false when ENABLE_OCTAVE_BINDING is OFF
        logFatalError("Invalid ENABLE_OCTAVE_BINDING option '${ENABLE_OCTAVE_BINDING}'; choose between ON, OFF and AUTO.")
    endif ()
endif ()

#------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)

#------------------------------------------------------

# Generate position independant (aka -fPIC) code even for static libs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#------------------------------------------------------

option(EXTRA_SYSTEM_LIBRARY_PATH "Path to add to default system library path for finding libs")
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH ${EXTRA_SYSTEM_LIBRARY_PATH})

# Dependencies
set(DETECT_HDF5 false CACHE BOOL "Disable HDF5 to avoid conflict with armadillo") # prevent issue #40
add_subdirectory(dependencies/armadillo-code)
include_directories(SYSTEM dependencies/armadillo-code/include)
include_directories(SYSTEM dependencies/optim/header_only_version)

#------------------------------------------------------

# Default behaviour is to build shared library
# Change its behaviour using -DBUILD_SHARED_LIBS:BOOL=OFF
if (NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS on
            CACHE BOOL "Global flag to cause add_library to create shared libraries if on"
            FORCE)
endif ()
if (BUILD_SHARED_LIBS)
    message(STATUS "Build shared library")
else ()
    message(STATUS "Build static library")
endif ()

#------------------------------------------------------

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#------------------------------------------------------

# Change RelWithDebInfo to compile assertions
SET("CMAKE_CXX_FLAGS_RELWITHDEBINFO"
        "-g -O2"
        CACHE STRING "Flags used by the compiler during release builds with debug info and assertions"
        FORCE)
SET("CMAKE_C_FLAGS_RELWITHDEBINFO"
        "-g -O2"
        CACHE STRING "Flags used by the compiler during release builds with debug info and assertions"
        FORCE)

# Add new build types
# --coverage option is used to compile and link code instrumented for coverage analysis.
# The option is a synonym for -fprofile-arcs -ftest-coverage (when compiling) and -lgcov (when linking).
set(CMAKE_CXX_FLAGS_COVERAGE
        "-g -O0 --coverage"
        CACHE STRING "Flags used by the C++ compiler during coverage builds."
        FORCE)
set(CMAKE_C_FLAGS_COVERAGE
        "-g -O0 --coverage"
        CACHE STRING "Flags used by the C compiler during coverage builds."
        FORCE)
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE
        "--coverage"
        CACHE STRING "Flags used for linking binaries during coverage builds."
        FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
        "--coverage"
        CACHE STRING "Flags used by the shared libraries linker during coverage builds."
        FORCE)
mark_as_advanced(
        CMAKE_CXX_FLAGS_COVERAGE
        CMAKE_C_FLAGS_COVERAGE
        CMAKE_EXE_LINKER_FLAGS_COVERAGE
        CMAKE_SHARED_LINKER_FLAGS_COVERAGE)

#------------------------------------------------------

# Required since RcppArmadillo uses it by defaut
# /D or -D definition headers are updated according to the compiler 'style'
# Use windows style for easy checking on Unix
add_definitions(/DARMA_32BIT_WORD)

#------------------------------------------------------

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wall -Wextra")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wall -Wextra")
    set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wsign-compare -Wunused -Wunused-member-function -Wunused-private-field")
endif ()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIBKRIGING_CXX_FLAGS}")

#------------------------------------------------------

# search for clang-tidy (while be used while adding library or executable)
find_program(CLANG_TIDY clang-tidy)
if (CLANG_TIDY)
    if (ENABLE_STATIC_ANALYSIS STREQUAL "ON")
        set(CXX_CLANG_TIDY ${CLANG_TIDY})
        message(STATUS "Static analysis requested and enabled while compiling.")
    elseif (ENABLE_STATIC_ANALYSIS STREQUAL "OFF")
        message(STATUS "Static analysis available but disabled as requested.")
    elseif (ENABLE_STATIC_ANALYSIS STREQUAL "AUTO")
        if (CMAKE_BUILD_TYPE MATCHES Debug)
            set(CXX_CLANG_TIDY ${CLANG_TIDY})
            message(STATUS "Static analysis using clang-tidy is enabled while compiling.")
        else ()
            message(STATUS "Static analysis available but disabled in ${CMAKE_BUILD_TYPE} mode.")
        endif ()
    else ()
        logFatalError("INTERNAL ERROR: value '${ENABLE_STATIC_ANALYSIS}' not managed")
    endif ()
else ()
    if (ENABLE_STATIC_ANALYSIS STREQUAL "ON")
        logFatalError("Static analysis requested but not available.")
    elseif (ENABLE_STATIC_ANALYSIS STREQUAL "OFF" OR ENABLE_STATIC_ANALYSIS STREQUAL "AUTO")
        message(STATUS "Static analysis not available.")
    else ()
        logFatalError("INTERNAL ERROR: value '${ENABLE_STATIC_ANALYSIS}' not managed")
    endif ()
endif ()

#------------------------------------------------------

# search for octave-config (for building Octave's mex library)
find_program(OCTAVE_CONFIG_EXECUTABLE
        NAMES octave-config
        )
if (OCTAVE_CONFIG_EXECUTABLE)
    if (ENABLE_OCTAVE_BINDING STREQUAL "ON")
        if (BUILD_SHARED_LIBS)
            logFatalError("Octave binding requires static library mode")
        else ()
            add_subdirectory(bindings/Octave)
            message(STATUS "Octave binding enabled")
        endif ()
    elseif (ENABLE_OCTAVE_BINDING STREQUAL "OFF")
        message(STATUS "Octave binding available but disabled as requested.")
    elseif (ENABLE_OCTAVE_BINDING STREQUAL "AUTO")
        if (BUILD_SHARED_LIBS)
            message(STATUS "Octave binding disabled since it requires static library mode")
        else ()
            add_subdirectory(bindings/Octave)
            message(STATUS "Octave binding enabled")
        endif ()
    else ()
        logFatalError("INTERNAL ERROR: value '${ENABLE_OCTAVE_BINDING}' not managed")
    endif ()
else ()
    if (ENABLE_OCTAVE_BINDING STREQUAL "ON")
        logFatalError("Octave binding requested but not available.")
    elseif (ENABLE_OCTAVE_BINDING STREQUAL "OFF" OR ENABLE_OCTAVE_BINDING STREQUAL "AUTO")
        message(STATUS "Octave binding not available.")
    else ()
        logFatalError("INTERNAL ERROR: value '${ENABLE_OCTAVE_BINDING}' not managed")
    endif ()
endif ()

#------------------------------------------------------

add_subdirectory(src)

#------------------------------------------------------

set(CATCH_MODULE_PATH "${LIBKRIGING_SOURCE_DIR}/dependencies/Catch2")
include("${CATCH_MODULE_PATH}/contrib/ParseAndAddCatchTests.cmake")
add_subdirectory("${CATCH_MODULE_PATH}")

enable_testing()
add_subdirectory(tests)

add_custom_target(run_unit_tests
        COMMAND ${CMAKE_CTEST_COMMAND} -j ${PROCESSOR_COUNT}
        COMMENT "Executing unit tests."
        DEPENDS all_test_binaries
        )

#------------------------------------------------------

# unit tests coverage

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Coverage")
    find_program(LCOV lcov)
    if (NOT LCOV)
        logFatalError("lcov not found, cannot perform coverage.")
    endif ()

    # coveralls.io does not support striped paths
    #find_program (SED NAMES sed)
    #if (NOT SED)
    #    logFatalError("Unable to find sed")
    #else()
    #    # message(STATUS "sed found at ${SED}")
    #endif (NOT SED)

    # Don't forget '' around each pattern
    set(LCOV_EXCLUDE_PATTERN "'${LIBKRIGING_SOURCE_DIR}/dependencies/*'")

    add_custom_target(coverage
            # Cleanup previously generated profiling data
            COMMAND ${LCOV} --base-directory ${LIBKRIGING_SOURCE_DIR} --directory ${LIBKRIGING_BINARY_DIR} --zerocounters
            # Initialize profiling data with zero coverage for every instrumented line of the project
            # This way the percentage of total lines covered will always be correct, even when not all source code files were loaded during the test(s)
            COMMAND ${LCOV} --base-directory ${LIBKRIGING_SOURCE_DIR} --directory ${LIBKRIGING_BINARY_DIR} --capture --initial --output-file coverage_base.info
            # Run tests
            COMMAND ${CMAKE_CTEST_COMMAND} -j ${PROCESSOR_COUNT}
            # Collect data from executions
            COMMAND ${LCOV} --base-directory ${LIBKRIGING_SOURCE_DIR} --directory ${LIBKRIGING_BINARY_DIR} --capture --output-file coverage_ctest.info
            # Combine base and ctest results
            COMMAND ${LCOV} --add-tracefile coverage_base.info --add-tracefile coverage_ctest.info --output-file coverage_full.info
            # Extract only project data (--no-capture or --remove options may be used to select collected data)
            COMMAND ${LCOV} --remove coverage_full.info ${LCOV_EXCLUDE_PATTERN} --output-file coverage_filtered.info
            COMMAND ${LCOV} --extract coverage_filtered.info '${LIBKRIGING_SOURCE_DIR}/*' --output-file coverage.info
            # coveralls.io does not support striped paths
            #COMMAND ${SED} -i.bak 's|SF:${LIBKRIGING_SOURCE_DIR}/|SF:|g' coverage.info
            DEPENDS all_test_binaries
            COMMENT "Running test coverage."
            WORKING_DIRECTORY "${LIBKRIGING_BINARY_DIR}"
            )

    find_program(GENHTML genhtml)
    if (NOT GENHTML)
        message(WARNING "genhtml not found, cannot perform report-coverage.")
    else ()
        add_custom_target(coverage-report
                COMMAND ${CMAKE_COMMAND} -E remove_directory "${LIBKRIGING_BINARY_DIR}/coverage"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBKRIGING_BINARY_DIR}/coverage"
                COMMAND ${GENHTML} -o coverage -t "${CMAKE_PROJECT_NAME} test coverage" --ignore-errors source --legend --num-spaces 4 coverage.info
                COMMAND ${LCOV} --list coverage.info
                DEPENDS coverage
                COMMENT "Building coverage html report."
                WORKING_DIRECTORY "${LIBKRIGING_BINARY_DIR}"
                )
    endif ()
else ()
    add_custom_target(coverage
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMAND ${CMAKE_COMMAND} -E echo "*** Use CMAKE_BUILD_TYPE=Coverage option in cmake configuration to enable code coverage ***"
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMENT "Inform about not available code coverage."
            )
    add_custom_target(coverage-report DEPENDS coverage)
endif ()

#------------------------------------------------------

# search for clang-format and add target
find_program(CLANG_FORMAT clang-format)
if (CLANG_FORMAT)
    add_custom_target(clang-format
            COMMAND echo "running ${CLANG_FORMAT} ..."
            COMMAND ${CMAKE_COMMAND}
            -DLIBKRIGING_SOURCE_DIR="${LIBKRIGING_SOURCE_DIR}"
            -DCLANG_FORMAT="${CLANG_FORMAT}"
            -P ${LIBKRIGING_SOURCE_DIR}/cmake/ClangFormatProcess.cmake)
    message(STATUS "clang-format target for updating code format is available")
else ()
    message(WARNING "clang-format no found; clang-format target is not available.")
    add_custom_target(clang-format
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMAND ${CMAKE_COMMAND} -E echo "*** code formatting not available since clang-format has not been found ***"
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMENT "Inform about not available code format."
            )
endif ()

#------------------------------------------------------

# search for doxygen and add target
# Require dot, treat the other components as optional
# TODO: try https://codedocs.xyz for online docs
find_package(Doxygen
        QUIET
        COMPONENTS dot
        OPTIONAL_COMPONENTS mscgen dia)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${LIBKRIGING_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(doc # add ALL if build together with the code
            COMMAND echo "running doc generation ..."
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
    message(STATUS "doc target for documentation generation is available")
else ()
    message(WARNING "doxygen no found; doc target is not available.")
    add_custom_target(doc
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMAND ${CMAKE_COMMAND} -E echo "*** doc generation not available since doxygen has not been found ***"
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMENT "Inform about not available doc generation."
            )
endif ()

#------------------------------------------------------

# Custom install.lib target which does not compile tests
ADD_CUSTOM_TARGET(install.lib
        ${CMAKE_COMMAND}
        -DBUILD_TYPE=${CMAKE_BUILD_TYPE}
        -P ${CMAKE_BINARY_DIR}/cmake_install.cmake)
ADD_DEPENDENCIES(install.lib Kriging armadillo)
