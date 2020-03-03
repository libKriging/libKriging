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
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/installed" CACHE PATH "default install path" FORCE)
    # Force update for sub-libraries (to follow current installation directive)
    set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE CACHE BOOL "Installation prefix has been set" FORCE)
endif()

#------------------------------------------------------

set(CMAKE_CXX_STANDARD 11)

#------------------------------------------------------

# Generate position independant (aka -fPIC) code even for static libs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#------------------------------------------------------

option(EXTRA_SYSTEM_LIBRARY_PATH "Path to add to default system library path for finding libs")
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH ${EXTRA_SYSTEM_LIBRARY_PATH})

# Dependencies
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
endif()
if (BUILD_SHARED_LIBS)
    message(STATUS "Build shared library")
else()
    message(STATUS "Build static library")
endif()

#------------------------------------------------------

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#------------------------------------------------------

set(LIBKRIGING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LIBKRIGING_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

#------------------------------------------------------

# Change RelWithDebInfo to compile assertions
SET("CMAKE_CXX_FLAGS_RELWITHDEBINFO"
        "-g -O2"
        CACHE STRING "Flags used by the compiler during release builds with debug info and assertions"
        FORCE )
SET("CMAKE_C_FLAGS_RELWITHDEBINFO"
        "-g -O2"
        CACHE STRING "Flags used by the compiler during release builds with debug info and assertions"
        FORCE )

# Add new build types
# --coverage option is used to compile and link code instrumented for coverage analysis.
# The option is a synonym for -fprofile-arcs -ftest-coverage (when compiling) and -lgcov (when linking).
set(CMAKE_CXX_FLAGS_COVERAGE
        "-g -O0 --coverage"
        CACHE STRING "Flags used by the C++ compiler during coverage builds."
        FORCE )
set(CMAKE_C_FLAGS_COVERAGE
        "-g -O0 --coverage"
        CACHE STRING "Flags used by the C compiler during coverage builds."
        FORCE )
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE
        "--coverage"
        CACHE STRING "Flags used for linking binaries during coverage builds."
        FORCE )
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
        "--coverage"
        CACHE STRING "Flags used by the shared libraries linker during coverage builds."
        FORCE )
mark_as_advanced(
        CMAKE_CXX_FLAGS_COVERAGE
        CMAKE_C_FLAGS_COVERAGE
        CMAKE_EXE_LINKER_FLAGS_COVERAGE
        CMAKE_SHARED_LINKER_FLAGS_COVERAGE )

if(CMAKE_BUILD_TYPE)
    string(REGEX MATCH "(Debug|Release|RelWithDebInfo|MinSizeRel|Coverage)" VALID_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
    if(NOT VALID_BUILD_TYPE)
        message(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE: '${CMAKE_BUILD_TYPE}'")
    endif()
endif()

# Default build type is RelWIthDebInfo
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
            "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel Coverage."
            FORCE)
endif()

#------------------------------------------------------

# Required since RcppArmadillo uses it by defaut
# /D or -D definition headers are updated according to the compiler 'style'
# Use windows style for easy checking on Unix
add_definitions(/DARMA_32BIT_WORD)

#------------------------------------------------------

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wall -Wextra")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wall -Wextra")
    set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wsign-compare -Wunused -Wunused-member-function -Wunused-private-field")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIBKRIGING_CXX_FLAGS}")

#------------------------------------------------------

# search for clang-tidy (while be used while adding library or executable)
find_program(CLANG_TIDY clang-tidy)
if (CLANG_TIDY)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set(CXX_CLANG_TIDY ${CLANG_TIDY})
        message(STATUS "Static analysis using clang-tidy is enabled while compiling.")
    else ()
        message(STATUS "Static analysis available but disabled in ${CMAKE_BUILD_TYPE} mode.")
    endif ()
else ()
    message(STATUS "Static analysis not available.")
endif ()

#------------------------------------------------------

add_subdirectory(src)

#------------------------------------------------------

set(CATCH_MODULE_PATH "${LIBKRIGING_SOURCE_DIR}/dependencies/Catch2")
include("${CATCH_MODULE_PATH}/contrib/ParseAndAddCatchTests.cmake")
add_subdirectory("${CATCH_MODULE_PATH}")

add_subdirectory(tests)
enable_testing()

add_custom_target(run_unit_tests
        COMMAND ${CMAKE_CTEST_COMMAND} -j ${PROCESSOR_COUNT}
        COMMENT "Executing unit tests."
        DEPENDS all_test_binaries
        )

#------------------------------------------------------

# unit tests coverage

if("${CMAKE_BUILD_TYPE}" STREQUAL "Coverage")
    find_program(LCOV lcov)
    if(NOT LCOV)
        message(FATAL_ERROR "lcov not found, cannot perform coverage.")
    endif()

    # coveralls.io does not support striped paths
    #find_program (SED NAMES sed)
    #if (NOT SED)
    #    message (FATAL_ERROR "Unable to find sed")
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
    if(NOT GENHTML)
        message(WARNING "genhtml not found, cannot perform report-coverage.")
    else()
        add_custom_target(coverage-report
                COMMAND ${CMAKE_COMMAND} -E remove_directory "${LIBKRIGING_BINARY_DIR}/coverage"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBKRIGING_BINARY_DIR}/coverage"
                COMMAND ${GENHTML} -o coverage -t "${CMAKE_PROJECT_NAME} test coverage" --ignore-errors source --legend --num-spaces 4 coverage.info
                COMMAND ${LCOV} --list coverage.info
                DEPENDS coverage
                COMMENT "Building coverage html report."
                WORKING_DIRECTORY "${LIBKRIGING_BINARY_DIR}"
                )
    endif()
else()
    add_custom_target(coverage
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMAND ${CMAKE_COMMAND} -E echo "*** Use CMAKE_BUILD_TYPE=Coverage option in cmake configuration to enable code coverage ***"
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMENT "Inform about not available code coverage."
            )
    add_custom_target(coverage-report DEPENDS coverage)
endif()

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
            COMMENT "Inform about not available code coverage."
            )
endif()

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
endif()

#------------------------------------------------------

# Custom install.lib target whuch does not compile tests
ADD_CUSTOM_TARGET(install.lib
        ${CMAKE_COMMAND}
        -DBUILD_TYPE=${CMAKE_BUILD_TYPE}
        -P ${CMAKE_BINARY_DIR}/cmake_install.cmake)
ADD_DEPENDENCIES(install.lib Kriging armadillo)
