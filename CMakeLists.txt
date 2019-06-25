cmake_minimum_required(VERSION 3.11)

# CMake utils
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Forbids in-source builds (placed before PROJECT keyword)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# VERSION must follow pattern : <major>.<minor>.<patch>
set(KRIGING_VERSION_MAJOR 0)
set(KRIGING_VERSION_MINOR 0)
set(KRIGING_VERSION_PATCH 0)
set(KRIGING_VERSION "${KRIGING_VERSION_MAJOR}.${KRIGING_VERSION_MINOR}.${KRIGING_VERSION_PATCH}")

project(libKriging
        VERSION ${KRIGING_VERSION}
        DESCRIPTION "LibKriging")
# PORJECT_VERSION now contains also KRIGING_VERSION

#------------------------------------------------------

# Prevent from root system installation
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/installed" CACHE PATH "default install path" FORCE )
endif()

#------------------------------------------------------

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#------------------------------------------------------

set(CMAKE_CXX_STANDARD 11)

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

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wall -Wextra")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wall -Wextra")
    set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wsign-compare -Wunused -Wunused-member-function -Wunused-private-field")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIBKRIGING_CXX_FLAGS}")

#------------------------------------------------------

add_subdirectory(src)

#------------------------------------------------------

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
            # TODO verify what is the better input coverage_full or coverage_ctest
            #COMMAND ${LCOV} --extract coverage_full.info '${LIBKRIGING_SOURCE_DIR}/*' --output-file coverage.info
            COMMAND ${LCOV} --extract coverage_ctest.info '${LIBKRIGING_SOURCE_DIR}/*' --output-file coverage.info
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

endif()

# Previous implementation using gcovr
if("${CMAKE_BUILD_TYPE}" STREQUAL "Coverage-Alt")
    # https://www.gcovr.com/en/stable/
    find_program(GCOVR gcovr)
    if(NOT GCOVR)
        message(FATAL_ERROR "gcovr not found, cannot perform coverage.")
    endif()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        string(REPLACE "clang++" "llvm-cov" LLVM_COV "${CMAKE_CXX_COMPILER}")
        if(NOT EXISTS "${LLVM_COV}")
            message(FATAL_ERROR "could not find ${LLVM_COV}, cannot perform coverage (using g++ for coverage is recommended).")
        endif()

        set(GCOVR_EXTRA "--gcov-executable=${LLVM_COV} gcov")
    endif()

    # set (GCOVR_EXCLUDE
    #        -e "${LIBKRIGING_SOURCE_DIR}/src/main.cpp"
    #     )

    set(GCOVR_OPTIONS --object-directory="${LIBKRIGING_BINARY_DIR}" -r "${LIBKRIGING_SOURCE_DIR}/src" ${GCOVR_EXCLUDE} ${GCOVR_EXTRA})

    add_custom_target(coverage
            COMMAND ${GCOVR} ${GCOVR_OPTIONS} --exclude-unreachable-branches --sort-percentage
            DEPENDS run_unit_tests
            COMMENT "Running test coverage."
            )

    add_custom_target(coverage-report
            COMMAND ${CMAKE_COMMAND} -E remove_directory "${LIBKRIGING_BINARY_DIR}/coverage"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBKRIGING_BINARY_DIR}/coverage"
            COMMAND ${GCOVR} ${GCOVR_OPTIONS} --delete --html --html-details -o "${LIBKRIGING_BINARY_DIR}/coverage/index.html"
            DEPENDS coverage
            COMMENT "Building coverage html report."
            WORKING_DIRECTORY "${LIBKRIGING_BINARY_DIR}"
            )

endif()