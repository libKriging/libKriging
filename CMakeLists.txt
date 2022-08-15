cmake_minimum_required(VERSION 3.10)

# CMake utils
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

## Forbids in-source builds (placed before PROJECT keyword)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
# Partially disabled since Armadillo do not preserve its source directory when
# this is due to line: configure_file(${PROJECT_SOURCE_DIR}/examples/Makefile.cmake ${PROJECT_SOURCE_DIR}/examples/Makefile)
# set(CMAKE_DISABLE_SOURCE_CHANGES ON)

include(cmake/version.cmake)
include(cmake/configuration.cmake)

include(GetGitRevisionDescription)
git_describe(BUILD_TAG --tags --dirty=-d)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/lib/version.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)
set(VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/version.cpp")

#------------------------------------------------------

project(libKriging
        VERSION ${KRIGING_VERSION}
        DESCRIPTION "LibKriging")
# PROJECT_VERSION now contains also KRIGING_VERSION

#------------------------------------------------------

enable_language(CXX)

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

#------------------------------------------------------

#option(BUILD_SHARED_LIBS "Build the shared library" ON) # BUILD_SHARED_LIBS config should be after armadillo config
#option(BUILD_STATIC_LIBS "Build the static library" ON) # TODO not yet implemented; today, equivalent to BUILD_SHARED_LIBS=off 
option(LIBKRIGING_BENCHMARK_TESTS "Enable benchmark tests while testing" OFF)
option(ENABLE_COVERAGE "Enable coverage target" OFF)
option(ENABLE_MEMCHECK "Enable memcheck on tests" OFF)
option(SANITIZE "Enable sanitize feature" OFF) # could be THREAD, ADDRESS or LEAK
option(LBFGSB_SHOW_BUILD OFF)

set(PYBIND11_PYTHON_VERSION 3.6)

#------------------------------------------------------
# Check options

# Default build type is RelWIthDebInfo
if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
            "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel"
            FORCE)
else ()
    string(REGEX MATCH "^(Debug|Release|RelWithDebInfo|MinSizeRel)$" VALID_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
    if (VALID_BUILD_TYPE STREQUAL "")
        logFatalError("Invalid CMAKE_BUILD_TYPE: '${CMAKE_BUILD_TYPE}'")
    endif ()
endif ()

# ENABLE_STATIC_ANALYSIS default is OFF
string(TOUPPER "${ENABLE_STATIC_ANALYSIS}" ENABLE_STATIC_ANALYSIS)
if (NOT DEFINED ENABLE_STATIC_ANALYSIS OR ENABLE_STATIC_ANALYSIS STREQUAL "")
    set(ENABLE_STATIC_ANALYSIS "OFF" CACHE STRING
            "Enable static analysis; choose between ON, OFF and AUTO (if available and Debug mode)"
            FORCE)
else ()
    string(REGEX MATCH "^(ON|OFF|AUTO)$" VALID_STATIC_ANALYSIS "${ENABLE_STATIC_ANALYSIS}")
    if (VALID_STATIC_ANALYSIS STREQUAL "") # /!\ IF(VALID_STATIC_ANALYSIS) is false when ENABLE_STATIC_ANALYSIS is OFF
        logFatalError("Invalid ENABLE_STATIC_ANALYSIS option '${ENABLE_STATIC_ANALYSIS}'; choose between ON, OFF and AUTO.")
    endif ()
endif ()

# ENABLE_MATLAB_BINDING default is AUTO
string(TOUPPER "${ENABLE_MATLAB_BINDING}" ENABLE_MATLAB_BINDING)
if (NOT DEFINED ENABLE_MATLAB_BINDING OR ENABLE_MATLAB_BINDING STREQUAL "")
    set(ENABLE_MATLAB_BINDING "AUTO" CACHE STRING
            "Enable static analysis; choose between ON, OFF and AUTO (if available and Debug mode)"
            FORCE)
else ()
    string(REGEX MATCH "^(ON|OFF|AUTO)$" VALID_MATLAB_BINDING "${ENABLE_MATLAB_BINDING}")
    if (VALID_MATLAB_BINDING STREQUAL "") # /!\ IF(VALID_MATLAB_BINDING) is false when ENABLE_MATLAB_BINDING is OFF
        logFatalError("Invalid ENABLE_MATLAB_BINDING option '${ENABLE_MATLAB_BINDING}'; choose between ON, OFF and AUTO.")
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

if (ENABLE_OCTAVE_BINDING STREQUAL "ON" AND ENABLE_MATLAB_BINDING STREQUAL "ON")
    logFatalError("You cannot request both Octave and Matlab bindings as required")
endif()

# ENABLE_PYTHON_BINDING default is AUTO
string(TOUPPER "${ENABLE_PYTHON_BINDING}" ENABLE_PYTHON_BINDING)
if (NOT DEFINED ENABLE_PYTHON_BINDING OR ENABLE_PYTHON_BINDING STREQUAL "")
    set(ENABLE_PYTHON_BINDING "AUTO" CACHE STRING
            "Enable static analysis; choose between ON, OFF and AUTO (if available and Debug mode)"
            FORCE)
else ()
    string(REGEX MATCH "^(ON|OFF|AUTO)$" VALID_PYTHON_BINDING "${ENABLE_PYTHON_BINDING}")
    if (VALID_PYTHON_BINDING STREQUAL "") # /!\ IF(VALID_PYTHON_BINDING) is false when ENABLE_PYTHON_BINDING is OFF
        logFatalError("Invalid ENABLE_PYTHON_BINDING option '${ENABLE_PYTHON_BINDING}'; choose between ON, OFF and AUTO.")
    endif ()
endif ()

#------------------------------------------------------

if (USE_COMPILER_CACHE)
    # search for requested compiler cache (could be: ccache, sccache...)
    find_program(COMPILER_CACHE_EXECUTABLE ${USE_COMPILER_CACHE})
    if (COMPILER_CACHE_EXECUTABLE)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${COMPILER_CACHE_EXECUTABLE}")
        message(STATUS "Using ${COMPILER_CACHE_EXECUTABLE} as compiler cache for accelerated compilation")
    else ()
        logFatalError("Requested compiler cache '${USE_COMPILER_CACHE}' is not available")
    endif ()
endif ()

#------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)

#------------------------------------------------------

# Can be locally overridden using set_target_properties + https://cmake.org/cmake/help/latest/prop_tgt/LANG_VISIBILITY_PRESET.html
# set(CMAKE_CXX_VISIBILITY_PRESET hidden) # global setting is incompatible with armadillo => moved to Kriging target
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW) # Honor the visibility properties for all target types

#------------------------------------------------------

# Generate position independent (aka -fPIC) code even for static libs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#------------------------------------------------------

option(EXTRA_SYSTEM_LIBRARY_PATH "Path to add to default system library path for finding libs")
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH ${EXTRA_SYSTEM_LIBRARY_PATH})

# Dependencies
set(DETECT_HDF5 false CACHE BOOL "Disable HDF5 to avoid conflict with armadillo") # prevent issue #40
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW) # disable warning about missing version in project command
add_subdirectory(dependencies/armadillo-code)
include_directories(SYSTEM dependencies/armadillo-code/include)

add_library(blas ALIAS armadillo) # to use armadillo blas inside lbfgsb_cpp

if (NOT CMAKE_Fortran_COMPILER)
    set(CMAKE_Fortran_COMPILER gfortran)
endif()
if (NOT Fortran_LINK_FLAGS)
    set(Fortran_LINK_FLAGS "-lgfortran -lquadmath -lm")
endif()
set(ignoreUnusedVariable ${LBFGSB_SHOW_BUILD})
message(STATUS "Compiling lbfgsb_cpp as an external library")

add_subdirectory(dependencies/lbfgsb_cpp)
#set(lbfgsb_INSTALL_DIR "${CMAKE_BINARY_DIR}/lbfgsb-installed")
#set(lbfgsb_DIR "${lbfgsb_INSTALL_DIR}/lib/cmake/lbfgsb")
#find_package(lbfgsb CONFIG REQUIRED)
message(STATUS "Compilation of lbfgsb_cpp done")

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

if (ENABLE_COVERAGE)
    # --coverage option is used to compile and link code instrumented for coverage analysis.
    # The option is a synonym for -fprofile-arcs -ftest-coverage (when compiling) and -lgcov (when linking).
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g --coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()

#------------------------------------------------------

# Required since RcppArmadillo uses it by default
# /D or -D definition headers are updated according to the compiler 'style'
# Use windows style for easy checking on Unix
add_definitions(/DARMA_32BIT_WORD)
add_definitions(-DARMA_ALIEN_MEM_ALLOC_FUNCTION=lkalloc::malloc)
add_definitions(-DARMA_ALIEN_MEM_FREE_FUNCTION=lkalloc::free)
add_definitions(-DCARMA_DO_NOT_EXPORT_ALIEN_MEM_FUNCTIONS)
#add_definitions(-DCARMA_DEBUG)
#add_definitions(-DCARMA_EXTRA_DEBUG)

#------------------------------------------------------

string(TOUPPER "${SANITIZE}" SANITIZE)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wall -Wextra")
    if (WIN32 AND MINGW) # https://stackoverflow.com/questions/16596876/object-file-has-too-many-sections
        set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wa,-mbig-obj")
    endif()
    if (SANITIZE STREQUAL "THREAD")
        add_compile_options(-fsanitize=thread)
        add_link_options(-fsanitize=thread)
    elseif (SANITIZE STREQUAL "ADDRESS")
        add_compile_options(-fsanitize=address)
        add_compile_options(-fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    elseif (SANITIZE STREQUAL "LEAK")
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
            message(FATAL_ERROR "Leak Sanitizer is not available on ARM")
        endif()
        add_compile_options(-fsanitize=leak)
        add_compile_options(-fno-omit-frame-pointer)
        add_link_options(-fsanitize=leak)
    elseif (NOT SANITIZE STREQUAL "OFF")
        message(FATAL_ERROR "Bad SANITIZE value; should be THREAD, ADDRESS, LEAK or OFF")
    endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wall -Wextra")
    set(LIBKRIGING_CXX_FLAGS "${LIBKRIGING_CXX_FLAGS} -Wsign-compare -Wunused -Wunused-member-function -Wunused-private-field")
    if (SANITIZE STREQUAL "THREAD")
        add_compile_options(-fsanitize=thread)
        add_link_options(-fsanitize=thread)
    elseif (SANITIZE STREQUAL "ADDRESS")
        add_compile_options(-fsanitize=address)
        add_compile_options(-fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    elseif (SANITIZE STREQUAL "LEAK")
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
            message(FATAL_ERROR "Leak Sanitizer is not available on ARM")
        endif()
        add_compile_options(-fsanitize=leak)
        add_compile_options(-fno-omit-frame-pointer)
        add_link_options(-fsanitize=leak)
    elseif (NOT SANITIZE STREQUAL "OFF")
        message(FATAL_ERROR "Bad SANITIZE value; should be THREAD, ADDRESS, LEAK or OFF")
    endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # workaround like https://github.com/nlohmann/json/issues/1408
    # to avoid error like: carma\third_party\armadillo-code\include\armadillo_bits/arma_str.hpp(194): error C2039: '_snprintf': is not a member of 'std' (compiling source file carma\tests\src\bindings.cpp) 
    ADD_DEFINITIONS(-DHAVE_SNPRINTF)
    if (ENABLE_COVERAGE)
        logFatalError("Coverage is not supported with MSVC")
    endif()
    if (NOT SANITIZE STREQUAL "OFF")
        message(FATAL_ERROR "SANITIZE feature not support with ${CMAKE_CXX_COMPILER_ID}")
    endif ()
else()
    if (NOT SANITIZE STREQUAL "OFF")
        message(FATAL_ERROR "SANITIZE feature not support with ${CMAKE_CXX_COMPILER_ID}")
    endif ()
endif()

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

# valgrind must be checked before adding new subdirectories
# before include(CTest)
if (ENABLE_MEMCHECK)
    find_program(VALGRIND_EXECUTABLE valgrind)
    if (NOT VALGRIND_EXECUTABLE)
        MESSAGE(FATAL_ERROR "ENABLE_MEMCHECK is requested but valgrind executable cannot be found")
    endif()
    # https://valgrind.org/docs/manual/mc-manual.html#mc-manual.leaks
    set(MEMORYCHECK_COMMAND_OPTIONS "--tool=memcheck --leak-check=yes --num-callers=50 --error-exitcode=1 --errors-for-leak-kinds=definite")
    set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --show-reachable=yes ") # could be noisy
    set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} -v") # verbose run with details
    # set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} -q") # for quiet runs
    # set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --gen-suppressions=all --num-callers=8") # for setting up suppressions
    set(MEMORYCHECK_SUPPRESSIONS_FILE "${PROJECT_SOURCE_DIR}/.valgrind-suppressions")
    MESSAGE(STATUS "Memcheck on tests enabled")
endif()

#------------------------------------------------------

# search for octave-config (for building Octave's mex library)
find_program(OCTAVE_CONFIG_EXECUTABLE
             NAMES octave-config
        )
if (OCTAVE_CONFIG_EXECUTABLE)
    if (ENABLE_OCTAVE_BINDING STREQUAL "OFF")
        message(STATUS "Octave binding available but disabled as requested.")
    else()
        message(STATUS "Octave binding available.")
    endif()
else()
    message(STATUS "Octave binding not available.")
endif()

# set(Matlab_ROOT_DIR) # should be defined to help finder
find_package(Matlab COMPONENTS MAIN_PROGRAM)
if (Matlab_FOUND)
    if(ENABLE_MATLAB_BINDING STREQUAL "OFF")
        message(STATUS "Matlab binding available but disabled as requested.")
    else()
        message(STATUS "Matlab binding available.")
    endif()
else()
    message(STATUS "Matlab binding not available.")
endif()

set(OCTAVE_BINDING_MODE)
if (ENABLE_OCTAVE_BINDING STREQUAL "ON")
    if (ENABLE_MATLAB_BINDING STREQUAL "ON")
        logFatalError("INTERNAL ERROR: cannot mix ENABLE_OCTAVE_BINDING=on and ENABLE_MATLAB_BINDING=on")
    else()
        if (Matlab_FOUND AND ENABLE_MATLAB_BINDING STREQUAL "AUTO")
            message(STATUS "Matlab binding disabled since Octave is required.")
        endif()
    endif()
    if (OCTAVE_CONFIG_EXECUTABLE)
        if (BUILD_SHARED_LIBS)
            logFatalError("Octave binding requires static library mode (see BUILD_SHARED_LIBS=OFF)")
        else ()
            set(OCTAVE_BINDING_MODE Octave)
            add_subdirectory(bindings/Octave)
            message(STATUS "Octave binding enabled")
        endif ()
    else()
        logFatalError("Octave binding requested but not available.")
    endif()
elseif (ENABLE_MATLAB_BINDING STREQUAL "ON")
    if (ENABLE_OCTAVE_BINDING STREQUAL "ON")
        logFatalError("INTERNAL ERROR: cannot mix ENABLE_OCTAVE_BINDING=on and ENABLE_MATLAB_BINDING=on")
    else()
        if (OCTAVE_CONFIG_EXECUTABLE AND ENABLE_OCTAVE_BINDING STREQUAL "AUTO")
            message(STATUS "Octave binding disabled since Matlab is required.")
        endif()
    endif()
    if (Matlab_FOUND)
        if (BUILD_SHARED_LIBS)
            logFatalError("Matlab binding requires static library mode")
        else ()
            set(OCTAVE_BINDING_MODE Matlab)
            add_subdirectory(bindings/Octave)
            message(STATUS "Matlab binding enabled")
        endif ()
    else()
        logFatalError("Matlab binding requested but not available.")
    endif()
else()
    if (ENABLE_OCTAVE_BINDING STREQUAL "AUTO" AND BUILD_SHARED_LIBS)
        message(STATUS "Octave binding disabled since it requires static library mode")
        set(ENABLE_OCTAVE_BINDING "OFF")
    endif()
    if (ENABLE_MATLAB_BINDING STREQUAL "AUTO" AND BUILD_SHARED_LIBS)
        message(STATUS "Matlab binding disabled since it requires static library mode")
        set(ENABLE_MATLAB_BINDING "OFF")
    endif()
    
    if (ENABLE_MATLAB_BINDING STREQUAL "AUTO" AND 
        (ENABLE_OCTAVE_BINDING STREQUAL "OFF" OR NOT OCTAVE_CONFIG_EXECUTABLE))
        if (Matlab_FOUND)
            set(OCTAVE_BINDING_MODE Matlab)
            add_subdirectory(bindings/Octave)
            set(ENABLED_MATLAB_BINDING ON)
            message(STATUS "Matlab binding enabled")
        endif ()
    elseif(ENABLE_OCTAVE_BINDING STREQUAL "AUTO" AND
        (ENABLE_MATLAB_BINDING STREQUAL "OFF" OR NOT Matlab_FOUND))
        if (OCTAVE_CONFIG_EXECUTABLE)
            set(OCTAVE_BINDING_MODE Octave)
            add_subdirectory(bindings/Octave)
            set(ENABLED_OCTAVE_BINDING ON)
            message(STATUS "Octave binding enabled")
        endif()
    elseif(ENABLE_MATLAB_BINDING STREQUAL "AUTO" AND ENABLE_OCTAVE_BINDING STREQUAL "AUTO" 
            AND OCTAVE_CONFIG_EXECUTABLE AND Matlab_FOUND)
        logFatalError("Both Octave and Matlab are available and requested if available. You have to select explicitly the expected binding")
    endif()
endif()

if (ENABLE_MATLAB_BINDING STREQUAL "ON" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
#    message(FATAL_ERROR "MKL_FOUND=${MKL_FOUND}")
    add_definitions(-DARMA_BLAS_LONG)
endif()

#------------------------------------------------------

# search for python-config (for building Python module)

if (PYTHON_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH_SAVED ${CMAKE_PREFIX_PATH})
    list(APPEND CMAKE_PREFIX_PATH ${PYTHON_PREFIX_PATH})
endif()
find_package(PythonInterp ${PYBIND11_PYTHON_VERSION})
if (CMAKE_PREFIX_PATH_SAVED)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH_SAVED})
    unset(CMAKE_PREFIX_PATH_SAVED)
endif()
set(HAS_PYTHON_REQUIREMENT false)
if (PythonInterp_FOUND)
    message(STATUS "Checking Python requirements")
    set(HAS_PYTHON_REQUIREMENT true)
    set(PYTHON_REQUIREMENT_INFO "\tpython3 interpreter found: ${PYTHON_EXECUTABLE}")
    set(INDENTATION "   ")
    execute_process(COMMAND ${PYTHON_EXECUTABLE} ${LIBKRIGING_SOURCE_DIR}/bindings/Python/check_requirements.py --pretty --indent=${INDENTATION} requirements.txt dev-requirements.txt
                    RESULT_VARIABLE PYTHON_IMPORT_RETCODE)
    if (NOT PYTHON_IMPORT_RETCODE EQUAL "0")
        set(HAS_PYTHON_REQUIREMENT false)
    endif()
else()
    set(PYTHON_REQUIREMENT_INFO "\tpython3 interpreter not found")
endif()

if (HAS_PYTHON_REQUIREMENT)
    if (ENABLE_PYTHON_BINDING STREQUAL "ON" OR ENABLE_PYTHON_BINDING STREQUAL "AUTO")
        # force pybind11 to use the right python interpreter, it could be useful 
        # to define CMAKE_PREFIX_PATH with PYTHON_PREFIX_PATH (cf carma CMake config)
        # pybind11 is loading manually to be always available even if conditionally not loaded by carma
        if (PYTHON_PREFIX_PATH)
            set(CMAKE_PREFIX_PATH_SAVED ${CMAKE_PREFIX_PATH})
            list(APPEND CMAKE_PREFIX_PATH ${PYTHON_PREFIX_PATH})
        endif()
        add_subdirectory(${LIBKRIGING_SOURCE_DIR}/dependencies/pybind11 pybind11)
        if (CMAKE_PREFIX_PATH_SAVED)
            set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH_SAVED})
            unset(CMAKE_PREFIX_PATH_SAVED)
        endif()
        add_subdirectory(bindings/Python)
        set(ENABLED_PYTHON_BINDING ON)
        message(STATUS "Python binding enabled")
    elseif (ENABLE_PYTHON_BINDING STREQUAL "OFF")
        message(STATUS "PYTHON binding available but disabled as requested.")
    else ()
        logFatalError("INTERNAL ERROR: value '${ENABLE_PYTHON_BINDING}' not managed")
    endif ()
else ()
    message(STATUS "Python requirement failure:\n${PYTHON_REQUIREMENT_INFO}")
    if (ENABLE_PYTHON_BINDING STREQUAL "ON")
        logFatalError("Python binding requested but not available.")
    elseif (ENABLE_PYTHON_BINDING STREQUAL "OFF" OR ENABLE_PYTHON_BINDING STREQUAL "AUTO")
        message(STATUS "Python binding not available.")
    else ()
        logFatalError("INTERNAL ERROR: value '${ENABLE_PYTHON_BINDING}' not managed")
    endif ()
endif ()

#------------------------------------------------------

add_subdirectory(src)

#------------------------------------------------------

if(${CMAKE_VERSION} VERSION_EQUAL "3.18.0")
    # cf 
    # - issue https://github.com/libKriging/libKriging/issues/51
    # - comment https://gitlab.kitware.com/cmake/cmake/-/issues/21017#note_804926
    # - doc https://cmake.org/cmake/help/v3.18/release/3.18.html#id1
    logFatalError("CMake 3.18.0 contains an abnormal behavior; consider another version of CMake")
endif()
if (POLICY CMP0110)
    cmake_policy(SET CMP0110 NEW) # supports arbitrary characters in test names (requires in CMake 3.19)
endif()
set(CATCH_MODULE_PATH "${LIBKRIGING_SOURCE_DIR}/dependencies/Catch2")
add_subdirectory("${CATCH_MODULE_PATH}")
list(APPEND CMAKE_MODULE_PATH ${CATCH_MODULE_PATH}/contrib)

include(CTest)
add_subdirectory(tests)

add_custom_target(run_unit_tests
        COMMAND ${CMAKE_CTEST_COMMAND} -j ${PROCESSOR_COUNT}
        COMMENT "Executing unit tests."
        DEPENDS all_test_binaries
        )

#------------------------------------------------------

# unit tests coverage

if (ENABLE_COVERAGE)
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
    set(LCOV_EXCLUDE_PATTERN "'${LIBKRIGING_SOURCE_DIR}/dependencies/*'" "'${LIBKRIGING_BINARY_DIR}/dependencies/*'")

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
    exec_program(${CLANG_FORMAT} ARGS -version
            OUTPUT_VARIABLE CLANG_FORMAT_RAW_VERSION)
    string(REGEX MATCH "[1-9][0-9]*\\.[0-9]+\\.[0-9]+"
            CLANG_FORMAT_VERSION ${CLANG_FORMAT_RAW_VERSION})
    if (CLANG_FORMAT_VERSION VERSION_GREATER_EQUAL "9.0.0")
        add_custom_target(clang-format
                COMMAND echo "running ${CLANG_FORMAT} ..."
                COMMAND ${CMAKE_COMMAND}
                -DLIBKRIGING_SOURCE_DIR="${LIBKRIGING_SOURCE_DIR}"
                -DCLANG_FORMAT="${CLANG_FORMAT}"
                -P ${LIBKRIGING_SOURCE_DIR}/cmake/ClangFormatProcess.cmake)
        message(STATUS "clang-format target for updating code format is available")
    else()
        message(WARNING "incompatible clang-format found (<6.0.0); clang-format target is not available.")
        add_custom_target(clang-format
                COMMAND ${CMAKE_COMMAND} -E echo ""
                COMMAND ${CMAKE_COMMAND} -E echo "*** code formatting not available since clang-format version is incompatible ***"
                COMMAND ${CMAKE_COMMAND} -E echo ""
                COMMENT "Inform about not available code format."
                )
    endif()
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
if (ENABLED_PYTHON_BINDING)
    ADD_DEPENDENCIES(install.lib _pylibkriging)
endif()
if (ENABLED_OCTAVE_BINDING)
    ADD_DEPENDENCIES(install.lib mLibKriging)
endif()


