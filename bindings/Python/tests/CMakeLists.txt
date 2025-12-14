
macro(add_python_test)
    set(options WILL_FAIL)
    set(oneValueArgs NAME FILENAME)
    # https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#test-properties
    set(multiValueArgs PROPERTIES)

    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (ARGS_UNPARSED_ARGUMENTS)
        logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
    endif ()

    if (NOT ARGS_NAME)
        logFatalError("add_mex_function needs NAME")
    endif ()

    if (NOT ARGS_FILENAME)
        logFatalError("add_mex_function needs NAME")
    endif ()

    set(TEST_NAME "Python/${ARGS_NAME}")

    if (ENABLE_MEMCHECK)
        SET(VALGRIND_LOG_FILE "${CMAKE_BINARY_DIR}/Testing/Temporary/PythonMemoryChecker.${ARGS_NAME}.log")
        add_test(NAME ${TEST_NAME}
                COMMAND ${VALGRIND_EXECUTABLE} --show-leak-kinds=definite --log-file=${VALGRIND_LOG_FILE} ${PYTHON_EXECUTABLE} -m pytest -vv --valgrind --valgrind-log=${VALGRIND_LOG_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/${ARGS_FILENAME})
        set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "PYTHONMALLOC=malloc")
    else ()
        add_test(NAME ${TEST_NAME}
                 COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/${ARGS_FILENAME})
    endif ()

    if (ARGS_WILL_FAIL)
        set_tests_properties(${TEST_NAME}
                PROPERTIES
                WILL_FAIL TRUE)
    endif ()
    
    if (WIN32)
        set(SEP "\;")
    else()
        set(SEP ":")
    endif()
    
    set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${LIBKRIGING_PYTHON_BINARY_DIR}${SEP}${CMAKE_CURRENT_SOURCE_DIR}/../src")

    # Fix "cannot allocate memory in static TLS block" error
    # Preload OpenMP library before loading the Python extension module
    # This ensures OpenMP's TLS is allocated in the initial TLS block
    if(NOT WIN32)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            # Search for libgomp.so.1 runtime library (not the linker stub libgomp.so)
            # Note: Must use NAMES_PER_DIR to check all names before checking next directory
            # This prevents finding the linker stub in GCC directories before the runtime library

            # Search standard runtime library locations with explicit .so.1 suffix
            find_file(GOMP_RUNTIME_LIB
                      NAMES libgomp.so.1
                      PATHS /usr/lib/x86_64-linux-gnu
                            /lib/x86_64-linux-gnu
                            /usr/lib64
                            /lib64
                            /usr/lib
                            /lib
                      NO_DEFAULT_PATH)

            # If not found, try GCC library directories (Ubuntu 22.04 may have it here)
            if(NOT GOMP_RUNTIME_LIB)
                find_file(GOMP_RUNTIME_LIB
                          NAMES libgomp.so.1
                          PATHS /usr/lib/gcc/x86_64-linux-gnu
                                /usr/lib/gcc
                          PATH_SUFFIXES 11 12 13 14
                          NO_DEFAULT_PATH)
            endif()

            # Last resort: system-wide search
            if(NOT GOMP_RUNTIME_LIB)
                find_file(GOMP_RUNTIME_LIB NAMES libgomp.so.1)
            endif()

            if(GOMP_RUNTIME_LIB)
                set(OPENMP_LIB_PATH ${GOMP_RUNTIME_LIB})
            else()
                # Very last resort: use just the library name and let LD find it
                set(OPENMP_LIB_PATH "libgomp.so.1")
            endif()

            set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "LD_PRELOAD=${OPENMP_LIB_PATH}")
            message(STATUS "Python test ${TEST_NAME} will preload OpenMP: ${OPENMP_LIB_PATH}")
        endif()
    endif()

    set_tests_properties(${TEST_NAME}
            PROPERTIES
            WORKING_DIRECTORY "${LIBKRIGING_PYTHON_BINARY_DIR}"
            LABELS Python
            ${ARGS_PROPERTIES})
endmacro()

add_python_test(NAME canary FILENAME canary_test.py)
add_python_test(NAME loading FILENAME loading_test.py)
add_python_test(NAME random FILENAME random_generator_test.py)
add_python_test(NAME direct_binding FILENAME direct_binding_test.py)
add_python_test(NAME one_side_carma_binding FILENAME one_side_carma_binding_test.py)
add_python_test(NAME two_side_carma_binding FILENAME two_side_carma_binding_test.py)
add_python_test(NAME py_dict_to_cpp_test FILENAME py_dict_to_cpp_test.py )
add_python_test(NAME WrappedPyLinearRegression FILENAME WrappedPyLinearRegression_test.py)
add_python_test(NAME WrappedPyKrigingParametricTest FILENAME WrappedPyKriging_parametric_test.py)
add_python_test(NAME PyLinearRegression FILENAME PyLinearRegression_test.py)
add_python_test(NAME PyKrigingParametricTest FILENAME PyKriging_parametric_test.py)
add_python_test(NAME KrigingCopyTest FILENAME PyKriging_copy_test.py)
add_python_test(NAME binding_consistency FILENAME binding_consistency_test.py)
add_python_test(NAME KrigingDemo FILENAME pylibkriging_demo.py)
