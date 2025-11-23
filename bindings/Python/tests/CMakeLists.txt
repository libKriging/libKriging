
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

    # Preload OpenMP library to avoid "cannot allocate memory in static TLS block" error
    # This ensures libgomp/libomp is loaded before the Python extension module
    if(NOT WIN32 AND OpenMP_CXX_FOUND)
        if(OpenMP_libomp_LIBRARY)
            # For LLVM/Clang OpenMP (libomp)
            set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "LD_PRELOAD=${OpenMP_libomp_LIBRARY}")
        elseif(OpenMP_gomp_LIBRARY)
            # For GNU OpenMP (libgomp)
            set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "LD_PRELOAD=${OpenMP_gomp_LIBRARY}")
        elseif(EXISTS "/usr/lib/x86_64-linux-gnu/libgomp.so.1")
            # Fallback for Ubuntu/Debian
            set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1")
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
