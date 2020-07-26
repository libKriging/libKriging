
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

    add_test(NAME Python/${ARGS_NAME}
            COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/${ARGS_FILENAME})

    if (ARGS_WILL_FAIL)
        set_tests_properties(Python/${ARGS_NAME}
                PROPERTIES
                WILL_FAIL TRUE)
    endif ()
    set_tests_properties(Python/${ARGS_NAME}
            PROPERTIES
            WORKING_DIRECTORY "${LIBKRIGING_PYTHON_BINARY_DIR}"
            ENVIRONMENT "PYTHONPATH=${LIBKRIGING_PYTHON_BINARY_DIR}"
            LABELS Python
            ${ARGS_PROPERTIES})
endmacro()

add_python_test(NAME canary FILENAME canary_test.py)
add_python_test(NAME loading FILENAME loading_test.py)
add_python_test(NAME numpy FILENAME NumPy_test.py)
add_python_test(NAME LinearRegression FILENAME LinearRegression_test.py)
add_python_test(NAME PyLinearRegression FILENAME PyLinearRegression_test.py)
