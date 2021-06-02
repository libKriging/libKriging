add_executable(manage_test_crash manage_test_crash.c)

macro(add_octave_test)
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

    # to configure command line : http://kirste.userpage.fu-berlin.de/chemnet/use/info/octave/octave_7.html
    if (NOT ARGS_WILL_FAIL)
        add_test(NAME Octave/${ARGS_NAME}
                COMMAND ${OCTAVE_EXECUTABLE} --path ${LIBKRIGING_OCTAVE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_wrapper.m)
    else ()
        # requires crash management for Octave 4 (where exit command causes 'abort')
        add_test(NAME Octave/${ARGS_NAME}
                COMMAND manage_test_crash ${OCTAVE_EXECUTABLE} --path ${LIBKRIGING_OCTAVE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test_wrapper.m)
        set_tests_properties(Octave/${ARGS_NAME}
                PROPERTIES
                WILL_FAIL TRUE)
    endif ()
    set_tests_properties(Octave/${ARGS_NAME}
            PROPERTIES
            WORKING_DIRECTORY ${LIBKRIGING_OCTAVE_BINARY_DIR}
            ENVIRONMENT "TESTFILE=${CMAKE_CURRENT_SOURCE_DIR}/${ARGS_FILENAME}"
            LABELS Octave
            ${ARGS_PROPERTIES})
endmacro()

add_octave_test(NAME success FILENAME dummy_succeededtest.m)
add_octave_test(NAME failure FILENAME dummy_failingtest.m WILL_FAIL)
add_octave_test(NAME mLibKriging-Demo FILENAME mLibKriging_demo.m)
add_octave_test(NAME mLibKriging-BadUsage FILENAME mLibKriging_failingtest.m WILL_FAIL)
add_octave_test(NAME LinearRegression-Simple FILENAME LinearRegression_simpletest.m)
add_octave_test(NAME LinearRegression-Complete FILENAME LinearRegression_completest.m)
add_octave_test(NAME BindingConsistency FILENAME binding_consistency_test.m)
