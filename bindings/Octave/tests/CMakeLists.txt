macro(add_octave_test)
    set(options)
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
    add_test(NAME Octave/${ARGS_NAME}
            COMMAND ${OCTAVE_EXECUTABLE} --path ${LIBKRIGING_OCTAVE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${ARGS_FILENAME})
    set_tests_properties(Octave/${ARGS_NAME}
            PROPERTIES
            WORKING_DIRECTORY ${LIBKRIGING_OCTAVE_BINARY_DIR}
            LABELS Octave
            ${ARGS_PROPERTIES})
endmacro()

add_octave_test(NAME mLibKriging FILENAME mLibKriging_test.m)
add_octave_test(NAME mLibKriging-BadUsage FILENAME mLibKriging_failingtest.m PROPERTIES WILL_FAIL TRUE)
add_octave_test(NAME LinearRegression-Simple FILENAME LinearRegression_simpletest.m)
add_octave_test(NAME LinearRegression-Complete FILENAME LinearRegression_completest.m)
