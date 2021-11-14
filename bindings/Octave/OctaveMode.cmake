if (OCTAVE_BINDING_MODE STREQUAL "Octave")
    include(${LIBKRIGING_SOURCE_DIR}/cmake/configureOctave.cmake)
    set(OctaveMode_COMPILE_FLAGS "${OCT_CPPFLAGS} ${OCT_CXXPICFLAGS} ${OCT_CXXFLAGS}")
elseif (OCTAVE_BINDING_MODE STREQUAL "Matlab")
    set(OctaveMode_INCLUDE_DIRS ${Matlab_INCLUDE_DIRS})
else ()
    logFatalError("INTERNAL ERROR: OCTAVE_BINDING_MODE should be Octave or Matlab not '${OCTAVE_BINDING_MODE}'")
endif ()

macro(add_mex_function)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES LINK_LIBRARIES)

    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (ARGS_UNPARSED_ARGUMENTS)
        logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
    endif ()

    if (OCTAVE_BINDING_MODE STREQUAL "Octave")
        octave_add_mex(NAME ${ARGS_NAME}
                SOURCES ${ARGS_SOURCES}
                LINK_LIBRARIES ${ARGS_LINK_LIBRARIES})
    elseif (OCTAVE_BINDING_MODE STREQUAL "Matlab")
        matlab_add_mex(NAME ${ARGS_NAME}
                SRC ${ARGS_SOURCES}
                LINK_TO ${ARGS_LINK_LIBRARIES})
    endif ()
endmacro()

message(STATUS "Matlab_MAIN_PROGRAM =  ${Matlab_MAIN_PROGRAM}")


macro(add_mex_test)
    set(options WILL_FAIL)
    set(oneValueArgs NAME FILENAME)
    # https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#test-properties
    set(multiValueArgs PROPERTIES)
    
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (ARGS_UNPARSED_ARGUMENTS)
        logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
    endif ()

    get_filename_component(FUNCTION_NAME "${ARGS_FILENAME}" NAME_WE)

    set(TEST_SCRIPT "addpath('${CMAKE_CURRENT_SOURCE_DIR}'), addpath('${LIBKRIGING_OCTAVE_SOURCE_DIR}'), try, run('${FUNCTION_NAME}'), catch err, disp('An exception has been thrown during the execution'), disp(err), disp(err.stack), exit(1), end, exit(0)")

    if (ARGS_WILL_FAIL)
        # requires crash management for Octave 4 (where exit command causes 'abort')
        set(PRECOMMAND manage_test_crash)
    else()
        set(PRECOMMAND)
    endif()
    
    if (OCTAVE_BINDING_MODE STREQUAL "Octave")
        add_test(NAME ${OCTAVE_BINDING_MODE}/${ARGS_NAME}
                COMMAND ${PRECOMMAND} ${OCTAVE_EXECUTABLE} --path ${LIBKRIGING_OCTAVE_SOURCE_DIR} --eval "${TEST_SCRIPT}")
    elseif (OCTAVE_BINDING_MODE STREQUAL "Matlab")
        # see tools/run_matlab_command.sh and matlab_add_unit_test for startup detail
        string(REPLACE "/" "_" log_file_name "${ARGS_NAME}.log")
        add_test(NAME ${OCTAVE_BINDING_MODE}/${ARGS_NAME}
                COMMAND ${PRECOMMAND} ${Matlab_MAIN_PROGRAM} -logfile "${log_file_name}" -batch "${TEST_SCRIPT}")
    endif ()

    if (ARGS_WILL_FAIL)
        set_tests_properties(${OCTAVE_BINDING_MODE}/${ARGS_NAME}
                PROPERTIES
                WILL_FAIL TRUE)
    endif()
    
    set_tests_properties(${OCTAVE_BINDING_MODE}/${ARGS_NAME}
            PROPERTIES
            WORKING_DIRECTORY ${LIBKRIGING_OCTAVE_BINARY_DIR}
            LABELS ${OCTAVE_BINDING_MODE}
            ${ARGS_PROPERTIES})

endmacro()