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

    if (OCTAVE_BINDING_MODE STREQUAL "Octave")
        if (${ARGS_WILL_FAIL})
            octave_add_test(NAME ${ARGS_NAME}
                    FILENAME ${ARGS_FILENAME}
                    WILL_FAIL
                    PROPERTIES ${ARGS_PROPERTIES})
        else ()
            octave_add_test(NAME ${ARGS_NAME}
                    FILENAME ${ARGS_FILENAME}
                    PROPERTIES ${ARGS_PROPERTIES})
        endif ()
    elseif (OCTAVE_BINDING_MODE STREQUAL "Matlab")
        set(Matlab_UNIT_TESTS_CMD -nosplash -nodesktop -nodisplay)
        if (WIN32)
            set(Matlab_UNIT_TESTS_CMD ${Matlab_UNIT_TESTS_CMD} -wait)
        endif ()

        string(REPLACE "/" "_" log_file_name "${ARGS_NAME}.log")

        get_filename_component(FUNCTION_NAME "${ARGS_FILENAME}" NAME_WE)
        
        set(TEST_SCRIPT "addpath('${CMAKE_CURRENT_SOURCE_DIR}'), addpath('${LIBKRIGING_OCTAVE_SOURCE_DIR}'), try, run('${FUNCTION_NAME}'), catch err, disp('An exception has been thrown during the execution'), disp(err), disp(err.stack), exit(1), end, exit(0)")
        
        if (NOT ARGS_WILL_FAIL)
            add_test(NAME ${OCTAVE_BINDING_MODE}/${ARGS_NAME}
                    COMMAND ${Matlab_MAIN_PROGRAM} ${Matlab_UNIT_TESTS_CMD} -logfile "${log_file_name}" -r "${TEST_SCRIPT}")
        else ()
            # requires crash management for Octave 4 (where exit command causes 'abort')
            add_test(NAME ${OCTAVE_BINDING_MODE}/${ARGS_NAME}
                    COMMAND manage_test_crash ${Matlab_MAIN_PROGRAM} ${Matlab_UNIT_TESTS_CMD} -logfile "${log_file_name}" -r "${TEST_SCRIPT}")
            set_tests_properties(${OCTAVE_BINDING_MODE}/${ARGS_NAME}
                    PROPERTIES
                    WILL_FAIL TRUE)
        endif ()

        set_tests_properties(${OCTAVE_BINDING_MODE}/${ARGS_NAME}
                PROPERTIES
                WORKING_DIRECTORY ${LIBKRIGING_OCTAVE_BINARY_DIR}
                LABELS ${OCTAVE_BINDING_MODE}
                ${ARGS_PROPERTIES})
    endif ()
endmacro()