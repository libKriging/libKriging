if (OCTAVE_BINDING_MODE STREQUAL "Octave")
    include(${LIBKRIGING_SOURCE_DIR}/cmake/configureOctave.cmake)
    set(OctaveMode_COMPILE_FLAGS "${OCT_CPPFLAGS} ${OCT_CXXPICFLAGS} ${OCT_CXXFLAGS}")
elseif (OCTAVE_BINDING_MODE STREQUAL "Matlab")
    set(OctaveMode_INCLUDE_DIRS ${Matlab_INCLUDE_DIRS})
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # The MEX file runs in a MATLAB process that has already loaded its own MKL. Linking it to
        # another BLAS / LAPACK (e.g. the system MKL found by armadillo) mixes two MKL builds in the
        # process: the system LAPACK then calls the internal kernels of MATLAB's MKL and crashes
        # (segmentation violation in mkl_blas_*_dgemv from mkl_lapack_dlarf1, #180). MATLAB's own
        # BLAS / LAPACK libraries are linked first, so that the BLAS / LAPACK symbols of the MEX file
        # resolve to them; they are ILP64, as ARMA_BLAS_LONG (set for this build) expects.
        find_library(Matlab_MWLAPACK_LIBRARY NAMES mwlapack PATHS "${Matlab_ROOT_DIR}/bin/glnxa64" NO_DEFAULT_PATH)
        find_library(Matlab_MWBLAS_LIBRARY NAMES mwblas PATHS "${Matlab_ROOT_DIR}/bin/glnxa64" NO_DEFAULT_PATH)
        if (NOT Matlab_MWLAPACK_LIBRARY OR NOT Matlab_MWBLAS_LIBRARY)
            logFatalError("MATLAB BLAS / LAPACK libraries (libmwblas, libmwlapack) not found in ${Matlab_ROOT_DIR}/bin/glnxa64")
        endif ()
        # --no-as-needed: they come before the static libraries that use them (Kriging, armadillo)
        # on the link line, so --as-needed (default on Ubuntu) would drop them; once kept, their
        # symbols satisfy the later references and the BLAS / LAPACK of armadillo (libmkl_rt) is
        # dropped by --as-needed.
        set(OctaveMode_MATLAB_BLAS_LAPACK
                "-Wl,--no-as-needed" ${Matlab_MWLAPACK_LIBRARY} ${Matlab_MWBLAS_LIBRARY} "-Wl,--as-needed")
        message(STATUS "Matlab MEX BLAS / LAPACK: ${Matlab_MWLAPACK_LIBRARY} ${Matlab_MWBLAS_LIBRARY}")
    endif ()
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
                LINK_TO ${OctaveMode_MATLAB_BLAS_LAPACK} ${ARGS_LINK_LIBRARIES})
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
        # Octave's assert (comparison forms used by the tests), see tools/matlab/run_mex_tests.m
        set(TEST_SCRIPT "addpath('${LIBKRIGING_SOURCE_DIR}/tools/matlab/octave_compat'), ${TEST_SCRIPT}")
        add_test(NAME ${OCTAVE_BINDING_MODE}/${ARGS_NAME}
                COMMAND ${PRECOMMAND} ${Matlab_MAIN_PROGRAM} -logfile "${log_file_name}" -batch "${TEST_SCRIPT}")
    endif ()

    if (ARGS_WILL_FAIL)
        set_tests_properties(${OCTAVE_BINDING_MODE}/${ARGS_NAME}
                PROPERTIES
                WILL_FAIL TRUE)
    endif ()

    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set_tests_properties(${OCTAVE_BINDING_MODE}/${ARGS_NAME}
                PROPERTIES
                WORKING_DIRECTORY ${LIBKRIGING_OCTAVE_BINARY_DIR}/$<CONFIG>
                LABELS ${OCTAVE_BINDING_MODE}
                ${ARGS_PROPERTIES})
    else ()
        set_tests_properties(${OCTAVE_BINDING_MODE}/${ARGS_NAME}
                PROPERTIES
                WORKING_DIRECTORY ${LIBKRIGING_OCTAVE_BINARY_DIR}
                LABELS ${OCTAVE_BINDING_MODE}
                ${ARGS_PROPERTIES})
    endif ()

endmacro()