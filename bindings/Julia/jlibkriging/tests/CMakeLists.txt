macro(add_julia_test)
    set(options WILL_FAIL)
    set(oneValueArgs NAME FILENAME)
    set(multiValueArgs PROPERTIES)

    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT ARGS_NAME)
        message(FATAL_ERROR "add_julia_test needs NAME")
    endif ()

    if (NOT ARGS_FILENAME)
        message(FATAL_ERROR "add_julia_test needs FILENAME")
    endif ()

    set(TEST_NAME "Julia/${ARGS_NAME}")

    add_test(NAME ${TEST_NAME}
             COMMAND ${JULIA_EXECUTABLE}
                     --project=${CMAKE_CURRENT_SOURCE_DIR}/../julia
                     ${CMAKE_CURRENT_SOURCE_DIR}/${ARGS_FILENAME})

    if (ARGS_WILL_FAIL)
        set_tests_properties(${TEST_NAME} PROPERTIES WILL_FAIL TRUE)
    endif ()

    set_property(TEST ${TEST_NAME} APPEND PROPERTY
                 ENVIRONMENT "JLIBKRIGING_LIB_PATH=$<TARGET_FILE:libkriging_c>")
    set_tests_properties(${TEST_NAME} PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            LABELS Julia
            ${ARGS_PROPERTIES})
endmacro()

add_julia_test(NAME loading FILENAME loading_test.jl)
add_julia_test(NAME linear_regression FILENAME linear_regression_test.jl)
add_julia_test(NAME kriging_basic FILENAME kriging_basic_test.jl)
add_julia_test(NAME kriging_parametric FILENAME kriging_parametric_test.jl)
add_julia_test(NAME nugget_kriging FILENAME nugget_kriging_test.jl)
add_julia_test(NAME noise_kriging FILENAME noise_kriging_test.jl)
add_julia_test(NAME kriging_copy FILENAME kriging_copy_test.jl)
add_julia_test(NAME kriging_save_load FILENAME kriging_save_load_test.jl)
add_julia_test(NAME demo FILENAME jlibkriging_demo.jl)
