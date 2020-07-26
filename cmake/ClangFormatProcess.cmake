# --------------- runs clang-format in place using the style file ---------------

if(LIBKRIGING_SOURCE_DIR AND CLANG_FORMAT)
    # get C++ sources file list (ignoring packages)
    file(GLOB_RECURSE ALL_SOURCE_FILES
            ${LIBKRIGING_SOURCE_DIR}/src/**.[hc]pp
            ${LIBKRIGING_SOURCE_DIR}/tests/**.[hc]pp
            ${LIBKRIGING_SOURCE_DIR}/bindings/Octave/**.[hc]pp
            ${LIBKRIGING_SOURCE_DIR}/bindings/Python/**.[hc]pp
            ${LIBKRIGING_SOURCE_DIR}/bindings/R/rlibkriging/**.[hc]pp
            )

    # apply style to the file list
    foreach(SOURCE_FILE ${ALL_SOURCE_FILES})
        execute_process(COMMAND "${CLANG_FORMAT}" -style=file -verbose -i "${SOURCE_FILE}")
    endforeach()
endif()
