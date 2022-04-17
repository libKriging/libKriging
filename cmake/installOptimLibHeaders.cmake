# Rewrite in CMake `configure` script in optim lib since it is not portable for windows

set(OPTIM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/dependencies/optim)
set(OPTIM_INSTALLATION_DIR ${CMAKE_BINARY_DIR}/dependencies/optim/header_only_version)

message(STATUS "Installing optim lib in ${OPTIM_INSTALLATION_DIR}")
file(COPY ${OPTIM_SOURCE_DIR}/include/
        DESTINATION ${OPTIM_INSTALLATION_DIR}/)

set(OPTIM_DIRS constrained line_search unconstrained zeros)
set(OPTIM_BAD_FILE_NAMES optim_unconstrained optim_zeros)
set(OPTIM_DELIMITER_PATTERN "[OPTIM_BEGIN]")
string(LENGTH "${OPTIM_DELIMITER_PATTERN}" OPTIM_DELIMITER_PATTERN_SIZE)
        
foreach (DIR ${OPTIM_DIRS})
    file(GLOB file_names "${OPTIM_INSTALLATION_DIR}/${DIR}/*.hpp")
    foreach(FILE ${file_names})
        get_filename_component(FILE_WE ${FILE} NAME_WE)
        list(FIND OPTIM_BAD_FILE_NAMES ${FILE_WE} BAD_FILE_FOUND)
        if (BAD_FILE_FOUND EQUAL -1)
            file(READ ${FILE} FILE_HPP_CONTENT)
            string(REPLACE "#endif" "//\n" FILE_HPP_CONTENT "${FILE_HPP_CONTENT}")
            file(READ ${OPTIM_SOURCE_DIR}/src/${DIR}/${FILE_WE}.cpp FILE_CPP_CONTENT)
            string(FIND "${FILE_CPP_CONTENT}" "${OPTIM_DELIMITER_PATTERN}" TRUNCATION_INDEX)
            math(EXPR TRUNCATION_INDEX "${TRUNCATION_INDEX}+${OPTIM_DELIMITER_PATTERN_SIZE}+1")
            string(SUBSTRING "${FILE_CPP_CONTENT}" ${TRUNCATION_INDEX} -1 FILE_CPP_CONTENT)
            string(REPLACE "optimlib_inline" "inline" FILE_CPP_CONTENT "${FILE_CPP_CONTENT}")
            string(REPLACE "optim::" "" FILE_CPP_CONTENT "${FILE_CPP_CONTENT}")
            string(APPEND FILE_HPP_CONTENT "${FILE_CPP_CONTENT}")
            string(APPEND FILE_HPP_CONTENT "\n#endif\n")
            file(WRITE ${FILE} "${FILE_HPP_CONTENT}")
        endif()
    endforeach ()
endforeach ()
