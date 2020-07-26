if (NOT LIBKRIGING_SOURCE_DIR)
    message(FATAL_ERROR "LIBKRIGING_SOURCE_DIR not defined")
endif ()

include(${LIBKRIGING_SOURCE_DIR}/cmake/configuration.cmake)

if (NOT OCTAVE_CONFIG_EXECUTABLE)
    find_program(OCTAVE_CONFIG_EXECUTABLE NAMES octave-config)
endif ()

execute_process(COMMAND ${OCTAVE_CONFIG_EXECUTABLE} -p BINDIR
        OUTPUT_VARIABLE OCTAVE_BIN_PATHS
        OUTPUT_STRIP_TRAILING_WHITESPACE)

find_program(OCTAVE_EXECUTABLE
        HINTS ${OCTAVE_BIN_PATHS}
        NAMES octave-cli octave
        )
find_program(OCTAVE_MKOCTFILE
        HINTS ${OCTAVE_BIN_PATHS}
        NAMES mkoctfile
        )

# below: shell script to list all paramters of mkoctfile
#  for i in ALL_CFLAGS ALL_CXXFLAGS ALL_FFLAGS ALL_LDFLAGS BLAS_LIBS CC CFLAGS CPICFLAG CPPFLAGS CXX CXXFLAGS CXXPICFLAG DL_LD DL_LDFLAGS F77 F77_INTEGER8_FLAG FFLAGS FPICFLAG INCFLAGS INCLUDEDIR LAPACK_LIBS LDFLAGS LD_CXX LD_STATIC_FLAG LFLAGS LIBDIR LIBOCTAVE LIBOCTINTERP OCTAVE_LINK_OPTS OCTINCLUDEDIR OCTAVE_LIBS OCTAVE_LINK_DEPS OCTLIBDIR OCT_LINK_DEPS OCT_LINK_OPTS RDYNAMIC_FLAG SPECIAL_MATH_LIB XTRA_CFLAGS XTRA_CXXFLAGS ; do echo $i=$(mkoctfile -p $i); done
execute_process(COMMAND ${OCTAVE_MKOCTFILE} -p CPPFLAGS
        OUTPUT_VARIABLE OCT_CPPFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OCT_CPPFLAGS "${OCT_CPPFLAGS} -DMEX_DEBUG")
endif ()

execute_process(COMMAND ${OCTAVE_MKOCTFILE} -p CXXPICFLAG
        OUTPUT_VARIABLE OCT_CXXPICFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${OCTAVE_MKOCTFILE} -p ALL_CXXFLAGS
        OUTPUT_VARIABLE OCT_CXXFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
if (WIN32)
	# Provided options from mkoctfile could be better if they are split into target_include_directories and set_target_properties(... COMPILE_FLAGS ...)
	# Work around misunderstanding of compiler with backslashs include paths
	if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
		string(REPLACE "\\" "/" OCT_CXXFLAGS ${OCT_CXXFLAGS})
	else()
		logFatalError("Unexpected configuration : WIN32 + ${CMAKE_CXX_COMPILER_ID}")
	endif()
endif()
		
execute_process(COMMAND ${OCTAVE_MKOCTFILE} -p DL_LDFLAGS
        OUTPUT_VARIABLE OCT_DLLDFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(OCT_DLLDFLAGS) # transform in list
execute_process(COMMAND ${OCTAVE_MKOCTFILE} -p LDFLAGS
        OUTPUT_VARIABLE OCT_LDFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(OCT_LDFLAGS) # transform in list
#execute_process(COMMAND ${OCTAVE_MKOCTFILE} -p LFLAGS
#        OUTPUT_VARIABLE OCT_LFLAGS
#        OUTPUT_STRIP_TRAILING_WHITESPACE)
#separate_arguments(OCT_LFLAGS) # transform in list
#execute_process(COMMAND ${OCTAVE_MKOCTFILE} -p OCTAVE_LIBS
#        OUTPUT_VARIABLE OCT_LIBS
#        OUTPUT_STRIP_TRAILING_WHITESPACE)
#separate_arguments(OCT_LIBS) # transform in list

#foreach (VAR OCT_CPPFLAGS OCT_CXXPICFLAGS OCT_CXXFLAGS OCT_DLLDFLAGS OCT_LDFLAGS OCT_LFLAGS OCT_LIBS)
#    message(STATUS "${VAR} = ${${VAR}}")
#endforeach ()

execute_process(COMMAND ${OCTAVE_MKOCTFILE} -p OCTLIBDIR
        OUTPUT_VARIABLE OCTAVE_LIBRARIES_PATHS
        OUTPUT_STRIP_TRAILING_WHITESPACE)

find_library(OCTAVE_OCTINTERP_LIBRARY
        NAMES octinterp liboctinterp
        HINTS ${OCTAVE_LIBRARIES_PATHS}
        )
find_library(OCTAVE_OCTAVE_LIBRARY
        NAMES octave liboctave
        HINTS ${OCTAVE_LIBRARIES_PATHS}
        )
set(OCTAVE_LIBRARIES ${OCTAVE_OCTINTERP_LIBRARY})
list(APPEND OCTAVE_LIBRARIES ${OCTAVE_OCTAVE_LIBRARY})

execute_process(COMMAND ${OCTAVE_CONFIG_EXECUTABLE} -v
        OUTPUT_VARIABLE OCTAVE_VERSION_STRING
        OUTPUT_STRIP_TRAILING_WHITESPACE)

if (OCTAVE_VERSION_STRING)
    string(REGEX REPLACE "([0-9]+)\\..*" "\\1" OCTAVE_MAJOR_VERSION ${OCTAVE_VERSION_STRING})
    string(REGEX REPLACE "[0-9]+\\.([0-9]+).*" "\\1" OCTAVE_MINOR_VERSION ${OCTAVE_VERSION_STRING})
    string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" OCTAVE_PATCH_VERSION ${OCTAVE_VERSION_STRING})
endif ()


macro(add_mex_function)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES LINK_LIBRARIES)

    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (ARGS_UNPARSED_ARGUMENTS)
        logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
    endif ()

    if (NOT ARGS_NAME)
        logFatalError("add_mex_function needs NAME")
    endif ()

    if (NOT ARGS_SOURCES)
        logFatalError("add_mex_function needs SOURCES")
    endif ()

    add_library(${ARGS_NAME} MODULE ${ARGS_SOURCES})
    target_link_libraries(${ARGS_NAME} ${ARGS_LINK_LIBRARIES} ${OCTAVE_LIBRARIES})
    # https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#properties-on-targets
    set_target_properties(${ARGS_NAME} PROPERTIES
            PREFIX ""
            SUFFIX ".mex")
    #mkoctfile compile = CXX OCT_CPPFLAGS OCT_CXXPICFLAGS OCT_CXXFLAGS -I. -DMEX_DEBUG
    set_target_properties(${ARGS_NAME} PROPERTIES
            COMPILE_FLAGS "${OCT_CPPFLAGS} ${OCT_CXXPICFLAGS} ${OCT_CXXFLAGS}") 
    # https://stackoverflow.com/questions/48176641/linking-to-an-executable-under-osx-with-cmake si pb avec bundle_loader
    #mkoctfile link = CXX OCT_CXXFLAGS OCT_DLLDFLAGS OCT_LDFLAGS OCT_LFLAGS OCT_LIBS
    target_link_options(${ARGS_NAME}
            PRIVATE ${OCT_DLLDFLAGS} ${OCT_LDFLAGS}
            )
endmacro()

# https://cmake.org/cmake/help/latest/command/mark_as_advanced.html
#mark_as_advanced() # TODO
