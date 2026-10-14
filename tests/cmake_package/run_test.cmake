# Test of the installed CMake package (issue #127). Usage:
#   cmake -DSRC_DIR= -DWORK_DIR= -DCONFIG= -DGENERATOR= -DEXPECTED_VERSION= -DINSTALL_DIRS="a|b|c" -P run_test.cmake
# 1. install the library and its dependencies in a temporary prefix
# 2. configure, build and run a minimal project using find_package(libKriging)

set(prefix "${WORK_DIR}/prefix")
set(consumer_build "${WORK_DIR}/consumer-build")
file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

function(run_step name)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE result)
  if (NOT result EQUAL 0)
    message(FATAL_ERROR "Step '${name}' failed (${result})")
  endif ()
endfunction()

string(REPLACE "|" ";" INSTALL_DIRS "${INSTALL_DIRS}")
foreach (dir IN LISTS INSTALL_DIRS)
  if (NOT EXISTS "${dir}/cmake_install.cmake")
    file(GLOB children "${dir}/*")
    get_filename_component(parent "${dir}" DIRECTORY)
    file(GLOB siblings "${parent}/*")
    message(FATAL_ERROR "No cmake_install.cmake in '${dir}'\n  content: ${children}\n  parent content: ${siblings}")
  endif ()
  run_step("install ${dir}" ${CMAKE_COMMAND} --install "${dir}" --prefix "${prefix}" --config "${CONFIG}")
endforeach ()

set(configure_args -S "${SRC_DIR}" -B "${consumer_build}" -G "${GENERATOR}"
    -DCMAKE_BUILD_TYPE=${CONFIG} -DCMAKE_PREFIX_PATH=${prefix} -DEXPECTED_VERSION=${EXPECTED_VERSION})
if (PLATFORM)
  list(APPEND configure_args -A "${PLATFORM}")
endif ()
if (CXX_COMPILER)
  list(APPEND configure_args -DCMAKE_CXX_COMPILER=${CXX_COMPILER})
endif ()
run_step("configure consumer" ${CMAKE_COMMAND} ${configure_args})
run_step("build consumer" ${CMAKE_COMMAND} --build "${consumer_build}" --config "${CONFIG}")

# locate the executable (single or multi-config generators)
find_program(consumer_exe NAMES consumer consumer.exe
        PATHS "${consumer_build}" "${consumer_build}/${CONFIG}" NO_DEFAULT_PATH)
if (NOT consumer_exe)
  message(FATAL_ERROR "consumer executable not found in ${consumer_build}")
endif ()

# shared library lookup at runtime
if (WIN32)
  set(ENV{PATH} "${prefix}/bin;${prefix}/lib;$ENV{PATH}")
else ()
  set(ENV{LD_LIBRARY_PATH} "${prefix}/lib:${prefix}/lib64:$ENV{LD_LIBRARY_PATH}")
  set(ENV{DYLD_LIBRARY_PATH} "${prefix}/lib:$ENV{DYLD_LIBRARY_PATH}")
endif ()
run_step("run consumer" "${consumer_exe}")
