# Test of the installed CMake package (issue #127). Usage:
#   cmake -DSRC_DIR= -DWORK_DIR= -DCONFIG= -DGENERATOR= -DEXPECTED_VERSION= -DINSTALL_DIRS="a|b|c" -DINSTALLED_PREFIX= -P run_test.cmake
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

# Some CI scripts remove the compiled library directory from the build tree after a first ctest run, to check that
# the installed library is used (see tools/linux-macos/test.sh): the build tree cannot be installed any more then;
# use the installation already done in INSTALLED_PREFIX instead.
set(build_tree_complete TRUE)
foreach (dir IN LISTS INSTALL_DIRS)
  if (NOT EXISTS "${dir}/cmake_install.cmake")
    set(build_tree_complete FALSE)
  endif ()
endforeach ()

if (NOT build_tree_complete)
  file(GLOB_RECURSE installed_config "${INSTALLED_PREFIX}/*/cmake/libKriging/libKrigingConfig.cmake")
  if (NOT installed_config)
    message(FATAL_ERROR "Neither a complete build tree (${INSTALL_DIRS}) nor an installation in '${INSTALLED_PREFIX}'")
  endif ()
  message(STATUS "Build tree incomplete: use the installation in ${INSTALLED_PREFIX}")
  set(prefix "${INSTALLED_PREFIX}")
else ()
  foreach (dir IN LISTS INSTALL_DIRS)
    execute_process(COMMAND ${CMAKE_COMMAND} --install "${dir}" --prefix "${prefix}" --config "${CONFIG}"
        RESULT_VARIABLE result)
    if (NOT result EQUAL 0)
      # replay the install script with a trace to show where it fails
      execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${prefix} -DCMAKE_INSTALL_CONFIG_NAME=${CONFIG}
          --trace-expand -P "${dir}/cmake_install.cmake" OUTPUT_QUIET ERROR_VARIABLE trace)
      string(LENGTH "${trace}" trace_length)
      math(EXPR trace_start "${trace_length} - 3000")
      if (trace_start LESS 0)
        set(trace_start 0)
      endif ()
      string(SUBSTRING "${trace}" ${trace_start} -1 trace_tail)
      message(FATAL_ERROR "Install of '${dir}' failed (${result}); end of trace:\n${trace_tail}")
    endif ()
  endforeach ()
endif ()

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
