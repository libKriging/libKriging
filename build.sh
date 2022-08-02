#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

# BEGIN/END: OPENBLAS RTOOLS42 WINDOWS BUG REPRODUCER
# BEGIN 
[ -d build ] && ( echo "Removing previous build" && rm -fr build)

export PATH=/c/rtools42/x86_64-w64-mingw32.static.posix/bin:$PATH
export PATH="/c/Program Files/R/R-4.2.0/bin/x64":$PATH
export EXTRA_SYSTEM_LIBRARY_PATH=/ # required by script but not used in that case
# END

# Setup used/unused bindings
export ENABLE_R_BINDING=ON
export ENABLE_OCTAVE_BINDING=OFF
export ENABLE_MATLAB_BINDING=OFF
export ENABLE_PYTHON_BINDING=OFF

export MAKE_SHARED_LIBS=off

: ${R_HOME=$(R RHOME)}
if test -z "${R_HOME}"; then
   as_fn_error $? "Could not determine R_HOME." "$LINENO" 5
fi

# Static libKriging build (using libKriging/.ci)
# BEGIN
#cd libKriging
# END
{
.travis-ci/common/before_script.sh
} || {
echo "!!! Failed checking configuration !!!"
}

export CC=`${R_HOME}/bin/R CMD config CC`
export CXX=`${R_HOME}/bin/R CMD config CXX`
export FC=`${R_HOME}/bin/R CMD config FC`

# R workflow requires to use R cmd with full path.
# These declarations help to skip declaration without full path in libKriging build scripts.
export CMAKE_Fortran_COMPILER="$(${R_HOME}/bin/R CMD config FC | awk '{ print $1 }')"
export Fortran_LINK_FLAGS="$(${R_HOME}/bin/R CMD config FLIBS)"

BUILD_TEST=false \
MODE=Release \
EXTRA_CMAKE_OPTIONS="-DBUILD_SHARED_LIBS=${MAKE_SHARED_LIBS} -DEXTRA_SYSTEM_LIBRARY_PATH=${EXTRA_SYSTEM_LIBRARY_PATH}" \
${PWD}/.travis-ci/linux-macos/build.sh 

# BEGIN
export LIBKRIGING_PATH=${PWD}/${BUILD_DIR:-build}/installed
export PATH=${LIBKRIGING_PATH}/bin:${PATH}

cd bindings/R
make uninstall || true
make clean
MAKE_SHARED_LIBS=${MAKE_SHARED_LIBS} make

R_LIBS="$PWD/Rlibs" Rscript.exe /z/rlibkriging/libKriging/bindings/R/rlibkriging/tests/testthat/test-debug.R
# END