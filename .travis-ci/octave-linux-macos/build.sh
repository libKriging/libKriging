#!/usr/bin/env bash
set -eo pipefail

# to get readlink on MacOS (no effect on Linux)
if [[ -e /usr/local/opt/coreutils/libexec/gnubin/readlink ]]; then
    export PATH=/usr/local/opt/coreutils/libexec/gnubin:$PATH
fi
BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

[[ ${ENABLE_OCTAVE_BINDING} == "on" ]] || (echo "Octave binding must be 'on'" && exit 1 )

CC=gcc-7 CXX=g++-7 \
    EXTRA_CMAKE_OPTIONS="-DBUILD_SHARED_LIBS=off -DDETECT_HDF5=false ${EXTRA_CMAKE_OPTIONS}" \
    "${BASEDIR}"/../linux-macos/build.sh
