#!/usr/bin/env bash
# Build the "-gpu" source archive attached to each GitHub release:
#   gpu-package/libKriging-gpu_<version>_src.tar.gz
# It holds every tracked file, submodules included (a plain GitHub source
# archive lacks them), so that `tools/install-gpu.sh [cpp|python|r|octave|matlab|julia]`
# can compile libKriging and its bindings on the user's machine with every GPU
# backend found there. The binary packages of each release stay CPU-only.
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "${ROOT_DIR}"

if [ ! -f dependencies/armadillo-code/CMakeLists.txt ]; then
  echo "Submodules are missing: run 'git submodule update --init --recursive' first" >&2
  exit 1
fi

VERSION=${GIT_TAG:-$(awk '/^set\(KRIGING_VERSION_(MAJOR|MINOR|PATCH) /{gsub(/\)/,"",$2); v=v (v?".":"") $2} END{print v}' cmake/version.cmake)}
VERSION=${VERSION#v}
NAME="libKriging-gpu_${VERSION}_src"
PACKAGE_DIR=gpu-package
mkdir -p "${PACKAGE_DIR}"

FILE_LIST=$(mktemp)
trap 'rm -f "${FILE_LIST}"' EXIT
# Tracked files of the repository and of its submodules; notebooks and
# third-party documentation/test trees are not needed to build.
git ls-files --recurse-submodules \
  | grep -v -E '\.ipynb$|^dependencies/armadillo-code/(docs|tests1|tests2)/|^dependencies/(Catch2|pybind11)/docs/' \
  > "${FILE_LIST}"

tar czf "${PACKAGE_DIR}/${NAME}.tar.gz" --transform "s,^,${NAME}/," -T "${FILE_LIST}"
ls -l "${PACKAGE_DIR}/${NAME}.tar.gz"
