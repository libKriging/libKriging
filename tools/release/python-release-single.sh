#!/usr/bin/env bash
# Build a single Python wheel using manylinux_2_28 docker image
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

ARCH=$(uname -s)
echo "Ready to release from $ARCH"

if [ -z "${GIT_TAG}" ]; then
  echo "No valid version tag found" 
  exit 1
fi
echo "Release tag '${GIT_TAG}' in branch '$(git branch --show-current)'"

# Get Python version from environment
PYTHON_VERSION=${PYTHON_VERSION:-3.8}
echo "Building for Python ${PYTHON_VERSION}"

case $ARCH in
  Linux)
    docker run --rm \
      -e ROOT_DIR=/data \
      -e DEBUG_CI="${DEBUG_CI}" \
      -w /data \
      -v `pwd`:/data \
      quay.io/pypa/manylinux_2_28_x86_64 \
      /data/bindings/Python/tools/build_single_wheel.sh ${PYTHON_VERSION}
    ;;
  Darwin|MSYS_NT*|MINGW64_NT*)
    python3 ./bindings/Python/setup.py bdist_wheel
    # Temporarily disable tests to focus on build
    # pip install pylibkriging --no-index -f ./dist
    # pytest -s ./bindings/Python/tests/
    ;;
  *)
    echo "Unknown OS [$ARCH]"
    exit 1
    ;;
esac
