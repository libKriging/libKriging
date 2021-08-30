#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

# windows environment requires to load special tools
loadenv_sh=".travis-ci/${BUILD_NAME}/loadenv.sh"
if [ -e "$loadenv_sh" ]; then
  . "$loadenv_sh"
fi

ARCH=$(uname -s)
echo "Ready to release from $ARCH"

if [ -z "${GIT_TAG}" ]; then
  echo "No valid version tag found" 
  exit 1
fi
echo "Release tag '${GIT_TAG}' in branch '$(git branch --show-current)'"

case $ARCH in
  Linux)
    docker run --rm \
      -e ROOT_DIR=/data \
      -e DEBUG_CI="${DEBUG_CI}" \
      -w /data \
      -v `pwd`:/data \
      quay.io/pypa/manylinux2014_x86_64 /data/bindings/Python/tools/build_wheels.sh
    ;;
  Darwin|MSYS_NT*|MINGW64_NT*)
    python3 ./bindings/Python/setup.py bdist_wheel
    pip install pylibkriging --no-index -f ./dist
    # TODO add more tests
    # pytest ./bindings/Python/tests    
    ;;
  *)
    echo "Unknown OS [$ARCH]"
    exit 1
    ;;
esac  

















