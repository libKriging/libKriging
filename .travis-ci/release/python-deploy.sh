#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi
set -x

if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
    cd "${TRAVIS_BUILD_DIR}" || exit 1
fi

echo "Ready to deploy from $TRAVIS_OS_NAME"
ARCH=$(uname -s)

#TAG=$(git describe --tags --match "v[0-9]*" --exact-match)
TAG=$(echo "$TRAVIS_TAG" | grep -E '^v[[:digit:]]+\.[[:digit:]]+')
if [ -z "$TAG" ]; then
  echo "No version tag: skip deploy" 
  exit 0 
fi
echo "Release tag '$TAG' in branch '$TRAVIS_BRANCH'"

if [ "$ENABLE_PYTHON_BINDING" == "on" ]; then
  python3 -m pip install twine  
  case $ARCH in
    Linux)
      docker run --rm \
        -e ROOT_DIR=/data \
        -e DEBUG_CI="${DEBUG_CI}" \
        -w /data \
        -v `pwd`:/data \
        quay.io/pypa/manylinux2014_x86_64 /data/bindings/Python/tools/build_wheels.sh
      # TWINE_USERNAME is set from travis global environment
      # TWINE_PASSWORD is set to an API token in Travis settings
      python3 -m twine upload \
        --repository-url https://upload.pypi.org/legacy/ \
        ./dist/*-manylinux2014_x86_64.whl        
      ;;
    Darwin|MSYS_NT*)
      python3 ./bindings/Python/setup.py bdist_wheel
      # TWINE_USERNAME is set from travis global environment
      # TWINE_PASSWORD is set to an API token in Travis settings
      python3 -m twine upload \
        --repository-url https://upload.pypi.org/legacy/ \
        ./dist/*.whl
      ;;
    *)
      echo "Unknown OS [$ARCH]"
      ;;
  esac  
fi

















