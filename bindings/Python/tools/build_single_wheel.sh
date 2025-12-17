#!/bin/bash
# Build a single Python wheel for a specific Python version
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

# Get Python version from argument
PYVER=$1
if [ -z "$PYVER" ]; then
  echo "Usage: $0 <python-version>"
  echo "Example: $0 3.8"
  exit 1
fi

export PLAT=manylinux_2_28_x86_64

if [[ -z ${ROOT_DIR:+x} ]]; then
    ROOT_DIR=.
fi

# Fix git dubious ownership issue in docker
git config --global --add safe.directory "${ROOT_DIR}"

# Initialize submodules (pybind11, etc.)
cd "${ROOT_DIR}"
git submodule update --init --recursive

# Upgrade CMake if needed (manylinux images often have old CMake)
if command -v yum &> /dev/null; then
    yum install -y cmake3 2>/dev/null || yum install -y cmake 2>/dev/null || true
    if [ -f /usr/bin/cmake3 ]; then
        ln -sf /usr/bin/cmake3 /usr/local/bin/cmake
    fi
fi

# Map Python version to manylinux binary path
case $PYVER in
  3.8)
    PYBIN=/opt/python/cp38-cp38/bin
    ;;
  3.9)
    PYBIN=/opt/python/cp39-cp39/bin
    ;;
  3.10)
    PYBIN=/opt/python/cp310-cp310/bin
    ;;
  3.11)
    PYBIN=/opt/python/cp311-cp311/bin
    ;;
  3.12)
    PYBIN=/opt/python/cp312-cp312/bin
    ;;
  *)
    echo "Unsupported Python version: $PYVER"
    exit 1
    ;;
esac

# Install a system package required by our library
yum install -y openblas-devel

# Install Python requirements
"${PYBIN}/pip" install -r "${ROOT_DIR}"/bindings/Python/requirements.txt
"${PYBIN}/pip" install -r "${ROOT_DIR}"/bindings/Python/dev-requirements.txt

# Build wheel
"${PYBIN}/python3" "${ROOT_DIR}"/bindings/Python/setup.py bdist_wheel

# Bundle external shared libraries into the wheels
for whl in "${ROOT_DIR}"/dist/*.whl; do
    if auditwheel show "$whl"; then
        auditwheel repair "$whl" --plat "$PLAT" -w "${ROOT_DIR}"/dist
    else
        echo "Skipping non-platform wheel $whl"
    fi
done

# Remove non-manylinux wheels
find "${ROOT_DIR}"/dist/ -name "*-linux_*" -exec rm {} \;
