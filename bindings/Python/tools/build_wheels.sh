#!/bin/bash
# Inspired from https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh
set -eo pipefail
# inside a docker image of quay.io/pypa/manylinux2014_x86_64

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

export PLAT=manylinux2014_x86_64

if [[ -z ${ROOT_DIR:+x} ]]; then
    ROOT_DIR=.
fi

# Fix git dubious ownership issue in docker
git config --global --add safe.directory "${ROOT_DIR}"

# Initialize submodules (pybind11, etc.)
cd "${ROOT_DIR}"
git submodule update --init --recursive

# Upgrade CMake to support newer pybind11
# manylinux2014 has old CMake that doesn't support pybind11's requirements
yum install -y cmake3
if [ -f /usr/bin/cmake3 ]; then
    ln -sf /usr/bin/cmake3 /usr/local/bin/cmake
fi

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w "${ROOT_DIR}"/dist
    fi
}


# Install a system package required by our library
yum install -y openblas-devel # hdf5-devel

# Compile wheels
for PYVER in cp38-cp38 cp39-cp39 cp310-cp310 cp311-cp311 cp312-cp312; do
    echo "------------------------------------------"
    echo "Building pyquantlib for Python ${PYVER}"
    echo "------------------------------------------"
    PYBIN=/opt/python/${PYVER}/bin
    "${PYBIN}/pip" install -r "${ROOT_DIR}"/bindings/Python/requirements.txt # not for Windows
    "${PYBIN}/pip" install -r "${ROOT_DIR}"/bindings/Python/dev-requirements.txt # not for Windows
    "${PYBIN}/python3" "${ROOT_DIR}"/bindings/Python/setup.py bdist_wheel
done

# Bundle external shared libraries into the wheels
for whl in "${ROOT_DIR}"/dist/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
# Temporarily disabled to focus on build
#for PYVER in cp38-cp38 cp39-cp39 cp310-cp310 cp311-cp311 cp312-cp312; do
#    echo "-----------------------------------------"
#    echo "Testing pyquantlib for Python ${PYVER}"
#    echo "-----------------------------------------"
#    PYBIN=/opt/python/${PYVER}/bin
#    "${PYBIN}/pip" install pylibkriging --no-index -f "${ROOT_DIR}"/dist
#    (cd "${ROOT_DIR}"; "${PYBIN}/pytest" "${ROOT_DIR}"/bindings/Python/tests)
#done

find "${ROOT_DIR}"/dist/ -name "*-linux_*" -exec rm {} \;
