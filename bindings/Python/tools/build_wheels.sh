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

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w "${ROOT_DIR}"/dist
    fi
}


# Install a system package required by our library
yum install -y openblas-devel

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [ "${PYBIN}" == "/opt/python/cp35-cp35m/bin" ]; then continue; fi
    if [ "${PYBIN}" == "/opt/python/cp310-cp310/bin" ]; then continue; fi
    "${PYBIN}/pip" install -r "${ROOT_DIR}"/bindings/Python/dev-requirements.txt # not for Windows
    "${PYBIN}/python3" "${ROOT_DIR}"/bindings/Python/setup.py bdist_wheel
done

# Bundle external shared libraries into the wheels
for whl in "${ROOT_DIR}"/dist/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYBIN in /opt/python/*/bin; do
    if [ "${PYBIN}" == "/opt/python/cp35-cp35m/bin" ]; then continue; fi
    if [ "${PYBIN}" == "/opt/python/cp310-cp310/bin" ]; then continue; fi
    "${PYBIN}/pip" install pylibkriging --no-index -f "${ROOT_DIR}"/dist
    (cd "${ROOT_DIR}"; "${PYBIN}/pytest" "${ROOT_DIR}"/bindings/Python/tests)
done

find "${ROOT_DIR}"/dist/ -name "*-linux_*" -exec rm {} \;
