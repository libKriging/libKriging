#!/bin/bash
# Inspired from https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh
set -e -u -x

export PLAT=manylinux2014_x86_64

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /data/dist
    fi
}


# Install a system package required by our library
yum install -y openblas-devel

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [ "${PYBIN}" == "/opt/python/cp35-cp35m/bin" ]; then continue; fi
    "${PYBIN}/pip" install -r /data/bindings/Python/dev-requirements.txt
    "${PYBIN}/python3" /data/bindings/Python/setup.py bdist_wheel
done

# Bundle external shared libraries into the wheels
for whl in /data/dist/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYBIN in /opt/python/*/bin; do
    if [ "${PYBIN}" == "/opt/python/cp35-cp35m/bin" ]; then continue; fi
    "${PYBIN}/pip" install pylibkriging --no-index -f /data/dist
    (cd "$HOME"; "${PYBIN}/pytest" /data/bindings/Python/tests)
done