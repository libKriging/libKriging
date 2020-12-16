# pylibkriging, Python binding for LibKriging

This Python binding is built using [pybind11](https://github.com/pybind/pybind11) module.

## Prerequisites

* A compiler with C++11 support
* CMake > =3.1 

## Installation

With the `setup.py` file included, the `pip3 install` command will
invoke CMake and build the pybind11 module as specified in `CMakeLists.txt`.

## Test call

```shell script
python3 setup.py build --debug
python3 setup.py build # release if not --debug
python3 setup.py build develop
python3 setup.py sdist bdist_wheel
```

and then

```python
import pylibkriging
pylibkriging.add(1, 2)
```

To test after a rebuild
```python
import importlib
import pylibkriging as lk
importlib.reload(lk)
``` 

```shell
PYTHONPATH=../../../build/bindings/Python python3 -i -c "exec(open('LinearRegression.py').read())"
```

Deployment / upload:
````
# avec token in ~/.pypirc
pip3 install twine
python3 -m twine upload --repository pylibkriging dist/*
````


# Linux wheel build




Cf:
* https://github.com/pypa/manylinux
* https://github.com/pypa/python-manylinux-demo
* https://www.python.org/dev/peps/pep-0513/#rationale 
```
docker run -it --rm -v "$PWD":"/data" quay.io/pypa/manylinux2014_x86_64 /bin/bash
cd /data
/opt/python/cp36-cp36m/bin/python3 bindings/Python/setup.py sdist bdist_wheel
```