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
python3 setup.py build --release
python3 setup.py build develop
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