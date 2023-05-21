import platform

if platform.system() == "Windows":
    import os
    import sys

    shared_lib_paths = [os.path.join(os.path.dirname(__file__), 'shared_libs')]  # cf setup.py
    lk_path = os.environ.get("LIBKRIGING_DLL_PATH")
    if lk_path:
        for path in lk_path.split(os.pathsep):
            shared_lib_paths.append(path)

    # alternative method if lib/site-packages prefix is not reliable (requires update of setup.py)
    # import distutils  # https://docs.python.org/3/distutils/apiref.html#module-distutils.sysconfig
    # shared_lib_path = os.path.join(distutils.sysconfig.PREFIX, 'pylibkriging', 'shared_libs')

    if sys.version_info[:2] < (3, 8):  # < 3.8.0
        for path in shared_lib_paths:
            if os.path.isdir(path):
                os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
    else:
        for path in shared_lib_paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)

from _pylibkriging import *
from _pylibkriging import __version__, __build_type__

# Type alias to switch to the right binding
Kriging = WrappedPyKriging
NuggetKriging = WrappedPyNuggetKriging
NoiseKriging = WrappedPyNoiseKriging
LinearRegression = PyLinearRegression


def anykriging_load(filename):
    str_filename = str(filename)
    kind = anykriging_describe(str_filename)
    if kind == KrigingType.Kriging:
        return Kriging.load(str_filename)
    elif kind == KrigingType.NoiseKriging:
        return NoiseKriging.load(str_filename)
    elif kind == KrigingType.NuggetKriging:
        return NuggetKriging.load(str_filename)
    else:
        raise TypeError("Unknown Kriging type")
