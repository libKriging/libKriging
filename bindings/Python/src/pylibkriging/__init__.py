import platform

if platform.system() == "Windows":
    import os
    import sys

    shared_lib_path = os.path.join(os.path.dirname(__file__), 'shared_libs')  # cf setup.py

    # alternative method if lib/site-packages prefix is not reliable (requires update of setup.py)
    # import distutils  # https://docs.python.org/3/distutils/apiref.html#module-distutils.sysconfig
    # shared_lib_path = os.path.join(distutils.sysconfig.PREFIX, 'pylibkriging', 'shared_libs')

    if sys.version_info[:2] < (3, 8):  # < 3.8.0
        sys.path.append(shared_lib_path)
    else:
        os.add_dll_directory(shared_lib_path)

from _pylibkriging import *
from _pylibkriging import __version__, __build_type__
