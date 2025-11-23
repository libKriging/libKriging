import platform
import sys

# Fix "cannot allocate memory in static TLS block" error on Linux/Unix
# This occurs because OpenMP (libgomp/libomp) uses static TLS which has limited space
# when loaded via dlopen(). We preload it using RTLD_GLOBAL before importing the module.
if platform.system() != "Windows":
    import ctypes
    import os
    import glob

    # Try to preload OpenMP library (libgomp or libomp) with RTLD_GLOBAL
    # This ensures its TLS is allocated in the initial TLS block
    openmp_loaded = False

    # First, try common library paths with full paths
    common_paths = [
        '/usr/lib/x86_64-linux-gnu/libgomp.so.1',
        '/usr/lib/x86_64-linux-gnu/libgomp.so',
        '/usr/lib/aarch64-linux-gnu/libgomp.so.1',
        '/usr/lib/aarch64-linux-gnu/libgomp.so',
        '/usr/lib64/libgomp.so.1',
        '/usr/lib64/libgomp.so',
        '/usr/local/lib/libomp.so',
        '/usr/lib/libomp.so',
    ]

    # Also try to find via glob patterns
    glob_patterns = [
        '/usr/lib/*/libgomp.so*',
        '/usr/lib*/libgomp.so*',
        '/usr/local/lib/libomp.so*',
    ]

    for pattern in glob_patterns:
        common_paths.extend(glob.glob(pattern))

    for lib_path in common_paths:
        if os.path.exists(lib_path):
            try:
                # RTLD_GLOBAL (0x00100 | 0x00002) makes symbols available to subsequently loaded libraries
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                openmp_loaded = True
                break  # Successfully loaded
            except OSError:
                continue  # Try next library

    # Fallback: try by name only (relies on LD_LIBRARY_PATH)
    if not openmp_loaded:
        openmp_libs = ['libgomp.so.1', 'libgomp.so', 'libomp.so']
        for lib_name in openmp_libs:
            try:
                ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
                openmp_loaded = True
                break  # Successfully loaded
            except OSError:
                continue  # Try next library name
else:  # Windows
    import os

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
