import os
import re
import shlex
import sys
import platform
import subprocess
import argparse
from packaging.version import Version
from check_requirements import *

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def main():
    """
    main() is a wrapper to the whole function so that it can be used as imported function by root setup.py
    """

    extra_libs = []
    if platform.system() == "Windows":
        # Try to find DLLs, make Fortran DLLs optional as they may not be available
        optional_dlls = ['flang.dll', 'flangrti.dll']
        extra_libs = [find_in_path(f, required=(f not in optional_dlls)) for f in ['flang.dll', 'flangrti.dll', 'libomp.dll', 'openblas.dll']]
        extra_libs = [lib for lib in extra_libs if lib is not None]

    argparser = argparse.ArgumentParser(add_help=False)
    argparser.add_argument('--debug', action="store_true", help='compile in debug mode')
    global args  # will available to CMakeBuild functions
    args, unknown = argparser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown

    libKriging_path = Path(__file__).absolute().parents[3]
    print(f"LibKriging root directory is {libKriging_path}")
    os.chdir(libKriging_path)

    # Distribution variant. "" (default): the `pylibkriging` package, whose
    # published wheels are CPU-only. "gpu": the `pylibkriging-gpu` source
    # distribution, compiled at install time with every GPU backend found on
    # the installing machine (ENABLE_GPU_ITERATIVE=AUTO). Selected by
    # LIBKRIGING_PY_VARIANT, or by a VARIANT file next to this setup.py:
    # tools/release/python-sdist-gpu.sh writes that file only for the time of
    # `setup.py sdist`, so it ships inside the pylibkriging-gpu sdist (which
    # then builds the gpu variant with no environment variable) and never
    # lingers in a checkout used to build the CPU wheels.
    global variant
    variant_file = Path(__file__).absolute().parent / "VARIANT"
    variant = os.environ.get("LIBKRIGING_PY_VARIANT")
    if variant is None and variant_file.exists():
        variant = variant_file.read_text().strip()
    variant = (variant or "").strip().lower()
    if variant not in ("", "gpu"):
        eprint(f"Unknown LIBKRIGING_PY_VARIANT '{variant}' (expected '' or 'gpu')")
        exit(1)
    with open("cmake/version.cmake", "r") as file:
        data = file.read()

    version_major = re.search(r"^set\(KRIGING_VERSION_MAJOR (\d+)\)$", data, re.M)
    version_minor = re.search(r"^set\(KRIGING_VERSION_MINOR (\d+)\)$", data, re.M)
    version_patch = re.search(r"^set\(KRIGING_VERSION_PATCH (\d+)\)$", data, re.M)
    kriging_version = f"{version_major.group(1)}.{version_minor.group(1)}.{version_patch.group(1)}"

    # Packages should be installed in global environment (seen by CMake/C++)
    if not has_requirements("requirements.txt"):
        eprint("Mandatory requirements are not satisfied")
        exit(1)

    if args.debug and not has_requirements("dev-requirements.txt"):
        eprint("Dev/debug requirements are not satisfied")
        exit(1)

    # PyPI project page: the package README (absolute links only: PyPI does not resolve relative ones)
    readme = Path(__file__).absolute().parent / "README.md"
    try:
        long_description = readme.read_text(encoding="utf-8")
    except OSError:
        long_description = 'Python support for libKriging, the kriging library for performance and wide language support'

    setup(
        name='pylibkriging-gpu' if variant == 'gpu' else 'pylibkriging',
        packages=['pylibkriging'],
        version=kriging_version,
        author='Pascal Havé',
        author_email='hpwxf@haveneer.com',
        url="https://github.com/libKriging/libKriging",
        description='Python binding for LibKriging',
        long_description=long_description,
        long_description_content_type="text/markdown",
        ext_modules=[CMakeExtension('pylibkriging', sourcedir=".")],
        cmdclass=dict(build_ext=CMakeBuild),
        script_name='./bindings/Python/pylibkriging/setup.py',
        package_dir={'pylibkriging': 'bindings/Python/pylibkriging/src/pylibkriging'},
        # https://docs.python.org/3/distutils/setupscript.html#installing-package-data
        package_data={'pylibkriging': []},
        # https://docs.python.org/3/distutils/setupscript.html#installing-additional-files
        data_files=[('lib/site-packages/pylibkriging/shared_libs', extra_libs)],
        python_requires='>=3.7',
        install_requires=get_requirements("requirements.txt"),  # they should be in C++ build environment
        extras_require={
            # pylibkriging.sklearn (scikit-learn compatible estimators): `pip install pylibkriging[sklearn]`
            'sklearn': ['scikit-learn'],
        },
        zip_safe=False,
    )


def find_in_path(filename, required=True):
    fpath, fname = os.path.split(filename)
    if fpath:
        if os.path.isfile(fpath):
            return filename
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            test_file = os.path.join(path, filename)
            if os.path.isfile(test_file):
                return test_file
    if required:
        raise RuntimeError(f"Cannot find required file '{filename}'")
    return None


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        cmake_version = Version(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < Version('3.1.0'):
            raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        print('extdir:', extdir)
        print('Python executable', sys.executable)

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DENABLE_PYTHON_BINDING=on',
                      '-DENABLE_OCTAVE_BINDING=off',
                      '-DENABLE_MATLAB_BINDING=off',
                      '-DENABLE_JULIA_BINDING=off',
                      '-DBUILD_SHARED_LIBS=off',
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      f'-DKRIGING_VERSION={self.distribution.get_version()}'
                      ]
        # GPU backends: ENABLE_GPU_ITERATIVE=AUTO detects every backend
        # (CUDA, HIP, SYCL, Metal) on the building machine -- the default of
        # the "gpu" variant; OFF forces a CPU-only build -- what the release
        # scripts export for the published wheels. Without either, CMake's
        # per-backend defaults apply (ENABLE_CUDA_ITERATIVE=AUTO, others OFF).
        gpu_mode = os.environ.get('ENABLE_GPU_ITERATIVE', 'AUTO' if variant == 'gpu' else '')
        if variant == 'gpu':
            # No auditwheel step bundles shared dependencies into a source
            # install: link armadillo/lbfgsb statically into the extension.
            cmake_args += ['-DSTATIC_LIB=on']
        if gpu_mode:
            cmake_args += ['-DENABLE_GPU_ITERATIVE=' + gpu_mode]
        for backend in ('CUDA', 'HIP', 'SYCL', 'METAL'):
            value = os.environ.get(f'ENABLE_{backend}_ITERATIVE')
            if value:
                cmake_args += [f'-DENABLE_{backend}_ITERATIVE={value}']
        # Any further CMake options, e.g. LIBKRIGING_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90"
        cmake_args += shlex.split(os.environ.get('LIBKRIGING_CMAKE_ARGS', ''))

        cfg = 'Debug' if args.debug else 'Release'
        print('build mode:', cfg)
        build_args = ['--config', cfg]
        if variant == 'gpu':
            # End-user install from source: only the extension (and the
            # libraries it needs), not the C++ tests and benchmarks.
            build_args += ['--target', '_pylibkriging']

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        print("version", self.distribution.get_version())

        env = os.environ.copy()
        env['CXXFLAGS'] = env.get('CXXFLAGS', '')
        # CMake >= 4 (what `pip install` may fetch as a build dependency)
        # rejects the pinned dependencies' cmake_minimum_required < 3.5.
        env.setdefault('CMAKE_POLICY_VERSION_MINIMUM', '3.5')

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print(f"environment variables: {env}")
        print(f"CMAKE arguments: {cmake_args}")
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


if __name__ == '__main__':
    main()
