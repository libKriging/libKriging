# if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  # Using choco installer : should be loaded by env in CI workflow
  # export PATH=/c/Python37:/c/Python37/Scripts:$PATH
    
  ## Using miniconda (Python 3.7 is already included in Miniconda3)
  #export PATH=${HOME}/Miniconda3:$PATH
# fi

# CMake helper
if ( ! command -v cmake >/dev/null 2>&1 ); then
  CMAKE_EXE=$(find "/c/Program Files/CMake" -name "cmake.exe" |& grep -v "Permission denied") || true
  if [ -z "$CMAKE_EXE" ]; then
    echo "Cannot find CMake executable" >&2
    exit 1
  fi
  CMAKE_DIR=$(dirname "$CMAKE_EXE")
  export PATH="$CMAKE_DIR":$PATH
fi

# add OpenBLAS & cie DLL libraries to search path for CMake
export PATH=${HOME}/Miniconda3/Library/bin:$PATH

# Manage in __init__.py the loading of extra dlls
# (not loaded by default from PATH since Python ≥3.8)
# An alternative could be to reuse PATH either in LIBKRIGING_DLL_PATH or __init__.py.
export LIBKRIGING_DLL_PATH=${HOME}/Miniconda3/Library/bin

# Belt and braces for OpenBLAS 0.3.34 (OpenMathLib/OpenBLAS#6013/#6021, see
# the 0.3.33 pin in tools/windows/install.sh): its Zen 4 GEMM blocking
# override only fires when L2 reads as 1024 KB, and overflows the stack of
# dgemm_kernel_ZEN/HASWELL on AVX-512 Zen 4/5 hosts. Claiming 2 MB disables it.
export OPENBLAS_L2_SIZE=2048
