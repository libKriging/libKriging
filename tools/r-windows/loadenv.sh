# make
export PATH="/c/Program Files/make/make-4.3/bin":${PATH}

# R helper
if ( ! command -v R >/dev/null 2>&1 ); then
  R_EXE=$(find "/c/Program Files" -wholename "*/bin/x64/R.exe" |& grep -v "Permission denied") || true
  if [ -z "$R_EXE" ]; then
    echo "Cannot find R executable" >&2
    exit 1
  fi
  R_DIR=$(dirname "$R_EXE")
  export PATH="$R_DIR":$PATH
fi


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

# Using R from Anaconda
## R shortcut (and tools) by Anaconda
#export PATH=${HOME}/Miniconda3/Scripts:${PATH}

# In all cases, we need an access to libomp.dll, flang.dll, flangrti.dll, openblas.dll
export PATH=${HOME}/Miniconda3/Library/bin:$PATH

# Belt and braces for OpenBLAS 0.3.34 (OpenMathLib/OpenBLAS#6013/#6021, see
# the 0.3.33 pin in tools/windows/install.sh): its Zen 4 GEMM blocking
# override only fires when L2 reads as 1024 KB, and overflows the stack of
# dgemm_kernel_ZEN/HASWELL on AVX-512 Zen 4/5 hosts. Claiming 2 MB disables it.
export OPENBLAS_L2_SIZE=2048
