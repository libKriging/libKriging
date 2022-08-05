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
