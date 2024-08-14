# if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  # Using choco installer : should be load by env in .travis.yml
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