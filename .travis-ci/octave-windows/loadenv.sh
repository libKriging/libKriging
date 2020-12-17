# if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  # Using choco installer : should be load by env in .travis.yml
  # export PATH=/c/Python37:/c/Python37/Scripts:$PATH
    
  ## Using miniconda (Python 3.7 is already included in Miniconda3)
  #export PATH=${HOME}/Miniconda3:$PATH
# fi

# make
export PATH="/c/Program Files/make/make-4.3/bin":${PATH}

# Octave shortcut (and tools with absolute path) by Chocolatey
export PATH=/c/ProgramData/chocolatey/lib/octave.portable/tools/octave-5.2.0-w64/mingw64/bin/:$PATH
