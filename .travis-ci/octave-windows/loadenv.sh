# if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  # Using choco installer : should be load by env in .travis.yml
  # export PATH=/c/Python37:/c/Python37/Scripts:$PATH
    
  ## Using miniconda (Python 3.7 is already included in Miniconda3)
  #export PATH=${HOME}/Miniconda3:$PATH
# fi

# Octave MinGW (put first to ensure we use Octave's compiler toolchain)
export PATH=/c/ProgramData/Chocolatey/lib/octave.portable/tools/octave/mingw64/bin:${PATH}

# make
export PATH="/c/Program Files/make/make-4.3/bin":${PATH}
