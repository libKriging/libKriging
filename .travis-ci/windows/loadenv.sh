if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  # Using choco installer
  export PATH=/c/Python37:$PATH
    
  ## Using miniconda (Python 3.7 is already included in Miniconda3)
  #export PATH=${HOME}/Miniconda3:$PATH
fi

# add OpenBLAS DLL library path
export PATH=$HOME/Miniconda3/Library/bin:$PATH