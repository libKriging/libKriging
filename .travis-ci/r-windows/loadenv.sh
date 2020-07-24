# make
export PATH="/c/Program Files/make/make-4.3/bin":${PATH}


# Using R from Chocolatey
# R shortcut (and tools with absolute path) by Chocolatey
export PATH="/c/Program Files/R/R-3.6.0/bin":$PATH
# Required to disambiguate with Chocolatey GCC installation
export PATH="/c/Rtools/mingw_64/bin":$PATH


# Using R from Anaconda
## R shortcut (and tools) by Anaconda
#export PATH=${HOME}/Miniconda3/Scripts:${PATH}
#export PATH=$HOME/Miniconda3/Rtools/mingw_64/bin:$PATH


# In all cases, we need an access to libomp.dll, flang.dll, flangrti.dll, openblas.dll
export PATH=${HOME}/Miniconda3/Library/bin:$PATH
