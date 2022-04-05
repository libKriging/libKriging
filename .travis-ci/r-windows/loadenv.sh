# make
export PATH="/c/Program Files/make/make-4.3/bin":${PATH}

REXE=$(which R 2>/dev/null) || REXE=$(find "/c/Program Files" -wholename "*/bin/x64/R.exe")
RDIR=$(dirname "$REXE")
export PATH="$RDIR":$PATH

# Using R from Anaconda
## R shortcut (and tools) by Anaconda
#export PATH=${HOME}/Miniconda3/Scripts:${PATH}

# In all cases, we need an access to libomp.dll, flang.dll, flangrti.dll, openblas.dll
export PATH=${HOME}/Miniconda3/Library/bin:$PATH
