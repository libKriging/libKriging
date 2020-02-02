#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

"${BASEDIR}"/../windows/install.sh

# https://chocolatey.org/docs/commands-install
# https://chocolatey.org/packages/make
choco install -y make --version 4.3

# Using chocolatey and manual command
# https://chocolatey.org/packages/R.Project
choco install -y r.project --version 3.6.0
if [ ! -f "C:/Rtools/VERSION.txt" ]; then
    # https://cran.r-project.org/bin/windows/Rtools
    curl -Lo ${HOME}/Downloads/Rtools.exe https://cran.r-project.org/bin/windows/Rtools/Rtools35.exe
    "${BASEDIR}"/install_rtools.bat
else
	echo "Rtools installation detected: nothing to do"
fi
export PATH="/c/Rtools/mingw_64/bin":$PATH

# For R packaging
choco install -y zip

if [[ "$DEBUG_CI" != true ]]; then
echo
echo 'Add following paths:'
echo 'export PATH="/c/Program Files/make/make-4.3/bin":$PATH'
echo 'export PATH="/c/Program Files/R/R-3.6.0/bin":$PATH'
echo 'export PATH="/c/Rtools/mingw_64/bin":$PATH'
fi

## Using Anaconda
## -c r : means "from 'r' channel"
## https://anaconda.org/r/r
## https://anaconda.org/r/rtools
#${HOME}/Miniconda3/condabin/conda.bat install -y -c r r rtools
#export PATH=$HOME/Miniconda3/Rtools/mingw_64/bin:$PATH
#if [[ "$DEBUG_CI" != true ]]; then
#echo
#echo 'Add following paths:'
#echo 'export PATH="/c/Program Files/make/make-4.3/bin":$PATH'
#echo 'export PATH=$HOME/Miniconda3/Scripts:$PATH'
#echo 'export PATH=$HOME/Miniconda3/Rtools/mingw_64/bin:$PATH'
#fi

if [[ "$DEBUG_CI" != true ]]; then
# We need an access to libomp.dll, flang.dll, flangrti.dll, openblas.dll
# located in ${HOME}/Miniconda3/Library/bin
echo 'export PATH=$HOME/Miniconda3/Library/bin:$PATH'
echo
fi

# Crazy hack since R try to call 'g++ ' as compiler
# Message looks like:
# /usr/bin/sh: line 8: g++ : command not found
#       with a space here ^
ln -sf "$(command -v g++).exe" "$(command -v g++) .exe"