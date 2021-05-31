#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

"${BASEDIR}"/../windows/install.sh

# https://chocolatey.org/docs/commands-install
# https://chocolatey.org/packages/make
choco install -y --no-progress make --version 4.3

if [ "${TRAVIS}" == "true" ]; then
  # Using chocolatey and manual command
  # https://chocolatey.org/packages/R.Project
  choco install -y --no-progress r.project --version 4.0.0
  if [ ! -f "C:/Rtools/VERSION.txt" ]; then
      # https://cran.r-project.org/bin/windows/Rtools
      curl -s -Lo ${HOME}/Downloads/Rtools.exe https://cran.r-project.org/bin/windows/Rtools/Rtools35.exe
      "${BASEDIR}"/install_rtools.bat
  else
    echo "Rtools installation detected: nothing to do"
  fi
elif [ "${GITHUB_ACTIONS}" == "true" ]; then
  # Already installed
#  if [ ! -f "C:/Rtools/VERSION.txt" ]; then
#      # https://cran.r-project.org/bin/windows/Rtools
#      curl -s -Lo ${HOME}/Downloads/Rtools.exe https://cran.r-project.org/bin/windows/Rtools/rtools40-x86_64.exe
#      "${BASEDIR}"/install_rtools.bat
#  else
#    echo "Rtools installation detected: nothing to do"
#  fi
  : # do nothing
else
  echo "Unknown CI environment"
  exit 1
fi



# For R packaging
choco install -y --no-progress zip

. ${BASEDIR}/loadenv.sh

## Using Anaconda
## -c r : means "from 'r' channel"
## https://anaconda.org/r/r
## https://anaconda.org/r/rtools
#${HOME}/Miniconda3/condabin/conda.bat install -y -c r r rtools


# Crazy hack since R try to call 'g++ ' as compiler
# Message looks like:
# /usr/bin/sh: line 8: g++ : command not found
#       with a space here ^
ln -sf "$(command -v g++).exe" "$(command -v g++) .exe"
# should be replaced using ~/.R/Makevars config
