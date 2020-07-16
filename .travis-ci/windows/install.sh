#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

if [[ "$MODE" == "Coverage" ]]; then
    echo "Coverage not supported for Windows"
    travis_terminate 1
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f ${BASEDIR})

if [ ! -f "$HOME/Miniconda3/condabin/conda.bat" ]; then
	curl -o ${HOME}/Downloads/Miniconda3-latest-Windows-x86_64.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
	cd ${HOME}/Downloads
	${BASEDIR}/install_conda.bat
	cd ${BASEDIR}
fi
$HOME/Miniconda3/condabin/conda.bat update -y -n base -c defaults conda

# https://anaconda.org/search?q=blas
$HOME/Miniconda3/condabin/conda.bat install -y -n base -c conda-forge openblas liblapack
