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

if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  # ** Install python tools **
  
  ## Using miniconda (Python 3.7 is already included in Miniconda3)
  #export PATH=${HOME}/Miniconda3:$PATH
  
  ## Using Chocolatey (by default only includes Python2)
  choco install --no-progress -y python --version 3.7
  export PATH=/c/Python37:$PATH
  
  # ** Install PIP ** 
  
  ## Using miniconda3
  # https://anaconda.org/search?q=pip
  #$HOME/Miniconda3/condabin/conda.bat install -y -n base -c conda-forge pip openssl
  # error like: https://stackoverflow.com/questions/45954528/pip-is-configured-with-locations-that-require-tls-ssl-however-the-ssl-module-in
  # python -m pip install --progress-bar off pytest numpy scipy --upgrade
  
  ## Using Chocolatey
  #choco install --no-progress -y pip
  #python3 -m pip install --progress-bar off pip --upgrade
  
  ## By 'hand'
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py # curl already available with Chocolatey
  python get-pip.py
  
  # ** Install required Python libs ** 
  python -m pip install --progress-bar off pip --upgrade
  python -m pip install --progress-bar off pytest numpy scipy --upgrade
fi


if [[ "$ENABLE_OCTAVE_BINDING" == "on" ]]; then
  choco install -y --no-progress octave.portable
fi
