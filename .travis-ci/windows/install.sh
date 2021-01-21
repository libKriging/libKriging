#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

if [[ "${ENABLE_COVERAGE}" == "on" ]]; then
    echo "Coverage not supported for Windows"
    travis_terminate 1
fi

if [ ! -f "$HOME/Miniconda3/condabin/conda.bat" ]; then
	curl -s -o ${HOME}/Downloads/Miniconda3-latest-Windows-x86_64.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
	cd ${HOME}/Downloads
	${BASEDIR}/install_conda.bat
	cd ${BASEDIR}
fi
$HOME/Miniconda3/condabin/conda.bat update -y -n base -c defaults conda

# https://anaconda.org/search?q=blas
$HOME/Miniconda3/condabin/conda.bat install -y --quiet -n base -c conda-forge openblas liblapack

. ${BASEDIR}/loadenv.sh

if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  # ** Install python tools **
  
  ## Using Chocolatey (by default only includes Python2)
  # should be installed using choco in main .travis.yml
  
  # ** Install PIP if not yet available **   
  if ( ! python3 -m pip --version 2>/dev/null ); then
    ## Using miniconda3
    # https://anaconda.org/search?q=pip
    #$HOME/Miniconda3/condabin/conda.bat install -y -n base -c conda-forge pip openssl
    # error like: https://stackoverflow.com/questions/45954528/pip-is-configured-with-locations-that-require-tls-ssl-however-the-ssl-module-in

    ## Using Chocolatey
    #choco install --no-progress -y pip
    #python3 -m pip install --progress-bar off pip --upgrade

    ## By 'hand'
    curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py # curl already available with Chocolatey
    python3 get-pip.py
  fi
  
  # ** Install required Python libs ** 
  python3 -m pip install --progress-bar off pip --upgrade
  python3 -m pip install --progress-bar off pytest numpy scipy --upgrade
fi
