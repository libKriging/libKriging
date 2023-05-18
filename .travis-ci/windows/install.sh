#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)
test -f "${BASEDIR}"/loadenv.sh && . "${BASEDIR}"/loadenv.sh 

if [[ "${ENABLE_COVERAGE}" == "on" ]]; then
    echo "Coverage not supported for Windows"
    travis_terminate 1
fi

if [ ! -f "$HOME/Miniconda3/condabin/conda.bat" ]; then
	curl -s -o ${HOME}/Downloads/Miniconda3-latest-Windows-x86_64.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
	pushd ${HOME}/Downloads 
	"${BASEDIR}"/install_conda.bat
	popd
fi
$HOME/Miniconda3/condabin/conda.bat update -y -n base -c defaults conda

# https://anaconda.org/search?q=blas
$HOME/Miniconda3/condabin/conda.bat install -y --quiet -n base -c conda-forge openblas liblapack
# $HOME/Miniconda3/condabin/conda.bat install -c conda-forge fortran-compiler

# https://chocolatey.org/docs/commands-install
# required to compile fortran part
choco install mingw
choco install -y --no-progress make --version 4.3

if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  # Check if python is available (it could be a python wrapper given by Windows)
  if ( ! python -V | grep "Python 3." &>/dev/null ); then 
    echo "#########################################################"
    echo "Missing Python interpreter"
    echo "Go to https://www.python.org/downloads and install it"
    echo 'or `choco install -y --no-progress python3 --version 3.9`'
    echo "Don't forget to add python executable in PATH"
    echo "#########################################################"
    exit 1
  fi
  # ** Install python tools **
  
  ## Using Chocolatey (by default only includes Python2)
  # should be installed using choco in main .travis.yml
  
  # ** Install PIP if not yet available **   
  if ( ! python -m pip --version 2>/dev/null ); then
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
  python -m pip install --progress-bar off pip --upgrade
  python -m pip install --progress-bar off -r bindings/Python/requirements.txt --upgrade
  python -m pip install --progress-bar off -r bindings/Python/dev-requirements.txt --upgrade
fi
