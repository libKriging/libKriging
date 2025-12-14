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

# Install Miniconda for BLAS/LAPACK dependencies
if [ ! -f "$HOME/Miniconda3/condabin/conda.bat" ]; then
  echo "Installing Miniconda (provides BLAS/LAPACK)..."
  curl --insecure -s -o "${HOME}"/Downloads/Miniconda3-latest-Windows-x86_64.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
  pushd "${HOME}"/Downloads
  "${BASEDIR}"/../windows/install_conda.bat
  popd
fi
$HOME/Miniconda3/Scripts/conda.exe update -y -n base -c defaults conda
$HOME/Miniconda3/Scripts/conda.exe install -y --quiet -n base -c conda-forge openblas liblapack pkg-config

# https://chocolatey.org/docs/commands-install
# https://chocolatey.org/packages/make
choco install -y --no-progress make --version 4.3

if [[ "$ENABLE_OCTAVE_BINDING" == "on" ]]; then
  # select your version from https://community.chocolatey.org/packages/octave.portable#versionhistory
  # Don't forget to update loadenv.sh with the related path
  choco install -y --no-progress octave.portable --version=6.2.0
  
  # Note: We use Octave's built-in MinGW compiler instead of installing a separate mingw package
  # This avoids conflicts between different MinGW versions
  
  if [[ ! -e  /c/windows/system32/GLU32.DLL ]]; then
    # add missing GLU32.dll in travis-ci windows image
    # 64bit 10.0.14393.0	161.5 KB	U.S. English	OpenGL Utility Library DLL
    # found at https://fr.dllfile.net/microsoft/glu32-dll
    curl -s -o glu32.zip https://fr.dllfile.net/download/9439
    unzip glu32.zip
    mv glu32.dll /c/windows/system32/GLU32.DLL
  fi
fi
