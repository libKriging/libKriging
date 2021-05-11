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

if [[ "$ENABLE_OCTAVE_BINDING" == "on" ]]; then
  # select your version from https://community.chocolatey.org/packages/octave.portable#versionhistory
  # Don't forget to update loadenv.sh with the related path
  choco install -y --no-progress octave.portable --version 5.2.0.1
  if [[ ! -e  /c/windows/system32/GLU32.DLL ]]; then
    # add missing GLU32.dll in travis-ci windows image
    # 64bit 10.0.14393.0	161.5 KB	U.S. English	OpenGL Utility Library DLL
    # found at https://fr.dllfile.net/microsoft/glu32-dll
    curl -s -o glu32.zip https://fr.dllfile.net/download/9439
    unzip glu32.zip
    mv glu32.dll /c/windows/system32/GLU32.DLL
  fi
fi
