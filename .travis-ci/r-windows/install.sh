#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f ${BASEDIR})

${BASEDIR}/../windows/install.sh

# https://chocolatey.org/docs/commands-install
# https://chocolatey.org/packages/make
choco install -y make --version 4.2.1

# -c r : means "from 'r' channel"
# https://anaconda.org/r/r
# https://anaconda.org/r/rtools
${HOME}/Miniconda3/condabin/conda.bat install -y -c r r rtools
