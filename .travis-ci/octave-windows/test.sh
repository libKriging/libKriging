#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

export PATH=/c/ProgramData/chocolatey/lib/octave.portable/tools/octave-5.2.0-w64/mingw64/bin/:$PATH
"${BASEDIR}"/../linux-macos/test.sh
