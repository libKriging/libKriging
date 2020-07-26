#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

# to get readlink on MacOS (no effect on Linux)
if [[ -e /usr/local/opt/coreutils/libexec/gnubin/readlink ]]; then
    export PATH=/usr/local/opt/coreutils/libexec/gnubin:$PATH
fi
BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

"${BASEDIR}"/../linux-macos/install.sh
