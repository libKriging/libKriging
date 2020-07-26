#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

. ${BASEDIR}/loadenv.sh

"${BASEDIR}"/../linux-macos/test.sh
