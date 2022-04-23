#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(cd $(dirname "$0") ; pwd -P)
test -f ${BASEDIR}/loadenv.sh && . ${BASEDIR}/loadenv.sh 

"${BASEDIR}"/../linux-macos/install.sh
