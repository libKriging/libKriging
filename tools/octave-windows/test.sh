#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)
test -f "${BASEDIR}"/loadenv.sh && . "${BASEDIR}"/loadenv.sh 

# Tests (LinearAlgebra varying sizes / rapid fire) time out on Windows
export CTEST_EXCLUDE="varying sizes|rapid fire"

"${BASEDIR}"/../linux-macos/test.sh
