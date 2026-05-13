#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)
test -f "${BASEDIR}"/loadenv.sh && . "${BASEDIR}"/loadenv.sh 

# Tests 56 & 64 (LinearAlgebra varying sizes / rapid fire) time out on Windows
export CTEST_EXCLUDE="safe_chol_lower - varying sizes|rapid fire varying sizes"

"${BASEDIR}"/../linux-macos/test.sh
