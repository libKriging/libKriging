#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)
test -f "${BASEDIR}"/loadenv.sh && . "${BASEDIR}"/loadenv.sh 

export LIBKRIGING_PATH=${PWD}/${BUILD_DIR:-build}/installed
export PATH=${LIBKRIGING_PATH}/bin:${PATH}

cd bindings/R

# R_TEST_REPEAT: rerun `make test` up to N times, stopping at the first
# failure -- TEMPORARY, for the intermittent Windows segfault investigation
# (docs/dev/WindowsRSegfaultTestEstimNone.md). Unset/empty (the default)
# behaves exactly as before: one run.
N=${R_TEST_REPEAT:-1}
for i in $(seq 1 "$N"); do
  if [ "$N" -gt 1 ]; then
    echo "=== R test attempt $i/$N ==="
  fi
  if ! make test; then
    echo "make test failed on attempt $i/$N"
    # TEMPORARY: re-probe the crash on this same VM (see segfault-probe.sh)
    R_LIBS="$PWD/Rlibs" "${BASEDIR}"/../windows/segfault-probe.sh r "$PWD/rlibkriging/tests"
    exit 1
  fi
done
