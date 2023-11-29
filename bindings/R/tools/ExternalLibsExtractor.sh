#!/usr/bin/env bash
set -eo pipefail

if [[ ! -d "${LIBKRIGING_PATH}" ]]; then
  >&2 echo "$0 error:"
  >&2 echo "libKriging should be installed into \$LIBKRIGING_PATH directory first"
  exit 1
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)

TMPDIR=$(mktemp -d)

if [ -z "${CMAKE_BIN}" ]; then
  CMAKE_BIN=cmake
fi

ARMA_LIBS=$(${CMAKE_BIN} -DLIBKRIGING_PATH="${LIBKRIGING_PATH}" -B "${TMPDIR}" "${BASEDIR}" | sed -n -r -e 's/^-- EXTERNAL_LIBS=(.*)$/\1/p')
## avoid *-NOTFOUND to be displayed:
ARMA_LIBS=$(echo "$ARMA_LIBS" | sed -e 's/[^ ]*-NOTFOUND[^ ]*//g')

# Add double quotes around each path to protect them except when it starts with a - which should be a (composed) option like "-framework Accelerate"
echo "${ARMA_LIBS}" | while read l
  do
    if [[ "$l" == "-"* ]]; then
      echo -n " $l"
    else
      echo -n " \"$l\"";
    fi
  done
rm -fr "${TMPDIR}"
