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
OUT=`cmake -DLIBKRIGING_PATH="${LIBKRIGING_PATH}" -B "${TMPDIR}" "${BASEDIR}"`
ARMA_LIBS=$(echo "$OUT" | sed -n -r -e 's/^-- EXTERNAL_LIBS=(.*)$/\1/p')
rm -fr "${TMPDIR}"
