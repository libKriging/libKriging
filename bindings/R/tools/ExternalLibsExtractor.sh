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
ARMA_LIBS=$(cmake -DLIBKRIGING_PATH="${LIBKRIGING_PATH}" -B "${TMPDIR}" "${BASEDIR}" | sed -n -r -e 's/^-- EXTERNAL_LIBS=(.*)$/\1/p')
# avoid *-NOTFOUND to be displayed:
ARMA_LIBS=$(echo "$ARMA_LIBS" | sed -e 's/[^ ]*-NOTFOUND[^ ]*//g')
echo "${ARMA_LIBS}"
rm -fr "${TMPDIR}"
