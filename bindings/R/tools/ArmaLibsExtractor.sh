#!/usr/bin/env bash
set -eo pipefail

if [[ ! -d "${LIBKRIGING_PATH}" ]]; then
  >&2 echo "$0 error:"
  >&2 echo "libKriging should be installed into \$LIBKRIGING_PATH directory first"
  exit 1
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

TMPDIR=$(mktemp -d)
ARMA_LIBS=$(cmake -DLIBKRIGING_PATH="${LIBKRIGING_PATH}" -B "${TMPDIR}" "${BASEDIR}" | sed -n -r -e 's/^-- ARMA_LIBS=(.*)$/\1/p')
echo "${ARMA_LIBS}"
rm -fr "${TMPDIR}"