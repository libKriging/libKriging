#!/bin/sh

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

cd ${BASEDIR}/../.git/hooks
ln -sf ../../.git-hooks/pre-commit.py pre-commit
