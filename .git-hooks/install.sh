#!/bin/sh

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)

cd "${BASEDIR}"/../.git/hooks
ln -sf ../../.git-hooks/pre-commit.py pre-commit
