#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
    echo "PATH=$PATH"

    echo "clang-format config: $(command -v clang-format)"
    clang-format --version | sed 's/^/  /'
fi