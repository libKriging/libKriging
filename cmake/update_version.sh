#!/usr/bin/env bash
set -eo pipefail

if [ $# -ne 3 ]; then
  echo "libKriging version updater"
  echo "Usage: $0 MAJOR MINOR PATCH"
  exit 1
fi

BASEDIR=$(git rev-parse --show-toplevel)

KRIGING_VERSION_MAJOR=$1
KRIGING_VERSION_MINOR=$2
KRIGING_VERSION_PATCH=$3
DATE=$(date +'%Y-%m-%d')

# Update main libKriging version
cat > "${BASEDIR}"/cmake/version.cmake <<EOF
# VERSION must follow pattern : <major>.<minor>.<patch>
set(KRIGING_VERSION_MAJOR $KRIGING_VERSION_MAJOR)
set(KRIGING_VERSION_MINOR $KRIGING_VERSION_MINOR)
set(KRIGING_VERSION_PATCH $KRIGING_VERSION_PATCH)
set(KRIGING_VERSION "\${KRIGING_VERSION_MAJOR}.\${KRIGING_VERSION_MINOR}.\${KRIGING_VERSION_PATCH}")
EOF

# Update rlibKriging version
sed -i.bak -e "s/^Version: .*$/Version: ${KRIGING_VERSION_MAJOR}.${KRIGING_VERSION_MINOR}-${KRIGING_VERSION_PATCH}/" \
  "${BASEDIR}"/bindings/R/rlibkriging/DESCRIPTION
sed -i.bak -e "s/^Date: .*$/Date: $DATE/" \
  "${BASEDIR}"/bindings/R/rlibkriging/DESCRIPTION

# Update pylibKriging version in tests
sed -i.bak -e "s/m.__version__ == .*$/m.__version__ == '${KRIGING_VERSION_MAJOR}.${KRIGING_VERSION_MINOR}.${KRIGING_VERSION_PATCH}'/" \
  "${BASEDIR}"/bindings/Python/tests/loading_test.py

sed -i.bak -e "s/pkgs=\"rlibkriging_.*\.tgz\"/pkgs=\"rlibkriging_${KRIGING_VERSION_MAJOR}.${KRIGING_VERSION_MINOR}-${KRIGING_VERSION_PATCH}.tgz\"/" \
  "${BASEDIR}"/bindings/R/rlibkriging/tests/testthat/notest-LinearRegression.R

git commit -m "build: update version to ${KRIGING_VERSION_MAJOR}.${KRIGING_VERSION_MINOR}.${KRIGING_VERSION_PATCH}" \
  "${BASEDIR}"/cmake/version.cmake \
  "${BASEDIR}"/bindings/R/rlibkriging/DESCRIPTION \
  "${BASEDIR}"/bindings/Python/tests/loading_test.py \
  "${BASEDIR}"/bindings/R/rlibkriging/tests/testthat/notest-LinearRegression.R
  
git tag "v${KRIGING_VERSION_MAJOR}.${KRIGING_VERSION_MINOR}.${KRIGING_VERSION_PATCH}"