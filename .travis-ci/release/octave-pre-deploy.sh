#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

if [ "$MODE" != "Release" ]; then
  exit 0
fi

if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
    cd "${TRAVIS_BUILD_DIR}" || exit 1
fi

ARCH=$(uname -s)

#TAG=$(git describe --tags --match "v[0-9]*" --exact-match)
TAG=$(echo "$TRAVIS_TAG" | grep -E '^v[[:digit:]]+\.[[:digit:]]+')
if [ -z "$TAG" ]; then
  echo "No version tag: skip deploy" 
  exit 0 
fi
echo "Release tag '$TAG' in branch '$TRAVIS_BRANCH'"
echo "API_DEPLOY_TOKEN=$API_DEPLOY_TOKEN"

PREFIX=build/bindings/Octave
[ -d deploy ] && rm -fr deploy
mkdir deploy

cp -a build/installed/bindings/Octave/* deploy
DEPLOY_FILE=${PREFIX}_${ARCH}_${TAG}.tgz
tar czvf "${DEPLOY_FILE}" -C deploy .

if [ -e "$DEPLOY_FILE" ]; then
  echo "$DEPLOY_FILE" > ./DEPLOY_FILE
  echo "File $DEPLOY_FILE ready to deploy"
fi
