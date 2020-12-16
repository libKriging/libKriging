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

case $BUILD_NAME in
r-*)
  RVER=$(awk -F": " '/^Version:/ { print $2 }' ./bindings/R/rlibkriging/DESCRIPTION)
  PREFIX=bindings/R/rlibkriging
  [ -d deploy ] && rm -fr deploy
  mkdir deploy

  case $ARCH in
    Linux)
      tar xzvf "${PREFIX}_${RVER}_R_x86_64-pc-linux-gnu.tar.gz" -C deploy 
      cp -a build/installed/lib/libarmadillo.so.* deploy/rlibkriging/libs/
      cp -a build/installed/lib/libKriging.so.* deploy/rlibkriging/libs/
      DEPLOY_FILE=${PREFIX}_${ARCH}_${TAG}.tgz
      tar czvf "${DEPLOY_FILE}" -C deploy .
      ;;
    Darwin)
      tar xzvf "${PREFIX}_${RVER}.tgz" -C deploy 
      cp -a build/installed/lib/libarmadillo.*.dylib deploy/rlibkriging/libs/
      cp -a build/installed/lib/libKriging.*.dylib deploy/rlibkriging/libs/
      DEPLOY_FILE=${PREFIX}_${ARCH}_${TAG}.tgz
      tar czvf "${DEPLOY_FILE}" -C deploy .
      ;;
    MSYS_NT*) # Windows
      ARCH=Windows10
      unzip "${PREFIX}_${RVER}.zip" -d deploy
      cp -a build/installed/lib/libarmadillo.dll deploy/rlibkriging/libs/x64/
      cp -a build/installed/lib/libKriging.dll deploy/rlibkriging/libs/x64/
      DEPLOY_FILE=${PREFIX}_${ARCH}_${TAG}.zip
      (cd deploy && zip -FS -r "../${DEPLOY_FILE}" rlibkriging)
      find -name "*rlibkriging*" -type f
      ;;
    *)
      echo "Unknown OS [$ARCH]"
      ;;
  esac
  ;;
*)
  echo "Build [${BUILD_NAME}] not packaged"
  ;;
esac

if [ -e "$DEPLOY_FILE" ]; then
  echo "$DEPLOY_FILE" > ./DEPLOY_FILE
  echo "File $DEPLOY_FILE ready to deploy"
fi
