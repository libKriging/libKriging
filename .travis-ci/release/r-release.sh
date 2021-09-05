#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

if [ "$MODE" != "Release" ]; then
  echo  "Release mode is required for packaging"
  exit 1
fi

# windows environment requires to load special tools
loadenv_sh=".travis-ci/${BUILD_NAME}/loadenv.sh"
if [ -e "$loadenv_sh" ]; then
  . "$loadenv_sh"
fi

ARCH=$(uname -s)
echo "Ready to release for $ARCH"

if [ -z "${GIT_TAG}" ]; then
  echo "No valid version tag found" 
  exit 1
fi
echo "Release tag '${GIT_TAG}' in branch '$(git branch --show-current)'"

PREFIX="${PWD}"/bindings/R/rlibkriging

TMP_DIR=build/tmp
[ -d "${TMP_DIR}" ] && rm -fr "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

RVER=$(awk -F": " '/^Version:/ { print $2 }' "${PREFIX}"/DESCRIPTION)

case $ARCH in
  Linux)
    ARCHZ="${ARCH}-$(uname -m)"
    tar xzvf "${PREFIX}_${RVER}_R_$(uname -m)-pc-linux-gnu.tar.gz" -C "${TMP_DIR}"
    cp -a build/installed/lib/libarmadillo.so.* "${TMP_DIR}"/rlibkriging/libs/ || echo "No armadillo shared lib to copy"
    cp -a build/installed/lib/libKriging.so.* "${TMP_DIR}"/rlibkriging/libs/ || echo "No libKriging shared lib to copy"
    RELEASE_FILE=${PREFIX}_${GIT_TAG#v}_${ARCHZ}.tgz
    tar czvf "${RELEASE_FILE}" -C "${TMP_DIR}" .
    ;;
  Darwin)
    MACOS_VERSION=$(/usr/libexec/PlistBuddy -c "Print:ProductVersion" /System/Library/CoreServices/SystemVersion.plist)
    ARCHZ="macOS${MACOS_VERSION}-$(uname -m)"
    tar xzvf "${PREFIX}_${RVER}.tgz" -C "${TMP_DIR}"
    cp -a build/installed/lib/libarmadillo.*.dylib "${TMP_DIR}"/rlibkriging/libs/ || echo "Warning: no armadillo shared lib to copy"
    cp -a build/installed/lib/libKriging.*.dylib "${TMP_DIR}"/rlibkriging/libs/ || echo "Warning: no libKriging shared lib to copy"
    RELEASE_FILE=${PREFIX}_${GIT_TAG#v}_${ARCHZ}.tgz
    tar czvf "${RELEASE_FILE}" -C "${TMP_DIR}" .
    ;;
  MSYS_NT*|MINGW64_NT*) # Windows
    ARCHZ="$(uname -s | awk -F- '{ print $1$2 }')-$(uname -m)" # remove trailing release
    unzip "${PREFIX}_${RVER}.zip" -d "${TMP_DIR}"
    cp -a build/installed/lib/libarmadillo.dll "${TMP_DIR}"/rlibkriging/libs/x64/ || echo "Warning: no armadillo shared lib to copy"
    cp -a build/installed/lib/libKriging.dll "${TMP_DIR}"/rlibkriging/libs/x64/ || echo "Warning: no libKriging shared lib to copy"
    # RELEASE_FILE=${PREFIX}_${ARCHZ}_${GIT_TAG#v}.zip
    RELEASE_FILE="${PREFIX}_${GIT_TAG#v}.zip" # cannot rename for Windows
    (cd "${TMP_DIR}" && zip -FS -r "${RELEASE_FILE}" rlibkriging)
    ;;
  *)
    echo "Unknown OS [$ARCH]"
    exit 1
    ;;
esac

PACKAGE_DIR=R-package
[ -d "${PACKAGE_DIR}" ] && rm -fr "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}"

mv "${RELEASE_FILE}" "${PACKAGE_DIR}"  