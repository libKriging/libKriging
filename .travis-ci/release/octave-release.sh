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

PREFIX=build/bindings/Octave

PACKAGE_DIR=octave-package
[ -d "${PACKAGE_DIR}" ] && rm -fr "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}"

case $ARCH in
  Linux)
    ARCHZ="${ARCH}-$(uname -m)"
    ;;
  Darwin)
    MACOS_VERSION=$(/usr/libexec/PlistBuddy -c "Print:ProductVersion" /System/Library/CoreServices/SystemVersion.plist)
    ARCHZ="macOS${MACOS_VERSION}-$(uname -m)"
    ;;
  MSYS_NT*|MINGW64_NT*) # Windows
    ARCHZ="$(uname -s | awk -F- '{ print $1$2 }')-$(uname -m)" # remove trailing release
    ;;
  *)
    echo "Unknown OS [$ARCH]"
    exit 1
    ;;
esac

RELEASE_FILE="${PACKAGE_DIR}"/mLibKriging_${GIT_TAG#v}_${ARCHZ}.tgz
tar czvf "${RELEASE_FILE}" -C build/installed/bindings/Octave .
