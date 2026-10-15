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
loadenv_sh="tools/${BUILD_NAME}/loadenv.sh"
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

TMP_DIR=${BUILD_DIR:-build}/tmp
[ -d "${TMP_DIR}" ] && rm -fr "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

RVER=$(awk -F": " '/^Version:/ { print $2 }' "${PREFIX}"/DESCRIPTION)
 
# shorten tag to match R requirements.
GIT_TAG=${GIT_TAG//-*} # remove everything after first -

case $ARCH in
  Linux)
    ARCHZ="${ARCH}-$(uname -m)"
    tar xzvf "${PREFIX}_${RVER}_R_$(uname -m)-pc-linux-gnu.tar.gz" -C "${TMP_DIR}"
    cp -a ${BUILD_DIR:-build}/installed/lib/libarmadillo.so.* "${TMP_DIR}"/rlibkriging/libs/ || echo "No armadillo shared lib to copy"
    cp -a ${BUILD_DIR:-build}/installed/lib/libKriging.so.* "${TMP_DIR}"/rlibkriging/libs/ || echo "No libKriging shared lib to copy"
    # Bundle the runtime libraries that a base R install does not provide
    # (OpenBLAS and its Fortran runtime, #387): R itself already brings
    # libblas / liblapack / libgomp / libstdc++. They are found next to
    # rlibkriging.so through a $ORIGIN RUNPATH (set on each bundled library
    # too, as RUNPATH is not transitive).
    LIBS_DIR="${TMP_DIR}"/rlibkriging/libs
    # OpenBLAS: always bundle the pthreads build when the runner has it (Debian
    # layout), whatever variant the system alternatives select: an OpenMP
    # build (libopenblaso) is the most deadlock-prone in forked R workers (#325)
    OPENBLAS_PTHREAD_DIR=/usr/lib/$(gcc -print-multiarch 2>/dev/null || echo "$(uname -m)-linux-gnu")/openblas-pthread
    ldd "${LIBS_DIR}"/rlibkriging.so | awk '/=> \// { print $3 }' | grep -E '/lib(openblas|gfortran|quadmath)[^/]*\.so' | while read -r lib; do
      name=$(basename "$lib")
      if [[ "$name" == libopenblas* && -e "${OPENBLAS_PTHREAD_DIR}/${name}" ]]; then
        lib="${OPENBLAS_PTHREAD_DIR}/${name}"
      fi
      echo "Bundling $lib as $name"
      cp -L "$lib" "${LIBS_DIR}/${name}"
    done
    for so in "${LIBS_DIR}"/*.so*; do
      patchelf --set-rpath '$ORIGIN' "$so"
    done
    if ldd "${LIBS_DIR}"/rlibkriging.so | grep -q "not found"; then
      echo "Unresolved libraries in rlibkriging.so:"
      ldd "${LIBS_DIR}"/rlibkriging.so | grep "not found"
      exit 1
    fi
    RELEASE_FILE=${PREFIX}_${GIT_TAG#v}_${ARCHZ}.tgz
    tar czvf "${RELEASE_FILE}" -C "${TMP_DIR}" .
    ;;
  Darwin)
    MACOS_VERSION=$(/usr/libexec/PlistBuddy -c "Print:ProductVersion" /System/Library/CoreServices/SystemVersion.plist)
    ARCHZ="macOS${MACOS_VERSION}-$(uname -m)"
    tar xzvf "${PREFIX}_${RVER}.tgz" -C "${TMP_DIR}"
    cp -a ${BUILD_DIR:-build}/installed/lib/libarmadillo.*.dylib "${TMP_DIR}"/rlibkriging/libs/ || echo "Warning: no armadillo shared lib to copy"
    cp -a ${BUILD_DIR:-build}/installed/lib/libKriging.*.dylib "${TMP_DIR}"/rlibkriging/libs/ || echo "Warning: no libKriging shared lib to copy"
    RELEASE_FILE=${PREFIX}_${GIT_TAG#v}_${ARCHZ}.tgz
    tar czvf "${RELEASE_FILE}" -C "${TMP_DIR}" .
    ;;
  MSYS_NT*|MINGW64_NT*) # Windows
    ARCHZ="$(uname -s | awk -F- '{ print $1$2 }')-$(uname -m)" # remove trailing release
    unzip "${PREFIX}_${RVER}.zip" -d "${TMP_DIR}"
    cp -a ${BUILD_DIR:-build}/installed/bin/libarmadillo.dll "${TMP_DIR}"/rlibkriging/libs/x64/ || echo "Warning: no armadillo shared lib to copy"
    cp -a ${BUILD_DIR:-build}/installed/bin/libKriging.dll "${TMP_DIR}"/rlibkriging/libs/x64/ || echo "Warning: no libKriging shared lib to copy"
    for extralib in openblas.dll libomp.dll flangrti.dll flang.dll
    do
      cp "${HOME}"/Miniconda3/Library/bin/$extralib "${TMP_DIR}"/rlibkriging/libs/x64/ || echo "Warning: no $extralib lib to copy"
    done
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