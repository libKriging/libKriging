#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)

"${BASEDIR}"/../windows/install.sh

# https://chocolatey.org/docs/commands-install
# https://chocolatey.org/packages/make
choco install -y --no-progress make --version 4.3

if [ "${GITHUB_ACTIONS:=false}" == "false" ]; then

  # Allow to fail
  REXE=$(which R 2>/dev/null) || REXE=$(find "/c/Program Files" -wholename "*/bin/x64/R.exe" |& grep -v "Permission denied") || true
  
  if [ -z "$REXE" ]; then
    echo "Missing R program"
    echo "Don't know how to install it"
    echo "You can try something like:"
    # https://chocolatey.org/packages/R.Project
    echo "\tchoco install -y --no-progress r.project --version 4.0.0"
    exit 1
  fi
  
  # Could be updated after install; cannot fail
  REXE=$(which R 2>/dev/null) || REXE=$(find "/c/Program Files" -wholename "*/bin/x64/R.exe" |& grep -v "Permission denied") || true
  if [ -z "$REXE" ]; then
    echo "Cannot find R executable" >&2
    exit 1
  fi
  RVERSION=$("$REXE" --version 2>&1 | grep -o "R version [[:digit:]]\.[[:digit:]]")
  
  # https://cran.r-project.org/bin/windows/Rtools
  case "$RVERSION" in
    "R version 3.3"|"R version 3.4"|"R version 3.5"|"R version 3.6")
      RTOOLSURL=https://cran.r-project.org/bin/windows/Rtools/Rtools35.exe
      RTOOLSDIR=Rtools
      ;;
    "R version 4.0"|"R version 4.1")    
      RTOOLSURL=https://cran.r-project.org/bin/windows/Rtools/rtools40-x86_64.exe
      RTOOLSDIR=Rtools40
      ;;
    "R version 4.2")    
      RTOOLSURL=https://cran.r-project.org/bin/windows/Rtools/rtools42/files/rtools42-5168-5107.exe
      RTOOLSDIR=Rtools42
      ;;
    *)
      echo "Cannot found Rtools related to $RVERSION"
      exit 1
      ;;
  esac
  
  if [ ! -d /c/$RTOOLSDIR ]; then
    curl -s -Lo "${HOME}"/Downloads/Rtools.exe "$RTOOLSURL" 
    "${BASEDIR}"/install_rtools.bat $RTOOLSDIR
  fi
fi

# For R packaging
choco install -y --no-progress zip

test -f "${BASEDIR}"/loadenv.sh && . "${BASEDIR}"/loadenv.sh 

