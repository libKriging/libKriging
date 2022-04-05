#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

"${BASEDIR}"/../windows/install.sh

# https://chocolatey.org/docs/commands-install
# https://chocolatey.org/packages/make
choco install -y --no-progress make --version 4.3

# Allow to fail
REXE=$(which R 2>/dev/null) || REXE=$(find "/c/Program Files" -wholename "*/bin/x64/R.exe") || true

if [ -z "$REXE" ]; then
  if [ "${TRAVIS}" == "true" ]; then
    # Using chocolatey and manual command
    # https://chocolatey.org/packages/R.Project
    choco install -y --no-progress r.project --version 4.0.0
  else
    echo "Missing R program"
    echo "Don't know how to install it"
    exit 1
  fi
fi

# Could be updated after install; cannot fail
REXE=$(which R 2>/dev/null) || REXE=$(find "/c/Program Files" -wholename "*/bin/x64/R.exe")
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
    RTOOLSURL=https://cran.r-project.org/bin/windows/Rtools/rtools42/files/rtools42-5111-5107.exe
    RTOOLSDIR=Rtools42
    ;;
  *)
    echo "Cannot found Rtools related to $RVERSION"
    exit 1
    ;;
esac

if [ ! -d /c/$RTOOLSDIR ]; then
  if [ -d /c/Rtools ]; then
    echo "Conflicting existing /c/Rtools directory; aborting"
    exit 1
  fi
  curl -s -Lo "${HOME}"/Downloads/Rtools.exe "$RTOOLSURL" 
  "${BASEDIR}"/install_rtools.bat
  mv /c/Rtools /c/$RTOOLSDIR
fi

# For R packaging
choco install -y --no-progress zip

. ${BASEDIR}/loadenv.sh

## Using Anaconda
## -c r : means "from 'r' channel"
## https://anaconda.org/r/r
## https://anaconda.org/r/rtools
#${HOME}/Miniconda3/condabin/conda.bat install -y -c r r rtools
