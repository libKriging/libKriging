#!/usr/bin/env bash
set -eo pipefail

if [ "${GITHUB_ACTIONS:=false}" == "false" ]; then
  
  # ~/.R/Makevars will override values in $(R_HOME}/etc$(R_ARCH)/Makeconf
  # Rscript -e 'print(R.home(component="etc"))'
  # Rscript -e 'Sys.getenv("R_HOME")'
  # Rscript -e 'Sys.getenv("R_ARCH")'
  mkdir -p ~/.R
  
  # Could be updated after install; cannot fail
  REXE=$(which R 2>/dev/null) || REXE=$(find "/c/Program Files" -wholename "*/bin/x64/R.exe")
  RVERSION=$("$REXE" --version 2>&1 | grep -o "R version [[:digit:]]\.[[:digit:]]")
  
  # https://cran.r-project.org/bin/windows/Rtools
  case "$RVERSION" in
    "R version 3.3"|"R version 3.4"|"R version 3.5"|"R version 3.6")
      RTOOLSDIR=Rtools
      ;;
    "R version 4.0"|"R version 4.1")    
      RTOOLSDIR=Rtools40
      ;;
    "R version 4.2")    
      RTOOLSDIR=Rtools42
      ;;
    *)
      echo "Cannot found Rtools related to $RVERSION"
      exit 1
      ;;
  esac
  
  echo "# Custom configuration" > ~/.R/Makevars
  echo "BINPREF=C:/${RTOOLSDIR}/mingw64/bin/" >> ~/.R/Makevars
  echo 'SHLIB_CXXLD=$(BINPREF)g++' >> ~/.R/Makevars
  # R CMD config SHLIB_CXXLD # updated config

fi