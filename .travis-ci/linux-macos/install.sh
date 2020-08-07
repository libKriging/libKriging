#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

if [[ "$ENABLE_COVERAGE" == "on" ]]; then
    gem install coveralls-lcov
fi

if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  if ( command -v python3 >/dev/null 2>&1 && 
       python3 -m pip --version >/dev/null 2>&1 ); then
    # python3 -m pip install pip # --upgrade # --progress-bar off
    python3 -m pip install pytest numpy # --upgrade # --progress-bar off
  fi
fi

if [[ "$ENABLE_OCTAVE_BINDING" == "on" ]]; then
  case "$(uname -s)" in
   Darwin)
     # using brew in .travis-ci.yml is too slow or fails with "update: true"
     brew install octave
     ;;

   Linux)
     sudo apt-get install -y software-properties-common
     sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key 291F9FF6FD385783
     sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
     sudo apt-get install -y cmake # requires cmake ≥3.13 for target_link_options
     test -d /usr/local/cmake-3.12.4 && sudo mv /usr/local/cmake-3.12.4 /usr/local/cmake-3.12.4.old # Overrides Travis installation
     # octave 4 is installed using travis
     ;;

   *)
     echo 'Unknown OS'
     exit 1 
     ;;
  esac
fi
