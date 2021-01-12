#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

if [[ "$ENABLE_COVERAGE" == "on" ]]; then
    sudo gem install coveralls-lcov
fi

if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  if ( command -v python3 >/dev/null 2>&1 && 
       python3 -m pip --version >/dev/null 2>&1 ); then
    # python3 -m pip install pip # --upgrade # --progress-bar off
    python3 -m pip install pytest numpy # --upgrade # --progress-bar off
  fi
  
  if [[ "$ENABLE_MEMCHECK" == "on" ]]; then 
    python3 -m pip install wheel # required ton compile pytest-valgrind
    python3 -m pip install pytest-valgrind
  fi
fi

if [[ "$ENABLE_COVERAGE" == "on" ]] && [[ "$ENABLE_MEMCHECK" == "on" ]]; then
  echo "Mixing coverage mode and memcheck is not supported"
  exit 1
fi

if [[ "$ENABLE_OCTAVE_BINDING" == "on" ]]; then
  case "$(uname -s)" in
   Darwin)
     # using brew in .travis-ci.yml is too slow or fails with "update: true"
     brew install octave
     ;;

   Linux)
     if [ "${TRAVIS}" == "true" ]; then
       # add kitware server signature cf https://apt.kitware.com       
       sudo apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common
       curl -s https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
       
       sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
       sudo apt-get install -y cmake # requires cmake ≥3.13 for target_link_options
       test -d /usr/local/cmake-3.12.4 && sudo mv /usr/local/cmake-3.12.4 /usr/local/cmake-3.12.4.old # Overrides Travis installation
       # octave 4 is installed using packager
     fi
     ;;

   *)
     echo 'Unknown OS'
     exit 1 
     ;;
  esac
fi
