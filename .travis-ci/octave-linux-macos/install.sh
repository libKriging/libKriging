#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

# to get readlink on MacOS (no effect on Linux)
if [[ -e /usr/local/opt/coreutils/libexec/gnubin/readlink ]]; then
    export PATH=/usr/local/opt/coreutils/libexec/gnubin:$PATH
fi
BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

"${BASEDIR}"/../linux-macos/install.sh

case "$(uname -s)" in
   Darwin)
     brew install octave gcc@7
     ;;

   Linux)
     sudo apt-get install -y software-properties-common
     sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key 291F9FF6FD385783
     sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
     sudo apt-get install -y cmake # requires cmake ≥3.13 for target_link_options
     test -d /usr/local/cmake-3.12.4 && sudo mv /usr/local/cmake-3.12.4 /usr/local/cmake-3.12.4.old # Overrides Travis installation
     # octave 4 is installer using travis
     ;;

   *)
     echo 'Unknown OS'
     exit 1 
     ;;
esac
