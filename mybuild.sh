#!/bin/bash
export BUILD_TEST=true
export ENABLE_OCTAVE_BINDING=OFF
export ENABLE_PYTHON_BINDING=OFF
#export USE_COMPILER_CACHE=true

play -n synth 0.1 sine 150 vol 0.5
.travis-ci/r-linux-macos/build.sh
if [[ $? != 0 ]]; then 
  play -n synth 0.1 sine 450 vol 0.5
else
  R -e "install.packages(list.files('./bindings/R/',pattern='.tar.gz',full=TRUE),repos=NULL)"
  play -n synth 0.1 sine 250 vol 0.5
fi