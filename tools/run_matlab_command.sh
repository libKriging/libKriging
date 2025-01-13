#!/bin/sh
# Executes a specified MATLAB command in batch mode. With MATLAB R2018b+,
# running this script is equivalent to executing "matlab -batch command" from
# the system prompt.
#
# Copyright 2020 The MathWorks, Inc.

# This file content comes from the following GHA's CI script
# - name: pre-install-matlab
#   uses: matlab-actions/setup-matlab@v1
#   if: matrix.enable_matlab == 'on'
# - name: pre-install-matlab2
#   uses: matlab-actions/run-command@v1
#   with:
#     command: disp(fileread("/home/runner/work/_actions/matlab-actions/run-command/v1/dist/bin/run_matlab_command.sh")), exit

usage() {
  echo ''
  echo '    Usage: run_matlab_command.sh command'
  echo ''
  echo '    command       - MATLAB script, statement, or function to execute.'
  echo ''
}

ver_less_than() {
  [ "$(ver_str "$1")" -lt "$(ver_str "$2")" ]
}

ver_str() {
  echo "$@" | awk -F. '{ printf("%d%03d%03d%09d\n", $1,$2,$3,$4); }'
}

command=$1
if [ -z "$command" ]; then
  usage
  exit 1
fi

if ! matlab_path="$(command -v matlab)" || [ -z "$matlab_path" ]; then
  echo "'matlab'"' command not found. Please make sure MATLAB_ROOT/bin is on'
  echo 'the system path, where MATLAB_ROOT is the full path to your MATLAB'
  echo 'installation directory.'
  exit 1
fi

# resolve symlink to target
while [ -h "$matlab_path" ]; do
  dir=$(dirname -- "$matlab_path")
  target=$(readlink "$matlab_path")
  matlab_path=$(cd "$dir" && cd "$(dirname -- "$target")" && pwd)/$(basename -- "$target")
done

matlab_root=$(dirname -- "$(dirname -- "$matlab_path")")

# try to discover the MATLAB version
if [ -f "$matlab_root"/VersionInfo.xml ]; then
  # get version tag contents
  matlab_ver=$(sed -n 's:.*<version>\(.*\)</version>.*:\1:p' <"$matlab_root"/VersionInfo.xml)
elif [ -f "$matlab_root"/toolbox/matlab/general/Contents.m ]; then
  # get version printed after "MATLAB Version"
  matlab_ver=$(grep -o 'MATLAB Version .*' <"$matlab_root"/toolbox/matlab/general/Contents.m | awk -v N=3 '{print $N}')
fi

# if version not discovered, assume worst-case version of 0
matlab_ver=${matlab_ver:-0}

# use -r to launch MATLAB versions below R2018b (i.e. 9.5), otherwise use -batch
if ver_less_than "$matlab_ver" '9.5'; then
  # define start-up options
  opts='-nosplash -nodesktop'
  if ! ver_less_than "$matlab_ver" '8.6'; then
    opts="$opts -noAppIcon"
  fi

  # escape single quotes in command
  exp=$(echo "$command" | sed "s/'/''/g")

  LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 matlab "$opts" -r "try,eval('$exp'),catch e,disp(getReport(e,'extended')),exit(1),end,exit"
else
  LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 matlab -batch "$command"
fi
