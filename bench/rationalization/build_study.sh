#!/bin/sh
# Builds a study driver of this directory with the flags of bench-fit-profile:
#   build_study.sh <build-dir> <driver.cpp> <output binary>
set -e
b=$(cd "$1" && pwd); src=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); out=$3
flags=$(python3 - "$b" <<'PY'
import json, shlex, sys
c = json.load(open(sys.argv[1] + '/compile_commands.json'))
e = [x for x in c if x['file'].endswith('bench-fit-profile.cpp')][0]
a = shlex.split(e['command'])[1:]
keep, skip = [], False
for t in a:
    if skip: skip = False; continue
    if t in ('-o', '-c'): skip = True; continue
    if t.endswith('.cpp'): continue
    keep.append(t)
print(' '.join(shlex.quote(t) for t in keep))
PY
)
eval c++ $flags -c "$src" -o "$out.o"
c++ -O2 "$out.o" -o "$out" -L"$b/src/lib" -lKriging -Wl,-rpath,"$b/src/lib" -fopenmp
rm -f "$out.o"
