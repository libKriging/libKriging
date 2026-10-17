#!/bin/bash
# Phase 4: one fit per (variant, seed), the variants interleaved seed by seed
# so that partial results stay comparable.
# Usage: run-phase4.sh <out.csv> <bin> <class> <optim> "<n list>" <seeds> <variant>...
#   variant = label:libdir[:VAR=value,...]  (libdir goes to LD_LIBRARY_PATH)
OUT=$1; BIN=$2; CLS=$3; OPTIM=$4; NS=$5; SEEDS=$6; shift 6
[ -f "$OUT" ] || echo "variant,mode,class,optim,n,d,seed,blas_before,blas_after,seconds,cpu_seconds,ll" > "$OUT"
for n in $NS; do
  for s in $(seq 1 "$SEEDS"); do
    for v in "$@"; do
      IFS=: read -r label libdir envs <<< "$v"
      line=$(env ${envs//,/ } LD_LIBRARY_PATH="$libdir" timeout 3600 "$BIN" fit "$CLS" "$OPTIM" "$n" 4 "$s" 2>/dev/null | grep '^fit,')
      [ -n "$line" ] || line="fit,$CLS,$OPTIM,$n,4,$s,,,FAIL,,"
      echo "$label,$line" >> "$OUT"
    done
  done
done
