#!/usr/bin/env bash
# TEMPORARY -- Windows segfault investigation, see
# docs/dev/WindowsRSegfaultTestEstimNone.md (section E).
#
# Called right after a failing test run, ON THE SAME VM (the crash has been
# shown to be deterministic per VM: a "bad" VM crashes the same tests every
# time, a "good" VM never does). Re-runs one known crasher under environment
# variants that discriminate between the remaining hypotheses, plus once
# under cdb to get the faulting module/stack from the first-chance access
# violation (before Catch2's or R's own handlers swallow it).
#
# Usage:
#   segfault-probe.sh core <build_dir>   # C++ Catch2 tests (Julia/Python jobs)
#   segfault-probe.sh r    <tests_dir>   # R binding (bindings/R/rlibkriging/tests)
#
# Never fails: it only prints. Grep the job log for "PROBE " for the summary.

set +e
MODE_PROBE=$1
DIR=$2

summary=()

run_variant() {
  # run_variant <label> <env assignments...> -- <cmd...>
  local label=$1
  shift
  local envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    envs+=("$1")
    shift
  done
  shift
  echo "::group::PROBE variant: ${label}"
  echo "env: ${envs[*]:-<none>}"
  echo "cmd: $*"
  env "${envs[@]}" "$@"
  local rc=$?
  echo "::endgroup::"
  echo "PROBE ${label}: exit ${rc}"
  summary+=("${label}: exit ${rc}")
}

find_cdb() {
  local c
  for c in "/c/Program Files (x86)/Windows Kits/10/Debuggers/x64/cdb.exe" \
    "/c/Program Files/Windows Kits/10/Debuggers/x64/cdb.exe"; do
    if [[ -x "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  command -v cdb.exe 2>/dev/null
}

# cdb: -g/-G skip initial/final breakpoints, -o follows child processes
# (Rscript.exe re-launches Rterm.exe). An access violation breaks on first
# chance by default, so the -c commands run at the faulting instruction.
CDB_CMDS='.echo PROBE_CDB_BREAK; .lastevent; .exr -1; r; kb 60; .echo PROBE_CDB_MODULES; lm; q'

echo "PROBE environment: NUMBER_OF_PROCESSORS=${NUMBER_OF_PROCESSORS:-?} OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-<unset>}"
powershell.exe -NoProfile -Command \
  "Get-CimInstance Win32_Processor | Format-List Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,L2CacheSize,L3CacheSize,MaxClockSpeed" 2>/dev/null

case "$MODE_PROBE" in
core)
  cd "$DIR" || exit 0
  # T1 crashes on bad VMs even single-threaded (Python Debug jobs, which run
  # with OMP_NUM_THREADS=1); T2 only crashed in multi-threaded jobs.
  T1_EXE=bin/KrigingNystromTest.exe
  T1_ARG="LLNystrom analytic gradient matches finite differences"
  T2_EXE=bin/LinearAlgebraTest.exe
  T2_ARG="LinearAlgebra - rapid fire varying sizes"
  for t in 1 2; do
    exe_var=T${t}_EXE
    arg_var=T${t}_ARG
    exe=${!exe_var}
    arg=${!arg_var}
    if [[ ! -f "$exe" ]]; then
      echo "PROBE T${t}: $exe not found, skipped"
      continue
    fi
    run_variant "T${t} baseline (verbose)" OPENBLAS_VERBOSE=2 -- "$exe" "$arg"
    run_variant "T${t} OPENBLAS_NUM_THREADS=1" OPENBLAS_NUM_THREADS=1 OPENBLAS_VERBOSE=2 -- "$exe" "$arg"
    run_variant "T${t} OMP_NUM_THREADS=1" OMP_NUM_THREADS=1 OPENBLAS_VERBOSE=2 -- "$exe" "$arg"
    run_variant "T${t} OPENBLAS_CORETYPE=Haswell" OPENBLAS_CORETYPE=Haswell OPENBLAS_VERBOSE=2 -- "$exe" "$arg"
    run_variant "T${t} OPENBLAS_CORETYPE=SkylakeX" OPENBLAS_CORETYPE=SkylakeX OPENBLAS_VERBOSE=2 -- "$exe" "$arg"
    run_variant "T${t} OPENBLAS_CORETYPE=Prescott" OPENBLAS_CORETYPE=Prescott OPENBLAS_VERBOSE=2 -- "$exe" "$arg"
  done
  CDB=$(find_cdb)
  if [[ -n "$CDB" && -f "$T1_EXE" ]]; then
    run_variant "T1 under cdb" -- "$CDB" -g -G -o -c "$CDB_CMDS" "$T1_EXE" "$T1_ARG"
  else
    echo "PROBE cdb not found (or T1 missing), skipped"
  fi
  ;;
r)
  cd "$DIR" || exit 0
  # The R crash was mis-attributed to test-estimnone.R: on a passing run the
  # next file testthat runs is test-KrigingCholCrash.R, whose first output
  # only comes after an n=1000 fit -- i.e. the first sizeable linear algebra
  # of the whole R suite. R1 runs that file alone; R2 isolates its n=1000
  # fit; R3 is the same fit at n=150 (Nystrom-test size).
  cat >probe_fit.R <<'EOF'
library(rlibkriging)
f <- function(X) apply(X, 1, function(x) prod(sin(2*pi*(x*(seq(0,1,l=1+length(x))[-1])^2))))
n <- as.integer(Sys.getenv("PROBE_N", "1000"))
set.seed(1234)
X <- matrix(runif(n*3), ncol=3)
y <- f(X)
rlibkriging:::linalg_check_chol_rcond(FALSE)
r <- try(Kriging(y, X, "gauss", regmodel="constant", normalize=FALSE, optim="BFGS", objective="LL"))
cat("PROBE_R fit n =", n, "returned class", class(r), "\n")
EOF
  run_variant "R1 test-KrigingCholCrash.R alone" -- Rscript -e "library(testthat); library(rlibkriging); test_file('testthat/test-KrigingCholCrash.R')"
  run_variant "R2 fit n=1000 baseline (verbose)" OPENBLAS_VERBOSE=2 -- Rscript probe_fit.R
  run_variant "R3 fit n=150" PROBE_N=150 -- Rscript probe_fit.R
  run_variant "R2 OPENBLAS_NUM_THREADS=1" OPENBLAS_NUM_THREADS=1 -- Rscript probe_fit.R
  run_variant "R2 OMP_NUM_THREADS=1" OMP_NUM_THREADS=1 -- Rscript probe_fit.R
  run_variant "R2 OPENBLAS_CORETYPE=Haswell" OPENBLAS_CORETYPE=Haswell -- Rscript probe_fit.R
  run_variant "R2 OPENBLAS_CORETYPE=SkylakeX" OPENBLAS_CORETYPE=SkylakeX -- Rscript probe_fit.R
  run_variant "R2 OPENBLAS_CORETYPE=Prescott" OPENBLAS_CORETYPE=Prescott -- Rscript probe_fit.R
  CDB=$(find_cdb)
  if [[ -n "$CDB" ]]; then
    run_variant "R2 under cdb" -- "$CDB" -g -G -o -c "$CDB_CMDS" "$(command -v Rscript)" probe_fit.R
  else
    echo "PROBE cdb not found, skipped"
  fi
  rm -f probe_fit.R
  ;;
*)
  echo "usage: $0 core|r <dir>"
  ;;
esac

echo "PROBE ===== summary (${MODE_PROBE}) ====="
for s in "${summary[@]}"; do
  echo "PROBE ${s}"
done
exit 0
