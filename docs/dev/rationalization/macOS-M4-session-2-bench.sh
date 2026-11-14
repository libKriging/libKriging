#!/usr/bin/env bash
# Session 2 timings (#472 sections 2, 3, 5). Waits for mains power and an idle machine, then runs every
# measurement; pauses again if the power is unplugged. Raw CSV next to this file.
# usage: macOS-M4-session-2-bench.sh <out dir> <progress log>
set -uo pipefail
OUT=${1:?out dir}; LOG=${2:?log}
mkdir -p "$OUT"
bdir() { # build name -> its bench directory (bash 3.2 on macOS has no associative arrays)
  case "$1" in
    seq) echo /Users/richet/lk-m4/build-seq/bench;; omp) echo /Users/richet/lk-m4/build-omp/bench;;
    omp200) echo /Users/richet/lk-m4/build-omp200/bench;; omp50) echo /Users/richet/lk-m4/build-omp50/bench;;
    gcd) echo /Users/richet/lk-gcd/build-gcd/bench;; gcd200) echo /Users/richet/lk-gcd/build-gcd200/bench;;
    nocheck) echo /Users/richet/lk-nocheck/build-nocheck/bench;;
    *) echo "unknown build $1" >&2; return 1;;
  esac
}
ROUNDS=${ROUNDS:-5}
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
# ALLOW_BATTERY=1: measure on battery (the conditions file records it); otherwise wait for mains power
ac_ok() { [ -n "${ALLOW_BATTERY:-}" ] || pmset -g batt | head -1 | grep -q "AC Power"; }
# REQUIRE_LPM_OFF=1: do not measure while macOS Low Power Mode is on (pmset -g: lowpowermode 1)
lpm_ok() { [ -z "${REQUIRE_LPM_OFF:-}" ] || [ "$(pmset -g | awk '/lowpowermode/{print $2}')" = "0" ]; }
idle_ok() {
  local idle; idle=$(top -l 2 -s 3 -n 0 | grep "CPU usage" | tail -1 | sed -E 's/.* ([0-9.]+)% idle.*/\1/')
  awk -v i="${idle:-0}" 'BEGIN{exit !(i>=88)}'
}
wait_ready() {
  local msg=0
  until ac_ok && lpm_ok && idle_ok; do
    if [ $msg -eq 0 ]; then log "waiting: AC=$(ac_ok && echo yes || echo NO) lowpower_off=$(lpm_ok && echo yes || echo NO) idle_ok=$(idle_ok && echo yes || echo NO)"; msg=1; fi
    sleep 30
  done
  [ $msg -eq 1 ] && log "ready"
}
snapshot() { { echo "## $1 $(date -u +%FT%TZ)"; pmset -g batt | head -2; pmset -g therm 2>&1 | head -3; pmset -g | grep -i lowpowermode; } >> "$OUT/macOS-M4-session-2-conditions.txt"; }
# time -l: "  real  user  sys" on the first line
timed() { # timed <tag> cmd...  -> sets T_REAL T_CPU T_FIT (the tag is only a label)
  shift
  local tf; tf=$(mktemp); local of; of=$(mktemp)
  /usr/bin/time -l "$@" > "$of" 2> "$tf"
  local l; l=$(grep -m1 " real " "$tf"); T_REAL=$(echo "$l" | awk '{print $1}'); T_CPU=$(echo "$l" | awk '{print $3+$5}')
  T_FIT=$(grep -m1 "^Total:" "$of" | awk '{print $2}'); rm -f "$tf" "$of"
}

: > "$OUT/macOS-M4-session-2-conditions.txt"
log "start (waiting for AC power and an idle machine)"; wait_ready; snapshot start

# ---------- A. per evaluation (section 2, Q4 break-even, Q10 baseline) ----------
HDR=$("$(bdir seq)/bench-threads" 10 2 1 matern5_2 --header | head -1)
F="$OUT/macOS-M4-session-2-eval.csv"; echo "build,round,$HDR" > "$F"
for r in $(seq 1 $ROUNDS); do wait_ready
  for n in 100 150 200 400 800 1500; do for b in seq gcd gcd200 omp omp200 omp50; do
    ac_ok || wait_ready
    echo "$b,$r,$("$(bdir $b)/bench-threads" $n 4 9 matern5_2 | tail -1)" >> "$F"
  done; done; log "A round $r done"
done; snapshot after-A

# ---------- B. full fits (section 2): wall (Total), process wall, CPU ----------
F="$OUT/macOS-M4-session-2-fits.csv"; echo "build,round,n,optim,fit_ms,proc_real_s,proc_cpu_s" > "$F"
for r in $(seq 1 $ROUNDS); do wait_ready
  for b in seq gcd gcd200 omp omp200; do   # baseline process cost (n = 10): data, warm-up busy-wait, start-up
    ac_ok || wait_ready; timed x "$(bdir $b)/bench-fit-profile" 10 4 BFGS; echo "$b,$r,10,BFGS,$T_FIT,$T_REAL,$T_CPU" >> "$F"; done
  for n in 200 500 1000 1500; do for o in BFGS BFGS4; do for b in seq gcd gcd200 omp omp200; do
    ac_ok || wait_ready; timed x "$(bdir $b)/bench-fit-profile" $n 4 $o; echo "$b,$r,$n,$o,$T_FIT,$T_REAL,$T_CPU" >> "$F"
  done; done; done; log "B round $r done"
done; snapshot after-B

# ---------- C. Q9: 4 threads (P-cores) vs 10 ----------
F="$OUT/macOS-M4-session-2-q9-pcores.csv"; echo "build,round,n,optim,percent,fit_ms,proc_real_s,proc_cpu_s" > "$F"
for r in $(seq 1 $ROUNDS); do wait_ready
  for n in 500 1500; do for o in BFGS BFGS4; do for p in 40 100; do for b in omp gcd200 seq; do
    ac_ok || wait_ready; LK_NUM_THREADS_PERCENT=$p timed x "$(bdir $b)/bench-fit-profile" $n 4 $o; echo "$b,$r,$n,$o,$p,$T_FIT,$T_REAL,$T_CPU" >> "$F"
  done; done; done; done; log "C round $r done"
done; snapshot after-C

# ---------- D. Q10: KMP_BLOCKTIME=0 (libomp spin), per evaluation ----------
F="$OUT/macOS-M4-session-2-q10-blocktime.csv"; echo "build,blocktime,round,$HDR" > "$F"
for r in $(seq 1 $ROUNDS); do wait_ready
  for n in 100 150 200 400 800; do for bt in default 0; do
    ac_ok || wait_ready
    if [ $bt = default ]; then row=$("$(bdir omp200)/bench-threads" $n 4 9 matern5_2 | tail -1); else row=$(KMP_BLOCKTIME=0 "$(bdir omp200)/bench-threads" $n 4 9 matern5_2 | tail -1); fi
    echo "omp200,$bt,$r,$row" >> "$F"
  done; done; log "D round $r done"
done; snapshot after-D

# ---------- E. Q5 complement: BFGS10 / n = 1500, VECLIB_MAXIMUM_THREADS=1 ----------
F="$OUT/macOS-M4-session-2-q5-veclib.csv"; echo "build,round,n,optim,veclib,fit_ms,proc_real_s,proc_cpu_s" > "$F"
for r in $(seq 1 $ROUNDS); do wait_ready
  for n in 800 1500; do for o in BFGS BFGS4 BFGS10; do for v in default 1; do for b in seq omp; do
    ac_ok || wait_ready
    if [ $v = default ]; then timed x "$(bdir $b)/bench-fit-profile" $n 4 $o; else VECLIB_MAXIMUM_THREADS=1 timed x "$(bdir $b)/bench-fit-profile" $n 4 $o; fi
    echo "$b,$r,$n,$o,$v,$T_FIT,$T_REAL,$T_CPU" >> "$F"
  done; done; done; done; log "E round $r done"
done; snapshot after-E

# ---------- F. Q8 follow-up: cost of the potri check in a full fit ----------
F="$OUT/macOS-M4-session-2-q8-potri-check.csv"; echo "build,round,n,optim,fit_ms,proc_real_s,proc_cpu_s" > "$F"
for r in $(seq 1 $ROUNDS); do wait_ready
  for n in 800 1500; do for o in BFGS BFGS4; do for b in seq nocheck; do
    ac_ok || wait_ready; timed x "$(bdir $b)/bench-fit-profile" $n 4 $o; echo "$b,$r,$n,$o,$T_FIT,$T_REAL,$T_CPU" >> "$F"
  done; done; done; log "F round $r done"
done; snapshot end
log "ALL DONE"; touch "$OUT/.bench-done"
