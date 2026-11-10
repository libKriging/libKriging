#!/usr/bin/env bash
# CI helper (GitHub-hosted Ubuntu runners): apt-get update / install with a time
# limit per attempt and retries, switching away from the runner's default
# mirror after a failed or timed-out attempt.
#
# usage: tools/linux-macos/apt-install.sh [--update] [apt-get options] [packages...]
#   --update            run apt-get update first (alone: update only)
#   apt-get options     passed through, e.g. --no-install-recommends
# environment:
#   APT_ATTEMPT_TIMEOUT seconds per attempt (default 600)
#   APT_ATTEMPTS        number of attempts (default 3)
#
# Why: the Azure apt mirror of the runners is sometimes so slow (one package
# every 2-30 s) that a single install used the whole job time limit
# (e.g. octave in the Coverage job of #441, 2026-11-06).

set -uo pipefail

update=0
opts=()
pkgs=()
for a in "$@"; do
  case "$a" in
    --update) update=1 ;;
    -*) opts+=("$a") ;;
    *) pkgs+=("$a") ;;
  esac
done

attempt_timeout=${APT_ATTEMPT_TIMEOUT:-600}
attempts=${APT_ATTEMPTS:-3}
apt_opts=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30
  -o DPkg::Lock::Timeout=120)

switch_mirror() {
  # GitHub runners resolve "mirror+file:/etc/apt/apt-mirrors.txt", whose first
  # entry is the Azure mirror; put the main Ubuntu archive first instead.
  if [ -f /etc/apt/apt-mirrors.txt ]; then
    printf '%s\n' "http://archive.ubuntu.com/ubuntu/" "http://security.ubuntu.com/ubuntu/" |
      sudo tee /etc/apt/apt-mirrors.txt >/dev/null
  fi
  for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
    [ -f "$f" ] && sudo sed -i 's#http://azure\.archive\.ubuntu\.com/ubuntu#http://archive.ubuntu.com/ubuntu#g' "$f"
  done
}

run() {
  # timeout runs as root, inside sudo: it signals apt-get itself. With timeout
  # outside sudo, the SIGKILL hit sudo only, and the orphaned apt-get kept
  # /var/lib/apt/lists/lock, so every retry failed on the lock
  # (Thread Sanitizer and julia jobs of #449, 2026-11-09).
  sudo timeout --kill-after=30 "$attempt_timeout" env DEBIAN_FRONTEND=noninteractive "$@"
}

release_apt() {
  # Kill any apt-get / dpkg left by a timed-out attempt, then wait (60 s at
  # most) until no process holds the apt and dpkg locks.
  sudo pkill -KILL -x apt-get 2>/dev/null || true
  sudo pkill -KILL -f '^/usr/lib/apt/methods/' 2>/dev/null || true
  local locks=(/var/lib/apt/lists/lock /var/lib/dpkg/lock /var/lib/dpkg/lock-frontend /var/cache/apt/archives/lock)
  for ((w = 0; w < 60; w++)); do
    sudo fuser "${locks[@]}" >/dev/null 2>&1 || return 0
    sleep 1
  done
  echo "::warning::apt locks still held after 60 s: $(sudo fuser -v "${locks[@]}" 2>&1 | tr '\n' ' ')"
}

for ((i = 1; i <= attempts; i++)); do
  ok=1
  if [ "$update" -eq 1 ]; then
    run apt-get "${apt_opts[@]}" update -q || ok=0
  fi
  if [ "$ok" -eq 1 ] && [ "${#pkgs[@]}" -gt 0 ]; then
    run apt-get "${apt_opts[@]}" install -qy "${opts[@]}" "${pkgs[@]}" || ok=0
  fi
  if [ "$ok" -eq 1 ]; then
    exit 0
  fi
  echo "::warning::apt attempt $i/$attempts failed or exceeded ${attempt_timeout}s: ${pkgs[*]:-update}"
  release_apt
  sudo dpkg --configure -a || true
  if [ "$i" -lt "$attempts" ]; then
    switch_mirror
    update=1
  fi
done
echo "::error::apt failed after $attempts attempts: ${pkgs[*]:-update}"
exit 1
