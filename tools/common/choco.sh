#!/usr/bin/env bash
# Shared helper: retry `choco install` to absorb transient failures from the
# Chocolatey community feed (notably HTTP 504 Gateway Timeout when fetching a
# package version, which otherwise fails the Windows CI/release jobs).
#
# Usage:   choco_install <package> [extra choco args...]
# Tunable: CHOCO_MAX_RETRIES (default 5), CHOCO_RETRY_DELAY seconds (default 15).
#
# Written to be safe under `set -e`: the inner choco call is guarded with `||`.
choco_install() {
  local attempt=1 max="${CHOCO_MAX_RETRIES:-5}" delay="${CHOCO_RETRY_DELAY:-15}" rc
  while true; do
    rc=0
    choco install --no-progress -y "$@" || rc=$?
    # 0 = success; 3010 / 1641 = success but a reboot was requested
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 3010 ] || [ "$rc" -eq 1641 ]; then
      return 0
    fi
    if [ "$attempt" -ge "$max" ]; then
      echo "choco_install: '$*' failed after ${max} attempt(s) (last rc=${rc})" >&2
      return "$rc"
    fi
    echo "choco_install: '$*' failed (attempt ${attempt}/${max}, rc=${rc}); retrying in ${delay}s..." >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}
