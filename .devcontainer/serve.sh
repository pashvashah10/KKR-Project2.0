#!/usr/bin/env bash
#
# Start the server and return immediately.
#
# `run.sh` ends in `exec uvicorn` and never exits, so it cannot be called
# directly from a devcontainer lifecycle hook --- doing exactly that is what
# left the codespace on "Setting up" forever. `setsid` detaches it from this
# shell's session so it survives when this script returns, and the redirect
# frees the hook's stdio so nothing is left waiting on a pipe.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG=/tmp/downside.log

if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8000/ 2>/dev/null; then
  echo "→ already running on port 8000"
  exit 0
fi

setsid nohup bash run.sh > "$LOG" 2>&1 < /dev/null &
disown 2>/dev/null || true

echo "→ starting the storefront in the background"
echo "  log: tail -f $LOG"

# Give it a moment so the PORTS tab picks 8000 up while the user is still
# looking at this output, then get out of the way regardless of the outcome.
for _ in $(seq 1 30); do
  if curl -sf -o /dev/null --max-time 1 http://127.0.0.1:8000/ 2>/dev/null; then
    echo "→ up on http://localhost:8000 — open the PORTS tab and click the globe icon"
    exit 0
  fi
  sleep 2
done

echo "→ still starting. Check with: tail -f $LOG"
exit 0
