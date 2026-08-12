#!/usr/bin/env bash
#
# Codespace build step. Must always exit --- if it hangs, the codespace never
# reports itself ready and the user stares at "Setting up your codespace".
#
# Chatty on purpose: this is the only window into a several-minute build, and
# silence is indistinguishable from a hang.

set -uo pipefail
cd "$(dirname "$0")/.."

echo "──────────────────────────────────────────────"
echo " Downside — setting up"
echo "──────────────────────────────────────────────"

echo "→ installing Python packages (2–4 minutes)"
if ! pip install --no-cache-dir \
      -r api/requirements.txt \
      -r engine/requirements.txt \
      -r api/requirements-dev.txt; then
  echo "!! pip install failed. Open a terminal and run:"
  echo "     pip install -r api/requirements.txt -r engine/requirements.txt"
  # Deliberately not fatal: a usable shell beats a failed build the user
  # cannot get into to diagnose.
  exit 0
fi

chmod +x run.sh 2>/dev/null || true

echo "→ checking the station bundle"
python3 - <<'PY' || echo "!! station bundle did not load; venues will fall back to reanalysis"
from api import stations
idx = stations.load_index()
print(f"   {len(idx):,} GHCN stations ready, no network needed")
PY

echo
echo "──────────────────────────────────────────────"
echo " Setup complete."
echo " The server starts automatically — watch the PORTS tab for port 8000,"
echo " or run ./run.sh yourself if it is not up."
echo "──────────────────────────────────────────────"
exit 0
