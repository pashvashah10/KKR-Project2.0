#!/usr/bin/env bash
#
# Start the storefront. Safe to run repeatedly.
#
#   ./run.sh              install if needed, seed in the background, serve on :8000
#   ./run.sh --no-seed    skip seeding (the site works, with no example venues)
#   PORT=9000 ./run.sh    serve somewhere else
#
# The design decision worth knowing: seeding happens in the *background* while
# the server is already up. Fitting eight venues against NOAA takes 5-20 minutes,
# and making someone stare at a terminal for that before seeing anything is the
# same mistake the storefront itself was built to avoid. The site is usable
# within seconds; example venues appear as each fit lands.

set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
SEED=1
[[ "${1:-}" == "--no-seed" ]] && SEED=0

# ---------------------------------------------------------------- secret
# Persisted rather than regenerated, so carts and resume links survive a
# restart. Gitignored --- this must never be committed.
if [[ -f .env.local ]]; then
  # shellcheck disable=SC1091
  source .env.local
fi
if [[ -z "${DOWNSIDE_SECRET:-}" ]]; then
  DOWNSIDE_SECRET="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"
  echo "export DOWNSIDE_SECRET=$DOWNSIDE_SECRET" > .env.local
  echo "→ generated a signing key and saved it to .env.local"
fi
export DOWNSIDE_SECRET

# -------------------------------------------------------------- packages
if ! python3 -c "import fastapi, numpy, jinja2, itsdangerous" 2>/dev/null; then
  echo "→ installing dependencies..."
  python3 -m pip install --quiet -r api/requirements.txt -r engine/requirements.txt
fi

# ------------------------------------------------------------------ seed
if [[ "$SEED" == "1" ]]; then
  ready="$(python3 -c 'from api import store; store.init(); print(len(store.list_example_sites()))' 2>/dev/null || echo 0)"
  if [[ "$ready" -lt 8 ]]; then
    echo "→ fitting example venues in the background ($ready/8 ready)"
    echo "  progress: tail -f seed.log        this takes 5-20 minutes and needs internet"
    nohup python3 -m api.seed > seed.log 2>&1 &
  else
    echo "→ 8 example venues already fitted"
  fi
fi

# ----------------------------------------------------------------- serve
echo
echo "→ starting Downside on http://localhost:$PORT"
echo
exec python3 -m uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
