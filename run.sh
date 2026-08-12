#!/usr/bin/env bash
# Starts the pipeline + dashboard on http://localhost:5050
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

export PORT="${PORT:-5050}"
export TICKS_PER_SECOND="${TICKS_PER_SECOND:-8}"
export ANOMALY_PROBABILITY="${ANOMALY_PROBABILITY:-0.015}"

echo "starting dashboard on http://localhost:${PORT}"
python3 -m app.server
