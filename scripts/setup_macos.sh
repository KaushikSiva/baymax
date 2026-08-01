#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
scripts/download_g1_mjcf.sh
scripts/setup_unitree_policy.sh

echo "Setup complete. Run: scripts/run_baymax.sh validate"
