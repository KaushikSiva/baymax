#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ -x .venv/bin/python ]] || {
  echo "Run scripts/setup_macos.sh first." >&2
  exit 1
}
[[ -f assets/mujoco_menagerie/unitree_g1/g1_with_hands.xml ]] || {
  echo "Run scripts/download_g1_mjcf.sh first." >&2
  exit 1
}
[[ -f vendor/unitree_rl_gym/deploy/pre_train/g1/motion.pt ]] || {
  echo "Run scripts/setup_unitree_policy.sh first." >&2
  exit 1
}

ASSET_SOURCE="${BAYMAX_ASSET_SOURCE:-$HOME/Downloads/hospital_assets}"
MANIFEST="baymax_nurse/assets/local/manifest.json"
if [[ ! -f "$MANIFEST" && -d "$ASSET_SOURCE" ]]; then
  required=(
    lowpoly-medical-room.zip
    medical-examination-bed-game-ready-asset.zip
    aero-monitor.zip
    grandma-on-bench-free.zip
    boy.glb
  )
  ready=true
  for filename in "${required[@]}"; do
    [[ -f "$ASSET_SOURCE/$filename" ]] || ready=false
  done
  if [[ "$ready" == true ]]; then
    BLENDER_BIN="${BLENDER_BIN:-$(command -v blender || true)}"
    [[ -n "$BLENDER_BIN" ]] || {
      echo "Blender is required to prepare hospital assets." >&2
      exit 1
    }
    "$BLENDER_BIN" --background --python scripts/prepare_hospital_assets.py -- \
      --source-dir "$ASSET_SOURCE"
  fi
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Hospital art not found; running with procedural fallback patients." >&2
fi

MODE="${1:-gemini}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

case "$MODE" in
  validate)
    exec .venv/bin/python -m baymax_nurse --headless --validate-only "$@"
    ;;
  scripted)
    [[ "$(uname -s)" == "Darwin" ]] || {
      echo "Visible mode currently requires macOS; use validate on other hosts." >&2
      exit 1
    }
    exec .venv/bin/mjpython -m baymax_nurse --adapter scripted "$@"
    ;;
  gemini)
    [[ -n "${GEMINI_API_KEY:-}" ]] || {
      echo "Set GEMINI_API_KEY before launching Gemini mode." >&2
      exit 1
    }
    [[ "$(uname -s)" == "Darwin" ]] || {
      echo "Visible mode currently requires macOS; use validate on other hosts." >&2
      exit 1
    }
    exec .venv/bin/mjpython -m baymax_nurse --adapter gemini-er "$@"
    ;;
  *)
    echo "Usage: scripts/run_baymax.sh {gemini|scripted|validate} [options]" >&2
    exit 2
    ;;
esac
