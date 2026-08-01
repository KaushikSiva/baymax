#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/vendor/unitree_rl_gym"
REPOSITORY="https://github.com/unitreerobotics/unitree_rl_gym.git"
REVISION="276801e46c5d433564f24658bac64f254b7d2d4b"
POLICY_SHA256="cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d"
CONFIG_SHA256="73044e7d355c61915695c16d6e09eb3efef46eec1e3d708fd3eb9157dfe3bbbb"

mkdir -p "$(dirname "$DEST")"
if [[ ! -d "$DEST/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$REPOSITORY" "$DEST"
fi
git -C "$DEST" sparse-checkout init --cone
git -C "$DEST" sparse-checkout set deploy/pre_train/g1 deploy/deploy_mujoco/configs
git -C "$DEST" fetch --depth=1 origin "$REVISION"
git -C "$DEST" checkout --detach "$REVISION"

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

policy="$DEST/deploy/pre_train/g1/motion.pt"
config="$DEST/deploy/deploy_mujoco/configs/g1.yaml"
[[ "$(sha256_file "$policy")" == "$POLICY_SHA256" ]] || {
  echo "Official G1 policy hash mismatch" >&2
  exit 1
}
[[ "$(sha256_file "$config")" == "$CONFIG_SHA256" ]] || {
  echo "Official G1 config hash mismatch" >&2
  exit 1
}
echo "Pinned Unitree G1 locomotion policy ready: $REVISION"
