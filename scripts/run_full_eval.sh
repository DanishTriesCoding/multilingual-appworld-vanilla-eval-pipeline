#!/usr/bin/env bash
# Full vanilla baseline: test_normal + test_challenge, 8 rollouts each.
# Assumes your Qwen server is already up and `appworld download data` is done.
set -euo pipefail

CONFIG="${1:-configs/vanilla_qwen7b.yaml}"

echo "== sanity check =="
appworld-vanilla check -c "$CONFIG"

echo "== starting AppWorld environment server =="
appworld serve environment --port 8123 &
ENV_PID=$!
trap 'kill $ENV_PID 2>/dev/null || true' EXIT
sleep 10

for SPLIT in test_normal test_challenge; do
  echo "== $SPLIT =="
  appworld-vanilla run -c "$CONFIG" --set \
    run.split="$SPLIT" \
    run.experiment_prefix="vanilla_qwen2p5_7b_${SPLIT}" \
    env.remote_environment_url="http://localhost:8123"
done

for SPLIT in test_normal test_challenge; do
  appworld-vanilla report -c "$CONFIG" --set \
    run.split="$SPLIT" run.experiment_prefix="vanilla_qwen2p5_7b_${SPLIT}"
done
