#!/usr/bin/env bash
# Reproduce the final model end to end. Run from the repo root.
# Optional: pass a YAML file of overrides, e.g.  bash scripts/reproduce_final.sh my_overrides.yaml
set -euo pipefail
CONFIG_ARGS=()
if [ $# -ge 1 ]; then CONFIG_ARGS=(--config "$1"); fi

python -m src.prepare_data "${CONFIG_ARGS[@]}"   # cleaned silver, train/dev split, vocab
python -m src.train        "${CONFIG_ARGS[@]}"   # mixed pretrain + gold fine-tuning grid
# summary of the grid on dev: outputs/finetune_grid.csv
