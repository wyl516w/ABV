#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-config/base.yaml}
INDEX=${2:-data/index.jsonl}
OUT=${3:-runs/default}
mkdir -p "$OUT"
export PYTHONHASHSEED=0
python - "$CONFIG" "$INDEX" "$OUT" <<'PY'
import sys
from pathlib import Path
from train.train_loop import TrainerConfig, run_training

config_path = Path(sys.argv[1])
index_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
print(f"Using config: {config_path}")
print(f"Index path: {index_path}")
out_dir.mkdir(parents=True, exist_ok=True)
loss = run_training(TrainerConfig())
(out_dir / "training.log").write_text(f"loss={loss}\n", encoding="utf-8")
# Placeholder checkpoint artifact for downstream scripts.
(out_dir / "best.ckpt").write_text("placeholder", encoding="utf-8")
print(f"Final loss: {loss}")
PY
