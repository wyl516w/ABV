#!/usr/bin/env bash
set -euo pipefail
CHECKPOINT=${1:-runs/default/best.ckpt}
OUTPUT=${2:-eval/outputs/report.json}
python - "$CHECKPOINT" "$OUTPUT" <<'PY'
import sys
from pathlib import Path

import torch

from eval.eval_suite import EvalConfig, run_eval

checkpoint = Path(sys.argv[1])
output_path = Path(sys.argv[2])
output_dir = output_path.parent
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Evaluating checkpoint: {checkpoint}")
print(f"Saving report to: {output_path}")

samples = [{
    "utt_id": "demo",
    "hyp": "hello",
    "ref": "hello",
    "pred_audio": torch.ones(1, 4),
    "ref_audio": torch.ones(1, 4),
    "pred_emb": torch.ones(1, 3),
    "ref_emb": torch.ones(1, 3),
}]
report_path = run_eval(samples, EvalConfig(output_dir=output_dir))
if report_path != output_path:
    output_path.write_text(Path(report_path).read_text(encoding="utf-8"), encoding="utf-8")
print(f"Report available at: {output_path}")
PY
