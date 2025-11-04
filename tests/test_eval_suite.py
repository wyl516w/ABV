"""Tests for evaluation suite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from eval.eval_suite import EvalConfig, run_eval


def test_run_eval(tmp_path: Path) -> None:
    samples = [
        {
            "utt_id": "utt-1",
            "hyp": "hello",
            "ref": "hello",
            "pred_audio": torch.ones(1, 4),
            "ref_audio": torch.ones(1, 4),
            "pred_emb": torch.ones(1, 3),
            "ref_emb": torch.ones(1, 3),
        }
    ]
    path = run_eval(samples, EvalConfig(output_dir=tmp_path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report[0]["wer"] == 0.0
