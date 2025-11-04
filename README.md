# Unified Auditory Token Space

## Runbook

1. Prepare `index_av.jsonl` and `index_bc.jsonl` using the scripts in `scripts/`.
2. Run `python codec/offline_tokenize.py index_av.jsonl tokens_av.jsonl` to generate codec tokens (preferring `audio_clean`, otherwise `audio_ac`).
3. Train Stage T1 using `scripts/run_train.sh config/stages/stage_t1_av.yaml index_av.jsonl runs/stage_t1`.
4. Train Stage T2 to incorporate bone-conduction by enabling domain alignment and uncertainty heads.
5. Train Stage T3 with multimodal calibration enabled.
6. Perform inference with `python -m infer.synthesize` (see module docstrings for details).
7. Evaluate with `scripts/run_eval.sh runs/stage_t3/best.ckpt eval/outputs` to generate reports.

## FAQ

- **Repetition/Looping**: Adjust repetition penalties in the inference config.
- **Timing Misalignment**: Ensure codec hop (20 ms) aligns with dataset timestamps.
- **Target Leakage**: Never use `audio_clean` as conditioning input; it is a target signal only.
- **Domain Shift**: Use the BC domain alignment loss to bridge BC↔AC differences.
- **Video Quality Gating**: Drop or down-weight frames with high blur/low confidence before fusion.
