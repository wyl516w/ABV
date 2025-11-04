# Unified Auditory Token Space

This repository bundles an end-to-end multimodal pipeline that targets both
**speech generation** (autoregressive codec synthesis) and **speech
recognition** (phonetic/text metrics such as WER/CER/STOI/ESTOI/PESQ). The
training stack is organised so downstream tasks can reuse shared encoders and
fusion modules across air-conducted, bone-conducted, and audio-visual inputs.

## Runbook

1. Prepare `index_av.jsonl` and `index_bc.jsonl` using the scripts in `scripts/`.
2. Run `python codec/offline_tokenize.py index_av.jsonl tokens_av.jsonl` to generate codec tokens (preferring `audio_clean`, otherwise `audio_ac`).
3. Train Stage T1 using `scripts/run_train.sh config/stages/stage_t1_av.yaml index_av.jsonl runs/stage_t1`.
4. Train Stage T2 to incorporate bone-conduction by enabling domain alignment and uncertainty heads.
5. Train Stage T3 with multimodal calibration enabled.
6. Perform inference with `python -m infer.synthesize` (see module docstrings for details).
7. Evaluate with `scripts/run_eval.sh runs/stage_t3/best.ckpt eval/outputs` to generate reports.

### Dataset compatibility helpers

- **Generic AV/BC**: `scripts/prepare_index_av.py` and `scripts/prepare_index_bc.py` scan arbitrary audio/video folders.
- **ABCS Corpus**: `scripts/prepare_index_abcs.py` stitches synchronised air/bone audio, optional video, and speaker IDs captured in an ISO 3745 anechoic chamber.
- **CMLR**: `scripts/prepare_index_cmlr.py` consumes Mandarin lip-reading splits with their per-utterance transcripts.
- **LRS2/LRS3**: `scripts/prepare_index_lrs2.py` and `scripts/prepare_index_lrs3.py` traverse the official directory layouts, linking video clips with optional extracted audio and subtitle files.

Each script emits JSONL indices that match the dataset abstractions in
`datasets/`, allowing the same Lightning training loop to mix ABCS bone
conduction with CMLR/LRS audio-visual samples.

## FAQ

- **Repetition/Looping**: Adjust repetition penalties in the inference config.
- **Timing Misalignment**: Ensure codec hop (20 ms) aligns with dataset timestamps.
- **Target Leakage**: Never use `audio_clean` as conditioning input; it is a target signal only.
- **Domain Shift**: Use the BC domain alignment loss to bridge BC↔AC differences.
- **Video Quality Gating**: Drop or down-weight frames with high blur/low confidence before fusion.
