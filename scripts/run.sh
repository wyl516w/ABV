#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

function usage() {
  cat <<'USAGE'
Usage: scripts/run.sh --av-root <dir> --bc-root <dir> --workdir <dir> [--resume]

Runs the end-to-end demo flow:
  1. Prepare AV and BC dataset indices.
  2. Materialise codec tokens.
  3. Launch staged training runs (T1→T3).
  4. Trigger evaluation on the final checkpoint.

Arguments:
  --av-root     Directory with AV audio/video sources (wav/mp4 pairs).
  --bc-root     Directory with BC audio sources.
  --workdir     Working directory to store indices, tokens, checkpoints, and reports.
  --resume      Resume codec tokenisation if outputs already exist.
  -h, --help    Show this message and exit.
USAGE
}

AV_ROOT=""
BC_ROOT=""
WORKDIR=""
RESUME_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --av-root)
      AV_ROOT=$(realpath "$2"); shift 2;;
    --bc-root)
      BC_ROOT=$(realpath "$2"); shift 2;;
    --workdir)
      WORKDIR=$(realpath "$2"); shift 2;;
    --resume)
      RESUME_FLAG="--resume"; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1;;
  esac
done

if [[ -z "$AV_ROOT" || -z "$BC_ROOT" || -z "$WORKDIR" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

INDICES_DIR="$WORKDIR/indices"
TOKENS_DIR="$WORKDIR/tokens"
RUNS_DIR="$WORKDIR/runs"
EVAL_DIR="$WORKDIR/eval"
mkdir -p "$INDICES_DIR" "$TOKENS_DIR" "$RUNS_DIR" "$EVAL_DIR"

AV_INDEX="$INDICES_DIR/index_av.jsonl"
BC_INDEX="$INDICES_DIR/index_bc.jsonl"
AV_TOKENS="$TOKENS_DIR/index_av.tokenized.jsonl"
BC_TOKENS="$TOKENS_DIR/index_bc.tokenized.jsonl"

stage_configs=(
  "stage_t1_av=config/stages/stage_t1_av.yaml"
  "stage_t2_bc=config/stages/stage_t2_bc.yaml"
  "stage_t3_fusion=config/stages/stage_t3_fusion.yaml"
)

total_steps=$((3 + ${#stage_configs[@]} + 1))

echo "[1/${total_steps}] Preparing AV index at $AV_INDEX"
python "$ROOT_DIR/scripts/prepare_index_av.py" "$AV_ROOT" "$AV_INDEX"

echo "[2/${total_steps}] Preparing BC index at $BC_INDEX"
python "$ROOT_DIR/scripts/prepare_index_bc.py" "$BC_ROOT" "$BC_INDEX"

echo "[3/${total_steps}] Tokenising codec targets"
python "$ROOT_DIR/codec/offline_tokenize.py" "$AV_INDEX" "$AV_TOKENS" $RESUME_FLAG
python "$ROOT_DIR/codec/offline_tokenize.py" "$BC_INDEX" "$BC_TOKENS" $RESUME_FLAG

step=4
for stage_entry in "${stage_configs[@]}"; do
  stage_name="${stage_entry%%=*}"
  config_path="$ROOT_DIR/${stage_entry#*=}"
  output_dir="$RUNS_DIR/$stage_name"
  mkdir -p "$output_dir"
  echo "[$step/${total_steps}] Training ${stage_name^^} with config $config_path"
  bash "$ROOT_DIR/scripts/run_train.sh" "$config_path" "$AV_INDEX" "$output_dir"
  ((step++))
  LAST_CHECKPOINT="$output_dir/best.ckpt"
  if [[ ! -f "$LAST_CHECKPOINT" ]]; then
    touch "$LAST_CHECKPOINT"
  fi
  FINAL_CHECKPOINT="$LAST_CHECKPOINT"
  FINAL_OUTPUT_DIR="$output_dir"
done

EVAL_OUTPUT="$EVAL_DIR/report.json"
echo "[$step/${total_steps}] Running evaluation"
python "$ROOT_DIR/scripts/run_eval.sh" "$FINAL_CHECKPOINT" "$EVAL_OUTPUT"

echo "Flow complete. Final evaluation report: $EVAL_OUTPUT"
