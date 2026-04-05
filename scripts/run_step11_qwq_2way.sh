#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/jizhicfs/zeg/data_synthesis}
PYTHON_BIN=${PYTHON_BIN:-/jizhicfs/hymiezhao/miniconda3/envs/Diexie/bin/python}
CONFIG=${CONFIG:-$PROJECT_ROOT/configs/kg_math.yaml}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
API_KEY=${API_KEY:-dummy}
MODEL=${MODEL:-QwQ-32B}
TIMEOUT_SEC=${TIMEOUT_SEC:-600}

SYNTH_TOTAL_SHARDS=${SYNTH_TOTAL_SHARDS:-75}
MATH_TOTAL_SHARDS=${MATH_TOTAL_SHARDS:-75}
SYNTH_BACKTRACK_SHARDS=${SYNTH_BACKTRACK_SHARDS:-12}
MATH_BACKTRACK_SHARDS=${MATH_BACKTRACK_SHARDS:-0}

SYNTH_EXISTING_PREFIX=${SYNTH_EXISTING_PREFIX:-$PROJECT_ROOT/data/outputs/answered_qwq_7500_merged_20260406/synth}
SYNTH_WORK_DIR=${SYNTH_WORK_DIR:-$PROJECT_ROOT/data/outputs/answered_qwq_resume_20260406}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/data/outputs/debug/step11_qwq_resume_2way_logs}
PID_DIR=${PID_DIR:-$LOG_DIR/pids}

MATH_ROOT=${MATH_ROOT:-/jizhicfs/zeg/datasets/MATH/train}
MATH_INPUT_DIR=${MATH_INPUT_DIR:-$PROJECT_ROOT/MATH_step11_inputs}
MATH_EXISTING_PREFIX=${MATH_EXISTING_PREFIX:-$PROJECT_ROOT/MATH_step11_answers_merged_20260406/math_merged}
MATH_OUT_DIR=${MATH_OUT_DIR:-$PROJECT_ROOT/MATH_step11_answers_qwq_resume_20260406}

mkdir -p "$SYNTH_WORK_DIR" "$LOG_DIR" "$PID_DIR" "$MATH_INPUT_DIR" "$MATH_OUT_DIR"

completed_rows_from_prefix() {
  local prefix="$1"
  local status_path
  status_path="$(dirname "$prefix")/$(basename "$prefix").status.json"
  if [ -f "$status_path" ]; then
    "$PYTHON_BIN" -c "import json,sys; print(int(json.load(open(sys.argv[1], encoding='utf-8')).get('completed_rows', 0)))" "$status_path"
    return
  fi
  "$PYTHON_BIN" -c "import sys; from pathlib import Path; prefix=Path(sys.argv[1]); paths=sorted(prefix.parent.glob(prefix.stem + '_*.jsonl')); total=sum(sum(1 for line in p.open('r', encoding='utf-8') if line.strip()) for p in paths); print(total)" "$prefix"
}

calc_start_shard() {
  local completed_rows="$1"
  local total_shards="$2"
  local backtrack_shards="$3"
  local completed_shards
  local start_shard
  completed_shards=$((completed_rows / 100))
  start_shard=$((completed_shards - backtrack_shards + 1))
  if [ "$start_shard" -lt 1 ]; then
    start_shard=1
  fi
  if [ "$start_shard" -gt "$total_shards" ]; then
    start_shard="$total_shards"
  fi
  echo "$start_shard"
}

launch_worker() {
  local kind="$1"
  local start_shard="$2"
  local end_shard="$3"
  local out_prefix="$4"
  local resume_prefix="$5"
  local log_path="$6"
  local -a cmd
  local -a inputs
  local i
  local shard
  local input_path

  if [ "$start_shard" -gt "$end_shard" ]; then
    echo "[skip] $kind no work for range ${start_shard}-${end_shard}"
    return
  fi

  cmd=(
    "$PYTHON_BIN"
    "$PROJECT_ROOT/scripts/11_generate_answers.py"
    --config "$CONFIG"
    --out-prefix "$out_prefix"
    --start-index "$start_shard"
    --resume
    --backend gateway
    --base-url "$BASE_URL"
    --api-key "$API_KEY"
    --model "$MODEL"
    --easy-model "$MODEL"
    --medium-model "$MODEL"
    --hard-model "$MODEL"
    --timeout-sec "$TIMEOUT_SEC"
  )
  if [ -n "$resume_prefix" ]; then
    cmd+=(--resume-from-prefix "$resume_prefix")
  fi
  cmd+=(--inputs)

  inputs=()
  for ((i=start_shard; i<=end_shard; i++)); do
    printf -v shard "%02d" "$i"
    if [ "$kind" = "synth" ]; then
      input_path="$PROJECT_ROOT/data/outputs/synth_${shard}.jsonl"
    else
      input_path="$MATH_INPUT_DIR/math_train_${shard}.jsonl"
    fi
    inputs+=("$input_path")
  done
  cmd+=("${inputs[@]}")

  nohup "${cmd[@]}" >"$log_path" 2>&1 &
  echo "$!" >"$PID_DIR/$(basename "$log_path").pid"
  echo "[launched] $kind ${start_shard}-${end_shard} pid=$! log=$log_path out_prefix=$out_prefix"
}

if find "$MATH_INPUT_DIR" -maxdepth 1 -type f -name 'math_train_*.jsonl' | grep -q .; then
  echo "[prepare] Reusing existing MATH input shards under $MATH_INPUT_DIR"
else
  echo "[prepare] MATH input shards"
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/11b_generate_math_train_answers.py" \
    --project-root "$PROJECT_ROOT" \
    --math-root "$MATH_ROOT" \
    --config "$CONFIG" \
    --input-dir "$MATH_INPUT_DIR" \
    --out-prefix "$MATH_OUT_DIR/math_train" \
    --prepare-only
fi

synth_completed_rows=$(completed_rows_from_prefix "$SYNTH_EXISTING_PREFIX")
math_completed_rows=$(completed_rows_from_prefix "$MATH_EXISTING_PREFIX")

synth_start=$(calc_start_shard "$synth_completed_rows" "$SYNTH_TOTAL_SHARDS" "$SYNTH_BACKTRACK_SHARDS")
math_start=$(calc_start_shard "$math_completed_rows" "$MATH_TOTAL_SHARDS" "$MATH_BACKTRACK_SHARDS")

echo "[synth] completed_rows=$synth_completed_rows start_shard=$synth_start worker=${synth_start}-${SYNTH_TOTAL_SHARDS} resume_prefix=$SYNTH_EXISTING_PREFIX"
echo "[math] completed_rows=$math_completed_rows start_shard=$math_start worker=${math_start}-${MATH_TOTAL_SHARDS} resume_prefix=$MATH_EXISTING_PREFIX"

launch_worker synth "$synth_start" "$SYNTH_TOTAL_SHARDS" "$SYNTH_WORK_DIR/synth_resume" "$SYNTH_EXISTING_PREFIX" "$LOG_DIR/synth_resume.log"
launch_worker math "$math_start" "$MATH_TOTAL_SHARDS" "$MATH_OUT_DIR/math_resume" "$MATH_EXISTING_PREFIX" "$LOG_DIR/math_resume.log"

echo "[logs] tail -f $LOG_DIR/synth_resume.log"
echo "[logs] tail -f $LOG_DIR/math_resume.log"
