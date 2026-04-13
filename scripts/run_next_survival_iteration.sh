#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXE="${ROOT_DIR}/.venv/bin/python"
ADAPTER_DIR="${1:-${ROOT_DIR}/output/gemma4e2b-survival-lora-v2}"
EVAL_OUTPUT_DIR="${2:-${ROOT_DIR}/output/evals-v2}"

if [[ ! -x "${PYTHON_EXE}" ]]; then
  echo "Missing virtual environment at ${PYTHON_EXE}" >&2
  exit 1
fi

# --- Step 1: Train (dataset already rebuilt externally) ---
echo "=== Starting training ==="
"${PYTHON_EXE}" "${ROOT_DIR}/scripts/train_gemma_survival.py" \
  --model-name google/gemma-4-E2B-it \
  --dataset-path "${ROOT_DIR}/data/normalized/survival_training_candidates_merged.jsonl" \
  --output-dir "${ADAPTER_DIR}" \
  --num-train-epochs 1 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 2 \
  --learning-rate 5e-5 \
  --lora-r 16 \
  --lora-alpha 16 \
  --weight-decay 0.01 \
  --warmup-steps 20 \
  --save-strategy steps \
  --save-steps 200 \
  --logging-steps 10 \
  --seed 3407

# --- Step 3: Evaluate final checkpoint on both suites ---
echo "=== Evaluating on original suite ==="
"${PYTHON_EXE}" "${ROOT_DIR}/scripts/run_survival_eval.py" \
  --adapter-dir "${ADAPTER_DIR}" \
  --eval-file "${ROOT_DIR}/data/evals/survival_eval_suite.jsonl" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --max-new-tokens 512

echo "=== Evaluating on holdout suite ==="
"${PYTHON_EXE}" "${ROOT_DIR}/scripts/run_survival_eval.py" \
  --adapter-dir "${ADAPTER_DIR}" \
  --eval-file "${ROOT_DIR}/data/evals/survival_eval_holdout_v1.jsonl" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --max-new-tokens 512
