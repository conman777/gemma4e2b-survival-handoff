#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXE="${ROOT_DIR}/.venv/bin/python"
ADAPTER_DIR="${1:-${ROOT_DIR}/output/gemma4e2b-survival-lora-next}"
EVAL_OUTPUT_DIR="${2:-${ROOT_DIR}/output/evals}"

if [[ ! -x "${PYTHON_EXE}" ]]; then
  echo "Missing virtual environment at ${PYTHON_EXE}" >&2
  exit 1
fi

"${PYTHON_EXE}" "${ROOT_DIR}/scripts/train_gemma_survival.py" \
  --model-name google/gemma-4-E2B-it \
  --dataset-path "${ROOT_DIR}/data/normalized/survival_training_candidates_merged.jsonl" \
  --output-dir "${ADAPTER_DIR}" \
  --num-train-epochs 1 \
  --learning-rate 1e-4

"${PYTHON_EXE}" "${ROOT_DIR}/scripts/run_survival_eval.py" \
  --adapter-dir "${ADAPTER_DIR}" \
  --eval-file "${ROOT_DIR}/data/evals/survival_eval_suite.jsonl" \
  --output-dir "${EVAL_OUTPUT_DIR}"

"${PYTHON_EXE}" "${ROOT_DIR}/scripts/run_survival_eval.py" \
  --adapter-dir "${ADAPTER_DIR}" \
  --eval-file "${ROOT_DIR}/data/evals/survival_eval_holdout_v1.jsonl" \
  --output-dir "${EVAL_OUTPUT_DIR}"
