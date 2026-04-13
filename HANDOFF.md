# Remote Handoff

This repository snapshot is the minimal handoff set for another AI or remote GPU box to continue the survival fine-tuning project without the local caches, venv, or prior outputs.

## Included

- training and eval scripts
- the merged training dataset:
  - `data/normalized/survival_training_candidates_merged.jsonl`
- the curated gap-fix dataset:
  - `data/curated/survival_priority_gapfix.jsonl`
- both conversational eval suites:
  - `data/evals/survival_eval_suite.jsonl`
  - `data/evals/survival_eval_holdout_v1.jsonl`
- the source manifest for redownloading open sources later:
  - `data/open_sources/open_sources_manifest.json`
- training-pool summary metadata:
  - `data/normalized/normalized_summary.json`

## Excluded

- `.venv`
- `output`
- raw mirrored Hugging Face/web source downloads
- local test response files
- build caches

## Quick Start On A Remote Linux GPU

```bash
git clone <private-repo-url>
cd gemma4e2b-survival
bash scripts/setup_env_linux.sh
export HF_TOKEN=...
bash scripts/run_next_survival_iteration.sh
```

## Notes

- If the remote machine already has the base model cached, you only need the repo files plus Hugging Face auth.
- If you want to rebuild the merged dataset from raw sources later, use `scripts/download_open_survival_sources.py` with `data/open_sources/open_sources_manifest.json`.
- The current merged dataset is already included so another AI can begin training immediately without waiting for source downloads.
