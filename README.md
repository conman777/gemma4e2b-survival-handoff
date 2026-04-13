# Gemma 4 E2B Survival Fine-Tune

This folder contains a local Unsloth-based LoRA fine-tuning setup for `google/gemma-4-E2B-it`.

It is intentionally isolated from the Electron app in this repository.

## What This Setup Does

- Installs a Windows-friendly Python environment for Unsloth.
- Fine-tunes `google/gemma-4-E2B-it` with LoRA adapters instead of full fine-tuning.
- Uses a seed instruction dataset focused on wilderness, emergency, and practical survival guidance.
- Saves the adapter output locally so you can continue training or merge/export later.

## Hardware Notes

This machine reports an `NVIDIA GeForce RTX 5080` with about `16 GB` VRAM, which is enough for a practical E2B LoRA workflow.

## Prerequisites

1. A Hugging Face account.
2. Acceptance of the Gemma terms on the model page:
   `https://huggingface.co/google/gemma-4-E2B-it`
3. A Hugging Face access token with read permission.
4. Visual Studio C++ build tools if Unsloth needs native compilation on Windows.

## Files

- `scripts/setup_env.ps1`: create the local venv and install dependencies.
- `scripts/run_train.ps1`: run the curated LoRA fine-tune on the merged survival dataset.
- `scripts/run_next_survival_iteration.ps1`: normalize data, train the next adapter, and run both eval suites.
- `scripts/chat_with_adapter.py`: quick local test script after training.
- `scripts/train_gemma_survival.py`: main training script.
- `data/survival_seed.jsonl`: starter dataset you can expand.
- `data/curated/survival_priority_gapfix.jsonl`: hand-written survival-native gap-fix examples for weak eval categories.
- `data/evals/survival_eval_holdout_v1.jsonl`: second held-out conversational eval suite for the next run.

## Quick Start

Open PowerShell in this folder and run:

```powershell
cd C:\Users\conor\OneDrive\Personal\Documents\fine-tuning\gemma4e2b-survival
.\scripts\setup_env.ps1
.\scripts\login_hf.ps1
.\scripts\run_train.ps1
```

The default training run now:

- trains on `data/normalized/survival_training_candidates_merged.jsonl`
- writes the adapter to `output/gemma4e2b-survival-lora-curated`
- uses `1` epoch
- disables step-by-step eval by default so long runs are practical on the 16 GB RTX 5080
- saves periodic checkpoints every `500` optimizer steps during long runs

The normalization step now also:

- caps the largest generic first-aid sources so they do not dominate the training mix
- upweights the survival seed, the curated gap-fix set, bushcraft data, and the offline practical survival subset
- writes a `normalized_summary.json` that records the caps and repeat multipliers used for the merge

After training, test the adapter:

```powershell
.\.venv\Scripts\python.exe .\scripts\chat_with_adapter.py --adapter-dir .\output\gemma4e2b-survival-lora-curated --prompt "I have 24 hours stranded with no filter. What should I prioritize first?"
```

## Expanding The Dataset

The current dataset is only a starter set. For materially better results, add examples in the same `messages` format covering:

- climate-specific survival scenarios
- first-aid decision making
- gear tradeoffs
- shelter and fire choices
- risk assessment under uncertainty
- water sourcing and purification
- signaling and rescue priorities

High-quality domain data matters more than just adding volume. Keep answers practical, concise, and explicit about uncertainty and safety limits.

## Open Dataset Acquisition

The workspace now includes a local open-source/public-domain data pipeline:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_open_survival_sources.py --include-optional
.\.venv\Scripts\python.exe .\scripts\normalize_open_survival_sources.py
```

This will:

- mirror openly licensed Hugging Face datasets into `data/open_sources/hf`
- download public-domain government reference pages into `data/open_sources/web`
- write normalized training/eval-ready files into `data/normalized`

The main merged training candidate file is:

`data/normalized/survival_training_candidates_merged.jsonl`

## Offline Eval

There is also a small local held-out survival benchmark:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_survival_eval.py --adapter-dir .\output\gemma4e2b-survival-lora-curated --output-dir .\output\evals
```

This compares the base model and the local adapter on fixed prompts in:

`data/evals/survival_eval_suite.jsonl`

It writes timestamped JSON and Markdown summaries into:

`output/evals`

To score responses from any other model manually, create a JSONL file with:

`{"id":"overnight_low_water","response":"..."}`

for every prompt id in `data/evals/survival_eval_suite.jsonl`, then run:

```powershell
.\.venv\Scripts\python.exe .\scripts\score_survival_responses.py --responses-file .\some_model_responses.jsonl --label manual --model-name some_model --output-dir .\output\evals
```

## Next Iteration Command

For the next run, use the wrapper script so the data normalization and both eval suites happen in one pass:

```powershell
.\scripts\run_next_survival_iteration.ps1 -AdapterDir .\output\gemma4e2b-survival-lora-next
```

That script will:

- rebuild `data/normalized/survival_training_candidates_merged.jsonl`
- train a fresh adapter with a lower default learning rate of `1e-4`
- run the original eval suite
- run the new held-out suite in `data/evals/survival_eval_holdout_v1.jsonl`
