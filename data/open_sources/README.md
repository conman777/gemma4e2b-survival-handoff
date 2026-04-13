# Open Survival Sources

This folder contains openly licensed datasets and public-domain government reference pages collected for survival, emergency, preparedness, and first-aid fine-tuning work.

## Included source types

- Hugging Face datasets with explicit open licenses such as `CC-BY-4.0`, `MIT`, and `CC-BY-SA-4.0`
- U.S. government agency pages used as raw reference material, treated as public-domain unless the page says otherwise

## Safety and quality notes

- `recommended` sources are the default legal-safe batch to start with
- `optional` sources are open, but lower-trust or more synthetic and should be filtered carefully before training
- `eval` sources may be better as held-out benchmarks than direct training data

## Downloader

Run:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_open_survival_sources.py --include-optional
```

This mirrors sources into subfolders under `data\open_sources` and writes `download_summary.json`.
