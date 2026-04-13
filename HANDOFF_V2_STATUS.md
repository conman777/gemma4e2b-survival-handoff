# Iteration Status Report — Gemma 4 E2B Survival Fine-Tune

## Repo
`/root/gemma4e2b-survival-handoff`
Clone: `https://github.com/conman777/gemma4e2b-survival-handoff.git`

## Environment
- GPU: NVIDIA RTX PRO 6000 Blackwell (98 GB VRAM)
- PyTorch 2.8.0+cu128, Unsloth 2026.4.4
- Flash Attention 2 removed (Gemma 4 head dim >256 incompatible). Uses PyTorch SDPA.
- HF_TOKEN must be exported in each shell session.
- `.venv/bin/python` is a symlink to system `/usr/bin/python3`.

## What Has Been Tried

### Iteration 1 (v1) — Baseline Remote Run
- Config: LR=1e-4, batch=8, grad_accum=2, epochs=1, lora_r=16, lora_alpha=16
- Dataset: 17,989 rows (original merged mix)
- Eval: max_new_tokens=180
- Results (at 180 tokens):
  - Original suite: base=54.3, adapter=57.5 (+3.2)
  - Holdout suite: base=51.1, adapter=56.2 (+5.1)
- Output: `output/gemma4e2b-survival-lora-next` (deleted)

### Iteration 2 (v2) — Rebalanced Data + Lower LR
- Config: LR=5e-5, batch=8, grad_accum=2, epochs=1, lora_r=16, lora_alpha=16, warmup=20
- Dataset changes: 11,111 rows
  - Medical caps reduced: firstaidqa 3500→1500, medrescue 3000→1500 (classified by keyword, subsampled)
  - Curated repeats increased: seed 6x→12x, gapfix 6x→12x
  - 21 new curated examples added to `data/curated/survival_priority_gapfix.jsonl` (38 total now)
  - New examples target: water_tradeoff (3), first_aid field (3), heat (3), food_risk (3), navigation (2), signaling (2), flood (2), hypothermia (1), general (2)
- Eval: max_new_tokens=512
- Results (at 512 tokens):
  - Original suite: base=65.9, adapter=58.1 (**-7.8**)
  - Holdout suite: base=67.2, adapter=52.3 (**-14.9**)
- **REJECTED** — adapter worse than base on both suites
- Output: `output/gemma4e2b-survival-lora-v2`, `output/evals-v2/`

## Critical Discovery: Token Truncation Artifact

The v1 "gains" were largely a truncation artifact. At max_new_tokens=180, the base model's structured multi-step responses were heavily clipped (nearly all ended mid-sentence), artificially depressing base scores. When v2 raised the cap to 512:
- Base original: 54.3 → 65.9 (+11.6 from more tokens alone)
- Base holdout: 51.1 → 67.2 (+16.1)

The base model is much stronger than v1 suggested. Future runs MUST use max_new_tokens=512 for fair comparison.

## v2 Detailed Category Results

### Original Suite (base → adapter, delta)
| Category | Base | Adapter | Delta |
|---|---:|---:|---:|
| cold_exposure | 70.0 | 70.0 | 0.0 |
| first_aid | 100.0 | 85.0 | -15.0 |
| flood | 54.2 | 23.3 | -30.9 |
| food_risk | 46.7 | 23.3 | -23.4 |
| heat | 85.0 | 92.5 | **+7.5** |
| hypothermia | 31.7 | 77.5 | **+45.8** |
| injury_navigation | 77.5 | 62.5 | -15.0 |
| navigation | 70.0 | 46.7 | -23.3 |
| signaling | 46.7 | 23.3 | -23.4 |
| vehicle_winter | 61.7 | 100.0 | **+38.3** |
| water_shelter | 77.5 | 70.0 | -7.5 |
| water_tradeoff | 70.0 | 23.3 | **-46.7** |

### Holdout Suite (base → adapter, delta)
| Category | Base | Adapter | Delta |
|---|---:|---:|---:|
| cold_exposure | 70.0 | 77.5 | +7.5 |
| first_aid | 85.0 | 85.0 | 0.0 |
| flood | 54.2 | 54.2 | 0.0 |
| food_risk | 46.7 | 23.3 | -23.4 |
| heat | 70.0 | 70.0 | 0.0 |
| hypothermia | 54.2 | 70.0 | +15.8 |
| injury_navigation | 85.0 | 77.5 | -7.5 |
| navigation | 46.7 | 46.7 | 0.0 |
| signaling | 92.5 | 61.7 | -30.8 |
| vehicle_winter | 54.2 | 38.3 | -15.9 |
| water_shelter | 77.5 | 0.0 | **-77.5** |
| water_tradeoff | 70.0 | 23.3 | **-46.7** |

## Root Cause Analysis

1. **The adapter still makes responses shorter/less structured than the base model.** Even at 5e-5 LR, the training data's ~579 char average response style overrides the base model's ~800 char structured output. The keyword scorer rewards coverage, so shorter = lower scores.

2. **water_tradeoff is consistently catastrophic (-46.7 on both suites).** The adapter says "boil" and "settle" but never mentions source selection keywords ("moving water", "upstream", "avoid stagnant") or explicit risk language ("still risky", "tradeoff", "if severely dehydrated"). The 3 new curated examples for this category at 12x repeat were not enough to overcome the bulk data's style.

3. **Categories where adapter improves are ones with thin base coverage.** Hypothermia (+45.8/+15.8) and vehicle_winter (+38.3) improve because the base model is weakest there. But the adapter pays for these gains by degrading categories where the base was already strong.

4. **food_risk remains stuck.** Both base and adapter score 23-47. The model says "don't eat unknown food" but doesn't redirect to other priorities (water/shelter/signal).

5. **Signaling regresses badly.** The adapter never mentions "open area" / "line of sight" — the most important signaling concept.

## What the Scoring System Rewards

The scorer is keyword-based, not semantic:
- required_groups = 70% (must mention specific keywords)
- bonus_groups = 15%
- priority_pairs = 15% (keyword A must appear before keyword B)
- negative_groups = -15 points each

This means a response can be medically correct but score poorly if it uses synonyms the rubric doesn't recognize, or if it's too brief to mention all required concepts. The base model's verbose, structured style naturally hits more keywords.

## Files Modified in v2
- `data/curated/survival_priority_gapfix.jsonl` — 21 new examples added (38 total)
- `data/normalized/survival_priority_gapfix_messages.jsonl` — updated
- `data/normalized/survival_training_candidates_merged.jsonl` — rebuilt with new balance (11,111 rows)
- `scripts/normalize_open_survival_sources.py` — caps changed (firstaidqa=1500, medrescue=1500, seed=12x, gapfix=12x)
- `scripts/run_next_survival_iteration.sh` — full config made explicit, max_new_tokens=512

## Acceptance Criteria (unchanged)
Reject if:
- Overall average below base on either suite
- water_tradeoff, first_aid, or heat regress >10 vs base on either suite
- water_tradeoff <50 on either suite
- first_aid <60 on either suite

## What to Try Next

### Hypothesis: The problem is fundamentally about response style, not knowledge
The adapter has the right knowledge (it mentions correct concepts) but in a compressed format that misses rubric keywords. Two potential paths:

**Path A: Even lower LR (1e-5 or 2e-5)**
Rationale: Preserve more of the base model's verbose structured style. The adapter should shift knowledge/vocabulary subtly, not override response format. Risk: may learn nothing useful at all.

**Path B: Increase curated example response length to 600-900 chars**
Rationale: The curated examples currently average 452 chars. The base model outputs ~800 chars. If the high-repeat curated examples were longer and more structured (with headers, numbered steps), they would push the adapter toward the base model's style rather than away from it. Risk: need to rewrite the curated examples.

**Path C: Evaluate intermediate checkpoints**
Checkpoints were saved at steps 200, 400. Earlier checkpoints may preserve more base model behavior. The final checkpoint may be overtrained. Evaluating checkpoint-200 and checkpoint-400 vs the final would test this cheaply.

**Path D: Reconsider whether LoRA fine-tuning helps at all for this scorer**
The base model at 512 tokens scores 65.9/67.2. If the eval rubric primarily rewards verbosity and keyword coverage, fine-tuning that compresses responses will always hurt. The curated examples might be better used as few-shot context rather than LoRA training data.

### Recommended immediate next step
Evaluate checkpoint-200 and checkpoint-400 on both suites before making any other changes. If an earlier checkpoint scores closer to base while adding value on weak categories, that's the correct model selection strategy.
