# Dataset Triage

This file is the working keep/filter/exclude decision sheet for the current survival fine-tuning corpus.

## Decision meanings

- `keep`: include directly in the training pool after normal schema conversion and basic deduplication
- `filter`: source is useful, but must be subsetted, capped, or reviewed before training
- `exclude`: do not include in the training pool for now
- `hold`: not rejected, but blocked by license/access ambiguity or better left as reference material for now

## Current verdict

| Source | Decision | Why | Action |
|---|---|---|---|
| `data/survival_seed.jsonl` | `keep` | Best-aligned seed behavior for calm, practical survival advice | Keep as a core anchor source |
| `i-am-mushfiq/FirstAidQA` | `keep` | Strong first-aid coverage, explicit open license, easy QA format | Include directly |
| `belvisk/First-Aid-Dataset` | `keep` | Small, clean, openly licensed, useful as a supplementary first-aid set | Include directly |
| `ianktoo/crisis-data-v3-sft` | `keep` | Useful disaster/preparedness decision patterns, already in instruction format | Include directly |
| `AnneshaChowdhury/Crisis-QA` | `keep` for eval only | Good benchmark-style crisis reasoning, not ideal as conversational SFT data | Use as eval only |
| `hardrave/bushcraft_survival_gpt_oss_data_distilled` | `keep` | Strongest open wilderness/bushcraft source currently on disk | Keep as a core source, continue refusal/short-answer filtering |
| `badri55/First_aid__dataset` | `keep` | Tiny, but harmless and open; useful as light supplementary data | Include, but low weighting |
| `ericrisco/medrescue` | `filter` | Very large and relevant in parts, but broad enough to overwhelm the niche and drift into generic emergency medicine | Keep only the selected survival/first-aid/rescue sub-sources, cap volume, no diagnosis-heavy sources |
| `Heralax/us-army-fm-instruct` | `filter` | Valuable in places, but mixed with unrelated military doctrine and field operations | Keep the normalized subset on disk, but exclude it from the first serious training run until drift is reviewed again |
| `AquaV/US-Army-Survival-Sharegpt` | `hold` | Content looks good, but dataset license is unclear even if the underlying source manual is public domain | Verify dataset card/legal footing before training use |
| `BI55/MedText` | `filter` | Some trauma/injury value, but heavily clinical and diagnosis-oriented | Keep only trauma/injury/environmental-injury relevant rows |
| `Xueren/EMS-Knowledge` | `exclude` for training now | Raw chunks, not QA/instruction data; converting it would add another synthetic generation layer | Keep as reference only for now |
| `adrianf12/healthcare-qa-dataset` | `filter` | Small and plausible, but very healthcare-general rather than survival-focused | Cherry-pick only directly relevant trauma/CPR/field-first-aid rows |
| `community-datasets/disaster_response_messages` | `exclude` for training now | This is classification/reference data, not answer data | Keep as reference/eval material only |
| `mattwesney/CoT_Reasoning_Bushcraft_Survival` | `hold` | High-potential niche source, but gated; may also need careful handling if it contains reasoning traces | Request access later; if approved, review structure before use |
| `mattwesney/CoT_Reasoning_First_Responders_Triage_And_Emergencies` | `hold` | Potentially useful, but gated and likely reasoning-trace heavy | Request access later; review before use |
| CDC / NPS / NWS / Ready.gov / FEMA manuals | `hold` | High-value official references, but not training-ready rows | Use as reference material for manual curation and gap filling, not bulk auto-generated SFT |
| `fm21_76_army_survival` reference | `hold` | Canonical survival content, but current URL is an archive mirror rather than a first-party Army source | Fine as reference, but do not treat mirror status as equivalent to a polished dataset license signal |
| `fm4_25_11_army_first_aid` reference | `hold` | Strong reference manual, but still reference material rather than training-ready rows | Use for manual curation/gap fills |
| `fema_cert_manual` reference | `hold` | Good source for triage/search-and-rescue/disaster guidance, but reference-only for now | Use for manual curation/gap fills |

## Recommended training pool

### Use now

- `survival_seed.jsonl`
- `FirstAidQA`
- `First-Aid-Dataset` (`belvisk`)
- `crisis-data-v3-sft`
- `bushcraft_survival_gpt_oss_data_distilled`
- `First_aid__dataset` (`badri55`)

### Use after filtering

- `medrescue`
- `MedText`
- `healthcare-qa-dataset`

### Do not train on yet

- `us-army-fm-instruct` for the first serious run
- `EMS-Knowledge`
- `disaster_response_messages`
- gated `mattwesney` datasets
- official web/manual references
- `US-Army-Survival-Sharegpt` until license situation is confirmed

## Practical next step

The first-run training corpus is now ready. The next concrete actions are:

1. train the first serious `1`-epoch run on the current merged corpus
2. keep `us-army-fm-instruct` out of that run unless a later review shows the 122-row subset is genuinely clean
3. add a small held-out conversational eval set before the next iteration
4. review the web/manual references only for targeted gap filling, not bulk ingestion

The dataset-hunting phase is effectively over. Quality control and training are the next bottlenecks.

## Current post-filter counts

These are the current normalized row counts after applying the implemented filters.

| Source | Rows | Included in merged training pool |
|---|---:|---|
| `survival_seed` | 12 | yes |
| `FirstAidQA` | 5,550 | yes |
| `belvisk/First-Aid-Dataset` | 344 | yes |
| `crisis-data-v3-sft` | 1,770 | yes |
| `offline_practical_skills_qa_synthetic_filtered` | 500 | yes |
| `bushcraft_survival_gpt_oss_data_distilled` | 2,586 | yes |
| `badri55/First_aid__dataset` | 116 | yes |
| `medrescue` | 4,832 | yes |
| `us_army_fm_instruct` | 122 | no |
| `MedText` | 519 | yes |
| `healthcare_qa` | 12 | yes |
| `US-Army-Survival-Sharegpt` | 381 | no |
| `EMS-Knowledge` | 14,528 | no |
| `Crisis-QA` | 1,971 | eval only |

Current merged training rows after deduplication and core-source oversampling:

- `18,545`
