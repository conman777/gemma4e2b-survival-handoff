import argparse
import json
import random
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a practical survival instructor. Give calm, prioritized, low-drama guidance. "
    "Focus on water, shelter, signaling, first aid, and risk reduction. State uncertainty "
    "clearly and avoid pretending a dangerous shortcut is safe."
)

# Patterns that indicate a GPT-style refusal baked into source data.
REFUSAL_PATTERNS = re.compile(
    r"(?i)(I'?m sorry.{0,10}(can'?t|cannot|unable)|I can'?t (comply|help|assist)|"
    r"as an AI|I'?m not able to|I cannot provide)",
)

# Minimum acceptable assistant response length (chars).
MIN_ASSISTANT_LEN = 50

DATASET_ROW_CAPS = {
    # Large generic first-aid corpora can drown out the narrower
    # survival-first behavior we want on this benchmark.
    "firstaidqa": 3500,
    "medrescue": 3000,
}

DATASET_REPEAT_MULTIPLIERS = {
    # Small, high-alignment local datasets should appear more often than
    # their raw row counts suggest.
    "survival_seed": 6,
    "survival_priority_gapfix": 6,
    "bushcraft_survival_gpt_oss_data_distilled": 3,
    "offline_practical_skills_qa_synthetic_filtered": 2,
}

# Explicit training-pool membership. This prevents "hold" or reference-like
# datasets from silently entering the merged corpus just because they were
# normalized.
TRAINING_DATASETS = {
    "survival_seed",
    "survival_priority_gapfix",
    "firstaidqa",
    "first_aid_dataset_belvisk",
    "crisis_data_v3_sft",
    "offline_practical_skills_qa_synthetic_filtered",
    "bushcraft_survival_gpt_oss_data_distilled",
    "first_aid_dataset_badri55",
    "medrescue",
    "medtext",
    "healthcare_qa",
}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default=str(root / "data" / "open_sources" / "hf"),
    )
    parser.add_argument(
        "--output-root",
        default=str(root / "data" / "normalized"),
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records, path: Path):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_first_turn_pair(messages):
    if not isinstance(messages, list) or not messages:
        return "", ""
    user_text = ""
    assistant_text = ""
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user" and not user_text:
            user_text = content
        if role == "assistant":
            assistant_text = content
    return user_text, assistant_text


def cap_records(records, dataset_name: str):
    max_rows = DATASET_ROW_CAPS.get(dataset_name)
    if not max_rows or len(records) <= max_rows:
        return records, len(records)

    random.seed(42)
    capped = random.sample(records, max_rows)
    print(
        f"    {dataset_name}: capped from {len(records)} to {max_rows} rows "
        "(deterministic sample)"
    )
    return capped, len(records)


def normalize_seed_dataset(output_root: Path):
    root = Path(__file__).resolve().parents[1]
    seed_path = root / "data" / "survival_seed.jsonl"
    records = []
    for row in read_jsonl(seed_path):
        user_text, assistant_text = extract_first_turn_pair(row.get("messages"))
        if not is_valid_pair(user_text, assistant_text):
            continue
        if not passes_quality_filter(assistant_text):
            continue
        records.append(to_messages(user_text, assistant_text))

    output_path = output_root / "survival_seed_messages.jsonl"
    write_jsonl(records, output_path)
    return {"dataset": "survival_seed", "rows": len(records), "path": str(output_path)}


def normalize_curated_gapfix_dataset(output_root: Path):
    root = Path(__file__).resolve().parents[1]
    curated_path = root / "data" / "curated" / "survival_priority_gapfix.jsonl"
    if not curated_path.exists():
        return None

    records = []
    for row in read_jsonl(curated_path):
        user_text, assistant_text = extract_first_turn_pair(row.get("messages"))
        if not is_valid_pair(user_text, assistant_text):
            continue
        if not passes_quality_filter(assistant_text):
            continue
        records.append(to_messages(user_text, assistant_text))

    output_path = output_root / "survival_priority_gapfix_messages.jsonl"
    write_jsonl(records, output_path)
    return {
        "dataset": "survival_priority_gapfix",
        "rows": len(records),
        "path": str(output_path),
    }


def copy_jsonl_lines(paths, destination: Path):
    with destination.open("w", encoding="utf-8") as out_handle:
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    if line.strip():
                        out_handle.write(line)


def to_messages(user_text: str, assistant_text: str):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]
    }


def is_valid_pair(user_text, assistant_text):
    """Check that both sides are non-empty strings."""
    return (
        isinstance(user_text, str)
        and isinstance(assistant_text, str)
        and user_text.strip()
        and assistant_text.strip()
    )


def passes_quality_filter(assistant_text: str) -> bool:
    """Reject refusals and very short / truncated responses."""
    text = assistant_text.strip()
    if len(text) < MIN_ASSISTANT_LEN:
        return False
    if REFUSAL_PATTERNS.search(text):
        return False
    return True


# ---------------------------------------------------------------------------
# Existing dataset normalizers (with quality filters added)
# ---------------------------------------------------------------------------

def normalize_firstaidqa(input_root: Path, output_root: Path):
    records = []
    for row in read_jsonl(input_root / "firstaidqa" / "train.jsonl"):
        if not is_valid_pair(row.get("question"), row.get("answer")):
            continue
        if not passes_quality_filter(row["answer"]):
            continue
        records.append(to_messages(row["question"], row["answer"]))
    records, pre_cap_rows = cap_records(records, "firstaidqa")
    output_path = output_root / "firstaidqa_messages.jsonl"
    write_jsonl(records, output_path)
    return {
        "dataset": "firstaidqa",
        "rows": len(records),
        "path": str(output_path),
        "pre_cap_rows": pre_cap_rows,
    }


def normalize_belvisk(input_root: Path, output_root: Path):
    """Normalizes belvisk first-aid data.

    Some rows have a JSON-array-formatted user prompt (multiple phrasings).
    We pick the first phrasing from the array instead of training on the raw
    JSON list string.
    """
    records = []
    for row in read_jsonl(input_root / "first_aid_dataset_belvisk" / "train.jsonl"):
        question = row.get("question", "")
        answer = row.get("answer", "")
        # Fix list-format user prompts: '["How to treat X?", "What to do..."]'
        if isinstance(question, str) and question.strip().startswith("["):
            try:
                variants = json.loads(question)
                if isinstance(variants, list) and variants:
                    question = str(variants[0])
            except (json.JSONDecodeError, TypeError):
                pass
        if not is_valid_pair(question, answer):
            continue
        if not passes_quality_filter(answer):
            continue
        records.append(to_messages(question, answer))
    output_path = output_root / "first_aid_dataset_belvisk_messages.jsonl"
    write_jsonl(records, output_path)
    return {"dataset": "first_aid_dataset_belvisk", "rows": len(records), "path": str(output_path)}


def normalize_crisis_sft(input_root: Path, output_root: Path):
    records = []
    for row in read_jsonl(input_root / "crisis_data_v3_sft" / "train.jsonl"):
        if not is_valid_pair(row.get("instruction"), row.get("output")):
            continue
        user_parts = [row["instruction"].strip()]
        if row.get("input"):
            user_parts.append(row["input"].strip())
        user_text = "\n\n".join(part for part in user_parts if part)
        if not passes_quality_filter(row["output"]):
            continue
        records.append(to_messages(user_text, row["output"]))
    output_path = output_root / "crisis_data_v3_sft_messages.jsonl"
    write_jsonl(records, output_path)
    return {"dataset": "crisis_data_v3_sft", "rows": len(records), "path": str(output_path)}


def normalize_offline_practical(input_root: Path, output_root: Path):
    allowed_topics = {"Wilderness Survival Basics", "Basic First Aid", "Essential Knots"}
    records = []
    for row in read_jsonl(input_root / "offline_practical_skills_qa_synthetic" / "train.jsonl"):
        if row["topic"] not in allowed_topics:
            continue
        if not is_valid_pair(row.get("question"), row.get("answer")):
            continue
        if not passes_quality_filter(row["answer"]):
            continue
        records.append(
            {
                **to_messages(row["question"], row["answer"]),
                "source_topic": row["topic"],
            }
        )
    output_path = output_root / "offline_practical_skills_filtered_messages.jsonl"
    write_jsonl(records, output_path)
    return {
        "dataset": "offline_practical_skills_qa_synthetic_filtered",
        "rows": len(records),
        "path": str(output_path),
        "topics": sorted(allowed_topics),
    }


def normalize_bushcraft_distilled(input_root: Path, output_root: Path):
    records = []
    for row in read_jsonl(input_root / "bushcraft_survival_gpt_oss_data_distilled" / "train.jsonl"):
        if not is_valid_pair(row.get("instruction"), row.get("output")):
            continue
        user_parts = [row["instruction"].strip()]
        if row.get("input"):
            user_parts.append(row["input"].strip())
        user_text = "\n\n".join(part for part in user_parts if part)
        if not passes_quality_filter(row["output"]):
            continue
        records.append(to_messages(user_text, row["output"]))
    output_path = output_root / "bushcraft_survival_distilled_messages.jsonl"
    write_jsonl(records, output_path)
    return {
        "dataset": "bushcraft_survival_gpt_oss_data_distilled",
        "rows": len(records),
        "path": str(output_path),
    }


def normalize_badri55(input_root: Path, output_root: Path):
    records = []
    for row in read_jsonl(input_root / "first_aid_dataset_badri55" / "train.jsonl"):
        responses = row.get("responses") or []
        patterns = row.get("patterns") or []
        if not responses:
            continue
        answer = responses[0]
        if not passes_quality_filter(answer):
            continue
        for pattern in patterns:
            if not is_valid_pair(pattern, answer):
                continue
            records.append(to_messages(pattern, answer))
    output_path = output_root / "first_aid_dataset_badri55_messages.jsonl"
    write_jsonl(records, output_path)
    return {
        "dataset": "first_aid_dataset_badri55",
        "rows": len(records),
        "path": str(output_path),
    }


def normalize_crisis_eval(input_root: Path, output_root: Path):
    records = []
    for row in read_jsonl(input_root / "crisis_qa" / "train.jsonl"):
        records.append(
            {
                "prompt": row["Question"].strip(),
                "expected_answer": row["Correct Answer"].strip(),
                "domain": row["Domain"].strip(),
                "distractors": [
                    value.strip()
                    for key, value in row.items()
                    if key.startswith("Distractor") and value
                ],
            }
        )
    output_path = output_root / "crisis_qa_eval.jsonl"
    write_jsonl(records, output_path)
    return {"dataset": "crisis_qa_eval", "rows": len(records), "path": str(output_path)}


# ---------------------------------------------------------------------------
# New dataset normalizers
# ---------------------------------------------------------------------------

def normalize_medrescue(input_root: Path, output_root: Path):
    """ericrisco/medrescue — emergency medicine and rescue QA.

    Columns: input, context, output, source (12 distinct values).
    We keep only the survival/emergency-relevant sub-sources to avoid
    pulling the model toward generic clinical medicine.
    """
    KEEP_SOURCES = {
        "extreme_firstaid_qa_dataset",
        "first_aid_dataset",
        "first_aid_qa_dataset",
        "rescue_qa_dataset",
        "synthetic_disaster_reports",
    }

    records = []
    skipped_sources = {}
    source_dir = input_root / "medrescue"
    if not source_dir.exists():
        return None

    for split_file in sorted(source_dir.glob("*.jsonl")):
        for row in read_jsonl(split_file):
            row_source = row.get("source", "")
            if row_source not in KEEP_SOURCES:
                skipped_sources[row_source] = skipped_sources.get(row_source, 0) + 1
                continue

            user_text = row.get("input", "")
            assistant_text = row.get("output", "")
            context = row.get("context", "")

            # Only prepend context if it's very short and essential to
            # disambiguate the question.  At inference the model won't
            # have reference paragraphs, so we avoid training that
            # dependency.
            if context and context.strip() and len(context) < 200:
                user_text = f"{user_text.strip()}\n\nContext: {context.strip()}"

            if not is_valid_pair(user_text, assistant_text):
                continue
            if not passes_quality_filter(assistant_text):
                continue
            records.append(to_messages(user_text, assistant_text))

    if skipped_sources:
        print(f"    medrescue: skipped non-survival sources: {skipped_sources}")

    records, pre_cap_rows = cap_records(records, "medrescue")

    output_path = output_root / "medrescue_messages.jsonl"
    write_jsonl(records, output_path)
    return {
        "dataset": "medrescue",
        "rows": len(records),
        "path": str(output_path),
        "kept_sources": sorted(KEEP_SOURCES),
        "pre_cap_rows": pre_cap_rows,
    }


def normalize_us_army_fm_instruct(input_root: Path, output_root: Path):
    """Heralax/us-army-fm-instruct — multi-turn Army Field Manual instruction data.

    Expected format: conversations column with list of {from: human/gpt, value: ...}
    or similar ShareGPT-style structure.  We filter for survival/medical/field-relevant
    topics by keyword matching.
    """
    survival_keywords = re.compile(
        r"(?i)(\bsurviv\w*|\bfirst.?aid\b|\bmedic\w*|\bcasualt\w*|\bwound\w*|"
        r"\bbleed\w*|\bfracture\w*|\bburn\w*|\bshock\b|\btriage\b|\bevacuat\w*|"
        r"\bshelter\b|\bwater\b|\brescue\b|\bhypotherm\w*|\bhypertherm\w*|"
        r"\bheat[- ]?(stroke|exhaust|illness)\b|\bfrostbite\b|\bdehydrat\w*|"
        r"\btourniquet\b|\bsplint\w*|\bbandag\w*|\bairway\b|\bbreath\w*|\bCPR\b|"
        r"\bpoison\w*|\bvenom\w*|\bsnake\w*|\binsect\w*|\bbear\b|\blightning\b|"
        r"\bflood\w*|\bwildfire\b|\bearthquake\b|\bhurricane\b|\btornado\b|"
        r"\bavalanche\b|\bdrown\w*|\bnavigation\b|\bcompass\b|\bmap(?:\s+reading)?\b|"
        r"\bfield hygiene\b|\bsanitat\w*|\bCBRN\b|\bdecontamin\w*|\bbivouac\b|"
        r"\bration\b|\bforag\w*|\bedible\b|\bMRE\b|\bcanteen\b|\bpurif\w*|"
        r"\bCASEVAC\b|\bMEDEVAC\b|\blitter\b|\bstretcher\b|\bcombat lifesaver\b)"
    )
    doctrine_reject = re.compile(
        r"(?i)(\barea of operations\b|\boperational area\b|\bAO\b|\bAOR\b|"
        r"\bjoint security area\b|\bJSA\b|\bnoncontiguous\b|\bclose air support\b|"
        r"\bCAS\b|\bSEAD\b|\bfire support\b|\bFSCOORD\b|\bFSE\b|\bHPT\b|"
        r"\bIPOE\b|\bISR\b|\bHUMINT\b|\bBCT\b|\bHROB\b|\bJICO\b|\bJTAC\b|"
        r"\bGI&S\b|\bLSCO\b|\bdeep area\b|\bcollection asset\b|\bintelligence\b|"
        r"\btargeting\b|\btarget development\b|\bunified action\b|\bcommander\b|"
        r"\bheadquarters\b|\bbrigade\b|\bcorps\b|\bcivil affairs\b|"
        r"\boperational contract support\b|\bsecurity force\b|\benemy\b|"
        r"\bwarfighting\b|\bcounterinsurg\w*|\bair defense\b|"
        r"\bdental\b|\boral\b|\bmaxillofacial\b|\bSBIRS\b|\bmissile\b|"
        r"\bdetainee\w*|\binternee\w*|\boccupying power\b|\bPOW\b|"
        r"\bprisoner of war\b|\brepatriat\w*|\btaxes?\b|\bwater production\b|"
        r"\bsupply chain\b|\bdistribution\b|\bsurvivability operations\b|"
        r"\btactical vehicles\b|\bweapons systems\b|\bcontamination markers\b|"
        r"\bplanning symbols\b|\bhide and surveillance\b|\bsurveillance site\b)"
    )

    records = []
    source_dir = input_root / "us_army_fm_instruct"
    if not source_dir.exists():
        return None

    for split_file in sorted(source_dir.glob("*.jsonl")):
        for row in read_jsonl(split_file):
            # Try ShareGPT conversation format first
            conversations = row.get("conversations") or row.get("conversation") or []
            if conversations:
                human_turns = [t for t in conversations if t.get("from") in ("human", "user")]
                gpt_turns = [t for t in conversations if t.get("from") in ("gpt", "assistant")]
                if human_turns and gpt_turns:
                    user_text = human_turns[0].get("value", "")
                    assistant_text = gpt_turns[0].get("value", "")
                else:
                    continue
            else:
                # Fallback to flat columns
                user_text = row.get("input") or row.get("instruction") or row.get("question") or ""
                assistant_text = row.get("output") or row.get("answer") or row.get("response") or ""

            if not is_valid_pair(user_text, assistant_text):
                continue

            # Filter to survival/medical relevance and reject broad doctrinal topics.
            combined = f"{user_text} {assistant_text}"
            if not survival_keywords.search(user_text):
                continue
            if doctrine_reject.search(combined):
                continue

            if not passes_quality_filter(assistant_text):
                continue
            records.append(to_messages(user_text, assistant_text))

    output_path = output_root / "us_army_fm_instruct_messages.jsonl"
    write_jsonl(records, output_path)
    return {
        "dataset": "us_army_fm_instruct",
        "rows": len(records),
        "path": str(output_path),
    }


def normalize_us_army_survival_sharegpt(input_root: Path, output_root: Path):
    """AquaV/US-Army-Survival-Sharegpt — ShareGPT conversations from FM 21-76.

    Expected format: conversations column with [{from: system/human/gpt, value: ...}].
    """
    records = []
    source_dir = input_root / "us_army_survival_sharegpt"
    if not source_dir.exists():
        return None

    for split_file in sorted(source_dir.glob("*.jsonl")):
        for row in read_jsonl(split_file):
            conversations = row.get("conversations") or row.get("conversation") or []
            if not conversations:
                continue

            human_turns = [t for t in conversations if t.get("from") in ("human", "user")]
            gpt_turns = [t for t in conversations if t.get("from") in ("gpt", "assistant")]

            if not human_turns or not gpt_turns:
                continue

            user_text = human_turns[0].get("value", "")
            assistant_text = gpt_turns[0].get("value", "")

            if not is_valid_pair(user_text, assistant_text):
                continue
            if not passes_quality_filter(assistant_text):
                continue
            records.append(to_messages(user_text, assistant_text))

    output_path = output_root / "us_army_survival_sharegpt_messages.jsonl"
    write_jsonl(records, output_path)
    return {
        "dataset": "us_army_survival_sharegpt",
        "rows": len(records),
        "path": str(output_path),
    }


def normalize_medtext(input_root: Path, output_root: Path):
    """BI55/MedText — doctor-validated clinical cases.

    Expected columns: Prompt, Completion (or similar).
    """
    keep_pattern = re.compile(
        r"(?i)(trauma|injur|fracture|sprain|strain|dislocat|burn|bleed|hemorrhag|"
        r"wound|lacerat|abrasion|concussion|head injur|chest injur|abdominal injur|"
        r"shock|tourniquet|splint|bandage|CPR|resuscit|airway|breathing|"
        r"dehydrat|hypotherm|frostbite|heat stroke|heat exhaustion|heat illness|"
        r"bite|sting|envenom|poison|drown|near[- ]drown|exposure|sunburn)"
    )
    records = []
    source_dir = input_root / "medtext"
    if not source_dir.exists():
        return None

    for split_file in sorted(source_dir.glob("*.jsonl")):
        for row in read_jsonl(split_file):
            user_text = (
                row.get("Prompt") or row.get("prompt")
                or row.get("input") or row.get("question") or ""
            )
            assistant_text = (
                row.get("Completion") or row.get("completion")
                or row.get("output") or row.get("answer") or ""
            )

            if not is_valid_pair(user_text, assistant_text):
                continue
            if not keep_pattern.search(f"{user_text} {assistant_text}"):
                continue
            if not passes_quality_filter(assistant_text):
                continue
            records.append(to_messages(user_text, assistant_text))

    output_path = output_root / "medtext_messages.jsonl"
    write_jsonl(records, output_path)
    return {
        "dataset": "medtext",
        "rows": len(records),
        "path": str(output_path),
    }


def normalize_ems_knowledge(input_root: Path, output_root: Path):
    """Xueren/EMS-Knowledge — raw EMS text chunks (not QA pairs).

    We convert these into instruction-style pairs by using the section/topic
    as a question prompt and the text chunk as the answer. Only chunks with
    clear topic headings produce good training rows.
    """
    records = []
    source_dir = input_root / "ems_knowledge"
    if not source_dir.exists():
        return None

    for split_file in sorted(source_dir.glob("*.jsonl")):
        for row in read_jsonl(split_file):
            topic = row.get("topic") or row.get("section") or row.get("title") or ""
            text = row.get("text") or row.get("content") or row.get("chunk") or ""

            if not topic.strip() or not text.strip():
                continue

            # Build a natural question from the topic
            user_text = f"Explain the key points about {topic.strip()} in an emergency medical context."
            assistant_text = text.strip()

            if not passes_quality_filter(assistant_text):
                continue
            records.append(to_messages(user_text, assistant_text))

    output_path = output_root / "ems_knowledge_messages.jsonl"
    write_jsonl(records, output_path)
    return {
        "dataset": "ems_knowledge",
        "rows": len(records),
        "path": str(output_path),
    }


def normalize_healthcare_qa(input_root: Path, output_root: Path):
    """adrianf12/healthcare-qa-dataset — small curated healthcare QA."""
    keep_pattern = re.compile(
        r"(?i)(CPR|concussion|stroke|FAST|bleed|bleeding|fracture|burn|wound|"
        r"lacerat|shock|tourniquet|splint|airway|breathing|drown|dehydrat|"
        r"hypotherm|heat stroke|heat exhaustion|bite|sting|poison|snake)"
    )
    records = []
    source_dir = input_root / "healthcare_qa"
    if not source_dir.exists():
        return None

    for split_file in sorted(source_dir.glob("*.jsonl")):
        for row in read_jsonl(split_file):
            user_text = (
                row.get("prompt") or row.get("question")
                or row.get("input") or row.get("instruction") or ""
            )
            assistant_text = (
                row.get("completion") or row.get("answer")
                or row.get("output") or row.get("response") or ""
            )

            if not is_valid_pair(user_text, assistant_text):
                continue
            if not keep_pattern.search(f"{user_text} {assistant_text}"):
                continue
            if not passes_quality_filter(assistant_text):
                continue
            records.append(to_messages(user_text, assistant_text))

    output_path = output_root / "healthcare_qa_messages.jsonl"
    write_jsonl(records, output_path)
    return {
        "dataset": "healthcare_qa",
        "rows": len(records),
        "path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Post-processing: deduplication
# ---------------------------------------------------------------------------

def _apply_dataset_oversampling(merged_path: Path, summaries: list):
    """Append extra copies of rows from high-priority datasets."""
    added_rows = 0
    applied = []

    with merged_path.open("a", encoding="utf-8") as merged_handle:
        for summary in summaries:
            repeats = DATASET_REPEAT_MULTIPLIERS.get(summary["dataset"], 1)
            if repeats <= 1:
                continue

            source_path = Path(summary["path"])
            if not source_path.exists():
                continue

            lines = [
                line
                for line in source_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            extra_repeats = repeats - 1
            for _ in range(extra_repeats):
                for line in lines:
                    merged_handle.write(line + "\n")
            added_rows += len(lines) * extra_repeats
            applied.append(f"{summary['dataset']}x{repeats}")

    if applied:
        print(
            f"  Dataset oversampling: added {added_rows} extra rows "
            f"({', '.join(applied)})"
        )


def deduplicate_merged(merged_path: Path) -> int:
    """Remove rows with duplicate user prompts, keeping the first occurrence."""
    seen_prompts = set()
    kept = []
    removed = 0

    with merged_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user_msg = next(
                (m["content"] for m in row["messages"] if m["role"] == "user"), ""
            )
            normalized_prompt = user_msg.strip().lower()
            if normalized_prompt in seen_prompts:
                removed += 1
                continue
            seen_prompts.add(normalized_prompt)
            kept.append(line)

    with merged_path.open("w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")

    return removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    # --- Existing datasets ---
    summaries = [
        normalize_seed_dataset(output_root),
        normalize_curated_gapfix_dataset(output_root),
        normalize_firstaidqa(input_root, output_root),
        normalize_belvisk(input_root, output_root),
        normalize_crisis_sft(input_root, output_root),
        normalize_offline_practical(input_root, output_root),
        normalize_bushcraft_distilled(input_root, output_root),
        normalize_badri55(input_root, output_root),
        normalize_crisis_eval(input_root, output_root),
    ]

    # --- New datasets (skip gracefully if not yet downloaded) ---
    new_normalizers = [
        normalize_medrescue,
        normalize_us_army_fm_instruct,
        normalize_us_army_survival_sharegpt,
        normalize_medtext,
        normalize_ems_knowledge,
        normalize_healthcare_qa,
    ]
    for normalizer in new_normalizers:
        result = normalizer(input_root, output_root)
        if result is not None:
            summaries.append(result)
            print(f"  [ok] {result['dataset']}: {result['rows']} rows")
        else:
            print(f"  [skip] {normalizer.__name__}: source not downloaded yet")

    summaries = [summary for summary in summaries if summary is not None]

    # --- Build merged training file ---
    merged_training_files = [
        summary["path"]
        for summary in summaries
        if summary["dataset"] in TRAINING_DATASETS
    ]
    # First merge without oversampling, then deduplicate, then apply
    # priority oversampling so intentional repeats aren't removed.
    merged_training_path = output_root / "survival_training_candidates_merged.jsonl"
    copy_jsonl_lines(merged_training_files, merged_training_path)

    # --- Deduplicate (before oversampling) ---
    dup_count = deduplicate_merged(merged_training_path)
    print(f"\n  Deduplication: removed {dup_count} duplicate user prompts")

    # --- Apply dataset-specific oversampling for high-alignment sources ---
    _apply_dataset_oversampling(merged_training_path, summaries)

    # Count final rows
    final_rows = 0
    with merged_training_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                final_rows += 1
    print(f"  Final merged training rows: {final_rows}")

    (output_root / "normalized_summary.json").write_text(
        json.dumps(
            {
                "system_prompt": SYSTEM_PROMPT,
                "artifacts": summaries,
                "merged_training_candidates": merged_training_files,
                "training_datasets": sorted(TRAINING_DATASETS),
                "dataset_row_caps": DATASET_ROW_CAPS,
                "dataset_repeat_multipliers": DATASET_REPEAT_MULTIPLIERS,
                "merged_training_path": str(merged_training_path),
                "deduplication": {"duplicates_removed": dup_count},
                "final_training_rows": final_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
