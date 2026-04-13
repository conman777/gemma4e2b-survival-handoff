import argparse
import json
import re
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-file",
        default=str(
            Path(__file__).resolve().parents[1] / "data" / "evals" / "survival_eval_suite.jsonl"
        ),
    )
    parser.add_argument("--responses-file", required=True)
    parser.add_argument("--label", default="manual")
    parser.add_argument("--model-name", default="manual_responses")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text):
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def match_group(text, group):
    matches = [keyword for keyword in group["keywords"] if keyword.lower() in text]
    return bool(matches), matches


def first_keyword_index(text, keywords):
    indices = [text.find(keyword.lower()) for keyword in keywords if keyword.lower() in text]
    return min(indices) if indices else None


def score_response(item, response_text):
    normalized = normalize_text(response_text)

    matched_required = []
    missing_required = []
    for group in item.get("required_groups", []):
        hit, keywords = match_group(normalized, group)
        if hit:
            matched_required.append({"name": group["name"], "keywords": keywords})
        else:
            missing_required.append(group["name"])

    matched_bonus = []
    for group in item.get("bonus_groups", []):
        hit, keywords = match_group(normalized, group)
        if hit:
            matched_bonus.append({"name": group["name"], "keywords": keywords})

    negative_hits = []
    for group in item.get("negative_groups", []):
        hit, keywords = match_group(normalized, group)
        if hit:
            negative_hits.append({"name": group["name"], "keywords": keywords})

    priority_hits = []
    priority_misses = []
    for pair in item.get("priority_pairs", []):
        earlier_idx = first_keyword_index(normalized, pair["earlier_keywords"])
        later_idx = first_keyword_index(normalized, pair["later_keywords"])
        if earlier_idx is not None and later_idx is not None and earlier_idx < later_idx:
            priority_hits.append(pair["name"])
        else:
            priority_misses.append(pair["name"])

    required_total = max(1, len(item.get("required_groups", [])))
    bonus_total = max(1, len(item.get("bonus_groups", [])))
    priority_total = max(1, len(item.get("priority_pairs", [])))

    required_score = len(matched_required) / required_total
    bonus_score = len(matched_bonus) / bonus_total if item.get("bonus_groups") else 1.0
    priority_score = len(priority_hits) / priority_total if item.get("priority_pairs") else 1.0

    weighted_score = (required_score * 70.0) + (bonus_score * 15.0) + (priority_score * 15.0)
    penalty = 15.0 * len(negative_hits)
    total_score = max(0.0, min(100.0, round(weighted_score - penalty, 1)))

    if total_score >= 85:
        verdict = "strong"
    elif total_score >= 70:
        verdict = "usable"
    elif total_score >= 50:
        verdict = "mixed"
    else:
        verdict = "weak"

    return {
        "score": total_score,
        "verdict": verdict,
        "matched_required": matched_required,
        "missing_required": missing_required,
        "matched_bonus": matched_bonus,
        "negative_hits": negative_hits,
        "priority_hits": priority_hits,
        "priority_misses": priority_misses,
    }


def build_summary(eval_rows, response_rows, label, model_name):
    eval_by_id = {row["id"]: row for row in eval_rows}
    response_by_id = {row["id"]: row for row in response_rows}

    missing_responses = [row_id for row_id in eval_by_id if row_id not in response_by_id]
    extra_responses = [row_id for row_id in response_by_id if row_id not in eval_by_id]
    if missing_responses:
        raise SystemExit(f"Responses file is missing ids: {', '.join(missing_responses)}")
    if extra_responses:
        raise SystemExit(f"Responses file has unknown ids: {', '.join(extra_responses)}")

    results = []
    for item in eval_rows:
        response_text = response_by_id[item["id"]]["response"]
        scored = score_response(item, response_text)
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "response": response_text,
                **scored,
            }
        )

    average_score = round(sum(row["score"] for row in results) / len(results), 1)
    category_scores = {}
    for row in results:
        category_scores.setdefault(row["category"], []).append(row["score"])
    category_summary = {
        category: round(sum(scores) / len(scores), 1)
        for category, scores in sorted(category_scores.items())
    }

    return {
        "eval_file": "",
        "prompt_count": len(eval_rows),
        "models": [
            {
                "label": label,
                "model_name": model_name,
                "average_score": average_score,
                "category_scores": category_summary,
                "results": results,
            }
        ],
    }


def write_outputs(output_dir, summary):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = output_path / f"survival_eval_manual_{timestamp}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_summary = summary["models"][0]
    lines = [
        "# Survival Response Scoring",
        "",
        f"## {model_summary['label']} ({model_summary['model_name']})",
        "",
        f"- Average score: `{model_summary['average_score']}`",
        "- Category scores:",
    ]
    for category, score in model_summary["category_scores"].items():
        lines.append(f"  - {category}: `{score}`")
    lines.extend(["", "| Prompt | Score | Verdict |", "|---|---:|---|"])
    for row in model_summary["results"]:
        lines.append(f"| `{row['id']}` | `{row['score']}` | `{row['verdict']}` |")
    lines.append("")

    md_path = output_path / f"survival_eval_manual_{timestamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main():
    args = parse_args()
    eval_rows = load_jsonl(args.eval_file)
    response_rows = load_jsonl(args.responses_file)
    summary = build_summary(eval_rows, response_rows, args.label, args.model_name)
    summary["eval_file"] = str(Path(args.eval_file).resolve())

    if args.output_dir:
        json_path, md_path = write_outputs(args.output_dir, summary)
        print(f"Wrote scoring JSON to {json_path}")
        print(f"Wrote scoring summary to {md_path}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
