import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import unsloth  # noqa: F401
import torch
from unsloth import FastModel


SYSTEM_PROMPT = (
    "You are a calm survival advisor. Prioritize immediate safety, injury control, "
    "exposure, water, shelter, signaling, and uncertainty-aware decision making. "
    "Give practical steps in priority order and avoid risky overconfidence."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-file",
        default=str(
            Path(__file__).resolve().parents[1] / "data" / "evals" / "survival_eval_suite.jsonl"
        ),
    )
    parser.add_argument("--base-model", default="google/gemma-4-E2B-it")
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("base", "adapter"),
        default=("base", "adapter"),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json-output-path", default="")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_eval_items(path):
    items = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


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


def generate_response(model, tokenizer, prompt, max_new_tokens):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def evaluate_model(label, model_name, items, max_seq_length, max_new_tokens):
    print(f"Loading {label} model: {model_name}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastModel.for_inference(model)

    results = []
    for index, item in enumerate(items, start=1):
        print(f"[{label}] {index}/{len(items)} {item['id']}")
        response = generate_response(model, tokenizer, item["prompt"], max_new_tokens)
        score = score_response(item, response)
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "response": response,
                **score,
            }
        )

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    average_score = round(sum(row["score"] for row in results) / len(results), 1)
    category_scores = {}
    for row in results:
        category_scores.setdefault(row["category"], []).append(row["score"])
    category_summary = {
        category: round(sum(scores) / len(scores), 1)
        for category, scores in sorted(category_scores.items())
    }

    return {
        "label": label,
        "model_name": model_name,
        "average_score": average_score,
        "category_scores": category_summary,
        "results": results,
    }


def write_outputs(output_dir, summary):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = output_path / f"survival_eval_{timestamp}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown_lines = ["# Survival Eval Summary", ""]
    for model_summary in summary["models"]:
        markdown_lines.append(
            f"## {model_summary['label']} ({model_summary['model_name']})"
        )
        markdown_lines.append("")
        markdown_lines.append(f"- Average score: `{model_summary['average_score']}`")
        markdown_lines.append("- Category scores:")
        for category, score in model_summary["category_scores"].items():
            markdown_lines.append(f"  - {category}: `{score}`")
        markdown_lines.append("")
        markdown_lines.append("| Prompt | Score | Verdict |")
        markdown_lines.append("|---|---:|---|")
        for row in model_summary["results"]:
            markdown_lines.append(
                f"| `{row['id']}` | `{row['score']}` | `{row['verdict']}` |"
            )
        markdown_lines.append("")

    if len(summary["models"]) == 2:
        left, right = summary["models"]
        markdown_lines.append("## Delta")
        markdown_lines.append("")
        markdown_lines.append(
            f"- `{right['label']}` minus `{left['label']}` average: "
            f"`{round(right['average_score'] - left['average_score'], 1)}`"
        )
        markdown_lines.append("")

    md_path = output_path / f"survival_eval_{timestamp}.md"
    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return json_path, md_path


def build_subprocess_command(args, model_label, json_output_path):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--eval-file",
        str(Path(args.eval_file).resolve()),
        "--models",
        model_label,
        "--max-seq-length",
        str(args.max_seq_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--json-output-path",
        str(json_output_path),
    ]
    if args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    if model_label == "base":
        command.extend(["--base-model", args.base_model])
    elif model_label == "adapter":
        command.extend(["--adapter-dir", args.adapter_dir])
    return command


def evaluate_in_subprocesses(args):
    model_labels = [label for label in args.models]
    with tempfile.TemporaryDirectory(prefix="survival-eval-") as temp_dir:
        summaries = []
        for label in model_labels:
            json_output_path = Path(temp_dir) / f"{label}.json"
            command = build_subprocess_command(args, label, json_output_path)
            subprocess.run(command, check=True)
            summaries.append(json.loads(json_output_path.read_text(encoding="utf-8")))
    return summaries


def main():
    args = parse_args()
    items = load_eval_items(args.eval_file)
    if args.limit > 0:
        items = items[: args.limit]

    model_requests = []
    if "base" in args.models:
        model_requests.append(("base", args.base_model))
    if "adapter" in args.models:
        if not args.adapter_dir:
            raise SystemExit("--adapter-dir is required when evaluating the adapter model")
        model_requests.append(("adapter", args.adapter_dir))

    if len(model_requests) > 1 and not args.json_output_path:
        summaries = evaluate_in_subprocesses(args)
    else:
        summaries = []
        for label, model_name in model_requests:
            summaries.append(
                evaluate_model(
                    label=label,
                    model_name=model_name,
                    items=items,
                    max_seq_length=args.max_seq_length,
                    max_new_tokens=args.max_new_tokens,
                )
            )

    combined = {
        "eval_file": str(Path(args.eval_file).resolve()),
        "prompt_count": len(items),
        "models": summaries,
    }

    if args.json_output_path:
        Path(args.json_output_path).write_text(json.dumps(summaries[0], indent=2), encoding="utf-8")
    elif args.output_dir:
        json_path, md_path = write_outputs(args.output_dir, combined)
        print(f"Wrote eval JSON to {json_path}")
        print(f"Wrote eval summary to {md_path}")

    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
