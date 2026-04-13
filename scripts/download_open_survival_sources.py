import argparse
import json
import re
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

import requests
from datasets import load_dataset


USER_AGENT = "gemma4e2b-survival-dataset-fetcher/1.0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "open_sources"
            / "open_sources_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[1] / "data" / "open_sources"),
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Download optional lower-trust sources in addition to recommended ones.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload sources even if local copies already exist.",
    )
    return parser.parse_args()


def read_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))["sources"]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_enabled_sources(sources, include_optional: bool):
    for source in sources:
        if source["status"] == "gated":
            yield source
            continue
        if source["status"] == "optional" and not include_optional:
            continue
        yield source


def clean_html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p>", "\n\n", html)
    html = re.sub(r"(?i)</h[1-6]>", "\n\n", html)
    html = re.sub(r"(?i)</li>", "\n", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = unescape(html)
    html = re.sub(r"\r\n?", "\n", html)
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def save_jsonl(dataset_split, path: Path):
    with path.open("w", encoding="utf-8") as handle:
        for row in dataset_split:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_hf_dataset(source, output_root: Path, force: bool):
    target_dir = output_root / "hf" / source["id"]
    ensure_dir(target_dir)
    marker_path = target_dir / "_complete.json"
    if marker_path.exists() and not force:
        return {
            "id": source["id"],
            "kind": source["kind"],
            "status": "skipped_existing",
            "target_dir": str(target_dir),
        }

    dataset_dict = load_dataset(source["dataset_id"])
    split_summaries = {}
    for split_name, split in dataset_dict.items():
        split_path = target_dir / f"{split_name}.jsonl"
        save_jsonl(split, split_path)
        split_summaries[split_name] = {
            "rows": len(split),
            "columns": split.column_names,
            "path": str(split_path),
        }

    source_metadata = {
        **source,
        "split_summaries": split_summaries,
    }
    write_json(target_dir / "source.json", source_metadata)
    write_json(marker_path, {"complete": True})
    return {
        "id": source["id"],
        "kind": source["kind"],
        "status": "downloaded",
        "target_dir": str(target_dir),
        "splits": split_summaries,
    }


def extension_from_url(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    if parsed.path.lower().endswith(".pdf") or "pdf" in content_type.lower():
        return ".pdf"
    if parsed.path.lower().endswith(".json") or "json" in content_type.lower():
        return ".json"
    return ".html"


def download_web_source(source, output_root: Path, force: bool):
    target_dir = output_root / "web" / source["id"]
    ensure_dir(target_dir)
    marker_path = target_dir / "_complete.json"
    if marker_path.exists() and not force:
        return {
            "id": source["id"],
            "kind": source["kind"],
            "status": "skipped_existing",
            "target_dir": str(target_dir),
        }

    response = requests.get(
        source["url"],
        timeout=60,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    extension = extension_from_url(source["url"], content_type)
    raw_path = target_dir / f"raw{extension}"

    if extension == ".pdf":
        raw_path.write_bytes(response.content)
        extracted_path = None
        text_length = None
    else:
        raw_path.write_text(response.text, encoding="utf-8")
        extracted_text = clean_html_to_text(response.text)
        extracted_path = target_dir / "extracted.txt"
        extracted_path.write_text(extracted_text, encoding="utf-8")
        text_length = len(extracted_text)

    source_metadata = {
        **source,
        "raw_path": str(raw_path),
        "extracted_path": str(extracted_path) if extracted_path else None,
        "content_type": content_type,
        "text_length": text_length,
    }
    write_json(target_dir / "source.json", source_metadata)
    write_json(marker_path, {"complete": True})
    return {
        "id": source["id"],
        "kind": source["kind"],
        "status": "downloaded",
        "target_dir": str(target_dir),
        "raw_path": str(raw_path),
        "extracted_path": str(extracted_path) if extracted_path else None,
        "text_length": text_length,
    }


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    sources = list(iter_enabled_sources(read_manifest(manifest_path), args.include_optional))
    results = []
    failures = []

    for source in sources:
        try:
            if source["status"] == "gated":
                result = {
                    "id": source["id"],
                    "kind": source["kind"],
                    "status": "gated_not_downloaded",
                    "notes": source.get("notes"),
                    "dataset_id": source.get("dataset_id"),
                }
            elif source["kind"] == "hf_dataset":
                result = download_hf_dataset(source, output_root, force=args.force)
            elif source["kind"] == "web_page":
                result = download_web_source(source, output_root, force=args.force)
            else:
                raise ValueError(f"Unsupported source kind: {source['kind']}")
            results.append(result)
            print(f"[ok] {source['id']} -> {result['status']}")
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": source["id"], "error": str(exc)})
            print(f"[error] {source['id']} -> {exc}")

    summary = {
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "include_optional": args.include_optional,
        "downloaded_or_skipped": results,
        "failures": failures,
    }
    write_json(output_root / "download_summary.json", summary)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
