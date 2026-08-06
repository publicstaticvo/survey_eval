"""Keep only paper content for Claude/LLM evaluation and save review-derived dates."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1] / "agent" / "test_inputs"
PDF_ROOT = ROOT / "pdf_content"
DATE_MANIFEST = ROOT / "pdf_content_publication_dates.json"
DEFAULT_DATE = "2026-06-30"


def _has_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_content(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_content(item) for item in value)
    return value is not None


def _review_timestamp(review: dict[str, Any]) -> int | None:
    values = []
    for key in ("cdate", "mdate"):
        value = review.get(key)
        if isinstance(value, (int, float)):
            values.append(int(value))
        elif isinstance(value, str) and value.strip().isdigit():
            values.append(int(value.strip()))
    return min(values) if values else None


def _date_from_timestamp(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date().isoformat()


def _date_from_record(record: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    timestamps = [_review_timestamp(review) for review in record.get("reviews") or [] if isinstance(review, dict)]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    if timestamps:
        timestamp_ms = min(timestamps)
        return {"publication_date": _date_from_timestamp(timestamp_ms), "timestamp_ms": timestamp_ms, "source": "earliest_review"}
    if previous and previous.get("publication_date"):
        return previous
    submission_date = record.get("submission_date")
    if isinstance(submission_date, str) and submission_date.strip():
        return {"publication_date": submission_date[:10], "source": "submission_date_fallback"}
    return {"publication_date": DEFAULT_DATE, "source": "default_fallback"}


def _filename_title(path: Path) -> str:
    stem = re.sub(r"^\d+_", "", path.stem)
    stem = re.sub(r"_[A-Za-z0-9]{8,12}$", "", stem)
    return re.sub(r"_+", " ", stem).strip()

def _paper_completeness(paper: dict[str, Any]) -> dict[str, bool]:
    return {
        "title": _has_content(paper.get("title")),
        "abstract": _has_content(paper.get("abstract")),
        "body": _has_content(paper.get("paragraphs")) or _has_content(paper.get("sections")),
        "references": _has_content(paper.get("citations")) or _has_content(paper.get("bibliography")),
    }


def main() -> None:
    previous = {}
    if DATE_MANIFEST.is_file():
        previous = json.loads(DATE_MANIFEST.read_text(encoding="utf-8"))
    manifest = {}
    counts = {key: 0 for key in ("title", "abstract", "body", "references")}
    processed = 0
    failures = []

    for path in sorted(PDF_ROOT.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            paper = record.get("paper")
            if not isinstance(paper, dict):
                raise ValueError("missing paper object")
            if not _has_content(paper.get("title")):
                paper["title"] = _filename_title(path)
            completeness = _paper_completeness(paper)
            for key, present in completeness.items():
                counts[key] += int(present)
            manifest[path.name] = _date_from_record(record, previous.get(path.name))
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps({"paper": paper}, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
            processed += 1
        except Exception as exc:
            failures.append({"file": path.name, "error": str(exc)})

    DATE_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"processed={processed}")
    print(f"complete_title={counts['title']}/{processed}")
    print(f"complete_abstract={counts['abstract']}/{processed}")
    print(f"complete_body={counts['body']}/{processed}")
    print(f"complete_references={counts['references']}/{processed}")
    print(f"date_manifest={DATE_MANIFEST}")
    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
