from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tools.utility.utils import normalize_text


GOLDEN_DIR = Path(__file__).resolve().parent
LABELED_PATH = GOLDEN_DIR / "missing_specific_references_labeled.jsonl"
DATA_DIR = GOLDEN_DIR / "data"
REVIEW_FIELDS_DIR = GOLDEN_DIR / "review_fields"
OUT_PATH = GOLDEN_DIR / "R5-label-audit.jsonl"
SUMMARY_PATH = GOLDEN_DIR / "R5-label-audit-summary.json"

# Manual adjudication after reading the 122 R5 evidence snippets.
# These are primarily missing-topic/method-family comments where references are used as examples,
# or cases without a concrete citable article.
STRICT_FALSE_INDEXES = {
    5, 9, 10, 11, 25, 30, 33, 59, 65, 85, 89, 90, 98, 118, 119, 122,
}

# Broader interpretation: still count topic-with-examples as R5 if the reviewer supplied citable works.
BROAD_FALSE_INDEXES = {5, 9, 98, 119}


LANDMARK_TERMS = (
    "landmark", "seminal", "foundational", "classic", "core paper", "core papers",
    "crucial", "essential", "important work", "important works", "major related work",
    "state-of-the-art", "sota", "award-winning", "first work", "one of the first",
    "early paper", "main paradigm", "highly relevant", "noteworthy",
)

STRONG_DIRECTIVE_TERMS = (
    "must", "should", "please", "need to", "needs to", "have to", "has to",
    "fails to", "failed to", "lack", "lacks", "missing", "omits", "omit",
    "not included", "not discussed", "not mentioned", "should be cited",
    "should be included", "should be added", "include", "add reference", "add missing",
)

SOFT_DIRECTIVE_TERMS = (
    "may", "could", "might", "worth", "consider", "helpful", "useful",
    "recommend", "suggest", "would like", "probably", "wonder if",
)

RATIONALE_TERMS = (
    "because", "since", "as ", "which ", "used for", "used to", "allow",
    "provides", "proposes", "introduces", "demonstrate", "demonstrates",
    "state-of-the-art", "crucial", "essential", "foundational", "core",
    "first", "early", "important", "relevant to", "related to", "helps",
    "support", "strengthen", "comprehensive", "up-to-date", "coverage",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    text = text.lower()
    return any(term in text for term in terms)


def reference_name(reference: dict[str, Any]) -> str:
    return reference.get("title") or reference.get("topic") or ((reference.get("location") or {}).get("link")) or ""


def has_concrete_reference(record: dict[str, Any]) -> bool:
    refs = [reference_name(item) for item in record.get("missed_references") or []]
    return any(ref and normalize_text(ref) not in {"rewardlearning", "rewardhacking", "goalmisgeneralization"} for ref in refs)


def paper_data_path(paper_id: int) -> Path | None:
    matches = list(DATA_DIR.glob(f"{paper_id:03d}_*.json"))
    return matches[0] if matches else None


def source_field(record: dict[str, Any]) -> str:
    path = paper_data_path(int(record["paper_id"]))
    evidence_norm = normalize_text(record.get("evidence", ""))
    field_from_review_fields = source_field_from_review_fields(record, evidence_norm)
    if field_from_review_fields != "unknown":
        return field_from_review_fields
    if not path:
        return "unknown"
    data = json.loads(path.read_text(encoding="utf-8"))
    best_field = "unknown"
    best_score = 0
    evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_norm))
    for review in data.get("reviews") or []:
        content = review.get("content") or {}
        for field, value in content.items():
            value_norm = normalize_text(value or "")
            if not evidence_norm or not value_norm:
                continue
            if evidence_norm in value_norm:
                return field
            value_tokens = set(re.findall(r"[a-z0-9]+", value_norm))
            overlap = len(evidence_tokens & value_tokens)
            if overlap > best_score:
                best_field = field
                best_score = overlap
    if best_score >= max(4, int(0.25 * len(evidence_tokens or []))):
        return best_field
    return "unknown"


def source_field_from_review_fields(record: dict[str, Any], evidence_norm: str) -> str:
    path = paper_data_path(int(record["paper_id"]))
    if not path or not evidence_norm:
        return "unknown"
    data = json.loads(path.read_text(encoding="utf-8"))
    forum_id = data.get("openreview_forum_id") or ""
    if not forum_id:
        return "unknown"
    candidates = (
        ("weaknesses", "strengths_and_weaknesses"),
        ("questions", "requested_changes"),
        ("strengths", "summary_of_contributions"),
    )
    best_field = "unknown"
    best_score = 0
    evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_norm))
    for file_stem, mapped_field in candidates:
        field_path = REVIEW_FIELDS_DIR / f"{file_stem}.json"
        if not field_path.exists():
            continue
        payload = json.loads(field_path.read_text(encoding="utf-8"))
        for item in payload.get(forum_id, []) or []:
            value_norm = normalize_text(item.get("text", ""))
            if evidence_norm in value_norm:
                return mapped_field
            value_tokens = set(re.findall(r"[a-z0-9]+", value_norm))
            overlap = len(evidence_tokens & value_tokens)
            if overlap > best_score:
                best_field = mapped_field
                best_score = overlap
    if best_score >= max(4, int(0.25 * len(evidence_tokens or []))):
        return best_field
    return "unknown"


def tone_bucket(field: str, evidence: str) -> str:
    text = evidence.lower()
    if field == "strengths_and_weaknesses":
        return "weakness_sharp"
    if field == "requested_changes":
        return "requested_change_actionable"
    if contains_any(text, ("must", "major limitation", "fails", "insufficient", "crucial", "essential")):
        return "sharp_even_if_not_weakness_field"
    return "comment_or_question_milder"


def audit_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = [record for record in read_jsonl(LABELED_PATH) if record.get("reason_label") == "R5"]
    outputs = []
    for index, record in enumerate(source, 1):
        evidence = clean_text(record.get("evidence", ""))
        lower = evidence.lower()
        strict_true = index not in STRICT_FALSE_INDEXES
        broad_true = index not in BROAD_FALSE_INDEXES
        landmark = strict_true and contains_any(lower, LANDMARK_TERMS)
        strong_directive = contains_any(lower, STRONG_DIRECTIVE_TERMS)
        soft_directive = contains_any(lower, SOFT_DIRECTIVE_TERMS)
        explicit_why = contains_any(lower, RATIONALE_TERMS)
        must_with_reason = strict_true and strong_directive and explicit_why
        field = source_field(record)
        outputs.append({
            "record_type": "r5_label_audit",
            "r5_index": index,
            "paper_id": record.get("paper_id"),
            "reviewer": record.get("reviewer"),
            "strict_true_r5": strict_true,
            "broad_true_r5": broad_true,
            "strict_false_reason": "" if strict_true else "topic_or_method_family_rather_than_specific_reference",
            "landmark_or_important_work": landmark,
            "must_cite_with_explicit_reason": must_with_reason,
            "soft_suggestion": strict_true and soft_directive and not must_with_reason,
            "source_field": field,
            "tone_bucket": tone_bucket(field, evidence),
            "evidence": evidence,
            "missed_references": record.get("missed_references") or [],
        })

    strict_true_records = [item for item in outputs if item["strict_true_r5"]]
    summary = {
        "unit": "original R5-labeled evidence record",
        "num_original_r5": len(outputs),
        "strict_true_r5": len(strict_true_records),
        "strict_false_r5": len(outputs) - len(strict_true_records),
        "broad_true_r5": sum(1 for item in outputs if item["broad_true_r5"]),
        "broad_false_r5": sum(1 for item in outputs if not item["broad_true_r5"]),
        "among_strict_true": {
            "landmark_or_important_work": sum(1 for item in strict_true_records if item["landmark_or_important_work"]),
            "must_cite_with_explicit_reason": sum(1 for item in strict_true_records if item["must_cite_with_explicit_reason"]),
            "soft_suggestion_without_must_reason": sum(1 for item in strict_true_records if item["soft_suggestion"]),
        },
        "source_field_frequency": dict(sorted(Counter(item["source_field"] for item in strict_true_records).items())),
        "tone_bucket_frequency": dict(sorted(Counter(item["tone_bucket"] for item in strict_true_records).items())),
        "strict_false_indexes": [item["r5_index"] for item in outputs if not item["strict_true_r5"]],
        "broad_false_indexes": [item["r5_index"] for item in outputs if not item["broad_true_r5"]],
    }
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()
    records, summary = audit_records()
    write_jsonl(args.out, [{"record_type": "overall", **summary}, *records])
    write_json(args.summary_out, summary)
    print(f"wrote {args.out} records={len(records) + 1}")
    print(f"wrote {args.summary_out}")


if __name__ == "__main__":
    main()



