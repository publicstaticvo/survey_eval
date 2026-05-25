from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_JSON = ROOT / "review.json"
OUTPUT_JSONL = ROOT / "semantic_review_weakness_inputs.jsonl"

NEGATIVE_FIELDS = [
    "weaknesses",
    "strengths_and_weaknesses",
    "requested_changes",
    "reasons_to_reject",
    "Reasons_to_reject",
    "questions",
    "Questions_for_the_Authors",
    "questions_for_the_authors",
    "additional_feedback",
    "additional_comments",
    "review",
    "metareview",
    "Relecture",
]


def strip_strengths(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"(?is)(?:\*\*)?\s*weaknesses?\s*:?\s*(?:\*\*)?", text)
    if match:
        return text[match.end() :].strip()
    text = re.sub(r"(?is)(?:\*\*)?\s*strengths?\s*:?\s*(?:\*\*)?.*?(?=(?:\*\*)?\s*weaknesses?\s*:)", "", text)
    return text.strip()


def review_text(review: dict) -> str:
    chunks = []
    for field in NEGATIVE_FIELDS:
        value = review.get(field, "")
        if not isinstance(value, str) or not value.strip():
            continue
        if field == "strengths_and_weaknesses":
            value = strip_strengths(value)
        chunks.append(f"[{field}]\n{value.strip()}")
    return "\n\n".join(chunks).strip()


def main() -> None:
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for forum_id, reviews in data.items():
            rows = []
            for index, review in enumerate(reviews, start=1):
                text = review_text(review)
                if text:
                    rows.append(
                        {
                            "review_index": index,
                            "reviewer_id": review.get("reviewer_id", ""),
                            "text": text,
                        }
                    )
            handle.write(json.dumps({"forum_id": forum_id, "reviews": rows}, ensure_ascii=False) + "\n")
    print(f"wrote {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
