import json
from pathlib import Path


DATA_DIR = Path("golden/data")
OUT_DIR = Path("golden")
FIELDS = ("strengths", "weaknesses", "questions", "requested_changes", "Relecture",
          'strengths_and_weaknesses', 'reasons_to_reject', 'reason_to_accept', "questions_for_the_authors", 
          'Reasons_to_reject', 'Reason_to_accept', "Questions_for_the_Authors")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_input_files() -> list[Path]:
    return sorted(path for path in DATA_DIR.glob("*.json") if not path.name.startswith("_"))


def reviewer_id(review: dict) -> str:
    return review["signature"] or review["review_id"]


def main():
    outputs, venues = {}, {}
    for path in iter_input_files():
        record = read_json(path)
        forum_id = record["openreview_forum_id"]
        domain = record['domain'].split("/")[0]
        venues[domain] = venues.get(domain, 0) + 1
        for review in record["reviews"]:
            content = review["content"]
            rid = reviewer_id(review)
            output = {"reviewer_id": rid}
            for field in FIELDS:
                text = content.get(field, "")
                if not isinstance(text, str) or not text.strip(): continue
                output[field] = text
            if len(output) == 1: continue
            outputs.setdefault(forum_id, []).append(output)

    print(venues)
    path = OUT_DIR / "review.json"
    path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {path} forums={len(outputs)}")


if __name__ == "__main__":
    main()
