from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tools.utility.utils import normalize_text
from golden.analyze_r5_missing_references import reference_key, title_match


GOLDEN_DIR = Path(__file__).resolve().parent
DEFAULT_COVERAGE = GOLDEN_DIR / "R5-reference-survey-unseen-coverage.jsonl"
DEFAULT_R5 = GOLDEN_DIR / "R5-openalex.jsonl"
DEFAULT_OUT = GOLDEN_DIR / "R5-unseen-review-reasons.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def ref_title(reference: dict[str, Any]) -> str:
    return reference.get("title") or reference.get("topic") or ""


def ref_location(reference: dict[str, Any]) -> tuple[str, str]:
    location = reference.get("location") or {}
    return normalize_text(location.get("type", "")), normalize_text(location.get("link", ""))


def ref_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_loc = ref_location(left)
    right_loc = ref_location(right)
    if left_loc[1] and right_loc[1] and left_loc == right_loc:
        return True
    left_title = ref_title(left)
    right_title = ref_title(right)
    if left_title and right_title and title_match(left_title, right_title):
        return True
    return reference_key(left) == reference_key(right)


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def first_sentences(text: str, limit: int = 2) -> str:
    text = clean_space(text)
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    selected = []
    for sentence in sentences:
        if sentence:
            selected.append(sentence)
        if len(selected) >= limit:
            break
    return " ".join(selected)


def strip_reference_list(text: str) -> str:
    text = re.sub(r"\[[0-9,\s]+\]", "", text or "")
    text = re.sub(r"\([A-Za-z][^)]{0,80},?\s+\d{4}[a-z]?\)", "", text)
    return clean_space(text)


def summarize_reason(reference: dict[str, Any], evidence_items: list[dict[str, Any]], query: str) -> str:
    title = ref_title(reference)
    topic = reference.get("topic") or ""
    evidence_texts = [item.get("evidence", "") for item in evidence_items if item.get("evidence")]
    combined = " ".join(evidence_texts)
    combined_lower = combined.lower()

    if topic:
        return (
            f"Reviewer used it as a missing reference for the topic '{topic}', arguing that the survey should cover this thread within {query}."
        )

    if "first" in combined_lower or "earliest" in combined_lower:
        return "Reviewer saw this as an early or foundational work that should be cited to make the historical lineage accurate."
    if "recent" in combined_lower or "new" in combined_lower or "latest" in combined_lower:
        return "Reviewer considered it a relevant recent work that the survey omitted, so including it would make the literature coverage more current."
    if "extreme multi-label" in combined_lower:
        return "Reviewer wanted the survey to cover extreme multi-label learning as a related retrieval/modeling category, and this work was cited as an example."
    if "privacy" in combined_lower and ("distributed learning" in combined_lower or "decentralized learning" in combined_lower):
        return "Reviewer suggested it as an additional privacy-related distributed learning reference that would broaden the survey's coverage."
    if "retrosynthesis" in query.lower() or "retrosynthetic" in query.lower():
        return retrosynthesis_reason(title, combined_lower)
    if "generative models" in combined_lower and "causal representation" in combined_lower:
        return "Reviewer thought the survey missed related work connecting generative modeling with causal representation learning."
    if "graph transformers" in combined_lower:
        return "Reviewer argued that Graph Transformers are central enough to the topic that the survey should at least mention key references in that line."
    if "joint-distribution fidelity" in combined_lower:
        return "Reviewer wanted stronger coverage of joint-distribution fidelity metrics, so this work was listed as a relevant evaluation reference."
    if "hypertuning" in combined_lower:
        return "Reviewer wanted the efficient fine-tuning section to include hypertuning-style methods as a relevant related line of work."
    if "came out after the submission" in combined_lower:
        return "Reviewer noted that this was a post-submission but closely related paper, useful for making the survey up to date."
    if "more related work on" in combined_lower or "works on using" in combined_lower:
        return "Reviewer identified it as part of a related subarea that the survey's current coverage did not adequately include."
    if "theoretical" in combined_lower or "theory" in combined_lower or "understanding" in combined_lower:
        return "Reviewer wanted this cited because it provides theoretical grounding or explanation for the methods discussed in the survey."
    if "benchmark" in combined_lower or "evaluation" in combined_lower or "empirical" in combined_lower:
        return "Reviewer treated it as important benchmark or evaluation work that would strengthen the survey's empirical comparison."
    if "compare" in combined_lower or "comparison" in combined_lower or "traditional" in combined_lower:
        return "Reviewer thought it was needed to compare the surveyed approach against closely related alternatives."
    if "application" in combined_lower or "applications" in combined_lower:
        return "Reviewer viewed it as a relevant application-side reference that would broaden the survey's coverage."
    if (
        "worth discussing" in combined_lower
        or "consider citing" in combined_lower
        or "should consider citing" in combined_lower
        or "additional references" in combined_lower
        or "references include" in combined_lower
        or "missed this citation" in combined_lower
    ):
        return "Reviewer proposed it as an additional relevant citation needed for more complete literature coverage."
    if "not mention" in combined_lower or "not mentioned" in combined_lower or "missing" in combined_lower:
        return "Reviewer explicitly pointed to this work as a relevant omitted citation for the survey's topic."

    context = strip_reference_list(first_sentences(combined, limit=2))
    if context:
        return "Reviewer connected this work to a nearby part of the survey and treated it as relevant missing related work."
    if title:
        return f"Reviewer listed '{title}' as a relevant missing work for the survey topic."
    return "Reviewer listed this as a relevant missing work, but the available extracted evidence does not give a more specific rationale."


def retrosynthesis_reason(title: str, combined_lower: str) -> str:
    title_lower = title.lower()
    if "pistachio" in title_lower:
        return "Reviewer wanted the survey to mention Pistachio because it is a dataset used for retrosynthesis generalization testing or pretraining."
    if "fusionretro" in title_lower or "root-aligned" in title_lower:
        return "Reviewer listed it as an additional single-step retrosynthesis model that would make the model coverage more exhaustive."
    if "gap" in title_lower or "re-evaluating" in title_lower or "models matter" in title_lower or "syntheseus" in title_lower:
        return "Reviewer saw it as part of the literature that evaluates retrosynthesis planners more holistically and discusses pitfalls in existing approaches."
    if "retro-fallback" in title_lower:
        return "Reviewer treated it as a relevant retrosynthetic search/planning algorithm that should be covered for completeness."
    if "drug" in combined_lower:
        return "Reviewer noted it as retrosynthesis-driven drug-design work, possibly adjacent to the survey's main backward-search scope."
    return "Reviewer considered it an additional relevant retrosynthesis reference needed to make the survey more exhaustive."


def build_evidence_index(r5_records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_paper_id = defaultdict(list)
    for record in r5_records:
        if record.get("record_type") == "overall":
            continue
        if record.get("reason_label") != "R5":
            continue
        paper_id = record.get("paper_id")
        if paper_id is None:
            continue
        by_paper_id[int(paper_id)].append(record)
    return by_paper_id


def find_evidence(reference: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matched = []
    for record in records:
        for missed in record.get("missed_references") or []:
            if ref_matches(reference, missed):
                matched.append(record)
                break
    if matched:
        return matched

    title = ref_title(reference)
    title_norm = normalize_text(title)
    if not title_norm:
        return []
    for record in records:
        evidence_norm = normalize_text(record.get("evidence", ""))
        if title_norm and title_norm in evidence_norm:
            matched.append(record)
    return matched


def build_reason_records(coverage_path: Path, r5_path: Path) -> list[dict[str, Any]]:
    coverage_records = [
        item for item in read_jsonl(coverage_path)
        if item.get("record_type") == "unseen_reference_coverage"
    ]
    evidence_by_paper = build_evidence_index(read_jsonl(r5_path))
    outputs = []
    missing_evidence = 0
    for item in coverage_records:
        paper_id = int(item["paper_id"])
        reference = item.get("missing_reference") or {}
        evidence_items = find_evidence(reference, evidence_by_paper.get(paper_id, []))
        if not evidence_items:
            missing_evidence += 1
        outputs.append({
            "record_type": "unseen_review_reason",
            "paper_id": paper_id,
            "query": item.get("query", ""),
            "missing_reference": reference,
            "in_reference_survey_references": item.get("in_reference_survey_references"),
            "reason_summary": summarize_reason(reference, evidence_items, item.get("query", "")),
            "num_review_evidence_items": len(evidence_items),
            "review_evidence": [
                {
                    "reviewer": evidence.get("reviewer", ""),
                    "evidence": evidence.get("evidence", ""),
                    "missed_references": evidence.get("missed_references") or [],
                }
                for evidence in evidence_items
            ],
        })

    outputs.insert(0, {
        "record_type": "overall",
        "num_unseen_coverage_records": len(coverage_records),
        "num_records_without_matched_review_evidence": missing_evidence,
        "coverage_frequency": dict(sorted(Counter(
            "in" if item.get("in_reference_survey_references") else "not_in"
            for item in coverage_records
        ).items())),
    })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--r5", type=Path, default=DEFAULT_R5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    outputs = build_reason_records(args.coverage, args.r5)
    write_jsonl(args.out, outputs)
    print(f"wrote {args.out} records={len(outputs)}")


if __name__ == "__main__":
    main()


