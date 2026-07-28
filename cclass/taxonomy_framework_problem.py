"""Comment-only taxonomy/framework detector.

The design follows recurring reviewer-comment clusters in
survey_eval/golden/review_analyze/taxonomy_framework_problem.jsonl:
- unclear category definitions or boundary conditions,
- overlapping or non-mutually-exclusive categories,
- mixed organizational axes in one hierarchy,
- likely misplacement of items under a taxonomy parent.

This module does not decide that a taxonomy issue is a score-bearing weakness or score cap.
It prepares localized evidence packets and validates LLM comments that can be
reported for human review.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable


PROBLEM_TYPES = {
    "UNCLEAR_BOUNDARY",
    "OVERLAPPING_CATEGORIES",
    "MIXED_ORGANIZING_AXES",
    "LIKELY_MISPLACEMENT",
    "NO_COMMENT",
}

PROMPT = """You are auditing a survey paper taxonomy or organizing framework.
Return JSON only.

Survey topic: {topic}

Candidate taxonomy artifact:
{artifact}

Nearby explanatory text:
{context}

Optional external evidence:
{external_evidence}

Decide whether this artifact supports a comment-only finding about the
survey taxonomy/framework. Use one problem_type from:
- UNCLEAR_BOUNDARY: category definitions or inclusion boundaries are unclear.
- OVERLAPPING_CATEGORIES: sibling categories are not mutually exclusive.
- MIXED_ORGANIZING_AXES: one hierarchy mixes method/task/dataset/application/etc.
- LIKELY_MISPLACEMENT: an item appears under a questionable parent category.
- NO_COMMENT: evidence is too weak.

Return exactly:
{
  "problem_type": "...",
  "comment": "one concise reviewer-facing comment",
  "implicated_labels": ["..."],
  "survey_quote": "short exact quote from the artifact or context",
  "evidence_summary": "why the evidence suggests this comment",
  "alternative_interpretation": "brief note on why competent reviewers may disagree",
  "confidence": 0.0
}
"""


@dataclass
class TaxonomyCandidate:
    artifact_id: str
    artifact_type: str
    artifact: str
    context: str
    source: str


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _iter_sections(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, str):
        return [{"title": "document", "text": document}]
    if isinstance(document, dict):
        sections = document.get("sections") or document.get("paper", {}).get("sections")
        if isinstance(sections, list):
            return [s for s in sections if isinstance(s, dict)]
        text = document.get("text") or document.get("content") or ""
        return [{"title": document.get("title", "document"), "text": text}]
    return []


def _iter_tables(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, dict):
        tables = document.get("tables") or document.get("paper", {}).get("tables") or []
        return [t for t in tables if isinstance(t, dict)]
    return []


def extract_candidates(document: Any, max_candidates: int = 24) -> list[TaxonomyCandidate]:
    candidates: list[TaxonomyCandidate] = []
    taxonomy_terms = re.compile(
        r"\b(taxonom|framework|categor|classification|hierarch|dimension|axis|group|type|family)\b",
        re.I,
    )
    risk_terms = re.compile(
        r"\b(mixed|other|misc|general|application|task|dataset|benchmark|method|model|metric|stage)\b",
        re.I,
    )

    for index, section in enumerate(_iter_sections(document), 1):
        title = _norm(section.get("title"))
        text = _norm(section.get("text") or section.get("content") or section.get("paragraphs"))
        combined = f"{title}. {text}"
        if taxonomy_terms.search(combined):
            snippet = combined[:2500]
            candidates.append(TaxonomyCandidate(
                artifact_id=f"section-{index}",
                artifact_type="section",
                artifact=title or snippet[:200],
                context=snippet,
                source=title or f"section {index}",
            ))
        elif risk_terms.search(title) and len(text) > 200:
            candidates.append(TaxonomyCandidate(
                artifact_id=f"section-{index}",
                artifact_type="section-title",
                artifact=title,
                context=combined[:1800],
                source=title or f"section {index}",
            ))

    for index, table in enumerate(_iter_tables(document), 1):
        caption = _norm(table.get("caption") or table.get("title"))
        body = _norm(table.get("text") or table.get("rows") or table.get("content"))
        if taxonomy_terms.search(f"{caption} {body}"):
            candidates.append(TaxonomyCandidate(
                artifact_id=f"table-{index}",
                artifact_type="table",
                artifact=caption or f"table {index}",
                context=body[:2500],
                source=caption or f"table {index}",
            ))

    return candidates[:max_candidates]


def build_prompt(candidate: TaxonomyCandidate, topic: str, external_evidence: str = "") -> str:
    return PROMPT.format(
        topic=topic,
        artifact=candidate.artifact,
        context=candidate.context,
        external_evidence=external_evidence or "None",
    )


def parse_json_response(response: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(response, dict):
        data = response
    else:
        text = response.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if match:
            text = match.group(1).strip()
        data = json.loads(text)
    problem_type = data["problem_type"]
    if problem_type not in PROBLEM_TYPES:
        raise ValueError(f"invalid problem_type: {problem_type}")
    for field in ["comment", "implicated_labels", "survey_quote", "evidence_summary", "alternative_interpretation", "confidence"]:
        if field not in data:
            raise ValueError(f"missing field: {field}")
    if not isinstance(data["implicated_labels"], list):
        raise ValueError("implicated_labels must be a list")
    confidence = data["confidence"]
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    return data


async def detect(
    document: Any,
    topic: str,
    llm_call: Callable[[str], Awaitable[str | dict[str, Any]]],
    external_evidence: str = "",
    max_candidates: int = 24,
) -> list[dict[str, Any]]:
    comments = []
    for candidate in extract_candidates(document, max_candidates=max_candidates):
        response = await llm_call(build_prompt(candidate, topic, external_evidence))
        data = parse_json_response(response)
        if data["problem_type"] == "NO_COMMENT":
            continue
        comments.append({
            "module": "cclass.taxonomy_framework_problem",
            "report_role": "Comment-only",
            "candidate": asdict(candidate),
            **data,
        })
    return comments


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare taxonomy/framework comment-only prompts.")
    parser.add_argument("input", type=Path, help="Parsed survey JSON, or plain text file.")
    parser.add_argument("--topic", default="", help="Survey topic string.")
    parser.add_argument("--output", type=Path, help="Write candidate prompt packets as JSONL.")
    args = parser.parse_args()

    raw = args.input.read_text(encoding="utf-8")
    try:
        document: Any = json.loads(raw)
    except json.JSONDecodeError:
        document = raw
    rows = [
        {"candidate": asdict(c), "prompt": build_prompt(c, args.topic)}
        for c in extract_candidates(document)
    ]
    output = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if args.output:
        args.output.write_text(output + ("\n" if output else ""), encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
