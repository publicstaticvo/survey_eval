"""Adequacy-signal evidence-support detector.

The design follows recurring reviewer-comment clusters in
evidence_support_insufficient.jsonl after excluding hallucination, missing
reference, and writing-only concerns:
- unsupported strong claims,
- weak motivation chains,
- empirical-support gaps,
- proof/citation gaps,
- over-strong quantitative or universal claims.

The module prepares evidence packets and validates LLM comments. It returns
Adequacy records that may contribute to a calibrated score cap, because the
amount of support required for a survey-level argument is a graded expert judgment.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable


PROBLEM_TYPES = {
    "UNSUPPORTED_STRONG_CLAIM",
    "WEAK_MOTIVATION_CHAIN",
    "EMPIRICAL_SUPPORT_GAP",
    "PROOF_OR_CITATION_GAP",
    "OVERSTRONG_QUANTIFIER",
    "NO_COMMENT",
}

PROMPT = """You are auditing whether a survey claim has enough visible support.
Return JSON only.

Survey topic: {topic}

Candidate claim span:
{claim}

Local support from the same section:
{local_support}

Retrieved cited or literature-pool evidence:
{retrieved_support}

Decide whether this evidence supports an adequacy record about evidence
support. Use one problem_type from:
- UNSUPPORTED_STRONG_CLAIM: a strong conclusion lacks visible support.
- WEAK_MOTIVATION_CHAIN: a need/motivation claim skips a key link.
- EMPIRICAL_SUPPORT_GAP: empirical, benchmark, or sample-size support is needed.
- PROOF_OR_CITATION_GAP: theorem, derivation, or factual support lacks proof/citation.
- OVERSTRONG_QUANTIFIER: numerical, universal, or causal wording is too strong.
- NO_COMMENT: evidence is too weak.

Return exactly:
{
  "problem_type": "...",
  "comment": "one concise reviewer-facing comment",
  "claim_quote": "short exact quote from the candidate claim",
  "support_type_needed": "citation|proof|empirical|motivation|nuance|other",
  "available_support": "what support is visible",
  "missing_support": "what support appears absent or thin",
  "alternative_interpretation": "brief note on why competent reviewers may disagree",
  "confidence": 0.0
}
"""


@dataclass
class EvidenceSupportCandidate:
    candidate_id: str
    claim: str
    local_support: str
    source: str
    trigger: str


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


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9'\"])", text)
    return [_norm(part) for part in parts if len(_norm(part)) > 35]


def _trigger(sentence: str) -> str | None:
    patterns = [
        ("OVERSTRONG_QUANTIFIER", r"\b(always|never|all|none|indisputable|guarantee[sd]?|prove[sn]?|must|only|entirely)\b|\b\d+(?:\.\d+)?%\b"),
        ("EMPIRICAL_SUPPORT_GAP", r"\b(experiment|empirical|benchmark|evaluation|sample size|ablation|result|outperform|state-of-the-art|SOTA)\b"),
        ("PROOF_OR_CITATION_GAP", r"\b(theorem|proof|derive|derivation|lemma|citation needed|uncited|without citation|not cited)\b"),
        ("WEAK_MOTIVATION_CHAIN", r"\b(motivat|why|because|therefore|thus|hence|need to|allows? .* overcome|helps?)\b"),
        ("UNSUPPORTED_STRONG_CLAIM", r"\b(show|demonstrate|confirm|establish|reveal|indicate|suggest|conclude|critical|crucial|foundation|significant)\b"),
    ]
    lowered = sentence.lower()
    for name, pattern in patterns:
        if re.search(pattern, lowered):
            return name
    return None


def extract_candidates(document: Any, max_candidates: int = 40) -> list[EvidenceSupportCandidate]:
    candidates: list[EvidenceSupportCandidate] = []
    for section_index, section in enumerate(_iter_sections(document), 1):
        title = _norm(section.get("title")) or f"section {section_index}"
        text = _norm(section.get("text") or section.get("content") or section.get("paragraphs"))
        sents = _sentences(text)
        for sent_index, sent in enumerate(sents):
            trigger = _trigger(sent)
            if not trigger:
                continue
            window = sents[max(0, sent_index - 2): sent_index] + sents[sent_index + 1: sent_index + 3]
            candidates.append(EvidenceSupportCandidate(
                candidate_id=f"section-{section_index}-sent-{sent_index + 1}",
                claim=sent,
                local_support=" ".join(window)[:2200],
                source=title,
                trigger=trigger,
            ))
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def build_prompt(candidate: EvidenceSupportCandidate, topic: str, retrieved_support: str = "") -> str:
    return PROMPT.format(
        topic=topic,
        claim=candidate.claim,
        local_support=candidate.local_support or "None",
        retrieved_support=retrieved_support or "None",
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
    for field in [
        "comment",
        "claim_quote",
        "support_type_needed",
        "available_support",
        "missing_support",
        "alternative_interpretation",
        "confidence",
    ]:
        if field not in data:
            raise ValueError(f"missing field: {field}")
    confidence = data["confidence"]
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    return data


async def detect(
    document: Any,
    topic: str,
    llm_call: Callable[[str], Awaitable[str | dict[str, Any]]],
    retrieved_support: str = "",
    max_candidates: int = 40,
) -> list[dict[str, Any]]:
    comments = []
    for candidate in extract_candidates(document, max_candidates=max_candidates):
        response = await llm_call(build_prompt(candidate, topic, retrieved_support))
        data = parse_json_response(response)
        if data["problem_type"] == "NO_COMMENT":
            continue
        comments.append({
            "module": "cclass.evidence_support_insufficient",
            "report_role": "Adequacy signal",
            "candidate": asdict(candidate),
            **data,
        })
    return comments


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare evidence-support adequacy-signal prompts.")
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
