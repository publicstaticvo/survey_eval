from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def extract_json_text(response: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    text = response.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def iter_sections(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, str):
        return [{"title": "document", "text": document}]
    if isinstance(document, dict):
        sections = document.get("sections") or document.get("paper", {}).get("sections")
        if isinstance(sections, list):
            return [s for s in sections if isinstance(s, dict)]
        text = document.get("text") or document.get("content") or ""
        return [{"title": document.get("title", "document"), "text": text}]
    return []


def iter_tables(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, dict):
        tables = document.get("tables") or document.get("paper", {}).get("tables") or []
        return [t for t in tables if isinstance(t, dict)]
    return []


@dataclass
class LLMCommentCandidate:
    candidate_id: str
    artifact_type: str
    artifact: str
    context: str
    source: str


def candidate_as_dict(candidate: LLMCommentCandidate) -> dict[str, Any]:
    return asdict(candidate)
