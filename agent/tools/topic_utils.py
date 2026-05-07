from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterable


GENERIC_SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "background",
    "preliminary",
    "preliminaries",
    "method",
    "methods",
    "methodology",
    "experiment",
    "experiments",
    "evaluation",
    "result",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "future work",
    "future works",
    "future direction",
    "future directions",
    "limitation",
    "limitations",
    "open problem",
    "open problems",
    "open question",
    "open questions",
    "related work",
    "references",
    "acknowledgment",
    "acknowledgement",
    "appendix",
]


def normalize_heading(title: str) -> str:
    value = re.sub(r"^[\s\d.ivxIVX]+[.)-]?\s*", "", title or "")
    value = value.replace("&", " and ").replace("/", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value).strip().lower()


def is_generic_heading(title: str) -> bool:
    normalized = normalize_heading(title)
    if not normalized:
        return True
    return any(keyword in normalized for keyword in GENERIC_SECTION_KEYWORDS)


def paragraph_to_text(paragraph: list[dict[str, Any]]) -> str:
    return " ".join(sentence.get("text", "") for sentence in paragraph if sentence.get("text")).strip()


def paragraphs_to_text(paragraphs: Iterable[list[dict[str, Any]]]) -> str:
    return "\n\n".join(text for paragraph in paragraphs if (text := paragraph_to_text(paragraph)))


def section_text(section: dict[str, Any], include_children: bool = True) -> str:
    blocks = [paragraphs_to_text(section.get("paragraphs", []))]
    if include_children:
        for child in section.get("sections", []) or []:
            child_text = section_text(child, include_children=True)
            if child_text:
                blocks.append(child_text)
    return "\n\n".join(block for block in blocks if block)


def iter_sections(section: dict[str, Any]):
    for child in section.get("sections", []) or []:
        yield child
        yield from iter_sections(child)


def flatten_section_titles(paper: dict[str, Any]) -> list[dict[str, str]]:
    titles: list[dict[str, str]] = []

    def walk(section: dict[str, Any], parent_id: str = ""):
        section_id = str(section.get("section_id") or "")
        if not section_id and parent_id:
            section_id = parent_id
        title = (section.get("title") or "").strip()
        if title:
            titles.append({"section_id": section_id, "section_title": title})
        for child in section.get("sections", []) or []:
            walk(child, section_id)

    for top in paper.get("sections", []) or []:
        walk(top)
    return titles


def top_level_section_blocks(paper: dict[str, Any]) -> list[dict[str, str]]:
    blocks = []
    for section in paper.get("sections", []) or []:
        section_id = str(section.get("section_id") or "")
        title = (section.get("title") or "").strip()
        pre_subsection_text = paragraphs_to_text(section.get("paragraphs", []))
        full_text = section_text(section, include_children=True)
        blocks.append(
            {
                "section_id": section_id,
                "section_title": title,
                "pre_subsection_text": pre_subsection_text,
                "full_text": full_text,
            }
        )
    return blocks


def paper_id_aliases(paper: dict[str, Any]) -> set[str]:
    ids = set(paper.get("ids") or [])
    if paper.get("id"):
        ids.add(paper["id"])
    return {str(item).replace("https://openalex.org/", "") for item in ids if item}


def citation_id_set(citations: dict[str, Any]) -> set[str]:
    ids = set()
    for info in citations.values():
        metadata = info.get("metadata") or info
        ids.update(paper_id_aliases(metadata))
    return ids


def paper_text(paper: dict[str, Any]) -> str:
    return f"{paper.get('title', '')}. {paper.get('abstract', '') or ''}".strip()


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(value[: len(fmt)], fmt)
        except Exception:
            continue
    return None


def months_between(start: datetime, end: datetime) -> float:
    days = max(1, (end - start).days)
    return max(days / 30.4375, 1.0)


def unique_by_id(papers: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for paper in papers:
        if paper.get("id"):
            result[paper["id"]] = paper
    return result
