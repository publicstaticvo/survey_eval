from typing import Any, Dict, List

from ..utility.paper_elements import Paper, Section
from ..utility.content_walk import get_top_level_section_titles


DISCUSSION_KEYWORDS = [
    "future",
    "limitation",
    "open problem",
    "open question",
    "challenge",
    "discussion",
    "outlook",
    "conclusion",
    "prospect",
    "frontier",
    "unsolved",
    "direction",
]


def _title_structure_check(paper: Paper) -> tuple[bool, Dict[str, Any]]:
    title = (paper.title or "").lower()
    survey_markers = ["survey", "review", "overview", "taxonomy", "summary", "synthesis"]
    has_title_marker = any(marker in title for marker in survey_markers)
    titles = [title.lower() for title in get_top_level_section_titles(paper)]
    has_intro_like = any(any(marker in title for marker in ["intro", "background", "overview"]) for title in titles)
    return has_title_marker or has_intro_like, {"title_marker": has_title_marker, "top_level_titles": titles}


def _title_matches_discussion_keywords(title: str) -> bool:
    lowered = (title or "").lower()
    return any(keyword in lowered for keyword in DISCUSSION_KEYWORDS)


def _discussion_candidate_sections(paper: Paper) -> List[Dict[str, Any]]:
    candidates = []

    def _walk(section: Section, section_id: str, parent_titles: List[str]):
        title = section.name.strip()
        if title and _title_matches_discussion_keywords(title):
            candidates.append({"section_id": section_id, "title": title, "depth": len(parent_titles) + 1})
        else:
            for index, child in enumerate(section.children):
                child_id = f"{section_id}.{index + 1}" if section_id else str(index + 1)
                _walk(child, child_id, [*parent_titles, title])

    for index, section in enumerate(paper.children):
        _walk(section, str(index + 1), [])
    return candidates


def minimum_completion(paper: Paper) -> Dict[str, List]:
    structure_ok, structure_details = _title_structure_check(paper)
    if not structure_ok:
        return {"minimum_check": {"status": "fail", "stage": "title_structure", "details": structure_details}}

    discussion_candidates = _discussion_candidate_sections(paper)
    if not discussion_candidates:
        return {"minimum_check": {"status": "fail", "stage": "discussion_candidates"}}
    return {"minimum_check": {"status": "pass", "discussion_candidates": discussion_candidates}}
