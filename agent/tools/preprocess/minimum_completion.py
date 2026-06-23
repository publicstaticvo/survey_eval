from typing import Any, Dict, List
from .utils import get_top_level_section_titles


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


def _title_structure_check(paper: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
    title = (paper.get("title") or "").lower()
    survey_markers = ["survey", "review", "overview", "taxonomy", "summary", "synthesis"]
    has_title_marker = any(marker in title for marker in survey_markers)
    titles = [t.lower() for t in get_top_level_section_titles(paper)]
    has_intro_like = any(any(marker in t for marker in ["intro", "background", "overview"]) for t in titles)
    return has_title_marker or has_intro_like, {"title_marker": has_title_marker, "top_level_titles": titles}


def _title_matches_discussion_keywords(title: str) -> bool:
    lowered = (title or "").lower()
    return any(keyword in lowered for keyword in DISCUSSION_KEYWORDS)


def _discussion_candidate_sections(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = []

    def _walk(section: Dict[str, Any], parent_titles: List[str]):
        title = (section.get("title") or "").strip()
        if title and _title_matches_discussion_keywords(title):
            candidates.append(
                {
                    "section_id": section.get("section_id"),
                    "title": title,
                    "depth": len(parent_titles) + 1,
                }
            )
        else:
            for child in section.get("sections", []):  _walk(child, [*parent_titles, title])

    for section in paper.get("sections", []): _walk(section, [])
    return candidates


def minimum_completion(paper: Dict[str, Any]) -> Dict[str, List]:
    structure_ok, structure_details = _title_structure_check(paper)
    if not structure_ok:
        return {"minimum_check": {"status": "fail", "stage": "title_structure", "details": structure_details}}

    discussion_candidates = _discussion_candidate_sections(paper)
    if not discussion_candidates:
        return {"minimum_check": {"status": "fail", "stage": "discussion_candidates"}}
    return {
        "minimum_check": {
            "status": "pass",
            "discussion_candidates": discussion_candidates,
        }
    }
