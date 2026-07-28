from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator


BASE = Path(__file__).resolve().parent
PLAN_PATH = BASE / "defect_injection_plans.json"
INPUT_ROOT = BASE.parent / "agent" / "test_inputs" / "corrupted_surveys"
REPORT_PATH = INPUT_ROOT / "application_report.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalized(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def section_label(section: dict[str, Any]) -> str:
    return f"{section.get('section_id', '')} {normalized(section.get('title', ''))}".strip()


def walk_sections(
    sections: list[dict[str, Any]],
    parent_path: str = "",
) -> Iterator[tuple[dict[str, Any], str, list[dict[str, Any]], int]]:
    for index, section in enumerate(sections):
        path = section_label(section)
        full_path = f"{parent_path} > {path}" if parent_path else path
        yield section, full_path, sections, index
        yield from walk_sections(section.get("sections", []) or [], full_path)


def sentence_list(paragraph: Any) -> list[dict[str, Any]]:
    if isinstance(paragraph, dict):
        return paragraph.get("sentences", []) or []
    if isinstance(paragraph, list):
        return paragraph
    return []


def iter_sentence_nodes(paper: dict[str, Any]) -> Iterator[dict[str, Any]]:
    roots = [paper.get("abstract"), *paper.get("sections", []), *paper.get("limitation", []), *paper.get("appendix", [])]

    def walk_section(section: dict[str, Any]) -> Iterator[dict[str, Any]]:
        for paragraph in section.get("paragraphs", []) or []:
            for sentence in sentence_list(paragraph):
                if isinstance(sentence, dict):
                    yield sentence
        for child in section.get("sections", []) or []:
            if isinstance(child, dict):
                yield from walk_section(child)

    for root in roots:
        if isinstance(root, dict):
            yield from walk_section(root)


def find_section(root_sections: list[dict[str, Any]], wanted_path: str) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    for section, path, siblings, index in walk_sections(root_sections):
        if path == wanted_path:
            return section, siblings, index
    raise KeyError(f"Section path not found: {wanted_path}")


def find_sentence(
    root_sections: list[dict[str, Any]],
    section_path: str,
    paragraph_index: int,
    sentence_index: int,
) -> dict[str, Any]:
    section, _siblings, _index = find_section(root_sections, section_path)
    paragraph = (section.get("paragraphs", []) or [])[paragraph_index - 1]
    sentence = sentence_list(paragraph)[sentence_index - 1]
    if not isinstance(sentence, dict):
        raise TypeError(f"Target is not a sentence: {section_path} P{paragraph_index} S{sentence_index}")
    return sentence


def remove_markers_from_text(text: str, markers: set[str]) -> str:
    def replace_group(match: re.Match[str]) -> str:
        raw_items = [part.strip() for part in match.group(1).split(",")]
        kept = [item for item in raw_items if item not in markers]
        if not kept:
            return ""
        return f"[{', '.join(kept)}]"

    result = re.sub(r"~?\s*\[([^\[\]]+)\]", replace_group, text)
    result = re.sub(r"\s+([,.;:])", r"\1", result)
    return re.sub(r" {2,}", " ", result).strip()


def remove_citation_keys(paper: dict[str, Any], keys: set[str]) -> dict[str, int]:
    removed_mentions = 0
    removed_markers = 0
    for sentence in iter_sentence_nodes(paper):
        citations = sentence.get("citations")
        if not isinstance(citations, dict):
            continue
        markers = {str(marker) for marker, key in citations.items() if str(key) in keys}
        if not markers:
            continue
        sentence["citations"] = {marker: key for marker, key in citations.items() if str(marker) not in markers}
        old_text = str(sentence.get("text", ""))
        new_text = remove_markers_from_text(old_text, markers)
        removed_mentions += len(markers)
        removed_markers += int(old_text != new_text)
        sentence["text"] = new_text

    bibliography = paper.get("citations")
    removed_bibliography = 0
    if isinstance(bibliography, dict):
        for key in keys:
            if key in bibliography:
                del bibliography[key]
                removed_bibliography += 1
    return {
        "removed_sentence_citation_mentions": removed_mentions,
        "updated_texts": removed_markers,
        "removed_bibliography_entries": removed_bibliography,
    }


def collect_section_id_mapping(section: dict[str, Any], old_prefix: str, new_prefix: str, mapping: dict[str, str]) -> None:
    current_old = str(section.get("section_id", ""))
    if current_old:
        suffix = current_old[len(old_prefix) :] if current_old.startswith(old_prefix) else ""
        current_new = new_prefix + suffix
        mapping[current_old] = current_new
        section["section_id"] = current_new
    for child_index, child in enumerate(section.get("sections", []) or [], 1):
        child_old = str(child.get("section_id", ""))
        child_new = f"{new_prefix}.{child_index}"
        if child_old:
            collect_section_id_mapping(child, child_old, child_new, mapping)


def renumber_siblings_after_deletion(siblings: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sibling in siblings:
        sid = str(sibling.get("section_id", ""))
        if "." not in sid:
            continue
        parent_id = sid.rsplit(".", 1)[0]
        break
    else:
        return mapping

    for index, sibling in enumerate(siblings, 1):
        old_id = str(sibling.get("section_id", ""))
        if not old_id:
            continue
        new_id = f"{parent_id}.{index}"
        collect_section_id_mapping(sibling, old_id, new_id, mapping)
    return {old: new for old, new in mapping.items() if old != new}


def replace_section_references(paper: dict[str, Any], mapping: dict[str, str]) -> int:
    if not mapping:
        return 0
    changed = 0
    replacements = sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True)
    for sentence in iter_sentence_nodes(paper):
        text = str(sentence.get("text", ""))
        updated = text
        for old, new in replacements:
            updated = re.sub(rf"(?<![\d.]){re.escape(old)}(?![\d.])", new, updated)
        if updated != text:
            sentence["text"] = updated
            changed += 1
    return changed


def apply_plan(plan: dict[str, Any]) -> dict[str, Any]:
    target_path = INPUT_ROOT / f"{plan['slug']}.json"
    document = load_json(target_path)
    paper = document["full_text"]
    sections = paper.get("sections", []) or []

    fact_changes = []
    for error in plan["fact_errors"]:
        sentence = find_sentence(
            sections,
            error["section_path"],
            error["paragraph_index"],
            error["sentence_index"],
        )
        actual = normalized(sentence.get("text", ""))
        expected = normalized(error["original_sentence"])
        if actual != expected:
            raise ValueError(
                f"{plan['slug']} factual target changed at {error['section_path']} "
                f"P{error['paragraph_index']} S{error['sentence_index']}: {actual!r}"
            )
        sentence["text"] = error["modified_sentence"]
        fact_changes.append(error["error_type"])

    scope = plan["structural_contradiction"]
    sentence = find_sentence(
        sections,
        scope["section_path"],
        scope["paragraph_index"],
        scope["sentence_index"],
    )
    actual = normalized(sentence.get("text", ""))
    expected = normalized(scope["original_sentence"])
    if actual != expected:
        raise ValueError(f"{plan['slug']} structural target changed: {actual!r}")
    sentence["text"] = scope["modified_sentence"]

    deleted_keys = set(plan["non_overlap_constraints"]["deleted_citation_keys"])
    citation_report = remove_citation_keys(paper, deleted_keys)

    subsection = plan["citation_or_topic_missing"]["delete_method_subsection"]
    renumber_mapping: dict[str, str] = {}
    reference_updates = 0
    deleted_subsection = None
    if subsection.get("path"):
        _section, siblings, index = find_section(sections, subsection["path"])
        deleted_subsection = siblings.pop(index)
        renumber_mapping = renumber_siblings_after_deletion(siblings)
        reference_updates = replace_section_references(paper, renumber_mapping)

    write_json(target_path, document)
    return {
        "slug": plan["slug"],
        "fact_errors_applied": fact_changes,
        "structural_contradiction_applied": True,
        "citation_deletions": citation_report,
        "deleted_subsection": deleted_subsection.get("title") if deleted_subsection else None,
        "section_id_mapping": renumber_mapping,
        "section_reference_texts_updated": reference_updates,
    }


def main() -> None:
    plans = load_json(PLAN_PATH)
    report = [apply_plan(plan) for plan in plans]
    write_json(REPORT_PATH, report)
    print(json.dumps({"applied": len(report), "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
