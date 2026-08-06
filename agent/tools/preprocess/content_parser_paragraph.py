from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from ..utility.content_walk import paragraph_to_text
from ..utility.paper_elements import Paper, Section
from ..utility.tool_config import ToolConfig
from .content_parser_evidence import EvidenceAnchoredContentClient


class ParagraphContentParser:
    """Extract evidence-anchored content per paragraph and merge it by section."""

    def __init__(self, config: ToolConfig):
        self.llm = EvidenceAnchoredContentClient(config)
        self.last_report = {"module": "content_paragraph", "success_count": 0, "error_count": 0, "errors": []}

    @staticmethod
    def _paragraph_text(paragraph) -> str:
        return paragraph_to_text(paragraph, include_environments=True)

    def _targets(self, paper: Paper, only_missing: bool):
        targets = []

        def walk(section: Section):
            # 2.3 is structural extraction, so every textual section is eligible.
            if (not only_missing or (not section.parsed_contents or section.parsed_contents.get("parser_version") != "paragraph_v2")):
                for paragraph in section.paragraphs:
                    text = self._paragraph_text(paragraph)
                    if text:
                        targets.append((section, paragraph, text))
            for child in section.children:
                walk(child)

        if paper.abstract is not None:
            walk(paper.abstract)
        for section in paper.children:
            walk(section)
        for section in [*paper.limitation, *paper.appendix]:
            walk(section)
        return targets

    @staticmethod
    def _merge(section_results: list[dict[str, Any]]) -> dict[str, Any]:
        topics: list[str] = []
        topic_details: list[dict[str, Any]] = []
        objects: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []
        seen_objects = set()
        seen_topics = set()
        for result in section_results:
            for label, detail in zip(result.get("topics", []), result.get("topic_details", [])):
                if label not in seen_topics:
                    seen_topics.add(label)
                    topics.append(label)
                    topic_details.append(detail)
            for item in result.get("objects", []):
                key = (item.get("name"), tuple(item.get("citation_keys", [])), tuple(item.get("topics", [])))
                if key not in seen_objects:
                    seen_objects.add(key)
                    objects.append(item)
            discarded.extend(result.get("discarded_objects", []))
        return {"parser_version": "paragraph_v2", "topics": topics, "topic_details": topic_details, "objects": objects, "discarded_objects": discarded}

    async def __call__(self, paper: Paper, only_missing: bool = False) -> Paper:
        targets = self._targets(paper, only_missing)
        tasks = [asyncio.create_task(self.llm.call(inputs={
            "paper_title": paper.title,
            "section_title": section.name or "Abstract",
            "section_text": text,
        })) for section, _paragraph, text in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        section_index: dict[int, Section] = {}
        errors = []
        for index, ((section, paragraph, _), result) in enumerate(zip(targets, results)):
            key = id(section)
            section_index[key] = section
            if isinstance(result, Exception):
                errors.append({"index": index, "section": section.name, "error": repr(result)})
            else:
                paragraph.entities = [
                    {
                        "name": item["name"],
                        "sentence_has_citation": bool(item.get("citation_keys")),
                        "alias_pairs": "",
                        "alternative_names": [],
                        "evidence_span": item.get("evidence_span"),
                    }
                    for item in result.get("objects", [])
                    if item.get("name")
                ]
                paragraph.alias_pairs = []
                paragraph.entities_classified = True
                grouped[key].append(result)
        for key, section_results in grouped.items():
            section_index[key].parsed_contents = self._merge(section_results)
        self.last_report = {
            "module": "content_paragraph",
            "success_count": len(grouped),
            "error_count": len(errors),
            "errors": errors,
            "paragraph_count": len(targets),
        }
        logging.info("paragraph content parsing: %d paragraphs, %d sections, %d errors", len(targets), len(grouped), len(errors))
        return paper

