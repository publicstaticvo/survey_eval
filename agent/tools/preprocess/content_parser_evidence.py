from __future__ import annotations

import asyncio
import logging
from typing import Any

import jsonschema

from ..prompts import CONTENT_PARSE_WITH_TOPICS, CONTENT_PARSE_WITH_TOPICS_SCHEMA
from ..utility.citation_utils import citation_keys as normalize_citation_keys
from ..utility.content_walk import paragraphs_to_text
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper, Section
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json


class EvidenceAnchoredContentClient(AsyncChat):
    """Keep normalized object names, but require verbatim source evidence for every object."""

    PROMPT = CONTENT_PARSE_WITH_TOPICS.replace(
        "(1) research objects -- specific methods, models, datasets, benchmarks, or frameworks from the literature that this section discusses",
        "(1) research objects -- only named or explicitly identified methods, models, datasets, benchmarks, frameworks, tasks, or systems from the literature; do not extract generic noun phrases or abstract properties",
    ).replace(
        "- label: copied verbatim from the text, not invented.",
        "- label: a concise topic label; it may be normalized, but it must have a verbatim evidence_span from the section text unless it is the exact section title.",
    ).replace(
        "For each object, only assign it to a specific sub-topic if you can quote a verbatim span from section_text that directly ties that object to that sub-topic. If no such direct textual link exists, assign the object to the default section topic.",
        "For each object, name may be normalized and need not be a literal substring. However, evidence_span is mandatory for every non-null object name and must be copied verbatim from section_text. Use null name when no named research object is explicitly supported. Generic phrases such as objectives, engagement, architectures, or challenges are not objects unless the text identifies them as a named research artifact or cites a work.",
    )

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response: str, context: dict[str, Any]) -> dict[str, Any]:
        result = extract_json(response)
        if "entities" in result and "objects" not in result:
            result["objects"] = result.pop("entities")
        jsonschema.validate(result, CONTENT_PARSE_WITH_TOPICS_SCHEMA)
        source = context["source"]
        # The parser provides the canonical section topic. Model-generated topic aliases are
        # retained only when their evidence span can be verified in the paragraph.
        topics = [context["section_title"]]
        details = [{"label": context["section_title"], "anchor_type": "section_title", "evidence_span": None}]
        for topic in result["topics"]:
            label = str(topic.get("label", "")).strip()
            anchor_type = topic.get("anchor_type", "inferred")
            evidence_span = topic.get("evidence_span")
            if not label or label == context["section_title"]:
                continue
            if not evidence_span or not self.check.verify([evidence_span], source, min_char_len=8)[0]:
                continue
            topics.append(label)
            details.append({"label": label, "anchor_type": "text_span", "evidence_span": evidence_span})
        topic_set = set(topics)
        objects = []
        discarded_objects = []
        for item in result["objects"]:
            name = (item.get("name") or "").strip() or None
            evidence_span = item.get("evidence_span")
            if name and (not evidence_span or not self.check.verify([evidence_span], source, min_char_len=8)[0]):
                discarded_objects.append({"name": name, "reason": "missing_or_invalid_verbatim_evidence"})
                continue
            item_topics = [str(topic).strip() for topic in item["topics"]]
            if not set(item_topics) <= topic_set:
                discarded_objects.append({"name": name, "reason": "unknown_topic"})
                continue
            if evidence_span and not self.check.verify([evidence_span], source, min_char_len=8)[0]:
                discarded_objects.append({"name": name, "reason": "invalid_verbatim_evidence"})
                continue
            if name:
                objects.append({
                    "citation_keys": normalize_citation_keys(item["citation_keys"]),
                    "name": name,
                    "topics": item_topics,
                    "evidence_span": evidence_span,
                })
        return {"topics": topics, "topic_details": details, "objects": objects, "discarded_objects": discarded_objects}

    def _organize_inputs(self, inputs: dict[str, Any]):
        title = inputs["paper_title"]
        section_title = inputs["section_title"]
        section_text = inputs["section_text"]
        source = "\n".join(filter(None, [title, section_title, section_text]))
        return self.PROMPT.format(paper_title=title, section_title=section_title, section_text=section_text), {
            "source": source,
            "section_title": section_title,
        }


class EvidenceAnchoredContentParser:
    """Parse content and future-work sections with evidence-backed research objects."""

    def __init__(self, config: ToolConfig):
        self.llm = EvidenceAnchoredContentClient(config)
        self.last_report = {"module": "content_evidence", "success_count": 0, "error_count": 0, "errors": []}

    @staticmethod
    def _section_text(section: Section) -> str:
        return paragraphs_to_text(section.paragraphs, include_environments=True)

    def _targets(self, paper: Paper, only_missing: bool):
        targets = []
        def walk(section: Section):
            if section.functional_type in {"CONTENT", "FUTURE_WORK"} and self._section_text(section):
                if not only_missing or not section.parsed_contents:
                    targets.append(section)
            for child in section.children:
                walk(child)
        for section in paper.children:
            walk(section)
        for section in [*paper.limitation, *paper.appendix]:
            walk(section)
        return targets

    async def __call__(self, paper: Paper, only_missing: bool = False) -> Paper:
        targets = self._targets(paper, only_missing)
        tasks = [asyncio.create_task(self.llm.call(inputs={
            "paper_title": paper.title,
            "section_title": section.name,
            "section_text": self._section_text(section),
        })) for section in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        errors = []
        successes = 0
        for index, (section, result) in enumerate(zip(targets, results)):
            if isinstance(result, Exception):
                errors.append({"index": index, "section": section.name, "error": repr(result)})
                logging.error("EvidenceAnchoredContentParser failed for %r: %s", section.name, result)
            else:
                section.parsed_contents = result
                successes += 1
        self.last_report = {"module": "content_evidence", "success_count": successes, "error_count": len(errors), "errors": errors}
        return paper

