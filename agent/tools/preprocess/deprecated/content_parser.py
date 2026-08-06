from __future__ import annotations

import asyncio
from typing import Any

import jsonschema
import logging

from ..prompts import CONTENT_PARSE_WITH_TOPICS, CONTENT_PARSE_WITH_TOPICS_SCHEMA
from ..utility.citation_utils import citation_keys as normalize_citation_keys
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper, Section
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json
from ..utility.content_walk import paragraphs_to_text


class ContentParseWithTopicsClient(AsyncChat):
    PROMPT = CONTENT_PARSE_WITH_TOPICS

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        result = extract_json(response)
        if "entities" in result and "objects" not in result:
            result["objects"] = result.pop("entities")
        jsonschema.validate(result, CONTENT_PARSE_WITH_TOPICS_SCHEMA)
        source = context["source"]
        topics = []
        topic_details = []
        for topic in result["topics"]:
            if isinstance(topic, dict):
                label = (topic.get("label") or "").strip()
                anchor_type = topic.get("anchor_type") or "inferred"
                evidence_span = topic.get("evidence_span")
            else:
                label = str(topic).strip()
                anchor_type = "section_title" if label == context["section_title"] else "inferred"
                evidence_span = None
            assert label, "ContentParser: empty topic label"
            if evidence_span:
                verified, _ = self.check.verify([evidence_span], source, min_char_len=8)
                assert verified, f"ContentParser: evidence span {evidence_span} is not copied from source"
            elif anchor_type == "section_title":
                assert label == context["section_title"], f"ContentParser: section-title topic {label} does not match section title"
            else:
                verified, _ = self.check.verify([label], source, min_char_len=4)
                assert verified, f"ContentParser: inferred topic {label} is not copied from source {source}"
            topics.append(label)
            topic_details.append({
                "label": label,
                "anchor_type": anchor_type,
                "evidence_span": evidence_span,
            })
        topic_set = set(topics)
        objects = []
        for item in result["objects"]:
            name = (item["name"] or "").strip()
            if name:
                verified, _ = self.check.verify([name], source, min_char_len=4)
                assert verified, f"ContentParser: entity name {name} is not copied from source {source}"
            item_topics = [topic.strip() for topic in item["topics"]]
            assert set(item_topics) <= topic_set, f"ContentParser: topic set {set(item_topics)} is not subset of {topic_set}"
            evidence_span = item.get("evidence_span")
            if evidence_span:
                verified, _ = self.check.verify([evidence_span], source, min_char_len=8)
                assert verified, f"ContentParser: object evidence span {evidence_span} is not copied from source"
            objects.append({
                "citation_keys": normalize_citation_keys(item["citation_keys"]),
                "name": name,
                "topics": item_topics,
                "evidence_span": evidence_span,
            })
        return {"topics": topics, "topic_details": topic_details, "objects": objects}

    def _organize_inputs(self, inputs):
        paper_title = inputs["paper_title"]
        section_title = inputs["section_title"]
        section_text = inputs["section_text"]
        source = "\n".join(filter(None, [paper_title, section_title, section_text]))
        prompt = self.PROMPT.format(
            paper_title=paper_title,
            section_title=section_title,
            section_text=section_text,
        )
        return prompt, {"source": source, "section_title": section_title}


class ContentParser:
    def __init__(self, config: ToolConfig):
        self.last_report = {"module": "content", "success_count": 0, "error_count": 0, "errors": []}
        self.llm = ContentParseWithTopicsClient(config)

    def _content_sections(self, paper: Paper) -> list[Section]:
        sections = []

        def walk(section: Section):
            if section.functional_type == "CONTENT":
                sections.append(section)
            for child in section.children:
                walk(child)

        for section in paper.children: walk(section)
        return sections

    def _section_text(self, section: Section) -> str:
        return paragraphs_to_text(section.paragraphs, False)

    async def __call__(self, paper: Paper, only_missing: bool = False) -> Paper:
        targets = [
            section
            for section in self._content_sections(paper)
            if self._section_text(section) and (not only_missing or not section.parsed_contents)
        ]
        tasks = [
            asyncio.create_task(
                self.llm.call(inputs={
                    "paper_title": paper.title,
                    "section_title": section.name,
                    "section_text": self._section_text(section),
                })
            )
            for section in targets
        ]
        logging.info("Content parsing")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = 0
        errors = []
        for index, (section, result) in enumerate(zip(targets, results)):
            if isinstance(result, dict):
                section.parsed_contents = result
                success_count += 1
            else:
                error = result if isinstance(result, Exception) else TypeError(f"unexpected result type: {type(result).__name__}")
                errors.append({"index": index, "section": section.name, "error": repr(error)})
                logging.error("ContentParser failed for section %r: %s", section.name, error)
        self.last_report = {"module": "content", "success_count": success_count, "error_count": len(errors), "errors": errors}
        return paper
