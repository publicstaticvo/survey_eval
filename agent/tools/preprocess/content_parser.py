from __future__ import annotations

import asyncio
from typing import Any

import jsonschema
import logging

from ..prompts import CONTENT_PARSE_WITH_TOPICS, CONTENT_PARSE_WITH_TOPICS_SCHEMA
from ..utility.citation_utils import citation_keys as normalize_citation_keys
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper, Section
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json
from ..utility.content_walk import paragraphs_to_text


class ContentParseWithTopicsClient(AsyncChat):
    PROMPT = CONTENT_PARSE_WITH_TOPICS

    def _availability(self, response, context):
        result = extract_json(response)
        if "entities" in result and "objects" not in result:
            result["objects"] = result.pop("entities")
        jsonschema.validate(result, CONTENT_PARSE_WITH_TOPICS_SCHEMA)
        source = context["source"]
        topics = [topic.strip() for topic in result["topics"]]
        for topic in topics:
            assert topic in source, f"ContentParser: topic {topic} is not in source"
        topic_set = set(topics)
        objects = []
        for item in result["objects"]:
            name = (item["name"] or "").strip()
            if name:
                assert name in source, f"ContentParser: entity name {name} is not in source"
            item_topics = [topic.strip() for topic in item["topics"]]
            assert set(item_topics) <= topic_set, f"ContentParser: topic set {set(item_topics)} is not subset of {topic_set}"
            objects.append({
                "citation_keys": normalize_citation_keys(item["citation_keys"]),
                "name": name,
                "topics": item_topics,
            })
        return {"topics": topics, "objects": objects}

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
        return prompt, {"source": source}


class ContentParser:
    def __init__(self, config: ToolConfig):
        self.llm = ContentParseWithTopicsClient(config.llm_server_info, config.sampling_params)

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

    async def __call__(self, paper: Paper) -> Paper:
        targets = [section for section in self._content_sections(paper) if self._section_text(section)]
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
        for section, result in zip(targets, results):
            if isinstance(result, dict):
                section.parsed_contents = result
        return paper