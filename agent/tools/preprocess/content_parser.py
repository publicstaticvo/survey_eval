"""
content_parser.py
明确每个自然段都在说什么内容，涉及哪些topics。
"""
from __future__ import annotations

import asyncio
from typing import Any

import jsonschema
import tqdm

from ..prompts import (
    CONTENT_PARSE, CONTENT_PARSE_WITH_TOPICS,
    CONTENT_PARSE_SCHEMA, CONTENT_PARSE_WITH_TOPICS_SCHEMA
)
from ..utility.citation_utils import citation_keys as normalize_citation_keys
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig
from .utils import extract_json, paragraphs_to_text


class ContentParseClient(AsyncChat):
    PROMPT = CONTENT_PARSE

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, CONTENT_PARSE_SCHEMA)
        source = context["source"]
        objects = []
        for item in result["objects"]:
            name = (item["name"] or "").strip()
            if name:
                assert name in source
            objects.append({
                "citation_keys": normalize_citation_keys(item["citation_keys"]),
                "name": name,
            })
        return {"objects": objects}

    def _organize_inputs(self, inputs):
        paragraph = inputs["paragraph"]
        return self.PROMPT.format(paragraph=paragraph), {"source": paragraph}


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
            assert topic in source
        topic_set = set(topics)
        objects = []
        for item in result["objects"]:
            name = (item["name"] or "").strip()
            if name:
                assert name in source
            item_topics = [topic.strip() for topic in item["topics"]]
            assert set(item_topics) <= topic_set
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
        return self.PROMPT.format(
            paper_title=paper_title,
            section_title=section_title,
            section_text=section_text,
        ), {"source": source}


class ContentParser:
    def __init__(self, config: ToolConfig):
        self.llm = ContentParseWithTopicsClient(config.llm_server_info, config.sampling_params)
        self.paragraph_llm = ContentParseClient(config.llm_server_info, config.sampling_params)

    def _content_sections(self, paper: dict[str, Any]) -> list[dict[str, Any]]:
        sections = []

        def walk(section: dict[str, Any]):
            if section.get("functional_type") == "CONTENT":
                sections.append(section)
            for child in section.get("sections", []) or []:
                if isinstance(child, dict):
                    walk(child)

        for section in paper.get("sections", []) or []:
            if isinstance(section, dict):
                walk(section)
        return sections

    def _section_text(self, section: dict[str, Any]) -> str:
        return paragraphs_to_text(section.get("paragraphs", []) or [], False)

    async def __call__(self, paper: dict[str, Any]) -> dict[str, Any]:
        paper_title = paper.get("title", "")
        targets = [
            section
            for section in self._content_sections(paper)
            if self._section_text(section)
        ]
        tasks = [
            asyncio.create_task(
                self.llm.call(inputs={
                    "paper_title": paper_title,
                    "section_title": section.get("title", ""),
                    "section_text": self._section_text(section),
                })
            )
            for section in targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for section, result in tqdm.tqdm(
            zip(targets, results),
            total=len(targets),
            desc="content parse",
        ):
            if isinstance(result, dict):
                section["parsed_contents"] = result
        return paper
