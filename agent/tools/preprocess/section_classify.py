import logging
import asyncio
from typing import Any, List, Tuple

import jsonschema

from ..prompts import SECTION_CLASSIFICATION, SECTION_SCHEMA, CONTENT_TAGS
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper, Paragraph, Section
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json
from ..utility.content_walk import paragraph_to_text, paragraphs_to_text


class SectionClassificationClient(AsyncChat):
    PROMPT: str = SECTION_CLASSIFICATION

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        jsonschema.validate(result, SECTION_SCHEMA)
        result["content_tags"] = list(set(result["content_tags"]))
        if "GENERAL" in result["content_tags"] and len(result["content_tags"]) > 1:
            result["content_tags"].remove("GENERAL")
        return result

    def _text_sentences(self, paragraph: Paragraph) -> list:
        return [
            sentence
            for sentence in paragraph.sentences
            if sentence.environment_type == "text" and sentence.text.strip()
        ]

    def _section_preamble(self, section: Section) -> str:
        if not section.paragraphs: return ""
        return paragraphs_to_text(section.paragraphs, False)

    def _paragraph_preamble(self, paragraph: Paragraph) -> str:
        return paragraph_to_text(paragraph, False)

    def _organize_inputs(self, inputs):
        item = inputs["item"]
        is_paragraph = inputs.get("kind") == "paragraph"
        section_title = item.name if is_paragraph else item.name
        preamble = self._paragraph_preamble(item) if is_paragraph else self._section_preamble(item)
        prompt = self.PROMPT.format(
            DOCUMENT_TITLE=inputs.get("document_title", ""),
            PARENT_TITLE=inputs.get("parent_title", ""),
            SECTION_TITLE=section_title,
            PREAMBLE=preamble,
        )
        return prompt, {"item": item}


class SectionClassification:
    def __init__(self, config: ToolConfig):
        self.llm = SectionClassificationClient(config.llm_server_info, config.sampling_params)

    def _collect_targets(self, content: Section, parent_title: str = "") -> List[Tuple[Section | Paragraph, str, str]]:
        targets = []
        current_title = content.name
        for paragraph in content.paragraphs:
            if paragraph.name:
                targets.append((paragraph, "paragraph", current_title))
        for section in content.children:
            if section.name:
                targets.append((section, "section", parent_title))
            targets.extend(self._collect_targets(section, section.name))
        return targets

    async def __call__(self, paper_content: Paper) -> Paper:
        targets = self._collect_targets(paper_content)
        tasks = [
            asyncio.create_task(
                self.llm.call(inputs={
                    "item": item,
                    "kind": kind,
                    "document_title": paper_content.title,
                    "parent_title": parent_title,
                })
            )
            for item, kind, parent_title in targets
        ]
        logging.info(f"section classify for paper {paper_content.title}")
        for (item, _, _), result in zip(targets, await asyncio.gather(*tasks, return_exceptions=True)):
            if not isinstance(result, dict): continue
            item.functional_type = result["functional_type"]
            item.content_tags = result["content_tags"]
        return paper_content