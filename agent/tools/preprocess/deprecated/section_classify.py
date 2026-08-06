import logging
import asyncio
from typing import Any, List, Tuple

import jsonschema

from ..prompts import SECTION_CLASSIFICATION, SECTION_SCHEMA
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
        self.last_report = {"module": "section", "success_count": 0, "error_count": 0, "errors": []}
        self.llm = SectionClassificationClient(config.llm_server_info, config.sampling_params)

    def _collect_targets(
        self,
        content: Section,
        parent_title: str = "",
        only_missing: bool = False,
    ) -> List[Tuple[Section | Paragraph, str, str]]:
        targets = []
        current_title = content.name
        for paragraph in content.paragraphs:
            if paragraph.name and (not only_missing or not paragraph.functional_type):
                targets.append((paragraph, "paragraph", current_title))
        for section in content.children:
            if section.name and (not only_missing or not section.functional_type):
                targets.append((section, "section", parent_title))
            targets.extend(
                self._collect_targets(
                    section,
                    section.name,
                    only_missing=only_missing,
                )
            )
        return targets

    async def __call__(self, paper_content: Paper, only_missing: bool = False) -> Paper:
        targets = self._collect_targets(paper_content, only_missing=only_missing)
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
        logging.info(f"section classify for paper {paper_content.title} use {len(tasks)} sentences")
        success_count = 0
        errors = []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for index, ((item, _, _), result) in enumerate(zip(targets, results)):
            if isinstance(result, Exception):
                errors.append({"index": index, "item": getattr(item, "name", ""), "error": repr(result)})
                continue
            try:
                item.functional_type = result["functional_type"]
                success_count += 1
            except Exception as exc:
                errors.append({"index": index, "item": getattr(item, "name", ""), "error": repr(exc)})
        self.last_report = {"module": "section", "success_count": success_count, "error_count": len(errors), "errors": errors}
        return paper_content
