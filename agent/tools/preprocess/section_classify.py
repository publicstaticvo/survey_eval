import asyncio
from typing import Any, List, Tuple

import jsonschema

from ..prompts import SECTION_CLASSIFICATION, SECTION_SCHEMA, CONTENT_TAGS
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig
from .utils import extract_json, paragraphs_to_text, paragraph_to_text


class SectionClassificationClient(AsyncChat):
    PROMPT: str = SECTION_CLASSIFICATION

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        jsonschema.validate(result, SECTION_SCHEMA)
        result['content_tags'] = list(set(result['content_tags']))
        if 'GENERAL' in result['content_tags'] and len(result['content_tags']) > 1: 
            result['content_tags'].remove("GENERAL")
        return result

    def _text_sentences(self, paragraph: Any) -> List[dict[str, Any]]:
        if isinstance(paragraph, dict):
            sentences = paragraph.get("sentences", [])
        elif isinstance(paragraph, list):
            sentences = paragraph
        else:
            sentences = []
        return [
            sentence
            for sentence in sentences
            if (
                isinstance(sentence, dict)
                and sentence.get("environment_type", "text") == "text"
                and sentence.get("text", "").strip()
            )
        ]

    def _section_preamble(self, section: dict[str, Any]) -> str:
        paragraphs = section.get("paragraphs", [])
        if not paragraphs: return ""
        return paragraphs_to_text(paragraphs, False)
        # return " ".join(sentence["text"] for sentence in self._text_sentences(paragraphs[0])[:3])

    def _paragraph_preamble(self, paragraph: dict[str, Any]) -> str:
        return paragraph_to_text(paragraph, False)
        # return " ".join(sentence["text"] for sentence in self._text_sentences(paragraph)[:3])

    def _organize_inputs(self, inputs):
        item = inputs["item"]
        is_paragraph = inputs.get("kind") == "paragraph"
        section_title = item['name'] if is_paragraph else item['title']
        preamble = self._paragraph_preamble(item) if is_paragraph else self._section_preamble(item)
        return self.PROMPT.format(
            DOCUMENT_TITLE=inputs.get("document_title", ""),
            PARENT_TITLE=inputs.get("parent_title", ""),
            SECTION_TITLE=section_title,
            PREAMBLE=preamble,
        ), {"item": item}


class SectionClassification:
    def __init__(self, config: ToolConfig):
        self.llm = SectionClassificationClient(config.llm_server_info, config.sampling_params)

    def _collect_targets(
        self,
        content: dict[str, Any],
        parent_title: str = "",
    ) -> List[Tuple[list[dict[str, Any]], int, str, str]]:
        targets = []
        current_title = content.get("title", "")
        paragraphs = content.get("paragraphs", [])
        if isinstance(paragraphs, list):
            for index, paragraph in enumerate(paragraphs):
                if isinstance(paragraph, dict) and paragraph.get("name"):
                    targets.append((paragraphs, index, "paragraph", current_title))

        sections = content.get("sections", [])
        if not isinstance(sections, list):
            return targets
        for index, section in enumerate(sections):
            if not isinstance(section, dict): continue
            title = section.get("title", "")
            if title: targets.append((sections, index, "section", parent_title))
            targets.extend(self._collect_targets(section, title))
        return targets

    async def __call__(self, paper_content: dict[str, Any]) -> dict[str, Any]:
        document_title = paper_content.get("title", "")
        targets = self._collect_targets(paper_content)
        tasks = [
            asyncio.create_task(
                self.llm.call(inputs={
                    "item": container[index],
                    "kind": kind,
                    "document_title": document_title,
                    "parent_title": parent_title,
                })
            )
            for container, index, kind, parent_title in targets
        ]
        for (c, i, _, _), result in zip(targets, await asyncio.gather(*tasks, return_exceptions=True)):
            if not isinstance(result, dict): continue
            c[i] = {**c[i], **result}
        return paper_content
