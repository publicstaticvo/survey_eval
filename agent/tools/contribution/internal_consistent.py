import asyncio
import re
from typing import Any

from ..preprocess.utils import extract_json
from ..prompts import INTERNAL_CONSISTENT
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig


CHECK_FUNCTIONAL_TYPES = {"TAXONOMY", "CONTENT", "EVALUATION", "LIMITATION", "FUTURE_WORK"}


class InternalConsistentClient(AsyncChat):
    PROMPT: str = INTERNAL_CONSISTENT

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        assert isinstance(result.get("promised_topic", ""), str)
        sentence_ids = set(context["sentence_map"])
        subsection_ids = set(context["subsection_ids"])
        if "different_sentences" in result:
            assert all(item in sentence_ids for item in result["different_sentences"])
        if "different_subsections" in result:
            assert all(item in subsection_ids for item in result["different_subsections"])
        return result

    def _organize_inputs(self, inputs):
        sentence_list = inputs["sentence_list"] or "None"
        subsection_list = inputs["subsection_list"] or "None"
        return self.PROMPT.format(
            SECTION_TITLE=inputs["section_title"],
            SENTENCE_LIST=sentence_list,
            SUBSECTION_LIST=subsection_list,
        ), {
            "sentence_map": inputs["sentence_map"],
            "subsection_ids": inputs["subsection_ids"],
        }


class InternalConsistency:
    def __init__(self, config: ToolConfig):
        self.llm = InternalConsistentClient(config.llm_server_info, config.sampling_params)
        self.sentence_ratio_threshold = config.internal_consistency_sentence_ratio_threshold

    def _paragraph_sentences(self, paragraph):
        if isinstance(paragraph, dict):
            return paragraph.get("sentences", [])
        return paragraph if isinstance(paragraph, list) else []

    def _text_sentences(self, section: dict[str, Any]) -> list[str]:
        sentences = []
        for paragraph in section.get("paragraphs", []) or []:
            for sentence in self._paragraph_sentences(paragraph):
                if (
                    isinstance(sentence, dict)
                    and sentence.get("environment_type", "text") == "text"
                    and sentence.get("text", "").strip()
                ):
                    sentences.append(sentence["text"].strip())
        return sentences

    def _section_title_path(self, title_path: list[str]) -> str:
        return " > ".join(part for part in title_path if part)

    def _collect_targets(self, paper: dict[str, Any]):
        targets = []

        def walk(section: dict[str, Any], title_path: list[str]):
            current_path = [*title_path, section.get("title", "")]
            if section.get("functional_type") in CHECK_FUNCTIONAL_TYPES:
                targets.append((section, current_path))
            for child in section.get("sections", []) or []:
                if isinstance(child, dict):
                    walk(child, current_path)

        for section in paper.get("sections", []) or []:
            if isinstance(section, dict):
                walk(section, [])
        return targets

    def _input_for_section(self, section: dict[str, Any], title_path: list[str]) -> dict[str, Any]:
        sentences = self._text_sentences(section)
        sentence_map = {
            f"S{index}": sentence
            for index, sentence in enumerate(sentences, 1)
        }
        sentence_list = "\n".join(
            f"{sentence_id}: {sentence}"
            for sentence_id, sentence in sentence_map.items()
        )
        children = [
            child for child in section.get("sections", []) or []
            if isinstance(child, dict)
        ]
        subsection_ids = [
            str(child.get("section_id", "") or "").strip()
            for child in children
            if str(child.get("section_id", "") or "").strip()
        ]
        subsection_list = "\n".join(
            f"{child.get('section_id', '')} {child.get('title', '')}".strip()
            for child in children
        )
        return {
            "section": section,
            "section_title": self._section_title_path(title_path),
            "sentence_list": sentence_list or None,
            "sentence_map": sentence_map,
            "subsection_list": subsection_list or None,
            "subsection_ids": subsection_ids,
        }

    async def _check_section(self, item: dict[str, Any]) -> dict[str, Any]:
        result = await self.llm.call(inputs=item)
        different_sentences = result.get("different_sentences", []) or []
        different_subsections = result.get("different_subsections", []) or []
        total_sentences = max(1, len(item["sentence_map"]))
        sentence_ratio = len(different_sentences) / total_sentences
        internal_consistent = (
            not different_subsections
            and sentence_ratio <= self.sentence_ratio_threshold
        )
        different_sentence_items = [
            {"id": sentence_id, "text": item["sentence_map"][sentence_id]}
            for sentence_id in different_sentences
        ]
        return {
            "section_id": item["section"].get("section_id", ""),
            "section_title": item["section_title"],
            "internal_consistent": internal_consistent,
            "inconsistent": not internal_consistent,
            "promised_topic": result.get("promised_topic", ""),
            "different_sentences": different_sentence_items,
            "different_subsections": different_subsections,
            "inconsistency_evidence": {
                "sentences": different_sentence_items,
                "subsections": different_subsections,
            },
            "different_sentence_ratio": sentence_ratio,
        }

    async def __call__(self, paper: dict[str, Any]) -> dict[str, Any]:
        inputs = [
            self._input_for_section(section, title_path)
            for section, title_path in self._collect_targets(paper)
        ]
        tasks = [asyncio.create_task(self._check_section(item)) for item in inputs]
        checks = await asyncio.gather(*tasks)
        return {
            "checks": checks,
            "internal_consistent": all(item["internal_consistent"] for item in checks),
        }


ContributionInternalConsistency = InternalConsistency
