import asyncio
from typing import Any

from ..prompts import INTERNAL_CONSISTENT
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper, Section
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json


CHECK_FUNCTIONAL_TYPES = {"TAXONOMY", "CONTENT", "EVALUATION", "SCOPE", "FUTURE_WORK"}


class InternalConsistentClient(AsyncChat):
    PROMPT: str = INTERNAL_CONSISTENT

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        assert isinstance(result.get("promised_topic", ""), str)
        sentence_ids = set(context["sentence_map"])
        subsection_ids = set(context["subsection_ids"])
        if "different_sentences" in result:
            assert all(item in sentence_ids for item in result["different_sentences"]), f"InternalConsistentClient: Invalid differenct sentence ids"
        if "different_subsections" in result:
            assert all(item in subsection_ids for item in result["different_subsections"]), f"InternalConsistentClient: Invalid differenct section ids"
        return result

    def _organize_inputs(self, inputs):
        tagged_content = inputs["tagged_content"] or "None"
        subsection_list = inputs["subsection_list"] or "None"
        prompt = self.PROMPT.format(
            SECTION_TITLE=inputs["section_title"],
            TAGGED_CONTENT=tagged_content,
            SUBSECTION_LIST=subsection_list,
        )
        return prompt, {
            "sentence_map": inputs["sentence_map"],
            "subsection_ids": inputs["subsection_ids"],
        }


class InternalConsistency:
    def __init__(self, config: ToolConfig):
        self.llm = InternalConsistentClient(config.llm_server_info, config.sampling_params)
        self.sentence_ratio_threshold = config.internal_consistency_sentence_ratio_threshold

    def _text_sentences(self, section: Section) -> list[str]:
        sentences = []
        for paragraph in section.paragraphs:
            for sentence in paragraph.sentences:
                if sentence.environment_type == "text" and sentence.text.strip():
                    sentences.append(sentence.text.strip())
        return sentences

    def _tagged_content(self, section: Section) -> tuple[str, dict[str, str]]:
        sentence_map = {}
        paragraph_texts = []
        sentence_index = 1
        for paragraph in section.paragraphs:
            tagged_sentences = []
            for sentence in paragraph.sentences:
                if sentence.environment_type == "text" and sentence.text.strip():
                    sentence_id = f"S{sentence_index}"
                    text = sentence.text.strip()
                    sentence_map[sentence_id] = text
                    tagged_sentences.append(f"[{sentence_id}] {text}")
                    sentence_index += 1
            if tagged_sentences:
                paragraph_texts.append(" ".join(tagged_sentences))
        return "\n\n".join(paragraph_texts), sentence_map

    def _section_title_path(self, title_path: list[str]) -> str:
        return " > ".join(part for part in title_path if part)

    def _collect_targets(self, paper: Paper):
        targets = []

        def walk(section: Section, title_path: list[str]):
            current_path = [*title_path, section.name]
            if section.functional_type in CHECK_FUNCTIONAL_TYPES:
                targets.append((section, current_path))
            for child in section.children:
                walk(child, current_path)

        for section in paper.children:
            walk(section, [])
        return targets

    def _input_for_section(self, section: Section, title_path: list[str]) -> dict[str, Any]:
        tagged_content, sentence_map = self._tagged_content(section)
        children = section.children
        subsection_ids = [str(index + 1) for index, _ in enumerate(children)]
        subsection_list = "\n".join(
            f"{index + 1} {child.name}".strip()
            for index, child in enumerate(children)
        )
        return {
            "section": section,
            "section_title": self._section_title_path(title_path),
            "tagged_content": tagged_content or None,
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
            "section_id": "",
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

    async def __call__(self, paper: Paper) -> dict[str, Any]:
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