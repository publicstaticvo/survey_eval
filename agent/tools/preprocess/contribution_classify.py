import asyncio
import re
from typing import Any

import jsonschema

from ..prompts import CONTRIBUTION_CLASSIFICATION, CONTRIBUTION_SCHEMA
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig
from .utils import extract_json, split_content_to_paragraph


GRAPH_ENVIRONMENT_TYPES = {"figure", "figure*", "table", "table*", "tabular", "longtable"}


def _paragraph_sentences(paragraph):
    if isinstance(paragraph, dict):
        return paragraph.get("sentences", [])
    return paragraph if isinstance(paragraph, list) else []


def _walk_sections(paper: dict[str, Any]):
    def walk(section: dict[str, Any]):
        yield section
        for child in section.get("sections", []) or []:
            if isinstance(child, dict):
                yield from walk(child)

    for group_name in ("sections", "limitation", "appendix"):
        group = paper.get(group_name, [])
        if isinstance(group, dict):
            group = [group]
        for section in group or []:
            if isinstance(section, dict):
                yield from walk(section)


def _walk_paragraphs(paper: dict[str, Any]):
    abstract = paper.get("abstract")
    if isinstance(abstract, dict):
        for paragraph in abstract.get("paragraphs", []) or []:
            yield "", _paragraph_sentences(paragraph)
    for section in _walk_sections(paper):
        section_id = str(section.get("section_id", "") or "")
        for paragraph in section.get("paragraphs", []) or []:
            yield section_id, _paragraph_sentences(paragraph)


def build_section_enum_set(paper: dict[str, Any]) -> set[str]:
    SECTION_ENUM_SET = {"document", "this section"}
    for section in _walk_sections(paper):
        section_id = str(section.get("section_id", "") or "").strip()
        title = str(section.get("title", "") or "").strip()
        if section_id:
            SECTION_ENUM_SET.add(f"Section {section_id}")
        if title:
            SECTION_ENUM_SET.add(title)

    figure_index, table_index = 0, 0
    for _, paragraph in _walk_paragraphs(paper):
        for sentence in paragraph:
            if not isinstance(sentence, dict) or sentence.get("environment_type") not in GRAPH_ENVIRONMENT_TYPES:
                continue
            if sentence["environment_type"] in {"table", "table*", "tabular", "longtable"}:
                table_index += 1
                SECTION_ENUM_SET.add(f"Table {table_index}")
            else:
                figure_index += 1
                SECTION_ENUM_SET.add(f"Figure {figure_index}")
    return SECTION_ENUM_SET


def _section_lookup(paper: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    id_lookup, title_lookup = {}, {}
    for section in _walk_sections(paper):
        section_id = str(section.get("section_id", "") or "").strip()
        title = str(section.get("title", "") or "").strip()
        if section_id:
            id_lookup[section_id] = section_id
            id_lookup[f"Section {section_id}"] = section_id
        if title and section_id:
            title_lookup[title.casefold()] = section_id
    id_lookup.setdefault("Limitation", "Limitation")
    id_lookup.setdefault("Section Limitation", "Limitation")
    title_lookup.setdefault("limitation", "Limitation")
    return id_lookup, title_lookup


def normalize_claim_section(raw_section: str, current_section_id: str, paper: dict[str, Any]) -> str:
    section = str(raw_section or "").strip()
    if section == "this section":
        return current_section_id
    if section == "document" or re.match(r"^(?:Figure|Table)\s+\S+", section):
        return section

    id_lookup, title_lookup = _section_lookup(paper)
    if section in id_lookup:
        return id_lookup[section]

    section_match = re.match(r"^Section\s+(.+)$", section)
    if section_match:
        section_value = section_match.group(1).strip()
        if section_value in id_lookup:
            return id_lookup[section_value]
        if re.match(r"^\d", section_value):
            return section_value

    if re.match(r"^\d", section):
        return section

    section_id = title_lookup.get(section.casefold())
    assert section_id is not None
    return section_id


class ContributionClassificationClient(AsyncChat):
    PROMPT: str = CONTRIBUTION_CLASSIFICATION

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        SECTION_ENUM_SET = build_section_enum_set(context["paper"])
        jsonschema.validate(result, CONTRIBUTION_SCHEMA(SECTION_ENUM_SET))
        return result

    def _organize_inputs(self, inputs):
        sentence = inputs["sentence"]
        return self.PROMPT.format(
            S=sentence["text"],
            CONTEXT=inputs.get("context", ""),
        ), {"paper": inputs["paper"]}


class ContributionClassification:
    def __init__(self, config: ToolConfig):
        self.llm = ContributionClassificationClient(config.llm_server_info, config.sampling_params)

    def _is_target(self, sentence: dict[str, Any]) -> bool:
        return sentence.get("label") == "CONTRIBUTION" and bool(sentence.get("text", "").strip())

    def _paragraphs(self, paper: dict[str, Any]) -> list[list[dict[str, Any]]]:
        return [paragraph for _, paragraph in _walk_paragraphs(paper)]

    def _context(self, paragraph: list[dict[str, Any]], sentence_index: int) -> str:
        start = max(0, sentence_index - 2)
        end = min(len(paragraph), sentence_index + 3)
        context_sentences = [
            sentence.get("text", "").strip()
            for index, sentence in enumerate(paragraph[start:end], start)
            if (
                index != sentence_index
                and isinstance(sentence, dict)
                and sentence.get("environment_type", "text") == "text"
                and sentence.get("text", "").strip()
            )
        ]
        return " ".join(context_sentences)

    def _collect_targets(self, paper: dict[str, Any]) -> list[tuple[dict[str, Any], str, str]]:
        targets = []
        section_by_paragraph = {id(paragraph): section_id for section_id, paragraph in _walk_paragraphs(paper)}
        for paragraph in self._paragraphs(paper):
            if not isinstance(paragraph, list):
                continue
            current_section_id = section_by_paragraph.get(id(paragraph), "")
            for index, sentence in enumerate(paragraph):
                if isinstance(sentence, dict) and self._is_target(sentence):
                    targets.append((sentence, self._context(paragraph, index), current_section_id))
        return targets

    async def __call__(self, paper_content: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        targets = self._collect_targets(paper_content)
        tasks = [
            asyncio.create_task(
                self.llm.call(inputs={"sentence": sentence, "context": context, "paper": paper_content})
            )
            for sentence, context, _ in targets
        ]
        all_claims = {}
        for (sentence, _, current_section_id), result in zip(targets, await asyncio.gather(*tasks)):
            if result["excluded"]:
                if result["reason"] == "PRIOR_WORK_FALSE_POSITIVE":
                    sentence["label"] = "SUMMARY"
                else:
                    sentence["claims"] = []
                continue

            sentence_claims = []
            for claim in result["claims"]:
                section_key = normalize_claim_section(claim["section"], current_section_id, paper_content)
                normalized_claim = {
                    "section": section_key,
                    "type": claim["type"],
                    "target": claim["target"],
                }
                sentence_claims.append(normalized_claim)
                all_claims.setdefault(section_key, []).append({
                    "type": claim["type"],
                    "target": claim["target"],
                })
            sentence["claims"] = sentence_claims
        return paper_content, all_claims
