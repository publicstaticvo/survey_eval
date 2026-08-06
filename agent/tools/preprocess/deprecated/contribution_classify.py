import asyncio
import re
from typing import Any

import jsonschema
import logging

from ..prompts import CONTRIBUTION_CLASSIFICATION, CONTRIBUTION_SCHEMA, TEXTUAL_CLASSIFICATION, TEXTUAL_SCHEMA
from ..utility.content_walk import iter_sections_with_context
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper, Section, Sentence
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json


GRAPH_ENVIRONMENT_TYPES = {"figure", "figure*", "table", "table*", "tabular", "longtable"}
SELF_REFLECT = {"this section", "this chapter", "this subsection"}
SECTION_RANGE_RE = re.compile(r"^(?:Section\s+)?(?P<start>\d+(?:\.\d+)*)\s*-\s*(?P<end>\d+(?:\.\d+)*)$")


def _walk_sections(paper: Paper):
    for section, _title_path, section_id in iter_sections_with_context(paper):
        yield section, section_id


def _walk_paragraphs(paper: Paper):
    for paragraph in paper.paragraphs:
        yield "", paragraph.sentences
    if paper.abstract:
        for paragraph in paper.abstract.paragraphs:
            yield "", paragraph.sentences
    for section, section_id in _walk_sections(paper):
        for paragraph in section.paragraphs:
            yield section_id, paragraph.sentences


def build_section_enum_set(paper: Paper) -> set[str]:
    section_enum_set = {"document", *SELF_REFLECT}
    for section, section_id in _walk_sections(paper):
        title = section.name.strip()
        if section_id:
            section_enum_set.add(section_id)
            section_enum_set.add(f"Section {section_id}")
        if title:
            section_enum_set.add(title)

    figure_index, table_index = 0, 0
    for _, paragraph in _walk_paragraphs(paper):
        for sentence in paragraph:
            if sentence.environment_type not in GRAPH_ENVIRONMENT_TYPES:
                continue
            if sentence.environment_type in {"table", "table*", "tabular", "longtable"}:
                table_index += 1
                section_enum_set.add(f"Table {table_index}")
            else:
                figure_index += 1
                section_enum_set.add(f"Figure {figure_index}")
    return section_enum_set


def _section_id_parts(section_id: str) -> tuple[int, ...] | None:
    if not re.match(r"^\d+(?:\.\d+)*$", section_id):
        return None
    return tuple(int(part) for part in section_id.split("."))


def _expand_section_range(start: str, end: str, ordered_ids: list[str]) -> list[str]:
    id_set = set(ordered_ids)
    start_parts = _section_id_parts(start)
    end_parts = _section_id_parts(end)
    if start_parts and end_parts and len(start_parts) == len(end_parts) and start_parts[:-1] == end_parts[:-1]:
        step = 1 if start_parts[-1] <= end_parts[-1] else -1
        generated = [
            ".".join(str(part) for part in (*start_parts[:-1], value))
            for value in range(start_parts[-1], end_parts[-1] + step, step)
        ]
        if all(section_id in id_set for section_id in generated):
            return generated

    if start in id_set and end in id_set:
        start_index = ordered_ids.index(start)
        end_index = ordered_ids.index(end)
        if start_index <= end_index:
            return ordered_ids[start_index:end_index + 1]
        return ordered_ids[end_index:start_index + 1]
    return []


def _normalize_section_range(section: str, paper: Paper) -> str | None:
    match = SECTION_RANGE_RE.match(section)
    if not match:
        return None
    ordered_ids = [section_id for _, section_id in _walk_sections(paper) if re.match(r"^\d", section_id)]
    start = match.group("start")
    end = match.group("end")
    section_ids = _expand_section_range(start, end, ordered_ids)
    assert section_ids, "ContributionClassify: no section ids"
    return f"Section {section_ids[0]}-{section_ids[-1]}"


def _section_lookup(paper: Paper) -> tuple[dict[str, str], dict[str, str]]:
    id_lookup, title_lookup = {}, {}
    for section, section_id in _walk_sections(paper):
        title = section.name.strip()
        if section_id:
            id_lookup[section_id] = section_id
            id_lookup[f"Section {section_id}"] = section_id
        if title and section_id:
            title_lookup[title.casefold()] = section_id
    id_lookup.setdefault("Limitation", "Limitation")
    id_lookup.setdefault("Section Limitation", "Limitation")
    id_lookup.setdefault("Appendix", "Appendix")
    id_lookup.setdefault("Section Appendix", "Appendix")
    title_lookup.setdefault("limitation", "Limitation")
    title_lookup.setdefault("appendix", "Appendix")
    return id_lookup, title_lookup


def normalize_claim_section(raw_section: str, current_section_id: str, paper: Paper) -> str:
    section = str(raw_section or "").strip()
    if section in SELF_REFLECT:
        return current_section_id
    if section == "document":
        return section
    graph_match = re.match(r"^(?:Figure|Table)\s+(?P<index>\S+)$", section)
    if graph_match:
        assert re.match(r"^[1-9]\d*$", graph_match.group("index")), f"ContributionClassify: invalid index, {graph_match.group("index")}"
        return section

    normalized_range = _normalize_section_range(section, paper)
    if normalized_range is not None:
        return normalized_range

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
    assert section_id is not None, "ContributionClassify: section_id is none"
    return section_id


class ContributionClassificationClient(AsyncChat):
    PROMPT: str = CONTRIBUTION_CLASSIFICATION

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        # section_enum_set = build_section_enum_set(context["paper"])
        jsonschema.validate(result, CONTRIBUTION_SCHEMA)
        if not result["excluded"]:
            for claim in result["claims"]:
                normalize_claim_section(claim["section"], context["current_section_id"], context["paper"])
        return result

    def _organize_inputs(self, inputs):
        sentence = inputs["sentence"]
        prompt = self.PROMPT.format(S=sentence.text, CONTEXT=inputs.get("context", ""))
        return prompt, {
            "paper": inputs["paper"],
            "current_section_id": inputs["current_section_id"],
        }


class ContributionClassification:
    def __init__(self, config: ToolConfig):
        self.last_report = {"module": "contribution", "success_count": 0, "error_count": 0, "errors": []}
        self.llm = ContributionClassificationClient(config.llm_server_info, config.sampling_params)

    def _is_target(self, sentence: Sentence) -> bool:
        return sentence.label in {"CONTRIBUTION", "CONTRIBUTION+SCOPE"} and bool(sentence.text.strip())

    def _paragraphs(self, paper: Paper) -> list[list[Sentence]]:
        return [paragraph for _, paragraph in _walk_paragraphs(paper)]

    def _context(self, paragraph: list[Sentence], sentence_index: int) -> str:
        start = max(0, sentence_index - 2)
        end = min(len(paragraph), sentence_index + 3)
        context_sentences = [
            sentence.text.strip()
            for index, sentence in enumerate(paragraph[start:end], start)
            if index != sentence_index and sentence.environment_type == "text" and sentence.text.strip()
        ]
        return " ".join(context_sentences)

    def _classification_field(self) -> str:
        return getattr(self, "module_name", "contribution")

    def _collect_targets(
        self,
        paper: Paper,
        only_missing: bool = False,
    ) -> list[tuple[Sentence, str, str]]:
        targets = []
        section_by_paragraph = {id(paragraph): section_id for section_id, paragraph in _walk_paragraphs(paper)}
        for paragraph in self._paragraphs(paper):
            current_section_id = section_by_paragraph.get(id(paragraph), "")
            for index, sentence in enumerate(paragraph):
                if self._is_target(sentence) and (
                    not only_missing
                    or self._classification_field() not in sentence.classified_fields
                ):
                    targets.append((sentence, self._context(paragraph, index), current_section_id))
        return targets

    async def __call__(
        self,
        paper_content: Paper,
        only_missing: bool = False,
    ) -> tuple[Paper, dict[str, Any]]:
        targets = self._collect_targets(paper_content, only_missing=only_missing)
        tasks = [
            asyncio.create_task(
                self.llm.call(inputs={
                    "sentence": sentence,
                    "context": context,
                    "paper": paper_content,
                    "current_section_id": current_section_id,
                })
            )
            for sentence, context, current_section_id in targets
        ]
        all_claims = {}
        errors = []
        success_count = 0
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for index, ((sentence, _, current_section_id), result) in enumerate(zip(targets, results)):
            if isinstance(result, Exception):
                errors.append({"index": index, "sentence": sentence.text[:200], "error": repr(result)})
                continue
            try:
                success_count += 1
                if result["excluded"]:
                    if result["reason"] == "PRIOR_WORK_FALSE_POSITIVE":
                        sentence.label = "SUMMARY"
                    elif result["reason"] == "TEXTUAL_CLAIM":
                        sentence.label = "TEXTUAL"
                    else:
                        sentence.claims = []
                    field = self._classification_field()
                    if field not in sentence.classified_fields:
                        sentence.classified_fields.append(field)
                    continue
                sentence_claims = []
                for claim in result["claims"]:
                    section_key = normalize_claim_section(claim["section"], current_section_id, paper_content)
                    original_sentence = sentence.text.strip()
                    normalized_claim = {
                        "section": section_key,
                        "type": claim["type"],
                        "target": claim["target"],
                        "original_contribution_sentence": original_sentence,
                    }
                    sentence_claims.append(normalized_claim)
                    all_claims.setdefault(section_key, []).append({
                        "type": claim["type"],
                        "target": claim["target"],
                        "original_contribution_sentence": original_sentence,
                    })
                sentence.claims = sentence_claims
                field = self._classification_field()
                if field not in sentence.classified_fields:
                    sentence.classified_fields.append(field)
            except Exception as exc:
                errors.append({"index": index, "sentence": sentence.text[:200], "error": repr(exc)})
        self.last_report = {"module": getattr(self, "module_name", "contribution"), "success_count": success_count, "error_count": len(errors), "errors": errors}
        return paper_content, all_claims

class TextualClassificationClient(ContributionClassificationClient):
    PROMPT = TEXTUAL_CLASSIFICATION

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        jsonschema.validate(result, TEXTUAL_SCHEMA)
        if not result["excluded"]:
            for claim in result["claims"]:
                normalize_claim_section(claim["section"], context["current_section_id"], context["paper"])
        return result


class TextualClassification(ContributionClassification):
    def __init__(self, config: ToolConfig):
        self.llm = TextualClassificationClient(config.llm_server_info, config.sampling_params)
        self.module_name = "textual"
        self.last_report = {"module": "textual", "success_count": 0, "error_count": 0, "errors": []}

    def _is_target(self, sentence: Sentence) -> bool:
        return sentence.label == "TEXTUAL" and bool(sentence.text.strip())
