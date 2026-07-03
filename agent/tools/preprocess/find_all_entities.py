from __future__ import annotations

import asyncio
import re
from typing import Any

import jsonschema

from ..prompts import FIND_ALL_ENTITIES
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig
from .utils import extract_json, paragraph_to_text


FIND_ALL_ENTITIES_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "locally_cited": {"type": "boolean"},
                },
                "required": ["name", "locally_cited"],
                "additionalProperties": False,
            },
        },
        "alias_pairs": {
            "type": "array",
            "items": {
                "type": "array",
                "prefixItems": [
                    {"type": "string", "minLength": 1},
                    {"type": "string", "minLength": 1},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
        },
    },
    "required": ["entities", "alias_pairs"],
    "additionalProperties": False,
}


class FindAllEntitiesClient(AsyncChat):
    PROMPT: str = FIND_ALL_ENTITIES

    def _normalize_entity_name(self, name: str) -> str:
        name = name.strip()
        if re.search(r"[A-Z]s$", name):
            return name[:-1]
        return name

    def _contains_entity(self, paragraph: str, name: str) -> bool:
        return name.casefold() in paragraph.casefold()

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, FIND_ALL_ENTITIES_SCHEMA)
        paragraph = context["paragraph"]

        entities: dict[str, dict[str, Any]] = {}
        for item in result["entities"]:
            name = self._normalize_entity_name(item["name"])
            assert self._contains_entity(paragraph, name)
            key = name.casefold()
            if key in entities:
                entities[key]["locally_cited"] = entities[key]["locally_cited"] or item["locally_cited"]
            else:
                entities[key] = {"name": name, "locally_cited": item["locally_cited"]}

        alias_pairs = []
        for full_name, short_name in result["alias_pairs"]:
            full_name = self._normalize_entity_name(full_name)
            short_name = self._normalize_entity_name(short_name)
            full_key = full_name.casefold()
            short_key = short_name.casefold()
            assert full_key in entities and short_key in entities
            alias_pairs.append([entities[full_key]["name"], entities[short_key]["name"]])

        return {"entities": list(entities.values()), "alias_pairs": alias_pairs}

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(
            paragraph=inputs["paragraph"],
            query=inputs.get("query", "Computer Sciences"),
        ), {"paragraph": inputs["paragraph"]}


class FindAllEntities:
    def __init__(self, config: ToolConfig):
        self.find_entities = FindAllEntitiesClient(config.llm_server_info, config.sampling_params)

    def _paragraph_sentences(self, paragraph: Any) -> list[dict[str, Any]]:
        if isinstance(paragraph, dict):
            return paragraph.get("sentences", []) or []
        if isinstance(paragraph, list):
            return paragraph
        return []

    def _normalize_paragraphs(self, node: Any):
        if not isinstance(node, dict):
            return
        paragraphs = node.get("paragraphs", []) or []
        for index, paragraph in enumerate(paragraphs):
            if isinstance(paragraph, list):
                paragraphs[index] = {"sentences": paragraph}
            elif isinstance(paragraph, dict):
                paragraph.setdefault("sentences", paragraph.get("sentences", []) or [])
        for section in node.get("sections", []) or []:
            self._normalize_paragraphs(section)
        abstract = node.get("abstract")
        if isinstance(abstract, dict):
            self._normalize_paragraphs(abstract)

    def _iter_paragraphs(self, paper: dict[str, Any]):
        def walk(node: Any):
            if isinstance(node, dict):
                if "sentences" in node:
                    yield node
                    return
                for paragraph in node.get("paragraphs", []) or []:
                    if isinstance(paragraph, dict):
                        yield paragraph
                    elif isinstance(paragraph, list):
                        yield {"sentences": paragraph}
                for section in node.get("sections", []) or []:
                    yield from walk(section)

        abstract = paper.get("abstract")
        if isinstance(abstract, dict):
            yield from walk(abstract)
        yield from walk(paper)

    def _normalize_entity_name(self, name: str) -> str:
        name = name.strip()
        if re.search(r"[A-Z]s$", name):
            return name[:-1]
        return name

    def _entity_key(self, name: str) -> str:
        return self._normalize_entity_name(name).casefold()

    def _ensure_entity(self, entities_dict: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
        key = self._entity_key(name)
        if key not in entities_dict:
            entities_dict[key] = {
                "original_name": name,
                "sentence_has_citation": False,
                "alias_pairs": "",
                "alternative_names": [],
            }
        return entities_dict[key]

    def _add_alternative_name(self, info: dict[str, Any], name: str):
        if name != info["original_name"] and name not in info["alternative_names"]:
            info["alternative_names"].append(name)

    def _abbreviation_groups(self, query: str) -> list[str]:
        return re.findall(r"[A-Z][a-z]*", query)

    def _words(self, name: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+", name)

    def _abbreviation_match(self, query: str, candidate: str) -> bool:
        groups = self._abbreviation_groups(query)
        words = self._words(candidate)
        if not groups or len(groups) != len(words):
            return False
        return all(word.casefold().startswith(group.casefold()) for group, word in zip(groups, words))

    def _find_alternative_match(self, key: str, entities_dict: dict[str, dict[str, Any]]) -> str | None:
        query = entities_dict[key]["original_name"]
        for candidate_key, candidate in entities_dict.items():
            if candidate_key == key:
                continue
            if self._abbreviation_match(query, candidate["original_name"]):
                return candidate_key
        return None

    async def __call__(self, query: str, paper: dict[str, Any]) -> dict[str, Any]:
        self._normalize_paragraphs(paper)
        paragraphs = list(self._iter_paragraphs(paper))
        tasks = []
        task_paragraphs = []
        for paragraph in paragraphs:
            text = paragraph_to_text(paragraph, False)
            if not text:
                paragraph.setdefault("entities", [])
                paragraph.setdefault("alias_pairs", [])
                continue
            task_paragraphs.append((paragraph, text))
            tasks.append(asyncio.create_task(self.find_entities.call(inputs={"paragraph": text, "query": query})))

        entities_dict: dict[str, dict[str, Any]] = {}
        paragraph_entity_keys: dict[int, list[str]] = {}
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (paragraph, _text), result in zip(task_paragraphs, results):
            if not isinstance(result, dict):
                print(f"FindAllEntities {result}")
                paragraph.setdefault("entities", [])
                paragraph.setdefault("alias_pairs", [])
                continue

            paragraph["alias_pairs"] = result["alias_pairs"]
            keys = []
            for entry in result["entities"]:
                name = entry["name"]
                info = self._ensure_entity(entities_dict, name)
                self._add_alternative_name(info, name)
                info["sentence_has_citation"] = info["sentence_has_citation"] or entry["locally_cited"]
                keys.append(self._entity_key(info["original_name"]))

            for full_name, short_name in result["alias_pairs"]:
                full_info = self._ensure_entity(entities_dict, full_name)
                short_info = self._ensure_entity(entities_dict, short_name)
                short_info["alias_pairs"] = full_info["original_name"]

            paragraph_entity_keys[id(paragraph)] = list(dict.fromkeys(keys))

        for key, info in entities_dict.items():
            if info["sentence_has_citation"]:
                continue
            alias_name = info.get("alias_pairs") or ""
            alias_info = entities_dict.get(self._entity_key(alias_name)) if alias_name else None
            if alias_info and alias_info["sentence_has_citation"]:
                info["sentence_has_citation"] = True
                continue
            matched_key = self._find_alternative_match(key, entities_dict)
            if matched_key is not None:
                matched_info = entities_dict[matched_key]
                info["sentence_has_citation"] = info["sentence_has_citation"] or matched_info["sentence_has_citation"]
                if not info["alias_pairs"]:
                    info["alias_pairs"] = matched_info["original_name"]
                self._add_alternative_name(info, matched_info["original_name"])

        for paragraph in paragraphs:
            paragraph["entities"] = []
            for key in paragraph_entity_keys.get(id(paragraph), []):
                info = entities_dict[key]
                paragraph["entities"].append({
                    "name": info["original_name"],
                    "sentence_has_citation": info["sentence_has_citation"],
                    "alias_pairs": info["alias_pairs"],
                    "alternative_names": list(info["alternative_names"]),
                })
            paragraph.setdefault("alias_pairs", [])
        return paper




