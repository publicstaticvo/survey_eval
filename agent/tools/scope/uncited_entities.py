from __future__ import annotations

import asyncio
import re
from typing import Any

from ..prompts import FIND_ALL_ENTITIES
from ..utility.llmclient import AsyncChat
from ..utility.openalex import OPENALEX_SELECT, get_openalex_client
from ..utility.s2 import S2_DEFAULT_FIELDS, get_semantic_scholar_client
from ..utility.tool_config import ToolConfig
from .utils import extract_json


ENTITY_SEARCH_OPENALEX_SELECT = OPENALEX_SELECT
ENTITY_SEARCH_S2_SELECT = S2_DEFAULT_FIELDS


class FindAllEntities(AsyncChat):
    PROMPT: str = FIND_ALL_ENTITIES

    def _availability(self, response, context):
        result = extract_json(response)
        sentence = context["sentence"].lower()
        artifacts = []
        for artifact in result["artifacts"]:
            artifact = str(artifact).strip()
            if not artifact:
                continue
            assert artifact.lower() in sentence
            artifacts.append(artifact)
        return artifacts

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(sentence=inputs["sentence"]), {"sentence": inputs["sentence"]}


class UncitedEntities:
    """Find all scientific named entities in a paper and whether each entity is already cited."""

    def __init__(self, config: ToolConfig):
        self.config = config
        self.find_entities = FindAllEntities(config.llm_server_info, config.sampling_params)
        self.use_semantic_scholar = config.use_semantic_scholar()
        self.openalex = get_openalex_client(config)
        self.semantic_scholar = get_semantic_scholar_client(config) if self.use_semantic_scholar else None

    def _iter_sentences(self, paper: dict[str, Any]):
        def walk(node: Any):
            if isinstance(node, dict):
                for paragraph in node.get("paragraphs", []) or []:
                    yield from walk(paragraph)
                for section in node.get("sections", []) or []:
                    yield from walk(section)
            elif isinstance(node, list):
                for sentence in node:
                    if isinstance(sentence, dict) and sentence.get("text"):
                        yield sentence

        yield from walk(paper)

    def _has_citation(self, sentence: dict[str, Any]) -> bool:
        return bool(sentence.get("citations"))

    def _paper_text(self, paper: dict[str, Any]) -> str:
        return f"{paper.get('title', '')}\n{paper.get('abstract', '')}".lower()

    def _entity_matches_paper(self, entity: str, paper: dict[str, Any]) -> bool:
        return entity.lower() in self._paper_text(paper)

    async def _search_openalex(self, entity: str) -> list[dict[str, Any]]:
        payload = await self.openalex.search_works(
            search=entity,
            per_page=3,
            select=ENTITY_SEARCH_OPENALEX_SELECT,
        )
        return [
            paper
            for paper in payload.get("results", []) or []
            if self._entity_matches_paper(entity, paper)
        ]

    async def _search_semantic_scholar(self, entity: str) -> list[dict[str, Any]]:
        if self.semantic_scholar is None:
            return []
        payload = await self.semantic_scholar.search_works(
            search=entity,
            per_page=3,
            select=ENTITY_SEARCH_S2_SELECT,
        )
        return [
            paper
            for paper in payload.get("results", []) or []
            if self._entity_matches_paper(entity, paper)
        ]

    def _paper_key(self, paper: dict[str, Any]) -> str:
        for key in ("id", "paperId"):
            if paper.get(key):
                return str(paper[key]).replace("https://openalex.org/", "")
        title = re.sub(r"\s+", " ", paper.get("title", "") or "").strip().lower()
        return f"title:{title}" if title else ""

    async def _search_entity(self, entity: str) -> dict[str, Any]:
        matches = []
        searchers = [("openalex", self._search_openalex)]
        if self.use_semantic_scholar:
            searchers.append(("semantic_scholar", self._search_semantic_scholar))
        for source, search in searchers:
            try:
                papers = await search(entity)
            except Exception as exc:
                print(f"uncitedEntitySearch {source} {entity} {exc}")
                papers = []
            for paper in papers:
                item = dict(paper)
                item["source"] = source
                matches.append(item)
        unique = {}
        for paper in matches:
            key = self._paper_key(paper)
            if key and key not in unique:
                unique[key] = paper
        return {"entity": entity, "matched_papers": list(unique.values())}

    async def __call__(self, paper: dict[str, Any]) -> dict[str, Any]:
        sentences = list(self._iter_sentences(paper))
        tasks = [
            asyncio.create_task(self.find_entities.call(inputs={"sentence": sentence["text"]}))
            for sentence in sentences
        ]
        cited_entities, uncited_entities = set(), set()
        for sentence, result in zip(sentences, await asyncio.gather(*tasks, return_exceptions=True)):
            if not isinstance(result, list): continue
            has_citation = self._has_citation(sentence)
            sentence["entities"] = [
                {"name": entity, "sentence_has_citation": has_citation}
                for entity in result
            ]
            target = cited_entities if self._has_citation(sentence) else uncited_entities
            target.update(result)
        uncited_entities -= cited_entities

        search_tasks = [
            asyncio.create_task(self._search_entity(entity))
            for entity in sorted(uncited_entities)
        ]
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        return {
            "uncited_entities": [
                result
                for result in results
                if isinstance(result, dict)
            ]
        }
