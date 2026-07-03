from __future__ import annotations

import asyncio
import re
from typing import Any

import jsonschema

from ..prompts import EXTRACT_PROPOSED
from ..utility.citation_utils import citation_keys
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.openalex import OPENALEX_SELECT, get_openalex_client
from ..utility.s2 import S2_DEFAULT_FIELDS, get_semantic_scholar_client
from ..utility.tool_config import ToolConfig
from .utils import extract_json, extract_literature_pool_proposed_entities


ENTITY_SEARCH_OPENALEX_SELECT = OPENALEX_SELECT
ENTITY_SEARCH_S2_SELECT = S2_DEFAULT_FIELDS

EXTRACT_PROPOSED_SCHEMA = {
    "type": "object",
    "properties": {
        "proposed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "evidence_sentence": {"type": "string", "minLength": 1},
                },
                "required": ["name", "evidence_sentence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["proposed"],
    "additionalProperties": False,
}


class ExtractProposedClient(AsyncChat):
    PROMPT = EXTRACT_PROPOSED

    def __init__(self, llm, sampling_params: dict | None = None, evidence_check: EvidenceCheck | None = None):
        super().__init__(llm, sampling_params)
        self.evidence_check = evidence_check

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, EXTRACT_PROPOSED_SCHEMA)
        abstract = context["abstract"]
        proposed = []
        for item in result["proposed"]:
            ok, _confidence = self.evidence_check.verify([item["evidence_sentence"]], abstract) if self.evidence_check else (True, 1.0)
            assert ok
            proposed.append({"name": item["name"].strip(), "evidence_sentence": item["evidence_sentence"].strip()})
        return proposed

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(title=inputs["title"], abstract=inputs["abstract"]), {
            "abstract": inputs["abstract"],
        }


class UncitedEntities:
    """Find scientific named entities whose defining papers may be missing from citations."""

    def __init__(self, config: ToolConfig):
        self.config = config
        self.use_semantic_scholar = config.use_semantic_scholar()
        self.openalex = get_openalex_client(config)
        self.semantic_scholar = get_semantic_scholar_client(config) if self.use_semantic_scholar else None
        self.evidence_check = EvidenceCheck(config)
        self.extract_proposed = ExtractProposedClient(
            config.llm_server_info,
            config.sampling_params,
            evidence_check=self.evidence_check,
        )
        self._proposed_cache: dict[str, list[dict[str, str]]] = {}

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

        # Intentionally skip abstract: uncited-entity checking is body-only.
        yield from walk(paper)

    def _paragraph_citation_keys(self, paragraph: dict[str, Any]) -> list[str]:
        keys = []
        for sentence in paragraph.get("sentences", []) or []:
            if isinstance(sentence, dict):
                keys.extend(citation_keys(sentence.get("citations")))
        return list(dict.fromkeys(keys))

    def _metadata_sources(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        metadata = info.get("metadata") or {}
        if isinstance(metadata, dict) and ("openalex" in metadata or "semantic scholar" in metadata):
            return [paper for paper in metadata.values() if isinstance(paper, dict)]
        return [metadata] if isinstance(metadata, dict) and metadata else []

    def _paper_sources_for_keys(self, keys: list[str], paper_content_map: dict[str, Any] | None) -> list[dict[str, Any]]:
        papers = []
        paper_content_map = paper_content_map or {}
        for key in keys:
            info = paper_content_map.get(key)
            if isinstance(info, dict):
                papers.extend(self._metadata_sources(info))
        return [paper for paper in papers if paper.get("title")]

    def _all_cited_papers(self, paper_content_map: dict[str, Any] | None) -> list[dict[str, Any]]:
        papers = []
        for info in (paper_content_map or {}).values():
            if isinstance(info, dict):
                papers.extend(self._metadata_sources(info))
        return [paper for paper in papers if paper.get("title")]

    def _paper_text(self, paper: dict[str, Any]) -> str:
        return f"{paper.get('title', '')}\n{paper.get('abstract', '')}".strip()

    def _normalize_entity_name(self, name: str) -> str:
        name = re.sub(r"\s+", " ", name or "").strip()
        if re.search(r"[A-Z]s$", name):
            return name[:-1]
        return name

    def _match_key(self, name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z]+", "", self._normalize_entity_name(name)).casefold()

    def _abbreviation_groups(self, query: str) -> list[str]:
        return re.findall(r"[A-Z][a-z]*", query or "")

    def _words(self, name: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+", name or "")

    def _abbreviation_match(self, query: str, candidate: str) -> bool:
        groups = self._abbreviation_groups(query)
        words = self._words(candidate)
        if not groups or len(groups) != len(words):
            return False
        return all(word.casefold().startswith(group.casefold()) for group, word in zip(groups, words))

    def _entity_matches_name(self, entity: str, proposed_name: str) -> bool:
        if self._match_key(entity) == self._match_key(proposed_name):
            return True
        return self._abbreviation_match(entity, proposed_name) or self._abbreviation_match(proposed_name, entity)

    async def _paper_proposed(self, paper: dict[str, Any]) -> list[dict[str, str]]:
        key = self._paper_key(paper)
        if key in self._proposed_cache:
            return self._proposed_cache[key]
        title = str(paper.get("title") or "").strip()
        abstract = str(paper.get("abstract") or "").strip()
        if not title or not abstract:
            self._proposed_cache[key] = []
            return []
        try:
            proposed = await self.extract_proposed.call(inputs={"title": title, "abstract": abstract})
        except Exception as exc:
            print(f"extractProposed {title} {exc}")
            proposed = []
        self._proposed_cache[key] = proposed
        return proposed

    async def _paper_proposes_entity(self, entity_names: list[str], paper: dict[str, Any]) -> bool:
        proposed = await self._paper_proposed(paper)
        return any(
            self._entity_matches_name(entity, item["name"])
            for entity in entity_names
            for item in proposed
        )

    async def _entity_matches_paper(self, entity_names: list[str], paper: dict[str, Any]) -> bool:
        return await self._paper_proposes_entity(entity_names, paper)

    async def _search_openalex(self, entity_names: list[str]) -> list[dict[str, Any]]:
        payload = await self.openalex.search_works(
            search=entity_names[0],
            per_page=5,
            select=ENTITY_SEARCH_OPENALEX_SELECT,
        )
        matches = []
        for paper in payload.get("results", []) or []:
            if await self._entity_matches_paper(entity_names, paper):
                matches.append(paper)
        return matches

    async def _search_semantic_scholar(self, entity_names: list[str]) -> list[dict[str, Any]]:
        if self.semantic_scholar is None:
            return []
        payload = await self.semantic_scholar.search_works(
            search=entity_names[0],
            per_page=3,
            select=ENTITY_SEARCH_S2_SELECT,
        )
        matches = []
        for paper in payload.get("results", []) or []:
            if await self._entity_matches_paper(entity_names, paper):
                matches.append(paper)
        return matches

    def _paper_ids(self, paper: dict[str, Any]) -> set[str]:
        ids = set()
        for key in ("id", "paperId", "corpusId"):
            if paper.get(key):
                ids.add(str(paper[key]).replace("https://openalex.org/", ""))
        raw_ids = paper.get("ids")
        if isinstance(raw_ids, dict):
            ids.update(str(value).replace("https://openalex.org/", "") for value in raw_ids.values() if value)
        elif isinstance(raw_ids, (list, tuple, set)):
            ids.update(str(value).replace("https://openalex.org/", "") for value in raw_ids if value)
        return {item for item in ids if item}

    def _paper_title(self, paper: dict[str, Any]) -> str:
        return re.sub(r"\s+", " ", paper.get("title", "") or "").strip().lower()

    def _paper_key(self, paper: dict[str, Any]) -> str:
        ids = sorted(self._paper_ids(paper))
        if ids:
            return f"id:{ids[0]}"
        title = self._paper_title(paper)
        return f"title:{title}" if title else ""

    def _is_cited(self, candidate: dict[str, Any], cited_papers: list[dict[str, Any]]) -> bool:
        candidate_ids = self._paper_ids(candidate)
        candidate_title = self._paper_title(candidate)
        for cited in cited_papers:
            if candidate_ids and candidate_ids & self._paper_ids(cited):
                return True
            if candidate_title and candidate_title == self._paper_title(cited):
                return True
        return False

    def _variant_names(self, entity: dict[str, Any] | str) -> list[str]:
        if not isinstance(entity, dict):
            return [str(entity).strip()] if str(entity).strip() else []
        names = [str(entity.get("name") or "").strip()]
        alias = str(entity.get("alias_pairs") or "").strip()
        if alias:
            names.append(alias)
        for alternative in entity.get("alternative_names", []) or []:
            if alternative:
                names.append(str(alternative).strip())
        return list(dict.fromkeys(self._normalize_entity_name(name) for name in names if name))

    def _entity_groups(self, paragraphs: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[int, list[str]]]:
        parent: dict[str, str] = {}
        labels: dict[str, str] = {}
        paragraph_keys: dict[int, list[str]] = {}

        def find(key: str) -> str:
            parent.setdefault(key, key)
            if parent[key] != key:
                parent[key] = find(parent[key])
            return parent[key]

        def union(left: str, right: str):
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        entity_records = []
        for paragraph in paragraphs:
            keys = []
            for entity in paragraph.get("entities", []) or []:
                variants = self._variant_names(entity)
                if not variants:
                    continue
                variant_keys = [self._match_key(name) for name in variants if self._match_key(name)]
                if not variant_keys:
                    continue
                for name, key in zip(variants, variant_keys):
                    labels.setdefault(key, name)
                    find(key)
                for key in variant_keys[1:]:
                    union(variant_keys[0], key)
                entity_records.append((paragraph, entity, variant_keys[0]))
                keys.append(variant_keys[0])
            paragraph_keys[id(paragraph)] = list(dict.fromkeys(keys))

        groups: dict[str, dict[str, Any]] = {}
        key_to_group = {}
        for key in list(parent):
            root = find(key)
            key_to_group[key] = root
            groups.setdefault(root, {"names": [], "cited": False})
            if labels[key] not in groups[root]["names"]:
                groups[root]["names"].append(labels[key])

        remapped_paragraph_keys = {
            pid: list(dict.fromkeys(key_to_group.get(key, find(key)) for key in keys))
            for pid, keys in paragraph_keys.items()
        }

        for paragraph, entity, key in entity_records:
            root = key_to_group.get(key, find(key))
            if isinstance(entity, dict) and entity.get("sentence_has_citation"):
                groups[root]["cited"] = True
        return groups, remapped_paragraph_keys

    async def _paragraph_cites_group(
        self,
        group: dict[str, Any],
        paragraph: dict[str, Any],
        paper_content_map: dict[str, Any] | None,
    ) -> bool:
        papers = self._paper_sources_for_keys(self._paragraph_citation_keys(paragraph), paper_content_map)
        for paper in papers:
            if await self._paper_proposes_entity(group["names"], paper):
                return True
        return False

    def _apply_prefix_citation_rule(self, groups: dict[str, dict[str, Any]]):
        changed = True
        while changed:
            changed = False
            items = list(groups.items())
            for left_key, left in items:
                for right_key, right in items:
                    if left_key == right_key or left["cited"] == right["cited"]:
                        continue
                    left_names = [self._match_key(name) for name in left["names"]]
                    right_names = [self._match_key(name) for name in right["names"]]
                    if any(a and b and a != b and (a.startswith(b) or b.startswith(a)) for a in left_names for b in right_names):
                        left["cited"] = True
                        right["cited"] = True
                        changed = True

    def _literature_pool_matches(
        self,
        group: dict[str, Any],
        proposed_index: dict[str, dict[str, Any]],
        cited_papers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        matches = []
        seen = set()
        for proposed_name, paper in proposed_index.items():
            if not any(self._entity_matches_name(entity, proposed_name) for entity in group["names"]):
                continue
            if self._is_cited(paper, cited_papers):
                continue
            key = self._paper_key(paper)
            if key and key not in seen:
                seen.add(key)
                item = dict(paper)
                item["source"] = item.get("source", "literature_pool")
                matches.append(item)
        return matches

    async def _search_entity(
        self,
        group: dict[str, Any],
        cited_papers: list[dict[str, Any]],
        proposed_index: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        matches = self._literature_pool_matches(group, proposed_index, cited_papers)
        if not matches:
            searchers = [("openalex", self._search_openalex)]
            if self.use_semantic_scholar:
                searchers.append(("semantic_scholar", self._search_semantic_scholar))
            for source, search in searchers:
                try:
                    papers = await search(group["names"])
                except Exception as exc:
                    print(f"uncitedEntitySearch {source} {group['names'][0]} {exc}")
                    papers = []
                for paper in papers:
                    if self._is_cited(paper, cited_papers):
                        continue
                    item = dict(paper)
                    item["source"] = source
                    matches.append(item)
        unique = {}
        for paper in matches:
            key = self._paper_key(paper)
            if key and key not in unique:
                unique[key] = paper
        return {
            "entity": group["names"][0],
            "alternative_names": group["names"][1:],
            "matched_papers": list(unique.values()),
        }

    async def __call__(
        self,
        paper: dict[str, Any],
        cited_papers: list[dict[str, Any]] | None = None,
        paper_content_map: dict[str, Any] | None = None,
        literature_pool: Any = None,
    ) -> dict[str, Any]:
        cited_papers = cited_papers or self._all_cited_papers(paper_content_map)
        paragraphs = list(self._iter_paragraphs(paper))
        groups, paragraph_group_keys = self._entity_groups(paragraphs)

        for paragraph in paragraphs:
            for key in paragraph_group_keys.get(id(paragraph), []):
                if not groups[key]["cited"] and await self._paragraph_cites_group(groups[key], paragraph, paper_content_map):
                    groups[key]["cited"] = True

        self._apply_prefix_citation_rule(groups)
        proposed_index = extract_literature_pool_proposed_entities(literature_pool, cited_papers) if literature_pool else {}
        uncited_groups = [group for group in groups.values() if not group["cited"]]
        print(f"{len(uncited_groups)} uncited groups")
        search_tasks = [
            asyncio.create_task(self._search_entity(group, cited_papers, proposed_index))
            for group in sorted(uncited_groups, key=lambda item: item["names"][0].casefold())
        ]
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        return {
            "uncited_entities": [
                result
                for result in results
                if isinstance(result, dict) and result.get("matched_papers")
            ]
        }
