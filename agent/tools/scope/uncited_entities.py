from __future__ import annotations

import asyncio
import re
from typing import Any

import jsonschema

from ..prompts import JUDGE_UNCITED_BATCH, JUDGE_UNCITED_BATCH_ITEM_SCHEMA
from ..utility.citation_utils import citation_keys
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper, Paragraph
from ..utility.academic_engine import get_academic_engine
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json
from ..utility.content_walk import iter_paragraphs


class JudgeUncitedBatchClient(AsyncChat):
    PROMPT = JUDGE_UNCITED_BATCH

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)
    def _candidate_text(self, papers: list[dict[str, Any]]) -> str:
        blocks = []
        for idx, paper in enumerate(papers, start=1):
            title = str(paper.get("title") or "").strip()
            abstract = str(paper.get("abstract") or "").strip()
            blocks.append(f"[{idx}] Title: {title}\n    Abstract: {abstract}")
        return "\n".join(blocks)

    def _availability(self, response, context):
        result = extract_json(response)
        candidate_count = context["candidate_count"]
        schema = {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "minLength": 1},
                "results": {
                    "type": "array",
                    "items": JUDGE_UNCITED_BATCH_ITEM_SCHEMA,
                    "minItems": candidate_count,
                    "maxItems": candidate_count,
                },
                "most_likely_source": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1, "maximum": candidate_count},
                        {"type": "null"},
                    ],
                },
            },
            "required": ["entity", "results", "most_likely_source"],
            "additionalProperties": False,
        }
        jsonschema.validate(result, schema)
        expected_indexes = set(range(1, candidate_count + 1))
        actual_indexes = {item["paper_index"] for item in result["results"]}
        assert actual_indexes == expected_indexes, f"Indexes mismatch: {actual_indexes}, {expected_indexes}"
        papers = context["candidate_papers"]
        yes_indexes = {item["paper_index"] for item in result["results"] if item["decision"] == "yes"}
        assert (result["most_likely_source"] is None) == (not yes_indexes)
        if result["most_likely_source"] is not None:
            assert result["most_likely_source"] in yes_indexes, "Most_likely invalid"
        for item in result["results"]:
            evidence = item["evidence"]
            if item["decision"] == "yes":
                assert evidence, "Yes decision with no evidence"
            if evidence:
                paper = papers[item["paper_index"] - 1]
                source_text = f"{paper.get('title', '')}\n{paper.get('abstract', '')}"
                verified, _ = self.check.verify([evidence], source_text)
                assert verified, f"Evidence invalid: {evidence}"
        return result

    def _organize_inputs(self, inputs):
        papers = inputs["candidate_papers"]
        prompt = self.PROMPT.format(
            entity_name=inputs["entity_name"],
            candidate_papers=self._candidate_text(papers),
        )
        return prompt, {
            "candidate_count": len(papers),
            "candidate_papers": papers,
        }
    

class UncitedEntities:
    """Find scientific named entities whose defining papers may be missing from citations."""

    def __init__(self, config: ToolConfig):
        self.config = config
        self.academic_engine = get_academic_engine(config)
        self.evidence_check = EvidenceCheck(config)
        # self.extract_proposed = ExtractProposedClient(
        #     config.llm_server_info,
        #     config.sampling_params,
        #     evidence_check=self.evidence_check,
        # )
        self.judge_uncited_batch = JudgeUncitedBatchClient(config)
        self._proposed_cache: dict[str, list[dict[str, str]]] = {}

    def _paragraph_citation_keys(self, paragraph: Paragraph) -> list[str]:
        keys = []
        for sentence in paragraph.sentences:
            keys.extend(citation_keys(sentence.citations))
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

    async def _paper_proposes_entity(self, entity_names: list[str], candidate_papers: list[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
        papers = candidate_papers if isinstance(candidate_papers, list) else [candidate_papers]
        papers = [paper for paper in papers if paper.get("title") and paper.get("abstract")]
        entity_name = " / ".join(entity_names)
        if not papers:
            return {"entity": entity_name, "results": [], "most_likely_source": None}
        try:
            return await self.judge_uncited_batch.call(
                inputs={"entity_name": entity_name, "candidate_papers": papers}
            )
        except Exception as exc:
            print(f"judgeUncitedBatch {entity_name} {exc}")
            return {"entity": entity_name, "results": [], "most_likely_source": None}

    async def _search_academic_engine(self, entity_names: list[str]) -> list[dict[str, Any]]:
        payload = await self.academic_engine.search_works(
            search=entity_names[0],
            per_page=3,
        )
        return [paper for paper in payload.get("results", []) or [] if paper.get("title") and paper.get("abstract")]

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

    def _entity_groups(self, paragraphs: list[Paragraph]) -> tuple[dict[str, dict[str, Any]], dict[int, list[str]]]:
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
            for entity in paragraph.entities:
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
        paragraph: Paragraph,
        paper_content_map: dict[str, Any] | None,
    ) -> bool:
        papers = self._paper_sources_for_keys(self._paragraph_citation_keys(paragraph), paper_content_map)
        judgment = await self._paper_proposes_entity(group["names"], papers)
        return any(item["decision"] == "yes" for item in judgment["results"])

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

    async def _search_entity(
        self,
        group: dict[str, Any],
        cited_papers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            candidates = await self._search_academic_engine(group["names"])
        except Exception as exc:
            print(f"uncitedEntitySearch {self.config.default_academic_search_engine} {group['names'][0]} {exc}")
            candidates = []
        candidates = [paper for paper in candidates if not self._is_cited(paper, cited_papers)]
        judgment = await self._paper_proposes_entity(group["names"], candidates)
        yes_indexes = {item["paper_index"] for item in judgment["results"] if item["decision"] == "yes"}
        matches = []
        for idx, paper in enumerate(candidates, start=1):
            if idx not in yes_indexes:
                continue
            item = dict(paper)
            item["source"] = self.config.default_academic_search_engine
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
            "judge_results": judgment["results"],
            "most_likely_source": judgment["most_likely_source"],
        }

    async def __call__(
        self,
        paper: Paper,
        cited_papers: list[dict[str, Any]] | None = None,
        paper_content_map: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cited_papers = cited_papers or self._all_cited_papers(paper_content_map)
        paragraphs = list(iter_paragraphs(paper, include_appendix=True))
        groups, paragraph_group_keys = self._entity_groups(paragraphs)

        for paragraph in paragraphs:
            for key in paragraph_group_keys.get(id(paragraph), []):
                if not groups[key]["cited"] and await self._paragraph_cites_group(groups[key], paragraph, paper_content_map):
                    groups[key]["cited"] = True
        # Propagate cited status across prefix-related entity variants.
        self._apply_prefix_citation_rule(groups)
        uncited_groups = [group for group in groups.values() if not group["cited"]]
        print(f"{len(uncited_groups)} uncited groups")
        # Search remaining uncited groups.
        search_tasks = [
            asyncio.create_task(self._search_entity(group, cited_papers))
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
