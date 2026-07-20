from __future__ import annotations

import asyncio
import re
from datetime import timedelta
from typing import Any

from ..utility.academic_engine import get_academic_engine
from ..utility.citation_utils import citation_keys as normalize_citation_keys
from ..utility.tool_config import ToolConfig


OPENALEX_LITERATURE_POOL_SELECT = "id,title,abstract_inverted_index,cited_by_count,counts_by_year,publication_date,referenced_works"
S2_LITERATURE_POOL_SELECT = "paperId,title,abstract,year,publicationDate,citationCount,referenceCount,externalIds,venue"
TARGET_SECTION_TYPES = {"CONTENT", "TAXONOMY", "EVALUATION", ""}
TARGET_SENTENCE_LABELS = {"SUMMARY", "COMPARISON", "EVALUATION", "SYNTHESIS", ""}


class BuildLiteraturePool:
    """Build a literature pool from cited papers, neighbor expansion, and a local citation graph."""

    def __init__(self, config: ToolConfig):
        self.config = config
        self.eval_date = config.evaluation_date
        self.engine = get_academic_engine(config)
        self.engine_name = (config.default_academic_search_engine or "openalex").strip().lower()

    def _source_name(self) -> str:
        if self.engine_name in {"semantic_scholar", "semanticscholar", "semantic scholar", "s2"}:
            return "semantic scholar"
        return self.engine_name or "openalex"

    def _uses_semantic_scholar(self) -> bool:
        return self._source_name() == "semantic scholar"

    def _select_fields(self) -> str:
        return S2_LITERATURE_POOL_SELECT if self._uses_semantic_scholar() else OPENALEX_LITERATURE_POOL_SELECT

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

    def _paper_doi(self, paper: dict[str, Any]) -> str:
        candidates = [paper.get("doi")]
        raw_ids = paper.get("ids")
        if isinstance(raw_ids, dict):
            candidates.append(raw_ids.get("doi"))
        external_ids = paper.get("external_ids") or paper.get("externalIds") or {}
        for key, value in external_ids.items():
            if str(key).lower() == "doi":
                candidates.append(value)
        for value in candidates:
            if not value:
                continue
            doi = str(value).strip().lower()
            doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
            doi = re.sub(r"^doi:", "", doi)
            if doi:
                return doi
        return ""

    def _paper_key(self, paper: dict[str, Any]) -> str:
        doi = self._paper_doi(paper)
        if doi:
            return f"doi:{doi}"
        ids = sorted(self._paper_ids(paper))
        if ids:
            return f"id:{ids[0]}"
        title = re.sub(r"\s+", " ", paper.get("title", "") or "").strip().lower()
        return f"title:{title}" if title else ""

    def _paper_aliases(self, paper: dict[str, Any]) -> set[str]:
        aliases = set()
        doi = self._paper_doi(paper)
        if doi:
            aliases.add(f"doi:{doi}")
        aliases.update(f"id:{paper_id}" for paper_id in self._paper_ids(paper))
        title = re.sub(r"\s+", " ", paper.get("title", "") or "").strip().lower()
        if title:
            aliases.add(f"title:{title}")
        return aliases

    def _deduplicate_papers(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen_aliases = set()
        unique = []
        for paper in papers:
            aliases = self._paper_aliases(paper)
            if not aliases or seen_aliases & aliases:
                continue
            seen_aliases.update(aliases)
            unique.append(paper)
        return unique

    def _metadata_sources(self, info: dict[str, Any]) -> dict[str, dict[str, Any]]:
        metadata = info.get("metadata") or {}
        if not isinstance(metadata, dict):
            return {}
        if "openalex" in metadata or "semantic scholar" in metadata:
            return {source: paper for source, paper in metadata.items() if isinstance(paper, dict)}
        return {self._source_name(): metadata} if metadata else {}

    def _default_engine_paper(self, info: dict[str, Any]) -> dict[str, Any] | None:
        sources = self._metadata_sources(info)
        if not sources:
            return None
        preferred = sources.get(self._source_name())
        if preferred:
            return preferred
        for source in ("openalex", "semantic scholar"):
            if sources.get(source):
                return sources[source]
        return next(iter(sources.values()))

    def _add_to_pool(
        self,
        pool: dict[str, dict[str, Any]],
        pool_index: dict[str, str],
        paper: dict[str, Any],
        label: str,
        citation_key: str = "",
    ) -> str:
        if not paper or not paper.get("title"):
            return ""
        key = self._paper_key(paper)
        if not key:
            return ""
        aliases = self._paper_aliases(paper)
        existing = next((pool_index[alias] for alias in aliases if alias in pool_index), "")
        if existing:
            if citation_key:
                pool[existing].setdefault("citation_keys", [])
                if citation_key not in pool[existing]["citation_keys"]:
                    pool[existing]["citation_keys"].append(citation_key)
                pool_index[f"citation:{citation_key}"] = existing
            return existing
        pool[key] = {"paper": paper, "label": label}
        if citation_key:
            pool[key]["citation_keys"] = [citation_key]
            aliases.add(f"citation:{citation_key}")
        for alias in aliases:
            pool_index[alias] = key
        return key

    def _target_sections(self, paper: dict[str, Any]) -> list[dict[str, Any]]:
        sections = []

        def walk(section: dict[str, Any]):
            if section.get("functional_type", "") in TARGET_SECTION_TYPES:
                sections.append(section)
            for child in section.get("sections", []) or []:
                if isinstance(child, dict):
                    walk(child)

        for section in paper.get("sections", []) or []:
            if isinstance(section, dict):
                walk(section)
        return sections

    def _normalize_citations(self, citations: Any) -> list[str]:
        return normalize_citation_keys(citations)

    def _section_core_citation_keys(self, sections: list[dict[str, Any]]) -> list[str]:
        keys = []

        def walk(node: Any):
            if isinstance(node, dict):
                if "sentences" in node:
                    walk(node.get("sentences", []) or [])
                    return
                for paragraph in node.get("paragraphs", []) or []:
                    walk(paragraph)
                for child in node.get("sections", []) or []:
                    walk(child)
            elif isinstance(node, list):
                for sentence in node:
                    if isinstance(sentence, dict) and sentence.get("label", "") in TARGET_SENTENCE_LABELS:
                        keys.extend(self._normalize_citations(sentence.get("citations", [])))

        for section in sections:
            walk(section)
        return list(dict.fromkeys(keys))

    def _query_ids(self, paper: dict[str, Any]) -> list[str]:
        ids = []
        for key in ("id", "paperId"):
            if paper.get(key):
                ids.append(str(paper[key]).replace("https://openalex.org/", ""))
        ids.extend(self._paper_ids(paper))
        return list(dict.fromkeys(item for item in ids if item))

    async def _expand_one(self, paper: dict[str, Any], direction: str, filter: dict) -> list[dict[str, Any]]:
        method = self.engine.get_citations if direction == "cited_by" else self.engine.get_references
        for paper_id in self._query_ids(paper):
            try:
                if self._uses_semantic_scholar() and direction == "cited_by":
                    result = await method(paper_id, limit=9999, select=self._select_fields(), **filter)
                else:
                    result = await method(paper_id, limit=9999, select=self._select_fields(), filter=filter)
            except TypeError:
                result = await method(paper_id, limit=9999, fields=self._select_fields(), filter=filter)
            except Exception:
                continue
            papers = result.get("results", []) or []
            if papers:
                return papers
        return []

    def _referenced_work_aliases(self, work_id: str) -> list[str]:
        value = str(work_id or "").replace("https://openalex.org/", "").strip()
        return [f"id:{value}"] if value else []

    def _graph_dict(self, pool: dict[str, dict[str, Any]], edges: set[tuple[str, str]]) -> dict[str, Any]:
        nodes = sorted(pool)
        edges = {(source, target) for source, target in edges if source in pool and target in pool and source != target}
        out_counts = {node: 0 for node in nodes}
        for source, _target in edges:
            out_counts[source] += 1
        for key, count in out_counts.items():
            pool[key]["local_cited_by_count"] = count
            pool[key].setdefault("paper", {})["local_cited_by_count"] = count
        return {
            "nodes": nodes,
            "edges": [
                {"source": source, "target": target}
                for source, target in sorted(edges)
            ],
        }

    async def __call__(self, query: str, paper: dict[str, Any], paper_content_map: dict[str, Any] | None = None):
        paper_content_map = paper_content_map or paper.get("paper_content_map") or paper.get("citations") or {}
        to_publication_date = (self.eval_date - timedelta(days=90)).strftime("%Y-%m-%d")
        pool: dict[str, dict[str, Any]] = {}
        pool_index: dict[str, str] = {}
        edges: set[tuple[str, str]] = set()

        for citation_key, info in paper_content_map.items():
            cited_paper = self._default_engine_paper(info if isinstance(info, dict) else {})
            if cited_paper:
                self._add_to_pool(pool, pool_index, cited_paper, "cited_papers", citation_key=str(citation_key))

        print(f"BuildLiteraturePool starts with {len(pool)} cited papers")
        sections = self._target_sections(paper)
        seed_papers = []
        for key in self._section_core_citation_keys(sections):
            info = paper_content_map.get(key)
            if isinstance(info, dict):
                cited_paper = self._default_engine_paper(info)
                if cited_paper:
                    seed_papers.append(cited_paper)
        seed_papers = self._deduplicate_papers(seed_papers)

        async def _expand_labeled(seed: dict[str, Any], direction: str):
            try:
                candidates = await self._expand_one(seed, direction, {"to_publication_date": to_publication_date})
                return seed, direction, candidates, None
            except Exception as exc:
                return seed, direction, [], exc

        import tqdm
        expansion_tasks = []
        for direction in ("cited_by", "cites"):
            for seed in seed_papers:
                expansion_tasks.append(asyncio.create_task(_expand_labeled(seed, direction)))

        for task in tqdm.tqdm(asyncio.as_completed(expansion_tasks), total=len(expansion_tasks)):
            seed, direction, candidates, exc = await task
            if exc:
                print(f"literaturePoolExpand {direction} {exc}")
                continue
            seed_key = self._add_to_pool(pool, pool_index, seed, "cited_papers")
            for candidate in candidates:
                candidate_key = self._add_to_pool(pool, pool_index, candidate, direction)
                if not seed_key or not candidate_key:
                    continue
                if direction == "cited_by":
                    edges.add((seed_key, candidate_key))
                else:
                    edges.add((candidate_key, seed_key))

        for source_key, item in list(pool.items()):
            if item.get("label") == "cited_papers":
                continue
            for work_id in item.get("paper", {}).get("referenced_works", []) or []:
                target_key = next((pool_index[alias] for alias in self._referenced_work_aliases(work_id) if alias in pool_index), "")
                if target_key:
                    edges.add((target_key, source_key))
        graph = self._graph_dict(pool, edges)
        print(f"BuildLiteraturePool {len(pool)} neighbor, {len(graph['edges'])} graph edges")
        return {"literature_pool": pool, "citation_graph": graph}
