from __future__ import annotations

import asyncio
import logging
import re
from datetime import timedelta
from typing import Any

from ..utility.academic_engine import get_academic_engine
from ..utility.citation_utils import citation_keys as normalize_citation_keys
from ..utility.paper_elements import Paper, Section
from ..utility.tool_config import ToolConfig


OPENALEX_LITERATURE_POOL_SELECT = "id,title,abstract_inverted_index,cited_by_count,counts_by_year,publication_date,referenced_works"
S2_LITERATURE_POOL_SELECT = "paperId,title,abstract,year,publicationDate,citationCount,referenceCount,externalIds,venue"
TARGET_SECTION_TYPES = {"CONTENT", "TAXONOMY", "EVALUATION", ""}
TARGET_SENTENCE_LABELS = {"SUMMARY", "CONTRAST", "SYNTHESIS", ""}


class BuildLiteraturePool:
    """Build a literature pool from cited papers, neighbor expansion, and a local citation graph."""

    def __init__(self, config: ToolConfig):
        self.config = config
        self.eval_date = config.evaluation_date
        self.engine = get_academic_engine(config)
        self.engine_name = (config.default_academic_search_engine or "openalex").strip().lower()
        self.neighbor_search_limit = 9999

    def _source_name(self) -> str:
        if self.engine_name in {"semantic_scholar", "semanticscholar", "semantic scholar", "s2"}: return "semantic scholar"
        return self.engine_name or "openalex"

    def _uses_semantic_scholar(self) -> bool:
        return self._source_name() == "semantic scholar"

    def _select_fields(self) -> str:
        return S2_LITERATURE_POOL_SELECT if self._uses_semantic_scholar() else OPENALEX_LITERATURE_POOL_SELECT

    def _query_keywords(self, query: list[str]) -> list[str]:
        values = query if isinstance(query, list) else [query]
        return [
            re.sub(r"\s+", " ", str(item or "")).strip().casefold()
            for item in values
            if str(item or "").strip()
        ]

    def _text_contains_query_keywords(self, paper: dict[str, Any], query_keywords: list[str]) -> bool:
        if not query_keywords: return True
        text = re.sub(r"\s+", " ", f"{paper.get('title', '')}\n{paper.get('abstract', '')}").casefold()
        return all(keyword in text for keyword in query_keywords)

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
            if str(key).lower() == "doi": candidates.append(value)
        for value in candidates:
            if not value: continue
            doi = str(value).strip().lower()
            doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
            doi = re.sub(r"^doi:", "", doi)
            if doi: return doi
        return ""

    def _paper_key(self, paper: dict[str, Any]) -> str:
        doi = self._paper_doi(paper)
        if doi: return f"doi:{doi}"
        ids = sorted(self._paper_ids(paper))
        if ids: return f"id:{ids[0]}"
        title = re.sub(r"\s+", " ", paper.get("title", "") or "").strip().lower()
        return f"title:{title}" if title else ""

    def _paper_aliases(self, paper: dict[str, Any]) -> set[str]:
        aliases = set()
        doi = self._paper_doi(paper)
        if doi: aliases.add(f"doi:{doi}")
        aliases.update(f"id:{paper_id}" for paper_id in self._paper_ids(paper))
        title = re.sub(r"\s+", " ", paper.get("title", "") or "").strip().lower()
        if title: aliases.add(f"title:{title}")
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
        """获取与config的preferred搜索引擎相同的metadata"""
        metadata = info.get("metadata") or {}
        if not isinstance(metadata, dict): return {}
        if "openalex" in metadata or "semantic scholar" in metadata:
            return {source: paper for source, paper in metadata.items() if isinstance(paper, dict)}
        return {self._source_name(): metadata} if metadata else {}

    def _default_engine_paper(self, info: dict[str, Any]) -> dict[str, Any] | None:
        """paper_content_map中每一个被引用论文包含了openalex信息和S2信息，无preferred时优先返回openalex信息。"""
        sources = self._metadata_sources(info)
        if not sources: return
        preferred = sources.get(self._source_name())
        if preferred: return preferred
        for source in ("openalex", "semantic scholar"):
            if sources.get(source): return sources[source]
        return next(iter(sources.values()))

    def _add_to_pool(
        self,
        pool: dict[str, dict[str, Any]],
        pool_index: dict[str, str],
        paper: dict[str, Any],
        label: str,
        citation_key: str = "",
    ) -> str:
        if not paper or not paper.get("title"): return ""
        key = self._paper_key(paper)
        if not key: return ""
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

    def _pool_key_for(self, pool_index: dict[str, str], paper: dict[str, Any]) -> str:
        return next((pool_index[alias] for alias in self._paper_aliases(paper) if alias in pool_index), "")

    def _target_sections(self, paper: Paper) -> list[Section]:
        sections = []

        def walk(section: Section):
            if section.functional_type in TARGET_SECTION_TYPES:
                sections.append(section)
            for child in section.children:
                walk(child)

        for section in paper.children:
            walk(section)
        return sections

    def _section_core_citation_keys(self, sections: list[Section]) -> list[str]:
        keys = []

        def walk(section: Section):
            for paragraph in section.paragraphs:
                for sentence in paragraph.sentences:
                    if sentence.label in TARGET_SENTENCE_LABELS:
                        keys.extend(normalize_citation_keys(sentence.citations))
            for child in section.children:
                walk(child)

        for section in sections: walk(section)
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
                    result = await method(paper_id, limit=self.neighbor_search_limit, select=self._select_fields(), **filter)
                else:
                    result = await method(paper_id, limit=self.neighbor_search_limit, select=self._select_fields(), filter=filter)
            except TypeError:
                result = await method(paper_id, limit=self.neighbor_search_limit, fields=self._select_fields(), filter=filter)
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

    async def __call__(self, query: list[str], paper: Paper, paper_content_map: dict[str, Any] | None = None):
        """获取邻居扩展和关键词搜索的交集，构建引用关系图"""
        paper_content_map = paper_content_map or paper.references or {}
        to_publication_date = (self.eval_date - timedelta(days=90)).strftime("%Y-%m-%d")
        query_keywords = self._query_keywords(query)
        expansion_filter = {"to_publication_date": to_publication_date}
        pool: dict[str, dict[str, Any]] = {}
        pool_index: dict[str, str] = {}
        edges: set[tuple[str, str]] = set()

        for citation_key, info in paper_content_map.items():
            cited_paper = self._default_engine_paper(info if isinstance(info, dict) else {})
            if cited_paper:
                self._add_to_pool(pool, pool_index, cited_paper, "cited_papers", citation_key=str(citation_key))

        sections = self._target_sections(paper)
        seed_papers = []
        for key in self._section_core_citation_keys(sections):
            info = paper_content_map.get(key)
            if isinstance(info, dict):
                cited_paper = self._default_engine_paper(info)
                if cited_paper:
                    seed_papers.append(cited_paper)
        seed_papers = self._deduplicate_papers(seed_papers)
        logging.info(f"BuildLiteraturePool starts with {len(seed_papers)} seed papers and {len(pool)} pools")

        async def _expand_labeled(seed: dict[str, Any], direction: str):
            try:
                candidates = await self._expand_one(seed, direction, dict(expansion_filter))
                return seed, direction, candidates, None
            except Exception as exc:
                return seed, direction, [], exc

        expansion_tasks = [
            asyncio.create_task(_expand_labeled(seed, direction))
            for seed in seed_papers for direction in ("cited_by", "cites")
        ]

        for i, task in enumerate(asyncio.as_completed(expansion_tasks), 1):
            seed, direction, candidates, exc = await task
            if exc:
                logging.error(f"literaturePoolExpand {direction} {exc}")
                continue
            seed_key = self._pool_key_for(pool_index, seed)
            if not seed_key:
                logging.warning("Paper %s not in pool, this may be a bug", seed.get("title"))
                seed_key = self._add_to_pool(pool, pool_index, seed, "cited_papers")
            for candidate in candidates:
                if not self._text_contains_query_keywords(candidate, query_keywords): continue
                candidate_key = self._add_to_pool(pool, pool_index, candidate, direction)
                if not seed_key or not candidate_key: continue
                if direction == "cited_by": edges.add((seed_key, candidate_key))
                else: edges.add((candidate_key, seed_key))
            logging.info(f"Expansion progress {i}/{len(expansion_tasks)} Pool size {len(pool)}")

        for source_key, item in list(pool.items()):
            if item.get("label") == "cited_papers": continue
            for work_id in item.get("paper", {}).get("referenced_works", []) or []:
                target_key = next((pool_index[alias] for alias in self._referenced_work_aliases(work_id) if alias in pool_index), "")
                if target_key:
                    edges.add((target_key, source_key))
        graph = self._graph_dict(pool, edges)
        logging.info(f"BuildLiteraturePool {len(pool)} neighbor, {len(graph['edges'])} graph edges")
        return {"literature_pool": pool, "citation_graph": graph}
