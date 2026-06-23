from __future__ import annotations

import asyncio
import re
from datetime import timedelta
from typing import Any

from ..utility.academic_engine import get_academic_engine
from ..utility.tool_config import ToolConfig


OPENALEX_LITERATURE_POOL_SELECT = "id,title,cited_by_count,counts_by_year,publication_date"
S2_LITERATURE_POOL_SELECT = "paperId,title,year,publicationDate,citationCount,referenceCount,externalIds,venue"
TARGET_SECTION_TYPES = {"CONTENT", "TAXONOMY", "EVALUATION", ""}
TARGET_SENTENCE_LABELS = {"SUMMARY", "COMPARISON", "EVALUATION", "SYNTHESIS", ""}


class BuildLiteraturePool:
    """Build a literature pool from cited papers, neighbor expansion, and topic-keyword search papers."""

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
        if self._uses_semantic_scholar():
            return S2_LITERATURE_POOL_SELECT
        return OPENALEX_LITERATURE_POOL_SELECT

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
            if doi: return doi
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
        if not isinstance(metadata, dict): return {}
        if "openalex" in metadata or "semantic scholar" in metadata:
            return {source: paper for source, paper in metadata.items() if isinstance(paper, dict)}
        return {self._source_name(): metadata} if metadata else {}

    def _default_engine_paper(self, info: dict[str, Any]) -> dict[str, Any] | None:
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
    ):
        if not paper or not paper.get("title"): return
        key = self._paper_key(paper)
        if not key: return
        aliases = self._paper_aliases(paper)
        if any(alias in pool_index for alias in aliases): return
        pool[key] = {"paper": paper, "label": label}
        for alias in aliases:
            pool_index[alias] = key

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
        normalized = []
        for citation in citations or []:
            if isinstance(citation, dict):
                key = citation.get("key") or citation.get("ref_text")
            else:
                key = citation
            if key:
                normalized.append(str(key))
        return normalized

    def _section_core_citation_keys(self, sections: list[dict[str, Any]]) -> list[str]:
        keys = []

        def walk(node: Any):
            if isinstance(node, dict):
                for paragraph in node.get("paragraphs", []) or []:
                    walk(paragraph)
                for child in node.get("sections", []) or []:
                    walk(child)
            elif isinstance(node, list):
                for sentence in node:
                    if not isinstance(sentence, dict):
                        continue
                    if sentence.get("label", "") in TARGET_SENTENCE_LABELS:
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
        method = self.engine.get_references if direction == "cited_by" else self.engine.get_citations
        for paper_id in self._query_ids(paper):
            try:
                result = await method(
                    paper_id,
                    limit=9999,
                    select=self._select_fields(),
                    filter=filter
                )
            except TypeError:
                result = await method(
                    paper_id,
                    limit=9999,
                    fields=self._select_fields(),
                    filter=filter,
                )
            except Exception:
                continue
            papers = result.get("results", []) or []
            if papers:
                return papers
        return []

    async def __call__(self, query: str, paper: dict[str, Any], paper_content_map: dict[str, Any] | None = None):
        paper_content_map = paper_content_map or paper.get("paper_content_map") or paper.get("citations") or {}
        to_publication_date = (self.eval_date - timedelta(days=90)).strftime("%Y-%m-%d")
        pool: dict[str, dict[str, Any]] = {}
        pool_index: dict[str, str] = {}
        
        for info in paper_content_map.values():
            cited_paper = self._default_engine_paper(info if isinstance(info, dict) else {})
            if cited_paper:
                self._add_to_pool(pool, pool_index, cited_paper, "cited_papers")

        print(f"BuildLiteraturePool starts with {len(pool)} cited papers")
        sections = self._target_sections(paper)
        seed_papers = []
        for key in self._section_core_citation_keys(sections):
            info = paper_content_map.get(key)
            if isinstance(info, dict):
                cited_paper = self._default_engine_paper(info)
                if cited_paper: seed_papers.append(cited_paper)
        seed_papers = self._deduplicate_papers(seed_papers)
        
        async def _expand_labeled(seed: dict[str, Any], direction: str, label: str):
            try:
                candidates = await self._expand_one(seed, direction, {"to_publication_date": to_publication_date})
                return direction, label, candidates, None
            except Exception as exc:
                return direction, label, [], exc

        import tqdm
        expansion_results = {"cited_by": [], "cites": []}
        expansion_tasks = []
        for direction, label in (("cited_by", "cited_by"), ("cites", "cites")):
            for seed in seed_papers:
                expansion_tasks.append(asyncio.create_task(_expand_labeled(seed, direction, label)))

        for task in tqdm.tqdm(asyncio.as_completed(expansion_tasks), total=len(expansion_tasks)):
            direction, label, candidates, exc = await task
            if exc:
                print(f"literaturePoolExpand {direction} {exc}")
                continue
            expansion_results[label].extend(candidates)

        # for direction, label in (("cited_by", "cited_by"), ("cites", "cites")):
        #     for seed in tqdm.tqdm(seed_papers):
        #         direction, label, candidates, exc = await _expand_labeled(seed, direction, label)
        #         if exc:
        #             print(f"literaturePoolExpand {direction} {exc}")
        #             continue
        #         expansion_results[label].extend(candidates)   

        for label in ("cited_by", "cites"):
            for candidate in expansion_results[label]:
                self._add_to_pool(pool, pool_index, candidate, label)
        print(f"BuildLiteraturePool {len(pool)} neighbor")

        return {"literature_pool": pool}
