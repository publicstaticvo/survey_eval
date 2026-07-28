from __future__ import annotations

import re
from typing import Any

from ..utility.paper_elements import Paper
from ..utility.tool_config import ToolConfig
from .uncited_entities import UncitedEntities


class MissingPaperCheck:
    def __init__(self, config: ToolConfig):
        self.config = config
        self.uncited_entities = UncitedEntities(config)

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

    def _metadata_sources(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        metadata = info.get("metadata") or {}
        if isinstance(metadata, dict) and ("openalex" in metadata or "semantic scholar" in metadata):
            return [paper for paper in metadata.values() if isinstance(paper, dict)]
        return [metadata] if isinstance(metadata, dict) and metadata else []

    def _cited_papers(self, paper_content_map: dict[str, Any]) -> list[dict[str, Any]]:
        papers = []
        for info in paper_content_map.values():
            if isinstance(info, dict):
                papers.extend(self._metadata_sources(info))
        return [paper for paper in papers if paper.get("title")]

    def _is_cited(self, candidate: dict[str, Any], cited_papers: list[dict[str, Any]]) -> bool:
        candidate_ids = self._paper_ids(candidate)
        candidate_title = self._paper_title(candidate)
        for cited in cited_papers:
            if candidate_ids and candidate_ids & self._paper_ids(cited):
                return True
            if candidate_title and candidate_title == self._paper_title(cited):
                return True
        return False

    def _reference_survey_papers(self, reference_surveys: Any) -> list[dict[str, Any]]:
        papers = []
        reference_surveys = reference_surveys.get("reference_surveys", reference_surveys)
        if isinstance(reference_surveys, dict):
            values = reference_surveys.values()
        else:
            values = reference_surveys or []
        for item in values:
            if not isinstance(item, dict):
                continue
            for key in ("openalex", "semantic_scholar", "paper", "metadata"):
                paper = item.get(key)
                if isinstance(paper, dict) and paper.get("title"):
                    papers.append(paper)
                    break
        return papers

    def _add_missing(
        self,
        results: list[dict[str, Any]],
        seen: set[tuple[str, str, str]],
        paper: dict[str, Any],
        reason: str,
        **metadata,
    ):
        key = self._paper_key(paper)
        if not key:
            return
        dedupe_key = (reason, metadata.get("topic", ""), key)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        results.append({"paper": paper, "reason": reason, **metadata})

    async def __call__(
        self,
        queries: list[str],
        paper: Paper,
        paper_content_map: dict[str, Any],
        reference_surveys: Any = None,
        literature_pool: dict[str, Any] | list[dict[str, Any]] | None = None,
        citation_graph: dict[str, Any] | None = None,
        entity_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cited_papers = self._cited_papers(paper_content_map)
        missing, seen = [], set()

        for reference_paper in self._reference_survey_papers(reference_surveys):
            if not self._is_cited(reference_paper, cited_papers):
                self._add_missing(missing, seen, reference_paper, "reference_surveys")
        print(f"{len(missing)} missing papers")

        # Missing-topic detection now belongs to TopicCoverage (09), not source-level paper checks.
        entity_data = entity_data or await self.uncited_entities(paper, cited_papers, paper_content_map)
        for entity_item in entity_data.get("uncited_entities", []) or []:
            for candidate in entity_item.get("matched_papers", []) or []:
                self._add_missing(
                    missing,
                    seen,
                    candidate,
                    "uncited_entities",
                    entity_name=entity_item.get("entity", ""),
                )
        print(f"{len(missing)} missing papers")

        return {
            "source_evals": {
                "missing_papers": missing,
                "uncited_entities": entity_data.get("uncited_entities", []),
            }
        }
