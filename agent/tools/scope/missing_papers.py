from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import jsonschema
import networkx as nx

from ..prompts import REFERENCE_ANCHOR_RELEVANCE, REFERENCE_ANCHOR_RELEVANCE_SCHEMA
from ..utility.content_walk import iter_sections, paragraphs_to_text
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json
from .uncited_entities import UncitedEntities


class ReferenceAnchorClient(AsyncChat):
    PROMPT = REFERENCE_ANCHOR_RELEVANCE

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, REFERENCE_ANCHOR_RELEVANCE_SCHEMA)
        returned = {item["paper_id"]: item for item in result["papers"]}
        assert set(returned) == context["paper_ids"], "ReferenceAnchorClient must decide each candidate exactly once"
        for paper_id, item in returned.items():
            if item["relevant"]:
                assert item["reason"].strip()
                verified, _ = self.check.verify(
                    [item["verbatim_evidence"]], context["abstracts"][paper_id], min_char_len=8
                )
                assert verified, "Reference-anchor evidence must be copied verbatim from the abstract"
            else:
                assert not item["verbatim_evidence"].strip()
        return result

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(**inputs)
        return prompt, {"paper_ids": set(inputs["paper_ids"]), "abstracts": inputs["abstracts"]}


class MissingPaperCheck:
    """Generate evidence-linked missing-reference candidates."""

    def __init__(self, config: ToolConfig):
        self.config = config
        self.uncited_entities = UncitedEntities(config)
        self.reference_anchor = ReferenceAnchorClient(config)
        self.top_k = max(1, int(config.missing_paper_top_k))
        self.judge_batch_size = max(1, int(config.missing_paper_judge_batch_size))

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
        value = reference_surveys.get("reference_surveys", reference_surveys) if isinstance(reference_surveys, dict) else reference_surveys
        values = value.values() if isinstance(value, dict) else (value or [])
        papers = []
        for item in values:
            if not isinstance(item, dict):
                continue
            paper = next(
                (
                    item.get(key)
                    for key in ("openalex", "semantic_scholar", "paper", "metadata")
                    if isinstance(item.get(key), dict) and item[key].get("title")
                ),
                None,
            )
            papers.append(paper or item)
        return [paper for paper in papers if paper.get("title")]

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

    def _pool(self, literature_pool: Any) -> dict[str, dict[str, Any]]:
        if isinstance(literature_pool, dict):
            return literature_pool.get("literature_pool", literature_pool) or {}
        return {}

    def _graph(self, citation_graph: dict[str, Any] | None, pool: dict[str, Any]) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from((citation_graph or {}).get("nodes", []) or pool.keys())
        graph.add_edges_from(
            (str(edge["source"]), str(edge["target"]))
            for edge in (citation_graph or {}).get("edges", []) or []
        )
        return graph

    def _citation_key_map(self, pool: dict[str, dict[str, Any]]) -> dict[str, str]:
        return {
            str(citation_key): node
            for node, item in pool.items()
            for citation_key in item.get("citation_keys", []) or []
        }

    def _topic_records(self, paper: Paper, key_map: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        for section in iter_sections(paper):
            parsed = section.parsed_contents if isinstance(section.parsed_contents, dict) else {}
            topic_sources = {str(topic): set() for topic in parsed.get("topics", []) or []}
            for obj in parsed.get("objects", []) or []:
                sources = {key_map[str(key)] for key in obj.get("citation_keys", []) or [] if str(key) in key_map}
                for topic in obj.get("topics", []) or []:
                    topic_sources.setdefault(str(topic), set()).update(sources)
            context = paragraphs_to_text(section.paragraphs, False)[:6000]
            for topic, sources in topic_sources.items():
                if sources:
                    records.append({
                        "topic": topic,
                        "section": section.name,
                        "section_context": context,
                        "sources": sources,
                    })
        return records

    def _ranked_reference_candidates(
        self,
        paper: Paper,
        pool: dict[str, dict[str, Any]],
        graph: nx.DiGraph,
    ) -> tuple[list[dict[str, Any]], int]:
        key_map = self._citation_key_map(pool)
        cited_nodes = {node for node, item in pool.items() if item.get("label") == "cited_papers"}
        candidates = []
        anchor_nodes = set()
        for record in self._topic_records(paper, key_map):
            sources = record["sources"] & set(graph)
            if not sources:
                continue
            anchor_nodes.update(sources)
            personalization = {node: (1.0 / len(sources) if node in sources else 0.0) for node in graph}
            scores = nx.pagerank(graph, personalization=personalization, dangling=personalization)
            ranked_nodes = sorted(scores, key=lambda node: (scores[node], node), reverse=True)
            cutoff = len(ranked_nodes)
            if len(sources) >= 3:
                source_ranks = [ranked_nodes.index(node) for node in sources if node in scores]
                cutoff = max(source_ranks) + 1 if source_ranks else 0
            for rank, node in enumerate(ranked_nodes[:cutoff], 1):
                if node in cited_nodes or node not in pool:
                    continue
                paper_data = pool[node].get("paper", pool[node])
                if not paper_data.get("abstract"):
                    continue
                candidates.append({
                    **record,
                    "node": node,
                    "paper": paper_data,
                    "ppr": float(scores[node]),
                    "ppr_rank": rank,
                })
        candidates.sort(key=lambda item: (item["ppr"], -item["ppr_rank"]), reverse=True)
        return candidates, len(anchor_nodes)

    async def _judge_candidate(self, query: str, item: dict[str, Any]) -> dict[str, Any] | None:
        paper = item["paper"]
        candidate_text = json.dumps(
            [{"paper_id": item["node"], "title": paper.get("title", ""), "abstract": paper.get("abstract", "")}],
            ensure_ascii=False,
        )
        result = await self.reference_anchor.call(inputs={
            "query": query,
            "topic": item["topic"],
            "section_context": item["section_context"],
            "candidate_papers": candidate_text,
            "paper_ids": [item["node"]],
            "abstracts": {item["node"]: str(paper.get("abstract", ""))},
        })
        decision = result["papers"][0]
        if not decision["relevant"]:
            return None
        return {
            "paper": paper,
            "reason": "reference_anchoring",
            "topic": item["topic"],
            "section": item["section"],
            "risk": item["ppr"],
            "ppr_rank": item["ppr_rank"],
            "relevance_reason": decision["reason"],
            "evidence": decision["verbatim_evidence"],
        }

    async def _reference_anchoring(
        self,
        query: str,
        paper: Paper,
        literature_pool: Any,
        citation_graph: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], int]:
        pool = self._pool(literature_pool)
        graph = self._graph(citation_graph, pool)
        if not pool or not graph:
            return [], 0
        ranked, anchor_count = self._ranked_reference_candidates(paper, pool, graph)
        accepted = []
        seen_nodes = set()
        for start in range(0, len(ranked), self.judge_batch_size):
            batch = []
            scheduled = set()
            for item in ranked[start:start + self.judge_batch_size]:
                if item["node"] not in seen_nodes and item["node"] not in scheduled:
                    batch.append(item)
                    scheduled.add(item["node"])
            tasks = [asyncio.create_task(self._judge_candidate(query, item)) for item in batch]
            for completed in asyncio.as_completed(tasks):
                try:
                    result = await completed
                except Exception:
                    continue
                if result:
                    node = self._paper_key(result["paper"])
                    if node not in seen_nodes:
                        accepted.append(result)
                        seen_nodes.add(node)
                if len(accepted) >= self.top_k:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    break
            accepted.sort(key=lambda item: item["risk"], reverse=True)
            if len(accepted) >= self.top_k:
                break
        return accepted[:self.top_k], anchor_count

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
        query = " ".join(queries)
        cited_papers = self._cited_papers(paper_content_map)
        missing, seen = [], set()

        for reference_paper in self._reference_survey_papers(reference_surveys):
            if not self._is_cited(reference_paper, cited_papers):
                self._add_missing(missing, seen, reference_paper, "reference_surveys")

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

        anchors, anchor_count = await self._reference_anchoring(
            query, paper, literature_pool, citation_graph
        )
        for item in anchors:
            self._add_missing(
                missing,
                seen,
                item["paper"],
                "reference_anchoring",
                topic=item["topic"],
                section=item["section"],
                risk=item["risk"],
                ppr_rank=item["ppr_rank"],
                relevance_reason=item["relevance_reason"],
                evidence=item["evidence"],
            )

        reference_risks = [
            float(item.get("risk", 0.0))
            for item in missing
            if item.get("reason") == "reference_anchoring"
        ]
        return {
            "source_evals": {
                "missing_papers": missing,
                "uncited_entities": entity_data.get("uncited_entities", []),
                "reference_anchor_count": anchor_count,
                "reference_anchoring_risk": sum(reference_risks),
                "reference_anchoring_candidates": anchors,
            }
        }
