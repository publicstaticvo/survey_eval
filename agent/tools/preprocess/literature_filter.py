from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import jsonschema

from ..prompts import LITERATURE_POOL_RELEVANCE, LITERATURE_POOL_RELEVANCE_SCHEMA
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json


class LiteraturePoolRelevanceClient(AsyncChat):
    """Validate LLM decisions about candidate-paper relevance."""

    PROMPT = LITERATURE_POOL_RELEVANCE

    def __init__(self, config: ToolConfig):
        """Initialize the relevance client and verbatim-evidence checker."""
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        """Validate candidate coverage and verbatim evidence in an LLM response."""
        result = extract_json(response)
        jsonschema.validate(result, LITERATURE_POOL_RELEVANCE_SCHEMA)
        returned = {item["paper_id"]: item for item in result["papers"]}
        assert set(returned) == set(context["paper_ids"]), (
            "LiteraturePoolRelevance: response must decide every candidate exactly once"
        )
        for item in returned.values():
            if item["relevant"]:
                verified, _ = self.check.verify(
                    [item["verbatim_evidence"]],
                    context["candidate_text"],
                    min_char_len=8,
                )
                assert verified, (
                    "LiteraturePoolRelevance: verbatim evidence is not copied "
                    "from candidate title/abstract"
                )
            else:
                assert item["verbatim_evidence"] == "", (
                    "LiteraturePoolRelevance: irrelevant candidates must not provide evidence"
                )
        return result

    def _organize_inputs(self, inputs):
        """Build the prompt and validation context for a relevance batch."""
        prompt = self.PROMPT.format(**inputs)
        return prompt, {
            "paper_ids": inputs["paper_ids"],
            "candidate_text": inputs["candidate_text"],
        }


class LiteraturePoolFilter:
    """Filter a previously constructed full literature graph with an LLM."""

    def __init__(self, config: ToolConfig):
        """Initialize the filter without constructing or querying a literature graph."""
        self.config = config
        self.relevance_llm = LiteraturePoolRelevanceClient(config)

    def _query_text(self, query: list[str] | str, paper: Paper) -> str:
        """Combine the supplied query terms into the topic description used for filtering."""
        values = query if isinstance(query, list) else [query]
        text = " ".join(
            str(item or "").strip()
            for item in values
            if str(item or "").strip()
        )
        return re.sub(r"\s+", " ", text or paper.title or "").strip()

    def _graph_dict(
        self,
        pool: dict[str, dict[str, Any]],
        edges: set[tuple[str, str]],
    ) -> dict[str, Any]:
        """Project the full citation graph onto the papers retained by the filter."""
        nodes = sorted(pool)
        kept_edges = {
            (source, target)
            for source, target in edges
            if source in pool and target in pool and source != target
        }
        out_counts = {node: 0 for node in nodes}
        for source, _target in kept_edges:
            out_counts[source] += 1
        for key, count in out_counts.items():
            pool[key]["local_cited_by_count"] = count
            pool[key].setdefault("paper", {})["local_cited_by_count"] = count
        return {
            "nodes": nodes,
            "edges": [
                {"source": source, "target": target}
                for source, target in sorted(kept_edges)
            ],
        }

    def _candidate_record(self, paper_id: str, item: dict[str, Any]) -> dict[str, str]:
        """Reduce a candidate paper to the fields sent to the relevance model."""
        paper = item.get("paper", item)
        return {
            "paper_id": paper_id,
            "title": re.sub(r"\s+", " ", str(paper.get("title", ""))).strip(),
            "abstract": re.sub(
                r"\s+", " ", str(paper.get("abstract", "") or "")
            ).strip()[:1200],
        }

    def _candidate_batch_text(self, records: list[dict[str, str]]) -> str:
        """Serialize a candidate batch for the relevance prompt."""
        return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)

    async def _filter_candidates(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter candidates concurrently while preserving their input order."""
        batch_size = max(1, int(self.config.literature_pool_relevance_batch_size))
        semaphore = asyncio.Semaphore(
            max(1, int(self.config.literature_pool_relevance_concurrency))
        )

        async def decide(batch: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
            """Ask the relevance model to classify one candidate batch."""
            keys = [item["candidate_id"] for item in batch]
            records = [
                self._candidate_record(item["candidate_id"], item) for item in batch
            ]
            candidate_text = self._candidate_batch_text(records)
            async with semaphore:
                result = await self.relevance_llm.call(
                    inputs={
                        "query": query_text,
                        "papers": candidate_text,
                        "paper_ids": keys,
                        "candidate_text": candidate_text,
                    }
                )
            return {item["paper_id"]: item for item in result["papers"]}

        batches = [
            candidates[start : start + batch_size]
            for start in range(0, len(candidates), batch_size)
        ]
        results = await asyncio.gather(
            *(decide(batch) for batch in batches),
            return_exceptions=True,
        )
        decisions: dict[str, dict[str, Any]] = {}
        for result in results:
            if isinstance(result, Exception):
                logging.error("Literature-pool relevance batch failed: %s", result)
                continue
            decisions.update(result)
        return [
            {**candidate, "relevance_evidence": decisions[candidate["candidate_id"]]["verbatim_evidence"]}
            for candidate in candidates
            if candidate["candidate_id"] in decisions
            and decisions[candidate["candidate_id"]]["relevant"]
        ]

    async def __call__(
        self,
        query: list[str] | str,
        paper: Paper,
        literature_graph: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep cited papers and batch-filter all other papers in the full graph."""
        full_pool = literature_graph.get("literature_pool", {}) or {}
        full_graph = literature_graph.get("citation_graph", {}) or {}
        query_text = self._query_text(query, paper)

        cited_keys = {
            key
            for key, record in full_pool.items()
            if record.get("label") in {"cited_paper", "cited_papers"} or record.get("citation_keys")
        }
        candidates = [
            {
                "candidate_id": key,
                "paper": record.get("paper", {}),
                "label": record.get("label", ""),
                "retrieval": record.get("retrieval", {}),
                "retrieval_sources": record.get("retrieval_sources", []),
            }
            for key, record in full_pool.items()
            if key not in cited_keys
        ]
        relevant = await self._filter_candidates(query_text, candidates) if candidates else []
        relevant_by_key = {item["candidate_id"]: item for item in relevant}
        keep_keys = cited_keys | set(relevant_by_key)

        filtered_pool: dict[str, dict[str, Any]] = {}
        for key in keep_keys:
            if key not in full_pool:
                continue
            record = dict(full_pool[key])
            decision = relevant_by_key.get(key)
            if decision:
                record["relevance_evidence"] = decision["relevance_evidence"]
            filtered_pool[key] = record

        filtered_edges = {
            (edge["source"], edge["target"])
            for edge in full_graph.get("edges", []) or []
            if edge.get("source") in filtered_pool
            and edge.get("target") in filtered_pool
        }
        filtered_graph = self._graph_dict(filtered_pool, filtered_edges)
        return {
            "literature_pool": filtered_pool,
            "citation_graph": filtered_graph,
            "source_graph_stats": {
                "full_nodes": len(full_pool),
                "full_edges": len(full_graph.get("edges", []) or []),
                "filtered_nodes": len(filtered_pool),
                "filtered_edges": len(filtered_graph["edges"]),
            },
        }
