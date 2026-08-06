from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import jsonschema
import networkx as nx

from ..prompts import CITATION_WARRANT
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json


WARRANT_LABELS = {
    "concept_symbol_or_landmark_warrant",
    "attribution_warrant",
    "taxonomy_or_scope_warrant",
    "claim_support_or_counterevidence_warrant",
    "benchmark_dataset_evaluation_warrant",
    "recency_update_warrant",
    "weak_related_work_suggestion",
    "no_obligation",
}
CITATION_WARRANT_SCHEMA = {
    "type": "object",
    "properties": {
        "warrant_label": {"type": "string", "enum": sorted(WARRANT_LABELS)},
        "citation_obligation": {"type": "boolean"},
        "evidence": {"type": "string", "minLength": 1},
        "reasoning": {"type": "string", "minLength": 1},
    },
    "required": ["warrant_label", "citation_obligation", "evidence", "reasoning"],
    "additionalProperties": False,
}


class CitationWarrantClient(AsyncChat):
    PROMPT = CITATION_WARRANT

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, CITATION_WARRANT_SCHEMA)
        weak_labels = {"weak_related_work_suggestion", "no_obligation"}
        assert result["citation_obligation"] == (result["warrant_label"] not in weak_labels)
        verified, score = self.check.verify([result["evidence"]], context["textual_evidence"], min_char_len=8)
        assert verified, "citation-warrant evidence is not copied verbatim from supplied text"
        result["evidence_score"] = score
        return result

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(**inputs)
        textual_evidence = "\n".join([
            inputs["section_text"], inputs["cited_papers"], inputs["candidate_title"], inputs["candidate_abstract"],
        ])
        return prompt, {"textual_evidence": textual_evidence}


class TopicSpecificPapers:
    """Estimate subfield citation coverage from cited anchors and LLM-confirmed citation obligations."""

    def __init__(self, config: ToolConfig):
        self.search_limit = config.topic_papers_search_limit
        self.citation_warrant = CitationWarrantClient(config)

    def _pool(self, literature_pool: Any) -> dict[str, dict[str, Any]]:
        return literature_pool.get("literature_pool", literature_pool) if isinstance(literature_pool, dict) else {}

    def _graph(self, citation_graph: dict[str, Any] | None) -> nx.DiGraph:
        graph = nx.DiGraph()
        if citation_graph:
            graph.add_nodes_from(citation_graph.get("nodes", []) or [])
            graph.add_edges_from((str(edge["source"]), str(edge["target"])) for edge in citation_graph.get("edges", []) or [])
        return graph

    def _citation_key_map(self, pool: dict[str, dict[str, Any]]) -> dict[str, str]:
        mapping = {}
        for node, item in pool.items():
            for key in item.get("citation_keys", []) or []:
                mapping[str(key)] = str(node)
        return mapping

    def _section_text(self, section: dict[str, Any]) -> str:
        parts = []
        for paragraph in section.get("paragraphs", []) or []:
            if isinstance(paragraph, dict):
                parts.extend(str(sentence.get("text", "")) for sentence in paragraph.get("sentences", []) or [] if isinstance(sentence, dict))
            elif isinstance(paragraph, list):
                parts.extend(str(sentence.get("text", "")) for sentence in paragraph if isinstance(sentence, dict))
        return " ".join(part for part in parts if part).strip()[:6000]

    def _content_sections(self, paper: dict[str, Any]) -> list[dict[str, Any]]:
        sections = []
        def walk(section: dict[str, Any]):
            if section.get("functional_type") == "CONTENT" and isinstance(section.get("parsed_contents"), dict):
                sections.append(section)
            for child in section.get("sections", []) or []:
                if isinstance(child, dict):
                    walk(child)
        for section in paper.get("sections", []) or []:
            if isinstance(section, dict):
                walk(section)
        return sections

    def _topic_sources(self, section: dict[str, Any], key_map: dict[str, str]) -> dict[str, set[str]]:
        parsed = section["parsed_contents"]
        result = {str(topic): set() for topic in parsed.get("topics", []) or []}
        for item in parsed.get("objects", []) or []:
            nodes = {key_map[key] for key in item.get("citation_keys", []) or [] if key in key_map}
            for topic in item.get("topics", []) or []:
                result.setdefault(str(topic), set()).update(nodes)
        return {topic: nodes for topic, nodes in result.items() if nodes}

    def _summary(self, pool: dict[str, Any], node: str) -> dict[str, str]:
        paper = pool[node].get("paper", pool[node])
        return {"title": str(paper.get("title", "")), "abstract": str(paper.get("abstract", ""))[:1200]}

    def _candidate_nodes(self, graph: nx.DiGraph, sources: set[str], pool: dict[str, Any]) -> list[str]:
        candidates = set()
        for source in sources:
            if source in graph:
                candidates.update(graph.successors(source))
                candidates.update(graph.predecessors(source))
        candidates = {node for node in candidates if node in pool and pool[node].get("label") != "cited_papers"}
        def rank(node: str):
            direct = sum(graph.has_edge(source, node) or graph.has_edge(node, source) for source in sources)
            item = pool[node]
            paper = item.get("paper", item)
            return direct, int(paper.get("cited_by_count") or item.get("local_cited_by_count") or 0), node
        return sorted(candidates, key=rank, reverse=True)[:self.search_limit]

    async def _warrant(
        self,
        query: str,
        section: dict[str, Any],
        topic: str,
        sources: set[str],
        candidate: str,
        pool: dict[str, Any],
        graph: nx.DiGraph,
    ) -> dict[str, Any] | None:
        candidate_paper = pool[candidate].get("paper", pool[candidate])
        cited = json.dumps([self._summary(pool, node) for node in sorted(sources) if node in pool], ensure_ascii=False)
        graph_evidence = json.dumps({
            "direct_connections_to_cited_anchors": sum(graph.has_edge(source, candidate) or graph.has_edge(candidate, source) for source in sources),
        })
        warrant = await self.citation_warrant.call(inputs={
            "query": query,
            "section_title": str(section.get("title", "")),
            "topics": json.dumps([topic], ensure_ascii=False),
            "section_text": self._section_text(section),
            "cited_papers": cited,
            "candidate_title": str(candidate_paper.get("title", "")),
            "candidate_abstract": str(candidate_paper.get("abstract", ""))[:3000],
            "graph_evidence": graph_evidence,
        })
        if not warrant["citation_obligation"]:
            return None
        return {"node": candidate, "paper": self._summary(pool, candidate), "citation_warrant": warrant}

    async def __call__(
        self,
        query: str,
        paper: dict[str, Any],
        literature_pool: Any,
        citation_graph: dict[str, Any] | None = None,
        paper_content_map: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pool = self._pool(literature_pool)
        graph = self._graph(citation_graph)
        key_map = self._citation_key_map(pool)
        subfields = []
        tasks = []
        descriptors = []
        for section in self._content_sections(paper):
            for topic, sources in self._topic_sources(section, key_map).items():
                candidates = self._candidate_nodes(graph, sources, pool)
                descriptors.append((section, topic, sources))
                tasks.append(asyncio.gather(*(self._warrant(query, section, topic, sources, node, pool, graph) for node in candidates), return_exceptions=True))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        cited_total = 0
        eligible_total = 0
        for descriptor, result in zip(descriptors, results):
            section, topic, sources = descriptor
            obligations = [item for item in result if isinstance(item, dict)] if isinstance(result, list) else []
            cited_count = len(sources)
            eligible_count = cited_count + len(obligations)
            cited_total += cited_count
            eligible_total += eligible_count
            subfields.append({
                "section": str(section.get("title", "")),
                "topic": topic,
                "cited_anchor_count": cited_count,
                "citation_obligations": obligations,
                "reference_subfield_coverage": cited_count / eligible_count if eligible_count else 1.0,
            })
        return {"source_evals": {
            "reference_subfields": subfields,
            "reference_subfield_coverage": cited_total / eligible_total if eligible_total else 1.0,
            "reference_subfield_anchor_count": cited_total,
            "reference_subfield_obligation_count": eligible_total - cited_total,
        }}
