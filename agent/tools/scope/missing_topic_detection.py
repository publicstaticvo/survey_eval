from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import Any

import igraph as ig
import jsonschema
import leidenalg
import networkx as nx

from ..prompts import MISSING_TOPIC_DECISION, MISSING_TOPIC_DECISION_SCHEMA
from ..utility.content_walk import iter_sections
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _paper_source_text(paper: dict[str, Any]) -> str:
    return _normalized_text(f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')}")


def _format_papers(papers: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{index}] Title: {paper.get('title', '')}\nAbstract: {_normalized_text(paper.get('abstract', '')) or 'None'}"
        for index, paper in enumerate(papers, 1)
    )


class MissingTopicDecisionClient(AsyncChat):
    PROMPT = MISSING_TOPIC_DECISION

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, MISSING_TOPIC_DECISION_SCHEMA)
        for topic in result["covered_topics"]:
            assert topic in context["existing_topics"]
        decision = result["decision"]
        if decision == "COVERED":
            assert result["covered_topics"] and not result["community_name"] and not result["mixed_topics"] and not result["reason"]
        elif decision == "NOVEL":
            assert not result["covered_topics"] and result["community_name"] and not result["mixed_topics"] and result["reason"]
        elif decision == "UNRELATED":
            assert not result["covered_topics"] and not result["community_name"] and not result["mixed_topics"] and result["reason"]
        else:
            assert not result["covered_topics"] and not result["community_name"] and len(result["mixed_topics"]) >= 2 and result["reason"]
        for item in result["evidence"]:
            index = item["paper_index"]
            assert 1 <= index <= len(context["paper_texts"])
            verified, _ = self.check.verify([item["quote"]], context["paper_texts"][index - 1], min_char_len=8)
            assert verified, "Missing-topic evidence must be copied verbatim from a representative paper"
        return result

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(
            query=inputs["query"],
            existing_topics="\n".join(f"- {topic}" for topic in inputs["existing_topics"]),
            papers=_format_papers(inputs["papers"]),
        )
        return prompt, {
            "existing_topics": set(inputs["existing_topics"]),
            "paper_texts": [_paper_source_text(paper) for paper in inputs["papers"]],
        }


class MissingTopicDetector:
    """Rank multi-resolution Leiden communities by PR mass not reached from cited papers."""

    def __init__(self, config: ToolConfig):
        self.min_community_size = config.missing_topic_min_community_size
        self.min_community_size_ratio = config.missing_topic_min_community_size_ratio
        self.resolutions = tuple(float(value) for value in config.missing_topic_resolutions)
        self.top_k = max(1, int(config.missing_topic_top_k))
        self.llm_concurrency = max(1, int(config.missing_topic_llm_concurrency))
        self.representative_limit = max(1, int(config.missing_topic_representative_papers))
        self.topic_decision = MissingTopicDecisionClient(config)

    def _pool(self, literature_pool: Any) -> dict[str, dict[str, Any]]:
        if isinstance(literature_pool, dict):
            return literature_pool.get("literature_pool", literature_pool) or {}
        return {}

    def _directed_graph(self, literature_pool: Any, citation_graph: dict[str, Any] | None) -> nx.DiGraph:
        graph_data = citation_graph or (
            literature_pool.get("citation_graph", {}) if isinstance(literature_pool, dict) else {}
        )
        graph = nx.DiGraph()
        graph.add_nodes_from(graph_data.get("nodes", []) or self._pool(literature_pool).keys())
        graph.add_edges_from(
            (str(edge["source"]), str(edge["target"]))
            for edge in graph_data.get("edges", []) or []
            if edge.get("source") and edge.get("target") and edge["source"] != edge["target"]
        )
        return graph

    def _communities(self, graph: nx.Graph, resolution: float) -> list[frozenset[str]]:
        if not graph:
            return []
        nodes = list(graph)
        index = {node: idx for idx, node in enumerate(nodes)}
        edges = [(index[left], index[right]) for left, right in graph.edges()]
        if not edges:
            return [frozenset([node]) for node in nodes]
        leiden_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
        partition = leidenalg.find_partition(
            leiden_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=0,
        )
        return [frozenset(nodes[idx] for idx in community) for community in partition]

    def _existing_topics(self, paper: Paper) -> list[str]:
        topics = []
        for section in iter_sections(paper):
            parsed = section.parsed_contents if isinstance(section.parsed_contents, dict) else {}
            topics.extend(str(topic) for topic in parsed.get("topics", []) or [] if str(topic).strip())
        return list(dict.fromkeys(topics))

    def _paper(self, pool: dict[str, Any], node: str) -> dict[str, Any]:
        return pool[node].get("paper", pool[node])

    def _rank_communities(
        self,
        pool: dict[str, dict[str, Any]],
        graph: nx.DiGraph,
    ) -> list[dict[str, Any]]:
        if not graph:
            return []
        cited_nodes = {node for node, item in pool.items() if item.get("label") == "cited_papers"} & set(graph)
        global_pr = nx.pagerank(graph, max_iter=500)
        if cited_nodes:
            personalization = {
                node: (1.0 / len(cited_nodes) if node in cited_nodes else 0.0)
                for node in graph
            }
            cited_ppr = nx.pagerank(
                graph, personalization=personalization, dangling=personalization, max_iter=500
            )
        else:
            cited_ppr = {node: 0.0 for node in graph}

        uncited_nodes = set(graph) - cited_nodes
        undirected = graph.to_undirected().subgraph(uncited_nodes).copy()
        minimum_size = max(
            self.min_community_size,
            math.ceil(len(pool) * self.min_community_size_ratio),
        )
        unique: dict[frozenset[str], set[float]] = {}
        for resolution in self.resolutions:
            for community in self._communities(undirected, resolution):
                if len(community) >= minimum_size:
                    unique.setdefault(community, set()).add(resolution)

        ranked = []
        for nodes, resolutions in unique.items():
            pr_mass = sum(global_pr.get(node, 0.0) for node in nodes)
            ppr_mass = sum(cited_ppr.get(node, 0.0) for node in nodes)
            missing_score = pr_mass - ppr_mass
            # Alternative ablation: sum PR and PPR only over the community's top-K global-PR nodes.
            # top_nodes = sorted(nodes, key=lambda node: global_pr.get(node, 0.0), reverse=True)[:self.representative_limit]
            # missing_score = sum(global_pr.get(node, 0.0) - cited_ppr.get(node, 0.0) for node in top_nodes)
            representatives = sorted(
                nodes, key=lambda node: (global_pr.get(node, 0.0), node), reverse=True
            )[:self.representative_limit]
            papers = [
                {
                    "node": node,
                    "title": self._paper(pool, node).get("title", ""),
                    "abstract": self._paper(pool, node).get("abstract", ""),
                }
                for node in representatives
                if self._paper(pool, node).get("abstract")
            ]
            if papers:
                ranked.append({
                    "nodes": nodes,
                    "resolutions": sorted(resolutions),
                    "community_size": len(nodes),
                    "pr_mass": pr_mass,
                    "cited_ppr_mass": ppr_mass,
                    "missing_score": missing_score,
                    "papers": papers,
                })
        return sorted(ranked, key=lambda item: (item["missing_score"], item["pr_mass"]), reverse=True)

    async def _judge(self, query: str, topics: list[str], candidate: dict[str, Any]):
        decision = await self.topic_decision.call(inputs={
            "query": query,
            "existing_topics": topics,
            "papers": candidate["papers"],
        })
        return candidate, decision

    def _duplicate_novel(self, accepted: list[dict[str, Any]], candidate: dict[str, Any]) -> bool:
        name = candidate["community_name"].casefold().strip()
        for existing in accepted:
            if name and name == existing["community_name"].casefold().strip():
                return True
            overlap = len(set(candidate["nodes"]) & set(existing["nodes"]))
            if overlap / min(len(candidate["nodes"]), len(existing["nodes"])) >= 0.8:
                return True
        return False

    async def detect(
        self,
        queries: list[str],
        paper: Paper,
        literature_pool: Any,
        citation_graph: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        pool = self._pool(literature_pool)
        graph = self._directed_graph(literature_pool, citation_graph)
        if not pool or not graph:
            return [], [], [], {"missing_topic_risk": 0.0, "community_candidate_count": 0}

        candidates = self._rank_communities(pool, graph)
        topics = self._existing_topics(paper)
        query = " ".join(queries)
        detected: list[dict[str, Any]] = []
        mixed: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []

        for start in range(0, len(candidates), self.llm_concurrency):
            tasks = [
                asyncio.create_task(self._judge(query, topics, candidate))
                for candidate in candidates[start:start + self.llm_concurrency]
            ]
            for completed in asyncio.as_completed(tasks):
                try:
                    candidate, decision = await completed
                except Exception as exc:
                    logging.error("Missing-topic community decision failed: %s", exc)
                    continue
                report = {
                    **{key: value for key, value in candidate.items() if key != "nodes"},
                    "nodes": sorted(candidate["nodes"]),
                    **decision,
                }
                if decision["decision"] == "NOVEL" and not self._duplicate_novel(detected, report):
                    detected.append(report)
                    detected.sort(key=lambda item: item["missing_score"], reverse=True)
                    if len(detected) >= self.top_k:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        break
                elif decision["decision"] == "MIXED":
                    mixed.append(report)
                else:
                    discarded.append(report)
            if len(detected) >= self.top_k:
                break

        detected = detected[:self.top_k]
        metrics = {
            "missing_topic_risk": sum(max(0.0, item["missing_score"]) for item in detected),
            "missing_topic_candidate_count": len(detected),
            "community_candidate_count": len(candidates),
            "cited_anchor_count": sum(item.get("label") == "cited_papers" for item in pool.values()),
        }
        logging.info(
            "Missing-topic retained %d NOVEL communities from %d ranked candidates; %d MIXED communities were not recursed",
            len(detected), len(candidates), len(mixed),
        )
        return detected, mixed, discarded, metrics
