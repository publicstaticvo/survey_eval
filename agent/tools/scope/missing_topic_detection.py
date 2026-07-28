from __future__ import annotations

import asyncio
import jsonschema
import logging
import math
import re
from typing import Any

import igraph as ig
import leidenalg
import networkx as nx

from ..prompts import (
    MISSING_TOPIC_DECISION,
    MISSING_TOPIC_DECISION_SCHEMA,
    MISSING_TOPIC_NAME,
    MISSING_TOPIC_NAME_SCHEMA,
)
from ..utility.content_walk import iter_sections
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json


MAX_MISSING_TOPIC_EVIDENCE_PAPERS = 10


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _validate_paper_evidence(result: dict[str, Any], context: dict[str, Any], check: EvidenceCheck) -> None:
    paper_count = context["paper_count"]
    for item in result["evidence"]:
        index = item["paper_index"]
        assert 1 <= index <= paper_count, f"Invalid evidence index: {index} / {paper_count}"
        verified, _ = check.verify([item["quote"]], context["paper_texts"][index - 1])
        assert verified, f"Evidence not supported by paper {index}: {item['quote']}"

def _paper_source_text(paper: dict[str, Any]) -> str:
    """证据来源"""
    return _normalized_text(f"{paper.get('title', '')}\n{paper.get('abstract', '')}")


def _format_papers(papers: list[dict[str, Any]]) -> str:
    """输入"""
    return "\n\n".join(
        f"[{index}] Title: {paper.get('title', '')}\nAbstract: {_normalized_text(paper.get('abstract', '')) or 'None'}"
        for index, paper in enumerate(papers, 1)
    )


class MissingTopicDescriptorClient(AsyncChat):
    PROMPT = MISSING_TOPIC_NAME

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)
    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, MISSING_TOPIC_NAME_SCHEMA)
        topic_name = result["topic_name"].strip()
        query_terms = set(re.findall(r"[a-z0-9]+", context["query"].casefold()))
        name_terms = set(re.findall(r"[a-z0-9]+", topic_name.casefold()))
        assert name_terms != query_terms, f"name terms {name_terms} is query terms {query_terms}"
        _validate_paper_evidence(result, context, self.check)
        return result

    def _organize_inputs(self, inputs):
        papers = _format_papers(inputs["papers"])
        prompt = self.PROMPT.format(query=inputs["query"], papers=papers)
        return prompt, {
            "query": inputs["query"],
            "paper_count": len(inputs["papers"]),
            "paper_texts": [_paper_source_text(paper) for paper in inputs["papers"]],
        }


class MissingTopicDecisionClient(AsyncChat):
    PROMPT = MISSING_TOPIC_DECISION

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)
    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, MISSING_TOPIC_DECISION_SCHEMA)
        for topic in result["covered_topics"]:
            assert topic in context["existing_topics"], f"Topic {topic} not in existing topics {context['existing_topics']}"
        _validate_paper_evidence(result, context, self.check)
        return result

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(
            query=inputs["query"],
            existing_topics="\n".join(f"- {topic}" for topic in inputs["existing_topics"]),
            papers=_format_papers(inputs["papers"]),
        )
        return prompt, {
            "existing_topics": set(inputs["existing_topics"]),
            "paper_count": len(inputs["papers"]),
            "paper_texts": [_paper_source_text(paper) for paper in inputs["papers"]],
        }


class MissingTopicDetector:
    """Detect candidate missing research directions with Leiden and LLM decisions."""

    def __init__(self, config: ToolConfig):
        self.min_community_size = config.missing_topic_min_community_size
        self.min_community_size_ratio = config.missing_topic_min_community_size_ratio
        self.max_recursive_depth = config.missing_topic_recursive_depth
        self.leiden_resolution = config.missing_topic_leiden_resolution
        self.topic_descriptor = MissingTopicDescriptorClient(config)
        self.topic_decision = MissingTopicDecisionClient(config)

    def _pool(self, literature_pool: Any) -> dict[str, dict[str, Any]]:
        if isinstance(literature_pool, dict):
            return literature_pool.get("literature_pool", literature_pool) or {}
        return {}

    def _graph(self, literature_pool: Any, citation_graph: dict[str, Any] | None) -> nx.Graph:
        graph_data = citation_graph or (
            literature_pool.get("citation_graph", {}) if isinstance(literature_pool, dict) else {}
        )
        graph = nx.Graph()
        graph.add_nodes_from(graph_data.get("nodes", []) or self._pool(literature_pool).keys())
        graph.add_edges_from(
            (str(edge["source"]), str(edge["target"]))
            for edge in graph_data.get("edges", []) or []
            if edge.get("source") and edge.get("target") and edge["source"] != edge["target"]
        )
        return graph

    def _paper_relevance_score(self, paper: dict[str, Any]) -> int:
        values = [paper.get("local_cited_by_count"), paper.get("cited_by_count"), paper.get("citationCount")]
        return max((int(value) for value in values if isinstance(value, int)), default=0)

    def _representative_papers(
        self,
        nodes: list[str],
        pool: dict[str, dict[str, Any]],
        graph: nx.Graph,
    ) -> list[dict[str, Any]]:
        subgraph = graph.subgraph(nodes).copy()
        try:
            pagerank = nx.pagerank(subgraph) if subgraph.number_of_edges() else {node: 0.0 for node in nodes}
        except Exception:
            pagerank = {node: 0.0 for node in nodes}
        ranked_nodes = sorted(
            nodes,
            key=lambda node: (
                pagerank.get(node, 0.0),
                self._paper_relevance_score(pool.get(node, {}).get("paper", pool.get(node, {}))),
                node,
            ),
            reverse=True,
        )[:MAX_MISSING_TOPIC_EVIDENCE_PAPERS]
        return [
            {
                "node": node,
                "title": pool[node].get("paper", pool[node]).get("title", ""),
                "abstract": pool[node].get("paper", pool[node]).get("abstract", ""),
            }
            for node in ranked_nodes
            if node in pool
        ]

    def _communities(self, graph: nx.Graph) -> list[list[str]]:
        if graph.number_of_nodes() == 0:
            return []
        nodes = list(graph.nodes())
        node_index = {node: index for index, node in enumerate(nodes)}
        edges = [(node_index[source], node_index[target]) for source, target in graph.edges()]
        if not edges:
            return [[node] for node in nodes]
        leiden_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
        partition = leidenalg.find_partition(
            leiden_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.leiden_resolution,
            seed=0,
        )
        return [[nodes[index] for index in community] for community in partition]

    def _existing_topics(self, paper: Paper) -> list[str]:
        topics = []
        for section in iter_sections(paper):
            parsed = section.parsed_contents if isinstance(section.parsed_contents, dict) else {}
            topics.extend(str(topic) for topic in parsed.get("topics", []) or [] if str(topic).strip())
            for obj in parsed.get("objects", []) or []:
                if isinstance(obj, dict):
                    topics.extend(str(topic) for topic in obj.get("topics", []) or [] if str(topic).strip())
        return list(dict.fromkeys(topics))

    async def detect(
        self,
        queries: list[str],
        paper: Paper,
        literature_pool: Any,
        citation_graph: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        pool = self._pool(literature_pool)
        if not pool: return [], [], []
        full_graph = self._graph(literature_pool, citation_graph)
        candidate_nodes = {key for key, item in pool.items() if item.get("label") != "cited_papers"}
        candidate_graph = full_graph.subgraph(candidate_nodes).copy()
        minimum_size = max(
            self.min_community_size,
            math.ceil(candidate_graph.number_of_nodes() * self.min_community_size_ratio),
        )
        existing_topics = self._existing_topics(paper)
        query_text = " ".join(queries)
        detected: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []

        def discard_error(path: str, node_list: list[str], papers: list[dict[str, Any]], stage: str, exc: Exception) -> None:
            logging.error("Missing-topic %s failed for %s: %s", stage, path, exc)
            discarded.append({
                "community": path,
                "community_size": len(node_list),
                "status": "ERROR",
                "error_stage": stage,
                "error": str(exc),
                "papers": papers,
            })

        async def visit(nodes: list[str], depth: int, path: str) -> None:
            node_list = [node for node in nodes if node in pool]
            if len(node_list) < minimum_size: return
            papers = self._representative_papers(node_list, pool, candidate_graph)
            if not papers: return
            try:
                decision = await self.topic_decision.call(inputs={
                    "query": query_text,
                    "existing_topics": existing_topics,
                    "papers": papers,
                })
            except Exception as exc:
                logging.error(f"Decision {exc}")
                discard_error(path, node_list, papers, "decision", exc)
                return
            decision_report = {
                "community": path,
                "community_size": len(node_list),
                "papers": papers,
                "decision": decision["decision"],
                "covered_topics": decision["covered_topics"],
                "evidence": decision["evidence"],
            }
            papers_str = ','.join(f'"{x["title"]}"' for x in papers)
            logging.info(f'Community {path} Papers {papers_str} Decision {decision["decision"]} Topics {decision["covered_topics"]}')
            if decision["decision"] == "COVERED": return
            if decision["decision"] == "NOVEL":
                try:
                    descriptor = await self.topic_descriptor.call(inputs={"query": query_text, "papers": papers})
                except Exception as exc:
                    logging.error(f"Descriptor {exc}")
                    discard_error(path, node_list, papers, "descriptor", exc)
                    return
                content_tags = descriptor["content_tags"]
                logging.info(f'Community {path} Named {descriptor["topic_name"]} Content {content_tags}')
                if 'GENERAL' not in content_tags:
                    content_tag_priority = 0 if content_tags == ["METHOD"] else 1 if "METHOD" in content_tags else 2
                    detected.append({
                        **decision_report,
                        "topic_name": descriptor["topic_name"],
                        "content_tags": content_tags,
                        "content_tag_priority": content_tag_priority,
                        "descriptor_evidence": descriptor["evidence"],
                    })
                    return
            subgraph = candidate_graph.subgraph(node_list).copy()
            children = self._communities(subgraph)
            if depth >= self.max_recursive_depth:
                unresolved.append({**decision_report, "status": "MIXED", "reason": "maximum_recursive_depth"})
            elif len(children) <= 1:
                unresolved.append({**decision_report, "status": "MIXED", "reason": "indivisible_community"})
            else:
                await asyncio.gather(
                    *(visit(child, depth + 1, f"{path}.{index}") for index, child in enumerate(children, 1))
                )

        root_communities = self._communities(candidate_graph)
        await asyncio.gather(*(visit(nodes, 0, str(index)) for index, nodes in enumerate(root_communities, 1)))
        logging.info(
            "Missing-topic produced %d novel, %d unresolved, and %d error communities",
            len(detected), len(unresolved), len(discarded),
        )
        return detected, unresolved, discarded
