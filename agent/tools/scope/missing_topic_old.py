from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np

from ..utility.content_walk import iter_sections, iter_sentences
from ..utility.paper_elements import Paper, Section
from ..utility.sbert_client import SentenceTransformerClient
from ..utility.tool_config import ToolConfig


MAX_CENTRAL_PAPERS = 10


class MissingTopicOld:
    """Deprecated topic-paper diagnostics kept for RA and coherence experiments.

    Production missing-topic detection has moved to ``TopicCoverage``. This class keeps
    the older graph evidence helpers available for debugging: node/community
    ``E(p, T)``, ``E(p, cited_all)``, ``max_share`` calculations and title/abstract
    coherence summaries.
    """

    def __init__(self, config: ToolConfig):
        self.config = config
        self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)

    def _pool(self, literature_pool: Any) -> dict[str, dict[str, Any]]:
        if isinstance(literature_pool, dict):
            return literature_pool.get("literature_pool", literature_pool) or {}
        return {}

    def _graph(self, literature_pool: Any, citation_graph: dict[str, Any] | None = None) -> nx.Graph:
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

    def _citation_key_to_pool_key(self, pool: dict[str, dict[str, Any]]) -> dict[str, str]:
        mapping = {}
        for key, item in pool.items():
            for citation_key in item.get("citation_keys", []) or []:
                mapping[str(citation_key)] = key
        return mapping

    def _section_citation_number_map(self, section: Section) -> dict[str, str]:
        mapping = {}
        for sentence in iter_sentences(section):
            citations = sentence.citations
            if isinstance(citations, dict):
                for number, key in citations.items():
                    mapping[str(number)] = str(key)
            elif isinstance(citations, list):
                for key in citations:
                    mapping[str(key)] = str(key)
        return mapping

    def _resolve_section_citation_key(self, raw_key: Any, number_map: dict[str, str]) -> str:
        return number_map.get(str(raw_key), str(raw_key))

    def _topic_cited_nodes(self, paper: Paper, pool: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
        key_map = self._citation_key_to_pool_key(pool)
        topic_nodes: dict[str, set[str]] = defaultdict(set)
        for section in iter_sections(paper):
            parsed = section.parsed_contents if isinstance(section.parsed_contents, dict) else {}
            number_map = self._section_citation_number_map(section)
            section_topics = [str(topic) for topic in parsed.get("topics", []) or [] if str(topic).strip()]
            for obj in parsed.get("objects", []) or []:
                if not isinstance(obj, dict):
                    continue
                topics = [str(topic) for topic in obj.get("topics", []) or [] if str(topic).strip()] or section_topics
                for raw_key in obj.get("citation_keys", []) or []:
                    pool_key = key_map.get(self._resolve_section_citation_key(raw_key, number_map))
                    if not pool_key:
                        continue
                    for topic in topics:
                        topic_nodes[topic].add(pool_key)
        return dict(topic_nodes)

    def _cited_node_topics(self, topic_cited_nodes: dict[str, set[str]]) -> tuple[set[str], dict[str, list[str]]]:
        cited_all = set().union(*topic_cited_nodes.values()) if topic_cited_nodes else set()
        node_topics: dict[str, list[str]] = defaultdict(list)
        for topic, nodes in topic_cited_nodes.items():
            for node in nodes:
                node_topics[node].append(topic)
        return cited_all, dict(node_topics)

    def node_ra_shares(
        self,
        literature_pool: Any,
        paper: Paper,
        citation_graph: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        pool = self._pool(literature_pool)
        graph = self._graph(literature_pool, citation_graph)
        topic_cited_nodes = self._topic_cited_nodes(paper, pool)
        cited_all, cited_node_topics = self._cited_node_topics(topic_cited_nodes)
        candidates = sorted(set(graph.nodes()) - cited_all)
        rows = []
        for node in candidates:
            topic_scores: dict[str, float] = defaultdict(float)
            for neighbor in graph.neighbors(node) if node in graph else []:
                if neighbor not in cited_all:
                    continue
                degree = graph.degree(neighbor)
                topics = cited_node_topics.get(neighbor, [])
                if degree <= 0 or not topics:
                    continue
                contribution = 1.0 / degree / len(topics)
                for topic in topics:
                    topic_scores[topic] += contribution
            total = sum(topic_scores.values())
            if total <= 0:
                max_share, argmax_topics = 0.0, []
            else:
                max_score = max(topic_scores.values())
                max_share = max_score / total
                argmax_topics = sorted(
                    topic for topic, score in topic_scores.items()
                    if math.isclose(score, max_score, rel_tol=1e-12, abs_tol=1e-15)
                )
            rows.append({
                "node": node,
                "max_share": max_share,
                "argmax_topic": " | ".join(argmax_topics),
                "E_p_cited_all": total,
                "seed_neighbor_count": sum(1 for neighbor in graph.neighbors(node) if neighbor in cited_all) if node in graph else 0,
            })
        return rows

    def community_ra_share(
        self,
        nodes: list[str],
        literature_pool: Any,
        paper: Paper,
        citation_graph: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pool = self._pool(literature_pool)
        graph = self._graph(literature_pool, citation_graph)
        topic_cited_nodes = self._topic_cited_nodes(paper, pool)
        cited_all, cited_node_topics = self._cited_node_topics(topic_cited_nodes)
        topic_scores: dict[str, float] = defaultdict(float)
        for node in nodes:
            if node not in graph:
                continue
            for neighbor in graph.neighbors(node):
                if neighbor not in cited_all:
                    continue
                degree = graph.degree(neighbor)
                topics = cited_node_topics.get(neighbor, [])
                if degree <= 0 or not topics:
                    continue
                contribution = 1.0 / degree / len(topics)
                for topic in topics:
                    topic_scores[topic] += contribution
        total = sum(topic_scores.values())
        if total <= 0:
            return {"max_share_ra": 0.0, "argmax_topic_ra": "", "E_ra_C_cited_all": 0.0}
        argmax_topic, max_score = max(topic_scores.items(), key=lambda item: item[1])
        return {"max_share_ra": max_score / total, "argmax_topic_ra": argmax_topic, "E_ra_C_cited_all": total}

    def _paper_text(self, paper: dict[str, Any]) -> str:
        return f"{paper.get('title', '')}\n{paper.get('abstract', '')}".strip()

    def _central_nodes(self, nodes: list[str], graph: nx.Graph, top_k: int = MAX_CENTRAL_PAPERS) -> list[str]:
        subgraph = graph.subgraph(nodes).copy()
        try:
            pagerank = nx.pagerank(subgraph) if subgraph.number_of_edges() else {node: 0.0 for node in nodes}
        except Exception as exc:
            logging.warning("Community PageRank failed: %s", exc)
            pagerank = {node: 0.0 for node in nodes}
        return sorted(nodes, key=lambda node: (pagerank.get(node, 0.0), graph.degree(node) if node in graph else 0, node), reverse=True)[:top_k]

    def _mean_pairwise_similarity(self, embeddings: list[np.ndarray]) -> float:
        if len(embeddings) < 2:
            return 0.0
        matrix = np.vstack(embeddings)
        similarities = matrix @ matrix.T
        upper = similarities[np.triu_indices(len(embeddings), k=1)]
        return float(np.mean(upper)) if len(upper) else 0.0

    def community_coherence(
        self,
        communities: dict[str, list[str]],
        literature_pool: Any,
        citation_graph: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        pool = self._pool(literature_pool)
        graph = self._graph(literature_pool, citation_graph)
        node_texts = {
            node: self._paper_text(pool[node].get("paper", pool[node]))
            for nodes in communities.values()
            for node in nodes
            if node in pool and self._paper_text(pool[node].get("paper", pool[node]))
        }
        if not node_texts:
            return []
        nodes = list(node_texts)
        embeddings = self.sentence_transformer.embed([node_texts[node] for node in nodes])
        node_embeddings = {node: embeddings[index] for index, node in enumerate(nodes)}
        rows = []
        for community_id, community_nodes in communities.items():
            vectors = [node_embeddings[node] for node in community_nodes if node in node_embeddings]
            central_nodes = [node for node in self._central_nodes(community_nodes, graph) if node in node_embeddings]
            central_vectors = [node_embeddings[node] for node in central_nodes]
            rows.append({
                "community": community_id,
                "raw_intra_sim_mean": self._mean_pairwise_similarity(vectors),
                "central_intra_sim_mean": self._mean_pairwise_similarity(central_vectors),
            })
        return rows
