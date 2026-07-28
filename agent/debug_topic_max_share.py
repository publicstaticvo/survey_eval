from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys

import numpy as np
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

from tools.utility.sbert_client import SentenceTransformerClient


DEFAULT_OUTPUT_DIR = Path("agent/test_output/codex_surveys/diffusion_models")


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title or "")).strip().casefold()


def paper_ids(paper: dict[str, Any]) -> set[str]:
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


def pool_paper(item: dict[str, Any]) -> dict[str, Any]:
    paper = item.get("paper", item)
    return paper if isinstance(paper, dict) else {}


def paper_text(paper: dict[str, Any]) -> str:
    title = str(paper.get("title", "") or "").strip()
    abstract = str(paper.get("abstract", "") or "").strip()
    return "\n".join(part for part in [title, abstract] if part)


def build_pool_indexes(pool: dict[str, dict[str, Any]]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    id_to_node: dict[str, str] = {}
    title_to_node: dict[str, str] = {}
    citation_key_to_node: dict[str, str] = {}
    for node, item in pool.items():
        paper = pool_paper(item)
        for pid in paper_ids(paper):
            id_to_node[pid] = node
        title = normalize_title(paper.get("title", ""))
        if title:
            title_to_node[title] = node
        for citation_key in item.get("citation_keys", []) or []:
            citation_key_to_node[str(citation_key)] = node
    return id_to_node, title_to_node, citation_key_to_node


def community_paper_to_node(
    paper: dict[str, Any],
    id_to_node: dict[str, str],
    title_to_node: dict[str, str],
) -> str | None:
    for pid in paper_ids(paper):
        if pid in id_to_node:
            return id_to_node[pid]
    return title_to_node.get(normalize_title(paper.get("title", "")))


def iter_sections(node: dict[str, Any]):
    for section in node.get("sections", []) or []:
        if isinstance(section, dict):
            yield section
            yield from iter_sections(section)


def iter_sentences(node: Any):
    if isinstance(node, dict):
        if "sentences" in node:
            yield from iter_sentences(node.get("sentences", []) or [])
        for paragraph in node.get("paragraphs", []) or []:
            yield from iter_sentences(paragraph)
        for section in node.get("sections", []) or []:
            yield from iter_sentences(section)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and "text" in item:
                yield item
            else:
                yield from iter_sentences(item)


def section_citation_number_map(section: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sentence in iter_sentences(section):
        citations = sentence.get("citations")
        if isinstance(citations, dict):
            for number, key in citations.items():
                mapping[str(number)] = str(key)
        elif isinstance(citations, list):
            for key in citations:
                mapping[str(key)] = str(key)
    return mapping


def parsed_topic_citations(section: dict[str, Any]) -> dict[str, set[str]]:
    number_map = section_citation_number_map(section)
    result: dict[str, set[str]] = defaultdict(set)
    parsed = section.get("parsed_contents")
    if not isinstance(parsed, dict):
        return result
    section_topics = [str(topic) for topic in parsed.get("topics", []) or [] if str(topic).strip()]
    for obj in parsed.get("objects", []) or []:
        if not isinstance(obj, dict):
            continue
        obj_topics = [str(topic) for topic in obj.get("topics", []) or [] if str(topic).strip()]
        topics = obj_topics or section_topics
        for raw_key in obj.get("citation_keys", []) or []:
            key = number_map.get(str(raw_key), str(raw_key))
            for topic in topics:
                result[topic].add(key)
    return result


def build_topic_cited_nodes(classified_paper: dict[str, Any], citation_key_to_node: dict[str, str]) -> dict[str, set[str]]:
    topic_to_nodes: dict[str, set[str]] = defaultdict(set)
    for section in iter_sections(classified_paper):
        for topic, citation_keys in parsed_topic_citations(section).items():
            for citation_key in citation_keys:
                node = citation_key_to_node.get(citation_key)
                if node:
                    topic_to_nodes[topic].add(node)
    return {topic: nodes for topic, nodes in topic_to_nodes.items() if nodes}


def build_adjacency(edges: list[dict[str, Any]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target or source == target:
            continue
        source = str(source)
        target = str(target)
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def load_communities(
    missing_papers: dict[str, Any],
    id_to_node: dict[str, str],
    title_to_node: dict[str, str],
) -> dict[int, set[str]]:
    communities: dict[int, set[str]] = {}
    for item in missing_papers.get("source_evals", {}).get("missing_topics", []) or []:
        community_id = int(item["community"])
        nodes = set()
        for paper in item.get("papers", []) or []:
            if isinstance(paper, dict):
                node = community_paper_to_node(paper, id_to_node, title_to_node)
                if node:
                    nodes.add(node)
        communities[community_id] = nodes
    return communities


def build_cited_node_to_topics(topic_to_cited_nodes: dict[str, set[str]]) -> tuple[set[str], dict[str, list[str]]]:
    cited_all = set().union(*topic_to_cited_nodes.values()) if topic_to_cited_nodes else set()
    cited_node_to_topics: dict[str, list[str]] = defaultdict(list)
    for topic, nodes in topic_to_cited_nodes.items():
        for node in nodes:
            cited_node_to_topics[node].append(topic)
    return cited_all, cited_node_to_topics


def topic_degree_rows(topic_to_cited_nodes: dict[str, set[str]], adjacency: dict[str, set[str]]) -> list[dict[str, Any]]:
    rows = []
    for topic, nodes in topic_to_cited_nodes.items():
        degrees = [len(adjacency.get(node, set())) for node in nodes]
        if not degrees:
            continue
        rows.append({
            "topic": topic,
            "cited_count": len(nodes),
            "degree_max": max(degrees),
            "degree_mean": mean(degrees),
            "degree_median": median(degrees),
        })
    return sorted(rows, key=lambda row: row["degree_mean"], reverse=True)


def compute_max_share_rows(
    communities: dict[int, set[str]],
    adjacency: dict[str, set[str]],
    topic_to_cited_nodes: dict[str, set[str]],
    weight_mode: str = "none",
) -> list[dict[str, Any]]:
    cited_all, cited_node_to_topics = build_cited_node_to_topics(topic_to_cited_nodes)
    if weight_mode == "log":
        weights = {
            node: 1.0 / math.log1p(len(adjacency.get(node, set())))
            for node in cited_all
            if len(adjacency.get(node, set())) > 0
        }
    elif weight_mode == "ra":
        weights = {
            node: 1.0 / len(adjacency.get(node, set()))
            for node in cited_all
            if len(adjacency.get(node, set())) > 0
        }
    elif weight_mode == "none":
        weights = {}
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    rows = []
    for community_id, nodes in communities.items():
        topic_edge_counts: dict[str, float] = defaultdict(float)
        for node in nodes:
            for neighbor in adjacency.get(node, set()):
                if neighbor not in cited_all:
                    continue
                contribution = weights.get(neighbor, 0.0) if weight_mode != "none" else 1.0
                for topic in cited_node_to_topics[neighbor]:
                    topic_edge_counts[topic] += contribution
        total = sum(topic_edge_counts.values())
        if total == 0:
            max_share = 0.0
            argmax_topic = ""
        else:
            argmax_topic, max_edges = max(topic_edge_counts.items(), key=lambda item: item[1])
            max_share = max_edges / total
        rows.append({
            "community_id": community_id,
            "size": len(nodes),
            "max_share": max_share,
            "argmax_topic": argmax_topic,
            "E_C_cited_all": total,
        })
    return sorted(rows, key=lambda row: (row["max_share"], row["E_C_cited_all"], row["community_id"]))


def compute_node_ra_share_rows(
    adjacency: dict[str, set[str]],
    topic_to_cited_nodes: dict[str, set[str]],
) -> list[dict[str, Any]]:
    cited_all, cited_node_to_topics = build_cited_node_to_topics(topic_to_cited_nodes)
    weights = {
        node: 1.0 / len(adjacency.get(node, set()))
        for node in cited_all
        if len(adjacency.get(node, set())) > 0
    }
    graph_nodes = set(adjacency)
    for neighbors in adjacency.values():
        graph_nodes.update(neighbors)
    candidates = sorted(graph_nodes - cited_all)

    rows = []
    for node in candidates:
        topic_scores: dict[str, float] = defaultdict(float)
        for neighbor in adjacency.get(node, set()):
            if neighbor not in cited_all:
                continue
            topics = cited_node_to_topics.get(neighbor, [])
            if not topics:
                continue
            contribution = weights.get(neighbor, 0.0) / len(topics)
            for topic in topics:
                topic_scores[topic] += contribution
        total = sum(topic_scores.values())
        if total <= 0:
            max_share = 0.0
            argmax_topics = []
        else:
            max_score = max(topic_scores.values())
            argmax_topics = sorted(
                topic for topic, score in topic_scores.items()
                if math.isclose(score, max_score, rel_tol=1e-12, abs_tol=1e-15)
            )
            max_share = max_score / total
        rows.append({
            "node": node,
            "max_share": max_share,
            "argmax_topic": " | ".join(argmax_topics),
            "argmax_tie_count": len(argmax_topics),
            "E_p_cited_all": total,
            "degree": len(adjacency.get(node, set())),
        })
    return rows


def node_share_histogram(rows: list[dict[str, Any]], bin_width: float = 0.05) -> list[dict[str, Any]]:
    bins = int(round(1.0 / bin_width))
    histogram = [{"bin_start": index * bin_width, "bin_end": (index + 1) * bin_width, "count": 0} for index in range(bins)]
    for row in rows:
        value = min(1.0, max(0.0, float(row["max_share"])))
        index = min(bins - 1, int(value / bin_width))
        histogram[index]["count"] += 1
    total = len(rows)
    for item in histogram:
        item["fraction"] = item["count"] / total if total else 0.0
    return histogram


def node_share_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = np.array([float(row["max_share"]) for row in rows], dtype=float)
    zero_edges = sum(1 for row in rows if float(row["E_p_cited_all"]) <= 0)
    if len(values) == 0:
        return [{
            "candidate_count": 0,
            "zero_edge_count": 0,
            "zero_edge_fraction": 0.0,
            "mean": 0.0,
            "q1": 0.0,
            "median": 0.0,
            "q3": 0.0,
            "min": 0.0,
            "max": 0.0,
        }]
    return [{
        "candidate_count": int(len(values)),
        "zero_edge_count": int(zero_edges),
        "zero_edge_fraction": zero_edges / len(values),
        "mean": float(np.mean(values)),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q3": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }]


def node_share_topic_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fractional_counts: dict[str, float] = defaultdict(float)
    exclusive_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        topics = [topic for topic in str(row["argmax_topic"] or "").split(" | ") if topic]
        if not topics:
            fractional_counts["None"] += 1.0
            exclusive_counts["None"] += 1
            continue
        contribution = 1.0 / len(topics)
        for topic in topics:
            fractional_counts[topic] += contribution
        if len(topics) == 1:
            exclusive_counts[topics[0]] += 1
    total = len(rows)
    return [
        {
            "argmax_topic": topic,
            "fractional_count": count,
            "exclusive_count": exclusive_counts[topic],
            "fraction": count / total if total else 0.0,
        }
        for topic, count in sorted(fractional_counts.items(), key=lambda item: item[1], reverse=True)
    ]

def node_share_bimodality_diagnostic(histogram: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = [int(item["count"]) for item in histogram]
    peaks = []
    for index, count in enumerate(counts):
        left = counts[index - 1] if index > 0 else -1
        right = counts[index + 1] if index + 1 < len(counts) else -1
        if count > left and count > right:
            peaks.append(index)
    if len(peaks) < 2:
        return [{
            "bimodal": False,
            "left_peak_bin": "",
            "right_peak_bin": "",
            "valley_bin": "",
            "suggested_threshold": "",
            "note": "No clear two-peak structure in the 0.05-bin histogram.",
        }]
    top_peaks = sorted(peaks, key=lambda index: counts[index], reverse=True)[:2]
    left_peak, right_peak = sorted(top_peaks)
    if right_peak - left_peak <= 1:
        return [{
            "bimodal": False,
            "left_peak_bin": "",
            "right_peak_bin": "",
            "valley_bin": "",
            "suggested_threshold": "",
            "note": "The two highest local peaks are adjacent, so there is no intervening valley.",
        }]
    valley = min(range(left_peak + 1, right_peak), key=lambda index: counts[index])
    threshold = histogram[valley]["bin_end"]
    return [{
        "bimodal": True,
        "left_peak_bin": f"{histogram[left_peak]['bin_start']:.2f}-{histogram[left_peak]['bin_end']:.2f}",
        "right_peak_bin": f"{histogram[right_peak]['bin_start']:.2f}-{histogram[right_peak]['bin_end']:.2f}",
        "valley_bin": f"{histogram[valley]['bin_start']:.2f}-{histogram[valley]['bin_end']:.2f}",
        "suggested_threshold": threshold,
        "note": "Heuristic local-peak diagnostic; inspect the histogram before treating this as a natural threshold.",
    }]

def community_pagerank_scores(
    nodes: set[str],
    adjacency: dict[str, set[str]],
    damping: float = 0.85,
    iterations: int = 50,
) -> dict[str, float]:
    if not nodes:
        return {}
    node_list = sorted(nodes)
    n_nodes = len(node_list)
    scores = {node: 1.0 / n_nodes for node in node_list}
    internal_neighbors = {
        node: sorted(adjacency.get(node, set()) & nodes)
        for node in node_list
    }
    for _ in range(iterations):
        dangling = sum(scores[node] for node, neighbors in internal_neighbors.items() if not neighbors)
        base = (1.0 - damping) / n_nodes + damping * dangling / n_nodes
        next_scores = {node: base for node in node_list}
        for node, neighbors in internal_neighbors.items():
            if not neighbors:
                continue
            contribution = damping * scores[node] / len(neighbors)
            for neighbor in neighbors:
                next_scores[neighbor] += contribution
        scores = next_scores
    return scores


def community_degree_scores(nodes: set[str], adjacency: dict[str, set[str]]) -> dict[str, float]:
    return {node: float(len(adjacency.get(node, set()) & nodes)) for node in nodes}


def select_central_nodes(
    nodes: set[str],
    adjacency: dict[str, set[str]],
    method: str,
    top_k: int,
) -> list[str]:
    if top_k <= 0:
        return []
    if method == "pagerank":
        scores = community_pagerank_scores(nodes, adjacency)
    elif method == "degree":
        scores = community_degree_scores(nodes, adjacency)
    else:
        raise ValueError(f"Unknown centrality method: {method}")
    return sorted(
        nodes,
        key=lambda node: (scores.get(node, 0.0), len(adjacency.get(node, set())), node),
        reverse=True,
    )[:top_k]


def vector_similarity_stats(vectors: list[np.ndarray], prefix: str) -> dict[str, Any]:
    if len(vectors) < 2:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_pair_count": 0,
        }
    matrix = np.vstack(vectors)
    sims = matrix @ matrix.T
    upper = sims[np.triu_indices(len(vectors), k=1)]
    return {
        f"{prefix}_mean": float(np.mean(upper)),
        f"{prefix}_median": float(np.median(upper)),
        f"{prefix}_min": float(np.min(upper)),
        f"{prefix}_max": float(np.max(upper)),
        f"{prefix}_pair_count": int(len(upper)),
    }


def community_similarity_rows(
    communities: dict[int, set[str]],
    pool: dict[str, dict[str, Any]],
    sbert_url: str,
    adjacency: dict[str, set[str]],
    centrality_method: str = "pagerank",
    centrality_top_k: int = 10,
) -> list[dict[str, Any]]:
    node_texts = {
        node: paper_text(pool_paper(pool[node]))
        for nodes in communities.values()
        for node in nodes
        if node in pool and paper_text(pool_paper(pool[node]))
    }
    client = SentenceTransformerClient(sbert_url)
    nodes = list(node_texts)
    embeddings = client.embed([node_texts[node] for node in nodes]) if nodes else np.array([])
    node_embedding = {node: embeddings[index] for index, node in enumerate(nodes)}
    rows = []
    for community_id, community_nodes in communities.items():
        vectors = [node_embedding[node] for node in community_nodes if node in node_embedding]
        central_nodes = select_central_nodes(community_nodes, adjacency, centrality_method, centrality_top_k)
        central_vectors = [node_embedding[node] for node in central_nodes if node in node_embedding]
        rows.append({
            "community_id": community_id,
            "centrality_method": centrality_method,
            "centrality_top_k": centrality_top_k,
            "central_node_count": len(central_vectors),
            **vector_similarity_stats(vectors, "intra_sim"),
            **vector_similarity_stats(central_vectors, "central_intra_sim"),
        })
    return sorted(rows, key=lambda row: (row["central_intra_sim_mean"], row["intra_sim_mean"], row["community_id"]))

def comparison_rows(
    unweighted: list[dict[str, Any]],
    log_weighted: list[dict[str, Any]],
    ra_weighted: list[dict[str, Any]],
    similarity: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    unweighted_by_id = {row["community_id"]: row for row in unweighted}
    log_by_id = {row["community_id"]: row for row in log_weighted}
    ra_by_id = {row["community_id"]: row for row in ra_weighted}
    sim_by_id = {row["community_id"]: row for row in similarity or []}
    rows = []
    for community_id, ra_row in ra_by_id.items():
        unweighted_row = unweighted_by_id[community_id]
        log_row = log_by_id[community_id]
        rows.append({
            "community_id": community_id,
            "size": ra_row["size"],
            "max_share": unweighted_row["max_share"],
            "argmax_topic": unweighted_row["argmax_topic"],
            "max_share_weighted": log_row["max_share"],
            "argmax_topic_weighted": log_row["argmax_topic"],
            "max_share_ra": ra_row["max_share"],
            "argmax_topic_ra": ra_row["argmax_topic"],
            "E_ra_C_cited_all": ra_row["E_C_cited_all"],
            "argmax_changed_vs_log": log_row["argmax_topic"] != ra_row["argmax_topic"],
            "argmax_changed_vs_unweighted": unweighted_row["argmax_topic"] != ra_row["argmax_topic"],
            **sim_by_id.get(community_id, {
                "intra_sim_mean": 0.0,
                "intra_sim_median": 0.0,
                "intra_sim_min": 0.0,
                "intra_sim_max": 0.0,
                "intra_sim_pair_count": 0,
                "centrality_method": "",
                "centrality_top_k": 0,
                "central_node_count": 0,
                "central_intra_sim_mean": 0.0,
                "central_intra_sim_median": 0.0,
                "central_intra_sim_min": 0.0,
                "central_intra_sim_max": 0.0,
                "central_intra_sim_pair_count": 0,
            }),
        })
    return sorted(rows, key=lambda row: (row["max_share_ra"], row["E_ra_C_cited_all"], row["community_id"]))



def ordinal_ranks(rows: list[dict[str, Any]], column: str, reverse: bool = False) -> dict[int, int]:
    if reverse:
        sorted_rows = sorted(rows, key=lambda row: (-float(row[column]), int(row["community_id"])))
    else:
        sorted_rows = sorted(rows, key=lambda row: (float(row[column]), int(row["community_id"])))
    return {int(row["community_id"]): index + 1 for index, row in enumerate(sorted_rows)}


def rank_score(rank: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return (total - rank) / (total - 1)


def add_final_ranking_scores(
    rows: list[dict[str, Any]],
    max_share_weight: float,
    central_sim_weight: float,
    intra_sim_weight: float,
) -> list[dict[str, Any]]:
    total_weight = max_share_weight + central_sim_weight + intra_sim_weight
    if total_weight <= 0:
        raise ValueError("Final score weights must sum to a positive value.")

    total = len(rows)
    max_share_ranks = ordinal_ranks(rows, "max_share_ra", reverse=False)
    intra_sim_ranks = ordinal_ranks(rows, "intra_sim_mean", reverse=True)
    central_sim_ranks = ordinal_ranks(rows, "central_intra_sim_mean", reverse=True)

    scored_rows = []
    for row in rows:
        community_id = int(row["community_id"])
        max_share_rank = max_share_ranks[community_id]
        intra_sim_rank = intra_sim_ranks[community_id]
        central_sim_rank = central_sim_ranks[community_id]
        max_share_rank_score = rank_score(max_share_rank, total)
        intra_sim_rank_score = rank_score(intra_sim_rank, total)
        central_sim_rank_score = rank_score(central_sim_rank, total)
        final_score = (
            max_share_weight * max_share_rank_score
            + central_sim_weight * central_sim_rank_score
            + intra_sim_weight * intra_sim_rank_score
        ) / total_weight
        scored_rows.append({
            **row,
            "max_share_ra_rank": max_share_rank,
            "intra_sim_mean_rank": intra_sim_rank,
            "central_intra_sim_mean_rank": central_sim_rank,
            "max_share_ra_rank_score": max_share_rank_score,
            "intra_sim_mean_rank_score": intra_sim_rank_score,
            "central_intra_sim_mean_rank_score": central_sim_rank_score,
            "final_score": final_score,
        })

    final_ranks = ordinal_ranks(scored_rows, "final_score", reverse=True)
    for row in scored_rows:
        row["final_score_rank"] = final_ranks[int(row["community_id"])]
    return sorted(scored_rows, key=lambda row: (-row["final_score"], row["community_id"]))

def print_table(rows: list[dict[str, Any]], headers: list[str], float_columns: set[str] | None = None) -> None:
    float_columns = float_columns or set()
    formatted = []
    for row in rows:
        item = dict(row)
        for column in float_columns:
            item[column] = f"{float(item[column]):.4f}"
        formatted.append(item)
    widths = {
        header: max([len(header), *(len(str(row[header])) for row in formatted)])
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in formatted:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def write_csv(rows: list[dict[str, Any]], output: Path, fieldnames: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose cited-topic hub pollution and compute log/RA weighted max_share(C).")
    parser.add_argument("--dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory containing 02/04/08 JSON outputs.")
    parser.add_argument("--classified-paper", type=Path, help="Path to 02_classified_paper.json.")
    parser.add_argument("--literature-pool", type=Path, help="Path to 04_literature_pool.json.")
    parser.add_argument("--missing-papers", type=Path, help="Path to 08_missing_papers.json.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path for RA/log comparison rows.")
    parser.add_argument("--degree-output", type=Path, help="Optional CSV output path for topic degree diagnostics.")
    parser.add_argument("--node-share-output", type=Path, help="Optional CSV output path for node-level RA max-share rows.")
    parser.add_argument("--node-share-histogram-output", type=Path, help="Optional CSV output path for node-level max-share histogram.")
    parser.add_argument("--node-share-topic-output", type=Path, help="Optional CSV output path for node-level argmax topic counts.")
    parser.add_argument("--node-share-summary-output", type=Path, help="Optional CSV output path for node-level max-share summary and bimodality diagnostic.")
    parser.add_argument("--similarity-output", type=Path, help="Optional CSV output path for intra-community similarity rows.")
    parser.add_argument("--sbert-url", default="http://172.18.36.90:8030", help="SBERT embedding service URL.")
    parser.add_argument("--skip-community-similarity", action="store_true", help="Skip SBERT community similarity computation when only graph diagnostics are needed.")
    parser.add_argument("--centrality-method", choices=["pagerank", "degree"], default="pagerank", help="Method for selecting central community papers.")
    parser.add_argument("--centrality-top-k", type=int, default=10, help="Number of central community papers used for central similarity.")
    parser.add_argument("--score-max-share-weight", type=float, default=0.5, help="Final score weight for low max_share_ra rank.")
    parser.add_argument("--score-central-sim-weight", type=float, default=0.3, help="Final score weight for high central top-K similarity rank.")
    parser.add_argument("--score-intra-sim-weight", type=float, default=0.2, help="Final score weight for high whole-community similarity rank.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classified_path = args.classified_paper or args.dir / "02_classified_paper.json"
    literature_path = args.literature_pool or args.dir / "04_literature_pool.json"
    missing_path = args.missing_papers or args.dir / "08_missing_papers.json"
    cache_path = args.dir / "04_literature_pool.cache.json"

    classified_paper = json.loads(classified_path.read_text(encoding="utf-8"))
    literature_data = json.loads(literature_path.read_text(encoding="utf-8"))
    metadata_data = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else literature_data
    missing_data = json.loads(missing_path.read_text(encoding="utf-8"))

    pool = metadata_data["literature_pool"]
    graph = literature_data["citation_graph"]
    id_to_node, title_to_node, citation_key_to_node = build_pool_indexes(pool)
    topic_to_cited_nodes = build_topic_cited_nodes(classified_paper, citation_key_to_node)
    communities = load_communities(missing_data, id_to_node, title_to_node)
    adjacency = build_adjacency(graph.get("edges", []) or [])

    node_share_rows = compute_node_ra_share_rows(adjacency, topic_to_cited_nodes)
    node_histogram_rows = node_share_histogram(node_share_rows)
    node_summary_rows = node_share_summary(node_share_rows)
    node_topic_rows = node_share_topic_counts(node_share_rows)
    node_bimodality_rows = node_share_bimodality_diagnostic(node_histogram_rows)
    degree_rows = topic_degree_rows(topic_to_cited_nodes, adjacency)
    print("Topic Cited_T degree diagnostics")
    print_table(
        degree_rows,
        ["topic", "cited_count", "degree_max", "degree_mean", "degree_median"],
        {"degree_mean", "degree_median"},
    )

    if args.skip_community_similarity:
        similarity_rows = []
    else:
        similarity_rows = community_similarity_rows(
            communities,
            pool,
            args.sbert_url,
            adjacency,
            args.centrality_method,
            args.centrality_top_k,
        )
    print("\nIntra-community title+abstract similarity")
    print_table(
        similarity_rows,
        ["community_id", "intra_sim_mean", "central_intra_sim_mean", "central_node_count", "centrality_method", "centrality_top_k", "intra_sim_median", "central_intra_sim_median", "intra_sim_min", "central_intra_sim_min", "intra_sim_max", "central_intra_sim_max", "intra_sim_pair_count", "central_intra_sim_pair_count"],
        {"intra_sim_mean", "intra_sim_median", "intra_sim_min", "intra_sim_max", "central_intra_sim_mean", "central_intra_sim_median", "central_intra_sim_min", "central_intra_sim_max"},
    )

    unweighted = compute_max_share_rows(communities, adjacency, topic_to_cited_nodes, weight_mode="none")
    log_weighted = compute_max_share_rows(communities, adjacency, topic_to_cited_nodes, weight_mode="log")
    ra_weighted = compute_max_share_rows(communities, adjacency, topic_to_cited_nodes, weight_mode="ra")
    rows = comparison_rows(unweighted, log_weighted, ra_weighted, similarity_rows)
    rows = add_final_ranking_scores(
        rows,
        args.score_max_share_weight,
        args.score_central_sim_weight,
        args.score_intra_sim_weight,
    )
    print("\nCommunity max_share RA comparison")
    print_table(
        rows,
        [
            "community_id", "size", "final_score", "final_score_rank",
            "max_share_ra", "max_share_ra_rank", "argmax_topic_ra",
            "central_intra_sim_mean", "central_intra_sim_mean_rank", "intra_sim_mean", "intra_sim_mean_rank",
            "E_ra_C_cited_all", "argmax_changed_vs_log", "argmax_changed_vs_unweighted",
        ],
        {"final_score", "max_share_weighted", "max_share_ra", "E_ra_C_cited_all", "intra_sim_mean", "central_intra_sim_mean"},
    )
    print("\nNode-level RA max_share summary")
    print_table(
        node_summary_rows,
        ["candidate_count", "zero_edge_count", "zero_edge_fraction", "mean", "q1", "median", "q3", "min", "max"],
        {"zero_edge_fraction", "mean", "q1", "median", "q3", "min", "max"},
    )
    print("\nNode-level RA max_share histogram")
    print_table(
        node_histogram_rows,
        ["bin_start", "bin_end", "count", "fraction"],
        {"bin_start", "bin_end", "fraction"},
    )
    print("\nNode-level RA bimodality diagnostic")
    print_table(
        node_bimodality_rows,
        ["bimodal", "left_peak_bin", "right_peak_bin", "valley_bin", "suggested_threshold", "note"],
        {"suggested_threshold"},
    )
    print("\nNode-level argmax topic counts")
    print_table(
        node_topic_rows,
        ["argmax_topic", "fractional_count", "exclusive_count", "fraction"],
        {"fraction"},
    )
    if args.node_share_output:
        write_csv(
            node_share_rows,
            args.node_share_output,
            ["node", "max_share", "argmax_topic", "argmax_tie_count", "E_p_cited_all", "degree"],
        )
    if args.node_share_histogram_output:
        write_csv(
            node_histogram_rows,
            args.node_share_histogram_output,
            ["bin_start", "bin_end", "count", "fraction"],
        )
    if args.node_share_topic_output:
        write_csv(
            node_topic_rows,
            args.node_share_topic_output,
            ["argmax_topic", "fractional_count", "exclusive_count", "fraction"],
        )
    if args.node_share_summary_output:
        write_csv(
            [*node_summary_rows, *node_bimodality_rows],
            args.node_share_summary_output,
            [
                "candidate_count", "zero_edge_count", "zero_edge_fraction", "mean", "q1", "median", "q3", "min", "max",
                "bimodal", "left_peak_bin", "right_peak_bin", "valley_bin", "suggested_threshold", "note",
            ],
        )
    if args.degree_output:
        write_csv(
            degree_rows,
            args.degree_output,
            ["topic", "cited_count", "degree_max", "degree_mean", "degree_median"],
        )
    if args.similarity_output:
        write_csv(
            similarity_rows,
            args.similarity_output,
            ["community_id", "intra_sim_mean", "central_intra_sim_mean", "central_node_count", "centrality_method", "centrality_top_k", "intra_sim_median", "central_intra_sim_median", "intra_sim_min", "central_intra_sim_min", "intra_sim_max", "central_intra_sim_max", "intra_sim_pair_count", "central_intra_sim_pair_count"],
        )
    if args.output:
        write_csv(
            rows,
            args.output,
            [
                "community_id", "size", "final_score", "final_score_rank",
                "max_share_ra_rank", "intra_sim_mean_rank", "central_intra_sim_mean_rank",
                "max_share_ra_rank_score", "intra_sim_mean_rank_score", "central_intra_sim_mean_rank_score",
                "max_share", "argmax_topic",
                "max_share_weighted", "argmax_topic_weighted",
                "max_share_ra", "argmax_topic_ra", "E_ra_C_cited_all",
                "argmax_changed_vs_log", "argmax_changed_vs_unweighted",
                "intra_sim_mean", "intra_sim_median", "intra_sim_min", "intra_sim_max", "intra_sim_pair_count",
                "centrality_method", "centrality_top_k", "central_node_count",
                "central_intra_sim_mean", "central_intra_sim_median", "central_intra_sim_min", "central_intra_sim_max", "central_intra_sim_pair_count",
            ],
        )


if __name__ == "__main__":
    main()






















