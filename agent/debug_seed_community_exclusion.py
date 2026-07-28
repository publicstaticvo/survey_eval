"""Test full-graph Leiden clustering followed by seed-containing community removal.

This intentionally strict baseline keeps only Leiden communities that contain no
paper cited by any extracted survey topic. Surviving communities are ranked by
the mean node-level Resource Allocation max-share score.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import igraph as ig
import leidenalg


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

from debug_topic_max_share import (
    DEFAULT_OUTPUT_DIR,
    build_adjacency,
    build_pool_indexes,
    build_topic_cited_nodes,
    compute_node_ra_share_rows,
    paper_text,
    pool_paper,
)


def run_leiden(
    graph_nodes: set[str],
    edges: list[dict[str, Any]],
    resolution: float,
    seed: int,
) -> list[set[str]]:
    nodes = sorted(graph_nodes)
    node_index = {node: index for index, node in enumerate(nodes)}
    graph_edges = [
        (node_index[str(edge["source"])], node_index[str(edge["target"])])
        for edge in edges
        if edge.get("source")
        and edge.get("target")
        and edge["source"] != edge["target"]
        and str(edge["source"]) in node_index
        and str(edge["target"]) in node_index
    ]
    graph = ig.Graph(n=len(nodes), edges=graph_edges, directed=False)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    return [{nodes[index] for index in membership} for membership in partition]


def internal_pagerank(nodes: set[str], adjacency: dict[str, set[str]]) -> dict[str, float]:
    if not nodes:
        return {}
    node_list = sorted(nodes)
    node_index = {node: index for index, node in enumerate(node_list)}
    edges = [
        (node_index[node], node_index[neighbor])
        for node in node_list
        for neighbor in adjacency.get(node, set())
        if neighbor in node_index and node < neighbor
    ]
    graph = ig.Graph(n=len(node_list), edges=edges, directed=False)
    scores = graph.pagerank(directed=False)
    return dict(zip(node_list, scores, strict=True))


def representative_papers(
    nodes: set[str],
    pool: dict[str, dict[str, Any]],
    adjacency: dict[str, set[str]],
    limit: int,
) -> list[dict[str, Any]]:
    scores = internal_pagerank(nodes, adjacency)
    ranked_nodes = sorted(
        nodes,
        key=lambda node: (
            scores.get(node, 0.0),
            len(adjacency.get(node, set())),
            node,
        ),
        reverse=True,
    )[:limit]
    return [
        {
            "node": node,
            "title": pool_paper(pool.get(node, {})).get("title", ""),
            "abstract": pool_paper(pool.get(node, {})).get("abstract", ""),
            "internal_pagerank": scores.get(node, 0.0),
            "degree": len(adjacency.get(node, set())),
        }
        for node in ranked_nodes
    ]


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-graph Leiden, discard every community containing a cited seed, and rank survivors."
    )
    parser.add_argument("--dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--representatives", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "seed_community_exclusion_ranking.csv",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "seed_community_exclusion_details.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classified_paper = json.loads((args.dir / "02_classified_paper.json").read_text(encoding="utf-8"))
    literature_data = json.loads((args.dir / "04_literature_pool.json").read_text(encoding="utf-8"))
    cache_path = args.dir / "04_literature_pool.cache.json"
    metadata_data = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else literature_data

    pool = metadata_data["literature_pool"]
    citation_graph = literature_data["citation_graph"]
    edges = citation_graph.get("edges", []) or []
    adjacency = build_adjacency(edges)
    _, _, citation_key_to_node = build_pool_indexes(pool)
    topic_to_cited_nodes = build_topic_cited_nodes(classified_paper, citation_key_to_node)
    cited_all = set().union(*topic_to_cited_nodes.values())
    graph_nodes = set(pool) | set(adjacency)
    for neighbors in adjacency.values():
        graph_nodes.update(neighbors)

    communities = run_leiden(graph_nodes, edges, args.resolution, args.seed)
    node_scores = {row["node"]: row for row in compute_node_ra_share_rows(adjacency, topic_to_cited_nodes)}
    surviving = [
        (community_id, nodes)
        for community_id, nodes in enumerate(communities)
        if not nodes & cited_all
    ]

    summary_rows = []
    details = []
    for community_id, nodes in surviving:
        scores = [float(node_scores[node]["max_share"]) for node in nodes if node in node_scores]
        argmax_counts = Counter(
            str(node_scores[node]["argmax_topic"])
            for node in nodes
            if node in node_scores and node_scores[node]["argmax_topic"]
        )
        row = {
            "community_id": community_id,
            "size": len(nodes),
            "mean_node_max_share_ra": mean(scores) if scores else 0.0,
            "median_node_max_share_ra": median(scores) if scores else 0.0,
            "min_node_max_share_ra": min(scores) if scores else 0.0,
            "max_node_max_share_ra": max(scores) if scores else 0.0,
            "zero_seed_edge_nodes": sum(
                1 for node in nodes if node in node_scores and float(node_scores[node]["E_p_cited_all"]) == 0.0
            ),
            "top_argmax_topics": " | ".join(
                f"{topic}: {count}" for topic, count in argmax_counts.most_common(5)
            ),
        }
        summary_rows.append(row)
        details.append({
            **row,
            "representative_papers": representative_papers(nodes, pool, adjacency, args.representatives),
        })

    summary_rows.sort(
        key=lambda row: (
            row["mean_node_max_share_ra"],
            row["median_node_max_share_ra"],
            row["community_id"],
        )
    )
    details_by_id = {item["community_id"]: item for item in details}
    details = [details_by_id[row["community_id"]] for row in summary_rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary_rows, args.output)
    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    args.details_output.write_text(
        json.dumps(
            {
                "resolution": args.resolution,
                "seed": args.seed,
                "graph_node_count": len(graph_nodes),
                "graph_edge_count": len(edges),
                "cited_seed_count": len(cited_all),
                "community_count": len(communities),
                "communities_with_cited_seed": len(communities) - len(surviving),
                "surviving_community_count": len(surviving),
                "surviving_node_count": sum(len(nodes) for _, nodes in surviving),
                "communities": details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Full-graph Leiden: "
        f"nodes={len(graph_nodes)}, edges={len(edges)}, communities={len(communities)}, "
        f"cited_seeds={len(cited_all)}"
    )
    print(
        "After deleting every seed-containing community: "
        f"surviving_communities={len(surviving)}, "
        f"surviving_nodes={sum(len(nodes) for _, nodes in surviving)}"
    )
    if not summary_rows:
        return
    print("community_id,size,mean_node_max_share_ra,median_node_max_share_ra,top_argmax_topics")
    for row in summary_rows:
        print(
            f"{row['community_id']},{row['size']},{row['mean_node_max_share_ra']:.4f},"
            f"{row['median_node_max_share_ra']:.4f},{row['top_argmax_topics']}"
        )


if __name__ == "__main__":
    main()
