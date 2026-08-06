from __future__ import annotations

"""Fit the acceptance-calibrated adequacy bottleneck with cached graph features."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import igraph
import leidenalg
import networkx as nx
import numpy as np
import pandas as pd

from survey_eval.golden.fit_acceptance_bottleneck import (
    category_indices,
    evaluate,
    firth_logistic_fit,
    load_integrity_counts,
    load_papers,
    raw_features,
)


def graph_features(pool_path: Path) -> dict[str, float]:
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    nodes = list((data.get("citation_graph") or {}).get("nodes") or [])
    raw_edges = (data.get("citation_graph") or {}).get("edges") or []
    edges = [
        (str(edge["source"]), str(edge["target"]))
        for edge in raw_edges
        if edge.get("source") != edge.get("target")
    ]
    seeds = set((data.get("citation_matches") or {}).values()) & set(nodes)
    if len(nodes) < 100 or len(edges) < 100 or len(seeds) < 3:
        return {
            "graph_usable": 0.0,
            "reference_ppr_mass": np.nan,
            "topic_community_coverage": np.nan,
            "graph_nodes": float(len(nodes)),
            "graph_edges": float(len(edges)),
            "matched_citations": float(len(seeds)),
        }

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    personalization = {node: 1.0 / len(seeds) if node in seeds else 0.0 for node in nodes}
    try:
        ppr = nx.pagerank(graph, alpha=0.85, personalization=personalization, max_iter=500)
    except nx.PowerIterationFailedConvergence:
        return {
            "graph_usable": 0.0,
            "reference_ppr_mass": np.nan,
            "topic_community_coverage": np.nan,
            "graph_nodes": float(len(nodes)),
            "graph_edges": float(len(edges)),
            "matched_citations": float(len(seeds)),
        }
    reference_ppr_mass = sum(ppr.get(node, 0.0) for node in seeds)

    node_index = {node: index for index, node in enumerate(nodes)}
    undirected_edges = [(node_index[source], node_index[target]) for source, target in edges]
    igraph_graph = igraph.Graph(n=len(nodes), edges=undirected_edges, directed=False)
    partition = leidenalg.find_partition(
        igraph_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=1.0,
        seed=20260729,
    )
    membership = partition.membership
    section_matches = data.get("section_citation_matches") or {}
    body_citations = {
        work_id
        for section, work_ids in section_matches.items()
        if str(section).strip()
        for work_id in work_ids
    }
    covered_communities = {
        membership[node_index[work_id]]
        for work_id in body_citations
        if work_id in node_index
    }
    topic_community_coverage = sum(
        ppr[node]
        for node in nodes
        if membership[node_index[node]] in covered_communities
    )
    return {
        "graph_usable": 1.0,
        "reference_ppr_mass": float(reference_ppr_mass),
        "topic_community_coverage": float(topic_community_coverage),
        "graph_nodes": float(len(nodes)),
        "graph_edges": float(len(edges)),
        "matched_citations": float(len(seeds)),
    }


def graph_category_indices(train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    indices = category_indices(train, target).drop(columns=["reference_proxy", "topic_proxy"])
    ordered_ref = np.sort(train["reference_ppr_mass"].to_numpy(float))
    ordered_topic = np.sort(train["topic_community_coverage"].to_numpy(float))
    indices["reference"] = np.searchsorted(
        ordered_ref,
        target["reference_ppr_mass"].to_numpy(float),
        side="right",
    ) / len(ordered_ref)
    indices["topic"] = np.searchsorted(
        ordered_topic,
        target["topic_community_coverage"].to_numpy(float),
        side="right",
    ) / len(ordered_topic)
    return indices


def evaluate_graph(df: pd.DataFrame, repeats: int, folds: int, seed: int):
    import survey_eval.golden.fit_acceptance_bottleneck as base

    original = base.category_indices
    try:
        base.category_indices = graph_category_indices
        return evaluate(df, repeats, folds, seed)
    finally:
        base.category_indices = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-content", type=Path, default=Path(__file__).parent / "pdf_content")
    parser.add_argument("--review-analysis", type=Path, default=Path(__file__).parent / "review_analyze")
    parser.add_argument("--pools", type=Path, default=Path(__file__).parent / "acceptance_fit" / "literature_pools_v2")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "acceptance_fit" / "graph_model")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    integrity = load_integrity_counts(args.review_analysis)
    rows = []
    for view in load_papers(args.pdf_content):
        if view.accepted is None:
            continue
        features = raw_features(view, integrity.get(view.paper_id, {}))
        features.update(graph_features(args.pools / f"{view.paper_id:03d}_pool.json"))
        rows.append(
            {
                "paper_id": view.paper_id,
                "forum_id": view.forum_id,
                "title": view.title,
                "venue": view.venue,
                "accepted": view.accepted,
                **features,
            }
        )
    all_df = pd.DataFrame(rows).sort_values("paper_id").reset_index(drop=True)
    usable = all_df.loc[all_df["graph_usable"] == 1.0].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.output_dir / "all_training_features.csv", index=False)
    usable.to_csv(args.output_dir / "graph_usable_training_features.csv", index=False)
    predictions, metrics = evaluate_graph(usable, args.repeats, args.folds, args.seed)
    predictions.to_csv(args.output_dir / "cross_validated_predictions.csv", index=False)
    indices = graph_category_indices(usable, usable)
    bottleneck = indices.min(axis=1).to_numpy(float)
    integrity_columns = [
        "integrity_internal",
        "integrity_factual",
        "integrity_taxonomy",
        "integrity_argument",
    ]
    design = np.column_stack([np.ones(len(usable)), bottleneck, usable[integrity_columns].to_numpy(float)])
    coefficients = firth_logistic_fit(design, usable["accepted"].to_numpy(int))
    model = {
        "training_n": int(len(usable)),
        "category_order": list(indices.columns),
        "integrity_columns": integrity_columns,
        "coefficients": {
            "intercept": float(coefficients[0]),
            "adequacy_bottleneck": float(coefficients[1]),
            **{column: float(coefficients[index + 2]) for index, column in enumerate(integrity_columns)},
        },
        "empirical_reference_distributions": {
            column: sorted(float(value) for value in usable[column].to_numpy(float))
            for column in [
                "gap_section_coverage",
                "gap_volume_per_section",
                "contrast_section_coverage",
                "contrast_volume_per_section",
                "synthesis_section_coverage",
                "synthesis_volume_per_section",
                "contribution_grounding",
                "reference_ppr_mass",
                "topic_community_coverage",
            ]
        },
    }
    (args.output_dir / "fitted_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics["all_labeled_n"] = int(len(all_df))
    metrics["graph_usable_n"] = int(len(usable))
    metrics["graph_usable_accepted"] = int(usable["accepted"].sum())
    metrics["graph_usable_rejected"] = int((1 - usable["accepted"]).sum())
    (args.output_dir / "fit_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "folds"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
