from __future__ import annotations

"""Diagnose reference-anchoring thresholds against reviewer-mentioned cases.

This is deliberately a retrieval diagnostic, not a precision benchmark.  A
reviewer-mentioned paper is a positive case, while an unmentioned paper in the
pool is *unlabelled*, not a verified negative.  The script therefore reports
case recall and availability separately and labels the optional candidate-level
F1 as a proxy.

The pool files are the materialized output of the literature-pool builder.  We
reconstruct the directed graph, apply the evaluation-date cutoff, remove works
already cited by the survey, and rank the remaining works with PPR seeded by
the survey's cited references.  Raw pool contents are copied to ``raw_pools``
for inspection before any cutoff or ranking filter is applied.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
from rapidfuzz import fuzz


PUNCT = re.compile(r"[^a-z0-9]+")


def norm(value: Any) -> str:
    return " ".join(PUNCT.sub(" ", str(value or "").lower()).split())


def title_tokens(value: Any) -> set[str]:
    return {token for token in norm(value).split() if len(token) > 2}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paper_date(record: dict[str, Any]) -> str:
    value = record.get("submission_date") or ""
    if value:
        return str(value)[:10]
    dates = [int(review.get("cdate")) for review in record.get("reviews", []) if review.get("cdate")]
    if dates:
        return datetime.fromtimestamp(min(dates) / 1000, tz=timezone.utc).date().isoformat()
    return "9999-12-31"


def paper_id_from_path(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def cited_titles(pdf_record: dict[str, Any]) -> set[str]:
    citations = ((pdf_record.get("paper") or {}).get("citations") or {}).values()
    return {norm(item.get("title")) for item in citations if isinstance(item, dict) and item.get("title")}


def candidate_text(item: dict[str, Any]) -> str:
    paper = item.get("paper", item)
    return f"{paper.get('title', '')} {paper.get('abstract', '')}"


def candidate_id(item: dict[str, Any], fallback: str) -> str:
    paper = item.get("paper", item)
    return str(paper.get("id") or fallback).rsplit("/", 1)[-1]


def publication_date(item: dict[str, Any]) -> str:
    paper = item.get("paper", item)
    return str(paper.get("publication_date") or paper.get("publicationDate") or "9999-12-31")[:10]


def load_cases(review_dir: Path) -> dict[int, list[dict[str, Any]]]:
    cases: dict[int, list[dict[str, Any]]] = defaultdict(list)
    path = review_dir / "missing_specific_references.jsonl"
    for row in load_jsonl(path):
        raw = row.get("missed_references") or []
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            title = value.get("title") if isinstance(value, dict) else value
            if title and norm(title):
                cases[int(row["paper_id"])].append({
                    "reviewer": row.get("reviewer", ""),
                    "title": str(title),
                    "evidence": row.get("evidence", ""),
                })
    return cases


def dedupe_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for case in cases:
        key = (case["reviewer"], norm(case["title"]))
        if key not in seen:
            seen.add(key)
            result.append(case)
    return result


def graph_and_candidates(pool_data: dict[str, Any], eval_date: str, cited: set[str]) -> tuple[nx.DiGraph, list[dict[str, Any]], dict[str, str]]:
    raw_pool = pool_data.get("literature_pool") or {}
    graph_data = pool_data.get("citation_graph") or {}
    graph = nx.DiGraph()
    graph.add_nodes_from(str(node) for node in (graph_data.get("nodes") or raw_pool.keys()))
    graph.add_edges_from((str(edge["source"]), str(edge["target"])) for edge in graph_data.get("edges", []) if edge.get("source") and edge.get("target"))
    cited_ids = set()
    cited_titles_by_id: dict[str, str] = {}
    for node, item in raw_pool.items():
        title = norm((item.get("paper") or {}).get("title"))
        if item.get("label") == "cited_papers" or title in cited:
            cited_ids.add(str(node))
            cited_titles_by_id[str(node)] = title
    cutoff = eval_date or "9999-12-31"
    candidates = []
    for node, item in raw_pool.items():
        node = str(node)
        date = publication_date(item)
        if date > cutoff or node in cited_ids:
            continue
        paper = item.get("paper") or item
        if not paper.get("title"):
            continue
        candidates.append({"node": node, "title": paper.get("title", ""), "abstract": paper.get("abstract", ""), "publication_date": date, "cited_by_count": int(paper.get("cited_by_count") or 0), "label": item.get("label", "")})
    return graph, candidates, cited_titles_by_id


def fuzzy_match(title: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    query = norm(title)
    exact = [item for item in candidates if norm(item["title"]) == query]
    if exact:
        return exact[0]
    best = max(candidates, key=lambda item: fuzz.token_set_ratio(query, norm(item["title"])), default=None)
    if best and fuzz.token_set_ratio(query, norm(best["title"])) >= 90:
        return best
    return None


def ppr_rank(graph: nx.DiGraph, candidates: list[dict[str, Any]], cited_ids: set[str]) -> list[dict[str, Any]]:
    nodes = set(graph) | {item["node"] for item in candidates}
    graph = graph.subgraph(nodes).copy()
    seeds = cited_ids & set(graph)
    if not seeds:
        return [{**item, "ppr": 0.0, "rank": index + 1} for index, item in enumerate(sorted(candidates, key=lambda x: (-x["cited_by_count"], x["node"]))) ]
    personalization = {node: (1.0 / len(seeds) if node in seeds else 0.0) for node in graph}
    scores = nx.pagerank(graph, personalization=personalization, dangling=personalization, max_iter=200)
    ranked = sorted(candidates, key=lambda item: (scores.get(item["node"], 0.0), item["cited_by_count"], item["node"]), reverse=True)
    return [{**item, "ppr": float(scores.get(item["node"], 0.0)), "rank": rank} for rank, item in enumerate(ranked, 1)]


def recall_curve(ranked: list[dict[str, Any]], matched_titles: set[str], raw_case_count: int) -> list[dict[str, Any]]:
    positives = {item["node"] for item in ranked if norm(item["title"]) in matched_titles}
    scores = sorted({item["ppr"] for item in ranked}, reverse=True)
    thresholds = [float(value) for value in scores]
    if not thresholds:
        thresholds = [0.0]
    rows = []
    for threshold in thresholds:
        selected = [item for item in ranked if item["ppr"] >= threshold]
        hit = len({item["node"] for item in selected} & positives)
        rows.append({"threshold": threshold, "selected_candidates": len(selected), "matched_cases_recalled": hit, "pool_available_cases": len(positives), "raw_recall": hit / raw_case_count if raw_case_count else 0.0, "available_recall": hit / len(positives) if positives else 0.0})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-content", type=Path, default=Path(__file__).parent / "pdf_content")
    parser.add_argument("--review-dir", type=Path, default=Path(__file__).parent / "review_analyze")
    parser.add_argument("--pool-dir", type=Path, default=Path(__file__).parent / "acceptance_fit" / "literature_pools_v2")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "reference_threshold_diagnostic")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "raw_pools").mkdir(exist_ok=True)
    cases_by_paper = {paper_id: dedupe_cases(cases) for paper_id, cases in load_cases(args.review_dir).items()}
    summary, curve_rows, case_rows = [], [], []
    for pdf_path in sorted(args.pdf_content.glob("*.json")):
        paper_id = paper_id_from_path(pdf_path)
        pool_path = args.pool_dir / f"{paper_id:03d}_pool.json"
        if not pool_path.exists():
            continue
        survey = json.loads(pdf_path.read_text(encoding="utf-8"))
        pool_data = json.loads(pool_path.read_text(encoding="utf-8"))
        raw_pool = pool_data.get("literature_pool") or {}
        (args.output_dir / "raw_pools" / pool_path.name).write_text(json.dumps({"paper_id": paper_id, "eval_date": paper_date(survey), "source": str(pool_path), "literature_pool": raw_pool, "citation_graph": pool_data.get("citation_graph", {})}, ensure_ascii=False), encoding="utf-8")
        cited = cited_titles(survey)
        graph, candidates, cited_ids = graph_and_candidates(pool_data, paper_date(survey), cited)
        ranked = ppr_rank(graph, candidates, set(cited_ids))
        case_list = cases_by_paper.get(paper_id, [])
        matched_titles = set()
        already_cited = 0
        unavailable = 0
        for case in case_list:
            title = norm(case["title"])
            if title in cited:
                already_cited += 1
                case_rows.append({"paper_id": paper_id, **case, "status": "already_cited", "matched_node": "", "rank": "", "ppr": ""})
                continue
            match = fuzzy_match(case["title"], candidates)
            if not match:
                unavailable += 1
                case_rows.append({"paper_id": paper_id, **case, "status": "not_in_pool_after_cutoff", "matched_node": "", "rank": "", "ppr": ""})
                continue
            matched_titles.add(norm(match["title"]))
            case_rows.append({"paper_id": paper_id, **case, "status": "pool_candidate", "matched_node": match["node"], "rank": next(item["rank"] for item in ranked if item["node"] == match["node"]), "ppr": next(item["ppr"] for item in ranked if item["node"] == match["node"])})
        curve = recall_curve(ranked, matched_titles, len(case_list))
        for row in curve:
            curve_rows.append({"paper_id": paper_id, "eval_date": paper_date(survey), **row})
        summary.append({"paper_id": paper_id, "eval_date": paper_date(survey), "raw_pool_size": len(raw_pool), "post_cutoff_candidates": len(candidates), "reviewer_cases": len(case_list), "already_cited": already_cited, "pool_available": len(matched_titles), "not_in_pool_after_cutoff": unavailable, "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges()})
    with (args.output_dir / "paper_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summary for key in row})); writer.writeheader(); writer.writerows(summary)
    with (args.output_dir / "case_matches.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in case_rows for key in row})); writer.writeheader(); writer.writerows(case_rows)
    with (args.output_dir / "threshold_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in curve_rows for key in row})); writer.writeheader(); writer.writerows(curve_rows)
    aggregate = {"papers": len(summary), "reviewer_cases": sum(row["reviewer_cases"] for row in summary), "already_cited": sum(row["already_cited"] for row in summary), "pool_available": sum(row["pool_available"] for row in summary), "not_in_pool_after_cutoff": sum(row["not_in_pool_after_cutoff"] for row in summary), "warning": "Unmentioned pool candidates are unlabeled; this output reports recall/availability, not a valid precision or F1 estimate."}
    (args.output_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
