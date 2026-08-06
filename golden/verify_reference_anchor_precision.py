"""Evaluate PPR-ranked reference candidates on the existing test outputs.

The saved artifacts do not serialize ContentParser's parsed topic objects.  We
therefore use the serialized CONTENT leaf sections as explicit, reproducible
subtopic units and record that provenance in the output.  Citation keys in the
section paragraphs provide the PPR personalization seeds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from agent.tools.scope.missing_papers import ReferenceAnchorClient
from agent.tools.utility.paper_elements import Paper
from agent.tools.utility.tool_config import ToolConfig


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_sections(sections: list[dict[str, Any]]):
    for section in sections:
        yield section
        yield from iter_sections(section.get("sections", []) or [])


def paragraph_sentences(section: dict[str, Any]) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    for paragraph in section.get("paragraphs", []) or []:
        if isinstance(paragraph, list):
            sentences.extend(item for item in paragraph if isinstance(item, dict))
        elif isinstance(paragraph, dict):
            sentences.append(paragraph)
    return sentences


def section_context(section: dict[str, Any]) -> str:
    text = " ".join(str(item.get("text", "")) for item in paragraph_sentences(section))
    return text[:6000]


def citation_seed_nodes(section: dict[str, Any], pool: dict[str, Any]) -> set[str]:
    key_to_node = {
        str(key): node
        for node, item in pool.items()
        for key in item.get("citation_keys", []) or []
    }
    seeds: set[str] = set()
    for sentence in paragraph_sentences(section):
        for key in (sentence.get("citations") or {}).keys():
            if str(key) in key_to_node:
                seeds.add(key_to_node[str(key)])
    return seeds


def paper_data(pool: dict[str, Any], node: str) -> dict[str, Any]:
    return pool[node].get("paper", pool[node])


def ranked_candidates(section: dict[str, Any], pool: dict[str, Any], graph_data: dict[str, Any], top_n: int = 50):
    graph = nx.DiGraph()
    graph.add_nodes_from(graph_data.get("nodes", []) or pool.keys())
    graph.add_edges_from(
        (str(edge["source"]), str(edge["target"]))
        for edge in graph_data.get("edges", []) or []
        if edge.get("source") in graph and edge.get("target") in graph
    )
    seeds = citation_seed_nodes(section, pool) & set(graph)
    if not seeds:
        return [], [], 0
    personalization = {node: (1.0 / len(seeds) if node in seeds else 0.0) for node in graph}
    scores = nx.pagerank(graph, personalization=personalization, dangling=personalization, max_iter=500)
    cited = {node for node, item in pool.items() if item.get("label") == "cited_papers"}
    rows = []
    for rank, node in enumerate(sorted(scores, key=lambda value: (scores[value], value), reverse=True), 1):
        if node in cited or node not in pool:
            continue
        paper = paper_data(pool, node)
        if not paper.get("title") or not paper.get("abstract"):
            continue
        rows.append({
            "rank": len(rows) + 1,
            "node": node,
            "ppr": float(scores[node]),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
        })
        if len(rows) >= top_n:
            break
    return rows, sorted(seeds), len(cited)


def subtopics(classified: dict[str, Any], pool: dict[str, Any]):
    """Return serialized CONTENT leaf sections with at least one citation seed."""
    result = []
    for section in iter_sections(classified.get("sections", []) or []):
        children = section.get("sections", []) or []
        if children or section.get("functional_type") != "CONTENT":
            continue
        seeds = citation_seed_nodes(section, pool)
        if not seeds:
            continue
        result.append({
            "topic": str(section.get("title", "")).strip(),
            "section_id": str(section.get("section_id", "")),
            "context": section_context(section),
            "section": section,
        })
    return result


async def judge_one(client: ReferenceAnchorClient, query: str, item: dict[str, Any]):
    papers = item["candidates"]
    ids = [paper["node"] for paper in papers]
    abstracts = {paper["node"]: paper["abstract"] for paper in papers}
    payload = json.dumps([
        {"paper_id": paper["node"], "title": paper["title"], "abstract": paper["abstract"]}
        for paper in papers
    ], ensure_ascii=False)
    result = await client.call(inputs={
        "query": query,
        "topic": item["topic"],
        "section_context": item["context"],
        "candidate_papers": payload,
        "paper_ids": ids,
        "abstracts": abstracts,
    })
    decisions = {row["paper_id"]: row for row in result["papers"]}
    for paper in papers:
        decision = decisions[paper["node"]]
        paper["relevant"] = bool(decision["relevant"])
        paper["reason"] = decision["reason"]
        paper["verbatim_evidence"] = decision["verbatim_evidence"]
    return item


def summarize_topic(item: dict[str, Any]) -> dict[str, Any]:
    rows = item["candidates"]
    return {
        "topic": item["topic"],
        "section_id": item["section_id"],
        "candidate_count": len(rows),
        "relevant_count": sum(bool(row.get("relevant")) for row in rows),
        "relevance_at": {
            str(k): (sum(bool(row.get("relevant")) for row in rows[:k]) / min(k, len(rows)))
            if rows else None
            for k in (3, 5, 10, 50)
        },
        "candidates": rows,
    }


async def run(args):
    config = ToolConfig.from_yaml(args.config)
    client = ReferenceAnchorClient(config)
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    survey_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and (path / "02_classified_paper.json").exists() and (path / "04_literature_pool.json").exists()
    )
    if args.max_surveys:
        survey_dirs = survey_dirs[:args.max_surveys]
    for survey_dir in survey_dirs:
        destination = output / survey_dir.name / "reference_anchor_top50.json"
        if destination.exists() and not args.force:
            continue
        classified = load_json(survey_dir / "02_classified_paper.json")
        pool_data = load_json(survey_dir / "04_literature_pool.json")
        pool = pool_data.get("literature_pool", {})
        graph = pool_data.get("citation_graph", {})
        query = survey_dir.name.replace("_", " ")
        work = []
        for topic in subtopics(classified, pool):
            candidates, seeds, cited_count = ranked_candidates(topic["section"], pool, graph, 50)
            if candidates:
                work.append({
                    "topic": topic["topic"], "section_id": topic["section_id"],
                    "context": topic["context"], "seed_nodes": seeds,
                    "cited_node_count": cited_count, "candidates": candidates,
                })
        logging.info("%s: %d subtopics, %d candidates", survey_dir.name, len(work), sum(len(x["candidates"]) for x in work))
        judged = []
        for start in range(0, len(work), args.concurrency):
            results = await asyncio.gather(*(judge_one(client, query, item) for item in work[start:start + args.concurrency]), return_exceptions=True)
            for item, result in zip(work[start:start + args.concurrency], results):
                if isinstance(result, Exception):
                    logging.error("%s / %s failed: %s", survey_dir.name, item["topic"], result)
                    item["llm_error"] = repr(result)
                    judged.append(item)
                else:
                    judged.append(result)
        payload = {
            "survey": survey_dir.name,
            "query": query,
            "subtopic_source": "serialized CONTENT leaf sections; parsed topic objects were not serialized in 02_classified_paper.json",
            "ranking": "directed Personalized PageRank seeded by citations in each subtopic section; all cited_papers excluded",
            "accuracy_definition": "PPR-ranked relevance@K, where relevance is the LLM judgment that the paper merits citation for this subtopic",
            "topics": [summarize_topic(item) for item in judged],
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="agent/test_output/codex_surveys")
    parser.add_argument("--output", default="agent/test_output/reference_anchor_validation")
    parser.add_argument("--config", default="agent.yaml")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-surveys", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
