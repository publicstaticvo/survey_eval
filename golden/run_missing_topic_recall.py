from __future__ import annotations

"""Evaluate missing-topic retrieval against reviewer-mentioned topics.

The script reproduces the production Leiden/PR/PPR community ranking and
uses LLM judgments only for two semantic decisions: paper-to-subtopic
membership and community-to-reviewer-topic matching. Keyword overlap is not
used as a relevance decision.
"""

import argparse
import asyncio
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import igraph as ig
import leidenalg
import networkx as nx
import jsonschema

from agent.tools.scope.missing_topic_detection import MissingTopicDetector
from agent.tools.utility.llmclient import AsyncChat
from agent.tools.utility.request_utils import SessionManager
from agent.tools.utility.tool_config import ToolConfig
from agent.tools.utility.utils import extract_json


PAPER_BATCH_PROMPT = """# Task
You are an expert literature-review analyst. For each candidate paper, decide independently whether its title and abstract concern one of the supplied reviewer-mentioned subtopics. A match is allowed only when the paper studies that subtopic itself or a direct subtopic of it. Reject papers about only a broader parent field, a sibling topic, generic background, or an unrelated use of the same words. Do not use citation proximity as evidence. Return only JSON.

Reviewer-mentioned subtopics:
{topics}

Candidate papers:
{papers}

Output schema:
{{"matches": [{{"paper_id": "...", "topic_ids": ["..."], "reason": "..."}}]}}
Include a candidate only when it matches at least one topic. topic_ids must be copied from the supplied topic IDs. The reason must briefly explain the title/abstract basis.
"""

PAPER_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "minLength": 1},
                    "topic_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["paper_id", "topic_ids", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["matches"],
    "additionalProperties": False,
}

MATCH_PROMPT = """# Task
You are adjudicating whether ranked literature communities retrieve reviewer-mentioned missing topics. A reviewer topic is recalled if at least one retained community contains representative papers that concern that topic or a direct child topic. A community about a broader parent field or a sibling topic is not a match. Return only JSON.

Reviewer topics:
{topics}

Ranked communities:
{communities}

Output schema:
{{"matches": [{{"topic_id": "...", "community_ranks": [1], "reason": "..."}}]}}
Only include supported matches. Use the supplied topic IDs and community ranks exactly.
"""

MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic_id": {"type": "string", "minLength": 1},
                    "community_ranks": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                    "reason": {"type": "string"},
                },
                "required": ["topic_id", "community_ranks", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["matches"],
    "additionalProperties": False,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paper_data(pool_item: dict[str, Any]) -> dict[str, Any]:
    return pool_item.get("paper", pool_item)


def paper_text(paper: dict[str, Any]) -> str:
    return f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')}".strip()


def existing_topics(classified: dict[str, Any]) -> list[str]:
    found: list[str] = []

    def walk(section: dict[str, Any]):
        parsed = section.get("parsed_contents") or {}
        for topic in parsed.get("topics", []) or []:
            if str(topic).strip():
                found.append(str(topic))
        for child in section.get("sections", []) or []:
            if isinstance(child, dict):
                walk(child)

    for section in classified.get("sections", []) or []:
        if isinstance(section, dict):
            walk(section)
    return list(dict.fromkeys(found))


def community_partition(graph: nx.Graph, resolution: float) -> list[frozenset[str]]:
    nodes = list(graph)
    if not nodes:
        return []
    index = {node: i for i, node in enumerate(nodes)}
    edges = [(index[u], index[v]) for u, v in graph.edges()]
    if not edges:
        return [frozenset([node]) for node in nodes]
    partition = leidenalg.find_partition(
        ig.Graph(n=len(nodes), edges=edges, directed=False),
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=0,
    )
    return [frozenset(nodes[i] for i in group) for group in partition]


def rank_communities(pool: dict[str, Any], graph_data: dict[str, Any], config: ToolConfig) -> list[dict[str, Any]]:
    graph = nx.DiGraph()
    graph.add_nodes_from(graph_data.get("nodes", []) or pool.keys())
    graph.add_edges_from(
        (str(edge["source"]), str(edge["target"]))
        for edge in graph_data.get("edges", []) or []
        if edge.get("source") and edge.get("target") and edge["source"] != edge["target"]
    )
    cited = {node for node, item in pool.items() if item.get("label") == "cited_papers"} & set(graph)
    pr = nx.pagerank(graph, max_iter=500)
    personalization = {node: (1.0 / len(cited) if node in cited else 0.0) for node in graph} if cited else None
    ppr = nx.pagerank(graph, personalization=personalization, dangling=personalization, max_iter=500) if cited else {node: 0.0 for node in graph}
    uncited = set(graph) - cited
    undirected = graph.to_undirected().subgraph(uncited).copy()
    minimum = max(config.missing_topic_min_community_size, math.ceil(len(pool) * config.missing_topic_min_community_size_ratio))
    unique: dict[frozenset[str], set[float]] = {}
    for resolution in config.missing_topic_resolutions:
        for community in community_partition(undirected, float(resolution)):
            if len(community) >= minimum:
                unique.setdefault(community, set()).add(float(resolution))
    ranked = []
    for nodes, resolutions in unique.items():
        pr_mass = sum(pr.get(node, 0.0) for node in nodes)
        ppr_mass = sum(ppr.get(node, 0.0) for node in nodes)
        reps = sorted(nodes, key=lambda node: (pr.get(node, 0.0), node), reverse=True)[:config.missing_topic_representative_papers]
        ranked.append({
            "nodes": sorted(nodes),
            "community_size": len(nodes),
            "resolutions": sorted(resolutions),
            "pr_mass": pr_mass,
            "ppr_mass": ppr_mass,
            "missing_score": pr_mass - ppr_mass,
            "papers": [{"node": node, **paper_data(pool[node])} for node in reps],
        })
    ranked.sort(key=lambda row: (row["missing_score"], row["pr_mass"]), reverse=True)
    return ranked


class PaperTopicJudge(AsyncChat):
    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, PAPER_BATCH_SCHEMA)
        valid_ids = set(context["paper_ids"])
        valid_topics = set(context["topic_ids"])
        for item in result["matches"]:
            assert item["paper_id"] in valid_ids
            assert set(item["topic_ids"]) <= valid_topics
        return result

    def _organize_inputs(self, inputs):
        topics = "\n".join(f"- {item['id']}: {item['text']}" for item in inputs["topics"])
        papers = "\n\n".join(f"ID: {item['id']}\n{paper_text(item)}" for item in inputs["papers"])
        return PAPER_BATCH_PROMPT.format(topics=topics, papers=papers), {
            "paper_ids": [item["id"] for item in inputs["papers"]],
            "topic_ids": [item["id"] for item in inputs["topics"]],
        }


class CommunityTopicJudge(AsyncChat):
    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, MATCH_SCHEMA)
        valid_topics = set(context["topic_ids"])
        max_rank = context["max_rank"]
        for item in result["matches"]:
            assert item["topic_id"] in valid_topics
            assert all(1 <= rank <= max_rank for rank in item["community_ranks"])
        return result

    def _organize_inputs(self, inputs):
        topics = "\n".join(f"- {item['id']}: {item['text']}" for item in inputs["topics"])
        communities = "\n\n".join(
            f"Rank {i}: size={c['community_size']}, Miss(C)={c['missing_score']:.8g}\n" +
            "\n".join(f"  - {p.get('title', '')}\n    {str(p.get('abstract', ''))[:1200]}" for p in c["papers"])
            for i, c in enumerate(inputs["communities"], 1)
        )
        return MATCH_PROMPT.format(topics=topics, communities=communities), {
            "topic_ids": [item["id"] for item in inputs["topics"]],
            "max_rank": len(inputs["communities"]),
        }


async def judge_pool_topics(judge: PaperTopicJudge, pool: dict[str, Any], topics: list[dict[str, str]], batch_size: int, concurrency: int) -> dict[str, set[str]]:
    rows = [{"id": node, **paper_data(item)} for node, item in pool.items()]
    results: dict[str, set[str]] = defaultdict(set)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(batch):
        async with semaphore:
            return await judge.call(inputs={"topics": topics, "papers": batch})

    tasks = [asyncio.create_task(one(rows[i:i + batch_size])) for i in range(0, len(rows), batch_size)]
    for task in asyncio.as_completed(tasks):
        result = await task
        for item in result["matches"]:
            for topic_id in item["topic_ids"]:
                results[topic_id].add(item["paper_id"])
    return results


def load_full_pool(path: Path) -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return pool
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item.get("pool_key") or item.get("paper", {}).get("id") or "")
            if key:
                pool[key] = {"label": item.get("label", ""), "paper": item.get("paper", {})}
    return pool


async def process_paper(pid: int, topic_rows: list[dict[str, Any]], args, config: ToolConfig, paper_judge, community_judge):
    pool_path = args.pool_dir / f"{pid:03d}_pool.json"
    classified_paths = list(args.classified_dir.glob(f"{pid:03d}_*/02_classified_paper.json"))
    if not pool_path.exists() or not classified_paths:
        return {"paper_id": pid, "status": "missing_input", "topic_count": len(topic_rows)}
    pool_data = json.loads(pool_path.read_text(encoding="utf-8"))
    pool = pool_data.get("literature_pool", {})
    ranked = rank_communities(pool, pool_data.get("citation_graph", {}), config)
    topics = [{"id": f"t{i}", "text": row["missing_content"]} for i, row in enumerate(topic_rows)]
    matched_nodes = await judge_pool_topics(paper_judge, pool, topics, args.batch_size, args.llm_concurrency)
    communities = ranked[: args.community_limit]
    match = await community_judge.call(inputs={"topics": topics, "communities": communities}) if communities else {"matches": []}
    recalled = {item["topic_id"] for item in match["matches"]}
    unrecalled = [topic for topic in topics if topic["id"] not in recalled]
    full_counts: dict[str, int] = {}
    full_path = args.full_pool_dir / f"{pid:03d}.jsonl"
    if unrecalled and full_path.exists():
        full_pool = load_full_pool(full_path)
        full_pool = dict(sorted(full_pool.items(), key=lambda item: int((item[1].get("paper", {}).get("cited_by_count") or 0)), reverse=True)[:args.full_scan_limit])
        full_matches = await judge_pool_topics(paper_judge, full_pool, unrecalled, args.batch_size, args.llm_concurrency)
        full_counts = {topic["id"]: len(full_matches.get(topic["id"], set())) for topic in unrecalled}
    per_topic = []
    for topic in topics:
        nodes = sorted(matched_nodes.get(topic["id"], set()))
        per_topic.append({
            "topic_id": topic["id"],
            "missing_content": topic["text"],
            "related_in_1000_graph": len(nodes),
            "related_node_ids": nodes,
            "related_in_full_graph": full_counts.get(topic["id"]) if topic["id"] in full_counts else None,
            "recalled": topic["id"] in recalled,
        })
    return {
        "paper_id": pid,
        "status": "ok",
        "topic_count": len(topics),
        "recalled_count": len(recalled),
        "recall": len(recalled) / len(topics) if topics else 1.0,
        "community_count": len(ranked),
        "top_communities": [{k: v for k, v in c.items() if k != "nodes"} for c in communities],
        "topics": per_topic,
    }


async def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    await SessionManager.init()
    config = ToolConfig.from_yaml(args.config)
    rows = load_jsonl(args.review_topics)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_topics: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        pid = int(row["paper_id"])
        for topic in row.get("missed_topics", []) or []:
            if topic.get("category") == "T5":
                continue
            key = re.sub(r"\s+", " ", str(topic.get("missing_content", "")).strip().casefold())
            if key and key not in seen_topics[pid]:
                seen_topics[pid].add(key)
                grouped[pid].append(topic)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                done[int(item["paper_id"])] = item
    paper_judge = PaperTopicJudge(config)
    community_judge = CommunityTopicJudge(config)
    paper_judge.timeout = 90
    community_judge.timeout = 90
    pending = [pid for pid in sorted(grouped) if pid not in done]
    sem = asyncio.Semaphore(args.paper_concurrency)

    async def run(pid):
        async with sem:
            logging.info("Processing paper %03d (%d reviewer topics)", pid, len(grouped[pid]))
            return await process_paper(pid, grouped[pid], args, config, paper_judge, community_judge)

    tasks = [asyncio.create_task(run(pid)) for pid in pending]
    for task in asyncio.as_completed(tasks):
        item = await task
        done[int(item["paper_id"])] = item
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
        logging.info("Finished paper %03d: %s", item["paper_id"], item["status"])
    summary = {
        "total_papers": len(done),
        "ok": sum(item.get("status") == "ok" for item in done.values()),
        "missing_input": sum(item.get("status") == "missing_input" for item in done.values()),
        "topic_count": sum(item.get("topic_count", 0) for item in done.values()),
        "recalled_count": sum(item.get("recalled_count", 0) for item in done.values()),
        "micro_recall": sum(item.get("recalled_count", 0) for item in done.values()) / max(1, sum(item.get("topic_count", 0) for item in done.values())),
        "macro_recall": sum(item.get("recall", 0.0) for item in done.values() if item.get("status") == "ok") / max(1, sum(item.get("status") == "ok" for item in done.values())),
    }
    (output.parent / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    await SessionManager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("survey_eval/agent.yaml"))
    parser.add_argument("--review-topics", type=Path, default=Path("survey_eval/golden/review_analyze/missing_specific_topics.jsonl"))
    parser.add_argument("--pool-dir", type=Path, default=Path("survey_eval/golden/acceptance_fit/literature_pools_v2"))
    parser.add_argument("--full-pool-dir", type=Path, default=Path("survey_eval/golden/openalex_literature_pools"))
    parser.add_argument("--classified-dir", type=Path, default=Path("survey_eval/golden/classified_papers"))
    parser.add_argument("--output", type=Path, default=Path("survey_eval/golden/missing_topic_recall/topic_recall.jsonl"))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--llm-concurrency", type=int, default=8)
    parser.add_argument("--paper-concurrency", type=int, default=2)
    parser.add_argument("--community-limit", type=int, default=20)
    parser.add_argument("--full-scan-limit", type=int, default=2000)
    asyncio.run(main(parser.parse_args()))
