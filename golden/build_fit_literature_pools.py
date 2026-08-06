from __future__ import annotations

"""Rebuild bounded OpenAlex graph pools for acceptance calibration.

The historical R5 artifact retained only aggregate statistics, not graph nodes
or edges.  This script therefore makes one 200-result OpenAlex search per
survey, matches the returned works to local references, and materializes a
bounded directed citation graph for later PPR and Leiden computation.
"""

import argparse
import asyncio
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from survey_eval.agent.tools.utility.openalex import SessionManager, get_openalex_client, index_to_abstract
from survey_eval.agent.tools.utility.tool_config import ToolConfig


STOPWORDS = {
    "a", "an", "and", "approach", "based", "by", "for", "from", "in", "into", "literature",
    "of", "on", "perspective", "review", "survey", "systematic", "the", "to", "toward",
    "towards", "with",
}
SURVEY_MARKERS = re.compile(
    r"\b(a|an|the)?\s*(systematic|comprehensive|scoping|literature)?\s*(survey|review)\b",
    re.IGNORECASE,
)
PUNCTUATION = re.compile(r"[^a-z0-9]+")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return " ".join(PUNCTUATION.sub(" ", str(text or "").lower()).split())


def query_terms(query: str) -> list[str]:
    return [term for term in normalize(query).split() if len(term) > 2 and term not in STOPWORDS]


def fallback_query(title: str) -> str:
    """Extract a broad title-derived field query rather than brittle keyphrases."""

    title = re.sub(r"\s+", " ", str(title or "")).strip()
    title = SURVEY_MARKERS.sub(" ", title)
    title = re.sub(r"\b(an overview|state of the art|a tutorial)\b", " ", title, flags=re.I)
    if ":" in title:
        head, tail = title.split(":", 1)
        if SURVEY_MARKERS.search(tail) or len(query_terms(head)) >= 2:
            title = head
    title = re.sub(r"^[^:]{0,18}:\s*", "", title) if title.lower().startswith(("survey:", "review:")) else title
    focus = re.search(r"\b(?:of|on|for)\s+(.+?)(?:\s+\b(?:in|with|under|using|from|through|across)\b|$)", title, re.I)
    candidate = focus.group(1) if focus else title
    tokens = query_terms(candidate)
    if len(tokens) < 2:
        tokens = query_terms(title)
    if len(tokens) < 2:
        tokens = [token for token in normalize(title).split() if len(token) > 2]
    return " ".join(tokens[:6])


def historic_queries(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    queries: dict[int, Counter[str]] = defaultdict(Counter)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("paper_id"), int) or not row.get("query"):
            continue
        queries[row["paper_id"]][str(row["query"]).strip()] += 1
    return {paper_id: counts.most_common(1)[0][0] for paper_id, counts in queries.items()}


def citation_titles(record: dict[str, Any]) -> dict[str, str]:
    citations = (record.get("paper") or {}).get("citations") or {}
    if not isinstance(citations, dict):
        return {}
    return {
        str(key): str(value.get("title") or "")
        for key, value in citations.items()
        if isinstance(value, dict) and str(value.get("title") or "").strip()
    }


def citation_keys_by_section(record: dict[str, Any]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)

    def visit(section: dict[str, Any], inherited: str) -> None:
        title = str(section.get("title") or inherited)
        for paragraph in section.get("paragraphs") or []:
            for sentence in paragraph or []:
                if not isinstance(sentence, dict):
                    continue
                citations = sentence.get("citations") or []
                if isinstance(citations, dict):
                    citations = citations.values()
                for citation in citations:
                    if isinstance(citation, dict):
                        key = citation.get("key") or citation.get("xml_id")
                    else:
                        key = citation
                    if key:
                        output[title].add(str(key))
        for child in section.get("sections") or []:
            if isinstance(child, dict):
                visit(child, title)

    for section in (record.get("paper") or {}).get("sections") or []:
        if isinstance(section, dict):
            visit(section, "")
    return dict(output)


def result_text(work: dict[str, Any]) -> str:
    return f"{work.get('title') or ''} {index_to_abstract(work.get('abstract_inverted_index')) or ''}".lower()


def work_key(work: dict[str, Any]) -> str:
    return str(work.get("id") or "").rsplit("/", 1)[-1]


def match_local_citations(
    citation_map: dict[str, str],
    works: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, float]]:
    """Map local references to query results using exact then conservative fuzzy title matching."""

    normalized = {work_key(work): normalize(str(work.get("title") or "")) for work in works}
    exact = defaultdict(list)
    for key, title in normalized.items():
        if title:
            exact[title].append(key)
    matched: dict[str, str] = {}
    scores: dict[str, float] = {}
    candidates = list(normalized.items())
    for citation_key, title in citation_map.items():
        local = normalize(title)
        if not local:
            continue
        if local in exact:
            matched[citation_key] = exact[local][0]
            scores[citation_key] = 100.0
            continue
        best_key, best_score = "", 0.0
        for candidate_key, candidate_title in candidates:
            score = fuzz.token_set_ratio(local, candidate_title)
            if score > best_score:
                best_key, best_score = candidate_key, score
        if best_score >= 92.0:
            matched[citation_key] = best_key
            scores[citation_key] = best_score
    return matched, scores


def select_pool(
    works: list[dict[str, Any]],
    query: str,
    citation_matches: dict[str, str],
    max_nodes: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]], bool]:
    """Keep matched citations first, then query results, then referenced neighbors."""

    terms = query_terms(query)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for rank, work in enumerate(works):
        text = result_text(work)
        overlap = sum(term in text for term in terms)
        scored.append((overlap, -rank, work))
    filtered = [work for overlap, _rank, work in scored if overlap > 0]
    relaxed = not filtered
    selected_results = filtered if filtered else works

    lookup = {work_key(work): work for work in works if work_key(work)}
    selected_ids: list[str] = []
    for work_id in citation_matches.values():
        if work_id in lookup and work_id not in selected_ids:
            selected_ids.append(work_id)
    for work in selected_results:
        key = work_key(work)
        if key and key not in selected_ids and len(selected_ids) < max_nodes:
            selected_ids.append(key)

    neighbor_sources: dict[str, list[str]] = defaultdict(list)
    for source_key in selected_ids:
        source = lookup.get(source_key)
        if not source:
            continue
        for raw_target in source.get("referenced_works") or []:
            target = str(raw_target).rsplit("/", 1)[-1]
            if target and target not in selected_ids:
                neighbor_sources[target].append(source_key)
    ranked_neighbors = sorted(neighbor_sources, key=lambda key: (-len(neighbor_sources[key]), key))
    for neighbor in ranked_neighbors:
        if len(selected_ids) >= max_nodes:
            break
        selected_ids.append(neighbor)

    cited_ids = set(citation_matches.values())
    pool: dict[str, dict[str, Any]] = {}
    for key in selected_ids:
        work = lookup.get(key)
        if work:
            label = "cited_papers" if key in cited_ids else "query_search"
            paper = work
        else:
            label = "referenced_neighbor"
            paper = {"id": f"https://openalex.org/{key}", "referenced_works": []}
        pool[key] = {
            "paper": paper,
            "label": label,
            "expanded_from": sorted(neighbor_sources.get(key, [])),
        }

    edge_set: set[tuple[str, str]] = set()
    for source_key, item in pool.items():
        for raw_target in item["paper"].get("referenced_works") or []:
            target_key = str(raw_target).rsplit("/", 1)[-1]
            if target_key in pool and target_key != source_key:
                edge_set.add((source_key, target_key))
    edges = [{"source": source, "target": target} for source, target in sorted(edge_set)]
    return pool, edges, relaxed


async def build_one(
    client: Any,
    source_path: Path,
    output_path: Path,
    historic: dict[int, str],
    max_nodes: int,
) -> dict[str, Any]:
    match = re.match(r"^(\d+)_", source_path.name)
    assert match
    paper_id = int(match.group(1))
    record = read_json(source_path)
    title = str(record.get("paper_title") or "")
    query = historic.get(paper_id) or fallback_query(title)
    if len(query_terms(query)) < 2:
        query = normalize(title)
    payload = await client.search_works(
        search=query,
        per_page=200,
        select="id,title,doi,cited_by_count,publication_date,abstract_inverted_index,referenced_works",
    )
    works = payload.get("results") or []
    citations = citation_titles(record)
    matches, scores = match_local_citations(citations, works)
    pool, edges, relaxed = select_pool(works, query, matches, max_nodes=max_nodes)
    sections = citation_keys_by_section(record)
    section_work_ids = {
        section: sorted({matches[key] for key in keys if key in matches})
        for section, keys in sections.items()
    }
    output = {
        "paper_id": paper_id,
        "paper_title": title,
        "query": query,
        "query_source": "r5_history" if paper_id in historic else "title_rule",
        "openalex_result_count": int(payload.get("count") or 0),
        "returned_results": len(works),
        "keyword_filter_relaxed": relaxed,
        "local_citation_count": len(citations),
        "matched_citation_count": len(matches),
        "citation_matches": matches,
        "citation_match_scores": scores,
        "section_citation_matches": section_work_ids,
        "literature_pool": pool,
        "citation_graph": {"nodes": sorted(pool), "edges": edges},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    return {
        "paper_id": paper_id,
        "title": title,
        "query": query,
        "query_source": output["query_source"],
        "returned_results": len(works),
        "pool_nodes": len(pool),
        "graph_edges": len(edges),
        "local_citations": len(citations),
        "matched_citations": len(matches),
        "keyword_filter_relaxed": relaxed,
        "status": "ok",
    }


async def main_async(args: argparse.Namespace) -> None:
    config = ToolConfig.from_yaml(args.config)
    await SessionManager.init()
    client = get_openalex_client(config)
    historic = historic_queries(args.r5_history)
    source_paths = sorted(args.pdf_content.glob("*.json"))
    if args.limit:
        source_paths = source_paths[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    rows = []
    for source_path in source_paths:
        paper_id = int(re.match(r"^(\d+)_", source_path.name).group(1))
        output_path = args.output_dir / f"{paper_id:03d}_pool.json"
        if output_path.exists() and not args.refresh:
            data = read_json(output_path)
            rows.append(
                {
                    "paper_id": paper_id,
                    "title": data.get("paper_title"),
                    "query": data.get("query"),
                    "query_source": data.get("query_source"),
                    "returned_results": data.get("returned_results"),
                    "pool_nodes": len(data.get("literature_pool") or {}),
                    "graph_edges": len((data.get("citation_graph") or {}).get("edges") or []),
                    "local_citations": data.get("local_citation_count"),
                    "matched_citations": data.get("matched_citation_count"),
                    "keyword_filter_relaxed": data.get("keyword_filter_relaxed"),
                    "status": "cached",
                }
            )
            continue
        try:
            row = await build_one(client, source_path, output_path, historic, args.max_nodes)
        except Exception as exc:
            row = {"paper_id": paper_id, "title": read_json(source_path).get("paper_title"), "status": f"error: {exc}"}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)

    fieldnames = sorted({key for row in rows for key in row})
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    await SessionManager.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-content", type=Path, default=Path(__file__).parent / "pdf_content")
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "agent.yaml")
    parser.add_argument("--r5-history", type=Path, default=Path(__file__).parent / "R5-openalex.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "acceptance_fit" / "literature_pools_v2")
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
