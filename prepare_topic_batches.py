from __future__ import annotations

import argparse
import asyncio
import random
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiohttp

from agent.tools.preprocess.get_reference_surveys import ReferenceSurveySelect
from agent.tools.utility.openalex import get_openalex_client
from agent.tools.utility.request_utils import HEADERS, SessionManager
from agent.tools.utility.s2 import get_semantic_scholar_client
from agent.tools.utility.tool_config import ToolConfig


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
TOPICS_TXT = ROOT / "topics.txt"
TOPICS_JSONL = ROOT / "topics.jsonl"
CACHE_JSON = ROOT / "reference_survey_llm_cache.json"
SAMPLE_JSON = ROOT / "topics_sample.json"

DOMAIN_ALIASES = {
    "NLP": "NLP",
    "Computer Vision": "CV",
    "Speech & Audio": "Speech",
    "Multimodal": "Multimodal",
    "Core ML Methods": "Core ML",
    "Extended ML Methods": "Extended ML",
    "Reinforcement Learning Subtopics": "RL",
    "Efficiency & Compression": "Efficiency",
    "Knowledge & Data": "Knowledge",
    "AI for Science & Applications": "Applications",
    "Safety & Trustworthiness": "Safety",
    "Cross-cutting": "Cross-cutting",
}

DOMAIN_QUOTAS = {
    "NLP": 7,
    "CV": 6,
    "Core ML": 5,
    "Extended ML": 2,
    "Applications": 2,
    "Knowledge": 2,
    "RL": 1,
    "Speech": 1,
    "Multimodal": 1,
    "Efficiency": 1,
    "Safety": 1,
    "Cross-cutting": 1,
}

SURVEY_TERMS = ("survey", "review", "overview", "summary", "comprehensive study")
TITLE_CASE_SMALL = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "via", "with"}
ACRONYMS = {
    "3D",
    "AI",
    "ASR",
    "BERT",
    "CLIP",
    "CV",
    "DDPM",
    "FL",
    "GAN",
    "GNN",
    "GPT",
    "KGE",
    "LLM",
    "MARL",
    "ML",
    "NAS",
    "NER",
    "NLP",
    "NLI",
    "NMT",
    "OCR",
    "OOD",
    "QA",
    "RAG",
    "RL",
    "RLHF",
    "SQL",
    "TTS",
    "VQA",
    "YOLO",
}


def normalize_topic_name(raw_title: str) -> str:
    name = re.sub(r"^A\s+SURVEY\s+ON\s+", "", raw_title.strip(), flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    words = []
    for idx, token in enumerate(name.split(" ")):
        pieces = token.split("-")
        normalized_pieces = []
        for piece in pieces:
            upper = piece.upper()
            lower = piece.lower()
            if upper in ACRONYMS:
                normalized_pieces.append(upper)
            elif lower in TITLE_CASE_SMALL and idx != 0:
                normalized_pieces.append(lower)
            else:
                normalized_pieces.append(piece[:1].upper() + piece[1:].lower())
        words.append("-".join(normalized_pieces))
    return " ".join(words)


def parse_topics(path: Path = TOPICS_TXT) -> list[dict[str, Any]]:
    topics: list[dict[str, Any]] = []
    domain = ""
    heading_re = re.compile(r"^#\s*[^\w]*(.+?)\s*\(\d+[^)]*\)")
    run_re = re.compile(r'^\./run\.sh\s+"([^"]+)"\s+"([^"]+)"\s+(\d+)')
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = heading_re.match(line)
            if match:
                heading = match.group(1).strip()
                domain = DOMAIN_ALIASES.get(heading, heading)
            continue
        match = run_re.match(line)
        if not match:
            continue
        title, keywords_text, num_papers = match.groups()
        topics.append(
            {
                "name": normalize_topic_name(title),
                "surveyg_title": title,
                "keywords": [item.strip() for item in keywords_text.split(",") if item.strip()],
                "keywords_text": keywords_text,
                "num_papers": int(num_papers),
                "domain": domain,
            }
        )
    return topics


def paper_key(paper: dict[str, Any]) -> str:
    external = paper.get("external_ids") or paper.get("externalIds") or {}
    for key in ("DOI", "doi", "ArXiv", "arXiv", "CorpusId", "MAG"):
        value = external.get(key)
        if value:
            return f"{key.lower()}:{str(value).lower()}"
    title = re.sub(r"[^a-z0-9]+", " ", (paper.get("title") or "").lower()).strip()
    year = str(paper.get("year") or (paper.get("publication_date") or "")[:4])
    return f"title:{title}:{year}"


def dedupe_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for paper in papers:
        if not (paper.get("title") or "").strip():
            continue
        key = paper_key(paper)
        if key not in merged:
            merged[key] = dict(paper)
            continue
        sources = set(str(merged[key].get("source", "")).split("+"))
        sources.add(str(paper.get("source", "")))
        merged[key]["source"] = "+".join(sorted(x for x in sources if x))
    return sorted(
        merged.values(),
        key=lambda paper: int(paper.get("cited_by_count") or paper.get("citationCount") or 0),
        reverse=True,
    )


def normalize_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def strict_reference_survey_count(cache_item: dict[str, Any]) -> int:
    titles = {
        normalize_title_key(item.get("title", ""))
        for item in cache_item.get("strict_reference_surveys", []) or []
        if normalize_title_key(item.get("title", ""))
    }
    return len(titles)


def is_reference_survey_candidate(paper: dict[str, Any]) -> bool:
    title = (paper.get("title") or "").lower()
    if not any(term in title for term in SURVEY_TERMS):
        publication_types = {str(x).lower() for x in paper.get("publicationTypes", []) or []}
        publication_types |= {str(x).lower() for x in paper.get("publication_types", []) or []}
        if "review" not in publication_types:
            return False
    return int(paper.get("cited_by_count") or paper.get("citationCount") or 0) > 10


async def search_openalex(topic: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    client = get_openalex_client(ToolConfig(default_academic_search_engine="openalex"))
    query = topic["name"]
    try:
        payload = await client.search_works(
            search=query,
            filter={"title.search": 'survey|summary|review|overview|"comprehensive study"'},
            per_page=limit,
            select="id,title,cited_by_count,publication_date,created_date,abstract_inverted_index,authorships,ids,doi",
        )
    except Exception as exc:
        print(f"OpenAlex search skipped {topic['name']} ({exc})", flush=True)
        payload = {"results": []}
    results = []
    for paper in payload.get("results", []) or []:
        if is_reference_survey_candidate(paper):
            item = dict(paper)
            item["source"] = "openalex"
            results.append(item)
    return results


async def search_s2(topic: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    query = f'{topic["name"]} survey review overview'
    client = get_semantic_scholar_client(ToolConfig(default_academic_search_engine="semantic scholar"))
    fields = client._normalize_fields(None)
    params = {"query": query, "offset": 0, "limit": limit}
    if fields:
        params["fields"] = fields
    session = SessionManager.get()
    payload = {}
    while True:
        try:
            async with session.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers=HEADERS,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 429:
                    print(429, end=" ", flush=True)
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                    continue
                resp.raise_for_status()
                payload = await resp.json()
                break
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            print(f"Error in s2 {type(exc)}")
            await asyncio.sleep(random.uniform(1.0, 2.0))
    payload = client._wrap_results(payload, limit=limit)
    results = []
    for paper in payload.get("results", []) or []:
        if is_reference_survey_candidate(paper):
            item = dict(paper)
            item["source"] = "semantic_scholar"
            results.append(item)
    return results


async def count_reference_surveys(topic: dict[str, Any], limit: int) -> dict[str, Any]:
    openalex_task = asyncio.create_task(search_openalex(topic, limit))
    s2_task = asyncio.create_task(search_s2(topic, limit))
    openalex_results, s2_results = await asyncio.gather(openalex_task, s2_task)
    candidates = dedupe_papers([*openalex_results, *s2_results])[:limit]
    # candidates = await search_openalex(topic, limit)
    if not candidates:
        return {
            "reference_surveys_count": 0,
            "candidate_count": 0,
            "openalex_count": len(openalex_results),
            "semantic_scholar_count": len(s2_results),
            "strict_reference_surveys": [],
            "partial_reference_surveys": [],
            "selected_titles": [],
        }

    config = ToolConfig()
    print(f"Topic {topic} to ReferenceSurveySelect")
    selector = ReferenceSurveySelect(config.llm_server_info, config.sampling_params)
    try:
        selected_by_tier = await selector.call(inputs={"query": topic["name"], "surveys": candidates})
    except Exception as exc:
        print(f"LLM reference survey selection failed: {topic['name']} ({exc})", flush=True)
        selected_by_tier = {"strict_reference_surveys": [], "partial_reference_surveys": []}

    selected = dedupe_papers(selected_by_tier["strict_reference_surveys"])
    return {
        "reference_surveys_count": len(selected),
        "candidate_count": len(candidates),
        "openalex_count": len(openalex_results),
        "semantic_scholar_count": len(s2_results),
        "strict_reference_surveys": [
            {
                "title": paper.get("title", ""),
                "reason": paper.get("reference_survey_reason", ""),
                "covered_subtopics": paper.get("covered_subtopics", []),
            }
            for paper in selected_by_tier.get("strict_reference_surveys", []) or []
        ],
        "partial_reference_surveys": [
            {
                "title": paper.get("title", ""),
                "reason": paper.get("reference_survey_reason", ""),
                "covered_subtopics": paper.get("covered_subtopics", []),
            }
            for paper in selected_by_tier.get("partial_reference_surveys", []) or []
        ],
        "selected_titles": [paper.get("title", "") for paper in selected if paper.get("title")],
    }


def load_cache(topics: list[dict[str, Any]], refresh: bool) -> dict[str, Any]:
    cache = {}
    if CACHE_JSON.exists() and not refresh:
        cache = json.loads(CACHE_JSON.read_text(encoding="utf-8"))
    topic_names = {topic["name"] for topic in topics}
    aliases = {
        "3d Point Cloud Processing": "3D Point Cloud Processing",
        "Low-Resource Nlp": "Low-Resource NLP",
        "Multilingual Nlp": "Multilingual NLP",
    }
    cleaned = {}
    for key, value in cache.items():
        normalized_key = aliases.get(key, key)
        if normalized_key in topic_names:
            normalized_value = dict(value)
            normalized_value["reference_surveys_count"] = strict_reference_survey_count(normalized_value)
            cleaned[normalized_key] = normalized_value
    return cleaned


async def enrich_topics(
    topics: list[dict[str, Any]],
    limit: int,
    refresh: bool,
    max_topics: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = load_cache(topics, refresh)
    cache_lock = asyncio.Lock()

    pending = [(idx, topic) for idx, topic in enumerate(topics, 1) if topic["name"] not in cache]
    if max_topics > 0:
        pending = pending[:max_topics]

    async def worker(idx: int, topic: dict[str, Any]) -> None:
        name = topic["name"]
        print(f"[{idx:03d}/{len(topics)}] counting reference surveys: {name}", flush=True)
        result = await count_reference_surveys(topic, limit)
        async with cache_lock:
            cache[name] = result
            CACHE_JSON.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    if pending:
        print(f"Processing {len(pending)} uncached topics", flush=True)
        for idx, topic in pending: await worker(idx, topic)
        # await asyncio.gather(*(worker(idx, topic) for idx, topic in pending))
    else:
        CACHE_JSON.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    if max_topics > 0:
        processed_names = {topic["name"] for _, topic in pending}
        subset = [topic for topic in topics if topic["name"] in processed_names]
        for topic in subset:
            topic.update(cache[topic["name"]])
        return subset, cache

    for topic in topics:
        name = topic["name"]
        topic.update(cache[name])
    return topics, cache


def maturity_bucket(count: int) -> str:
    if count >= 3:
        return "3+"
    if count >= 1:
        return "1-2"
    return "0"


def sample_topics(topics: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    selected_names: set[str] = set()
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for topic in topics:
        by_domain[topic["domain"]].append(topic)

    graph = next(topic for topic in topics if topic["name"] == "Graph Neural Networks")
    selected.append(graph)
    selected_names.add(graph["name"])

    for domain, quota in DOMAIN_QUOTAS.items():
        domain_selected = [topic for topic in selected if topic["domain"] == domain]
        remaining_quota = quota - len(domain_selected)
        if remaining_quota <= 0:
            continue
        candidates = [topic for topic in by_domain[domain] if topic["name"] not in selected_names]
        buckets = defaultdict(list)
        for topic in sorted(candidates, key=lambda x: (-int(x["reference_surveys_count"]), x["name"])):
            buckets[maturity_bucket(int(topic["reference_surveys_count"]))].append(topic)
        picks = []
        while len(picks) < remaining_quota:
            progressed = False
            for bucket in ("3+", "1-2", "0"):
                if len(picks) >= remaining_quota:
                    break
                if buckets[bucket]:
                    picks.append(buckets[bucket].pop(0))
                    progressed = True
            if not progressed:
                break
        for topic in picks:
            selected.append(topic)
            selected_names.add(topic["name"])

    selected = sorted(selected, key=lambda t: (list(DOMAIN_QUOTAS).index(t["domain"]), t["name"]))
    remaining = [topic for topic in topics if topic["name"] not in selected_names]
    return selected, remaining


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(topics: list[dict[str, Any]], selected: list[dict[str, Any]], remaining: list[dict[str, Any]]) -> None:
    jsonl_lines = []
    for topic in topics:
        jsonl_lines.append(
            json.dumps(
                {
                    "name": topic["name"],
                    "keywords": topic["keywords"],
                    "domain": topic["domain"],
                    "reference_surveys_count": int(topic["reference_surveys_count"]),
                },
                ensure_ascii=False,
            )
        )
    write_lines(TOPICS_JSONL, jsonl_lines)

    sample_payload = {
        "selected30": [
            {
                "name": topic["name"],
                "domain": topic["domain"],
                "reference_surveys_count": topic["reference_surveys_count"],
                "maturity_bucket": maturity_bucket(int(topic["reference_surveys_count"])),
            }
            for topic in selected
        ],
        "remaining70": [topic["name"] for topic in remaining],
        "domain_quotas": DOMAIN_QUOTAS,
    }
    SAMPLE_JSON.write_text(json.dumps(sample_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for batch_name, batch in (("30", selected), ("70", remaining)):
        names = [topic["name"] for topic in batch]
        write_lines(ROOT / f"topics{batch_name}.txt", names)

        surveyg_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        for topic in batch:
            surveyg_lines.append(
                f"./run.sh {sh_quote(topic['surveyg_title'])} {sh_quote(topic['keywords_text'])} {topic['num_papers']}"
            )
        write_lines(WORKSPACE / "SurveyG" / f"run{batch_name}.sh", surveyg_lines)

        surveygen_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        for topic in batch:
            surveygen_lines.append(f"python run_pipeline.py --research_topic {sh_quote(topic['name'])}")
        write_lines(WORKSPACE / "SurveyGen-I" / f"run{batch_name}.sh", surveygen_lines)

        arise_dir = WORKSPACE / "Git_projects" / "ARISE" / "ARISE_Source_Code"
        write_lines(arise_dir / f"topics{batch_name}.txt", names)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-topics", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    await SessionManager.init()
    try:
        topics = parse_topics()
        if len(topics) != 100:
            raise RuntimeError(f"Expected 100 topics, parsed {len(topics)}")
        topics, cache = await enrich_topics(
            topics,
            limit=args.limit,
            refresh=args.refresh,
            max_topics=args.max_topics,
        )
        if args.max_topics > 0:
            print(f"Debug run complete for {len(topics)} topics; cache entries={len(cache)}")
            return
        selected, remaining = sample_topics(topics)
        if len(selected) != 30 or len(remaining) != 70:
            raise RuntimeError(f"Expected 30/70 split, got {len(selected)}/{len(remaining)}")
        if "Graph Neural Networks" not in {topic["name"] for topic in selected}:
            raise RuntimeError("Graph Neural Networks must be selected")
        write_outputs(topics, selected, remaining)
        print(f"Wrote {TOPICS_JSONL}")
        print(f"Wrote {SAMPLE_JSON}")
    finally:
        try:
            await asyncio.wait_for(SessionManager.close(), timeout=5)
        except TimeoutError:
            print("Session close timed out; exiting after outputs/cache were written.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
