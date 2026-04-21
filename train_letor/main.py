import argparse
import asyncio
import json
import math
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.tools.llmclient import AsyncChat
from agent.tools.openalex import OPENALEX_SELECT, openalex_search_paper, to_openalex
from agent.tools.prompts import QUERY_EXPANSION_PROMPT, SURVEY_SPECIFIED_QUERY_EXPANSION
from agent.tools.query_expand import QUERY_SCHEMA, SURVEY_KEYWORDS
from agent.tools.request_utils import OpenAlexBudgetExceeded, RateLimit, SessionManager
from agent.tools.sbert_client import SentenceTransformerClient
from agent.tools.tool_config import ToolConfig
from agent.tools.utils import cosine_similarity_matrix, extract_json, valid_check


DATASET_PATH = Path(__file__).resolve().parent / "surge.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SURVEY_SELECT = f"{OPENALEX_SELECT},best_oa_location,locations,relevance_score"


class QueryExpansionLLMClient(AsyncChat):
    PROMPT = QUERY_EXPANSION_PROMPT

    def _availability(self, response, context):
        queries = extract_json(response)
        import jsonschema

        jsonschema.validate(queries, QUERY_SCHEMA)
        return queries


class SurveyInfoFetcher:
    def __init__(self, eval_date: datetime):
        self.eval_date = eval_date

    def _normalize_title(self, title: str) -> str:
        title = title.replace("\\\\", "")
        return re.sub(r"\s+", " ", re.sub(r"[:,.!?&]", " ", title or "")).strip()

    async def __call__(self, title: str, publication_date: str) -> Dict[str, Any]:
        normalized_title = self._normalize_title(title)
        results = await openalex_search_paper(
            "works",
            filter={"title.search": normalized_title},
            select=SURVEY_SELECT,
            per_page=10,
        )
        for paper in results.get("results", []):
            if not valid_check(normalized_title, paper.get("title", "")): continue
            return paper


class StrictQueryExpand:
    def __init__(self, config: ToolConfig):
        self.llm = QueryExpansionLLMClient(config.llm_server_info, config.sampling_params)
        self.eval_date = config.evaluation_date

    async def _request_for_papers(self, query: str, uplimit: int, select: str = f"{OPENALEX_SELECT},relevance_score") -> List[Dict[str, Any]]:
        queries = to_openalex(query)
        search = [*[("default.search", q) for q in queries], ("to_publication_date", self.eval_date.strftime("%Y-%m-%d"))]
        papers = []
        total = uplimit
        for page in range(1, (uplimit - 1) // 200 + 2):
            results = await openalex_search_paper("works", search, per_page=min(200, uplimit), select=select, page=page)
            total = min(total, results.get("count", 0) or total)
            papers.extend(results.get("results", []))
            if len(papers) >= total:
                break
        return papers[:uplimit]

    async def __call__(self, query: str, papers_for_each_query: int = 50):
        queries = await self.llm.call(inputs={"query": query})
        expanded_queries = [
            *(queries.get("core_anchor") or [])[:2],
            queries.get("theoretical_bridge"),
            queries.get("methodological_bridge"),
        ]
        library, valid_queries = {}, []
        for expanded_query in expanded_queries:
            papers = await self._request_for_papers(expanded_query, papers_for_each_query)
            if not papers: continue
            valid_queries.append(expanded_query)
            for paper in papers:
                current = library.get(paper["id"])
                if current is None or paper.get("relevance_score", 0) > current.get("relevance_score", 0):
                    library[paper["id"]] = paper | {"query": expanded_query}
        return {
            "queries": valid_queries,
            "library": library,
        }


class OracleFeatureCollector:
    def __init__(self, config: ToolConfig):
        self.eval_date = config.evaluation_date
        self.num_oracle_papers = config.num_oracle_papers
        self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)

    def _citation_count_by_eval_date(self, paper: dict):
        eval_year = int(self.eval_date.year)
        citation_count = paper.get("cited_by_count", 0) or 0
        if citation_count:
            for item in paper.get("counts_by_year", []):
                if item["year"] > eval_year:
                    citation_count -= item["cited_by_count"]
            return citation_count
        return sum(item["cited_by_count"] for item in paper.get("counts_by_year", []) if item["year"] <= eval_year)

    def _paper_age(self, paper):
        publication_date = paper.get("publication_date") or self.eval_date.strftime("%Y-%m-%d")
        year = int(self.eval_date.strftime("%Y")) - int(publication_date[:4])
        return max(1, year + 1)

    async def _fetch_neighbors(self, paper_id: str):
        neighbors = {}
        filters = [
            {"cites": paper_id, "to_publication_date": self.eval_date.strftime("%Y-%m-%d")},
            {"cited_by": paper_id, "to_publication_date": self.eval_date.strftime("%Y-%m-%d")},
        ]
        for filter_kwargs in filters:
            results = await openalex_search_paper("works", filter=filter_kwargs, per_page=200)
            for paper in results.get("results", []):
                if paper.get("id"):
                    neighbors[paper["id"]] = paper
        return neighbors, paper_id

    async def _expand_library(self, query: str, library: Dict[str, Any]):
        expanded = {**library}
        if len(expanded) >= self.num_oracle_papers:
            return expanded

        neighbor_sources, neighbor_papers, co_neighbors, long_tail = {}, {}, {}, {}
        tasks = [asyncio.create_task(self._fetch_neighbors(paper_id)) for paper_id in expanded]
        for task in asyncio.as_completed(tasks):
            neighbors, paper_id = await task
            for neighbor_id, paper in neighbors.items():
                if neighbor_id in expanded: continue
                neighbor_papers[neighbor_id] = paper
                neighbor_sources.setdefault(neighbor_id, set()).add(paper_id)

        for paper_id, paper in neighbor_papers.items():
            sources = neighbor_sources.get(paper_id, set())
            paper["neighbors_count"] = len(sources)
            if len(sources) >= 2:
                co_neighbors[paper_id] = paper
            elif len(sources) == 1:
                long_tail[paper_id] = paper

        ranked_co_neighbors = sorted(
            co_neighbors.items(),
            key=lambda item: item[1]["neighbors_count"] * 100 + math.log1p(max(0, self._citation_count_by_eval_date(item[1]))),
            reverse=True,
        )
        for paper_id, paper in ranked_co_neighbors:
            expanded[paper_id] = paper
            if len(expanded) >= self.num_oracle_papers:
                return expanded

        if long_tail and len(expanded) < self.num_oracle_papers:
            similarity = self._score_similarity(query, long_tail)
            ranked_long_tail = sorted(
                long_tail.items(),
                key=lambda item: similarity.get(item[0], 0.0) * 10 + math.log1p(max(0, self._citation_count_by_eval_date(item[1]))),
                reverse=True,
            )
            for paper_id, paper in ranked_long_tail:
                paper["query_similarity"] = similarity.get(paper_id, 0.0)
                expanded[paper_id] = paper
                if len(expanded) >= self.num_oracle_papers:
                    break
        return expanded

    def _score_similarity(self, query: str, papers: Dict[str, Any]):
        sentences = []
        paper_ids = list(papers)
        for paper_id in paper_ids:
            paper = papers[paper_id]
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            sentences.append(f"{title}. {abstract}".strip())
        embeddings = self.sentence_transformer.embed([query, *sentences])
        scores = cosine_similarity_matrix(embeddings[:1], embeddings[1:])[0].tolist()
        return {paper_id: score for paper_id, score in zip(paper_ids, scores)}

    def _local_citation_and_pagerank(self, oracle: Dict[str, Any]):
        graph = nx.DiGraph()
        graph.add_nodes_from(oracle)
        local_citation_count = {paper_id: 0 for paper_id in oracle}
        for source_id, paper in oracle.items():
            for target_id in paper.get("referenced_works", []):
                if target_id in oracle:
                    graph.add_edge(source_id, target_id)
                    local_citation_count[target_id] += 1
        pagerank = nx.pagerank(graph, alpha=0.85) if graph.number_of_nodes() else {}
        return local_citation_count, pagerank

    async def __call__(self, query: str, library: Dict[str, Any], survey_info: Dict[str, Any]) -> Dict[str, Any]:
        expanded = await self._expand_library(query, library)
        similarity = self._score_similarity(query, expanded)
        scored = []
        for paper_id, paper in expanded.items():
            sim = similarity.get(paper_id, 0.0)
            prestige = math.log1p(max(0, self._citation_count_by_eval_date(paper)))
            age = self._paper_age(paper)
            scored.append((paper_id, sim + 0.05 * prestige + 0.02 / age))
        scored.sort(key=lambda item: item[1], reverse=True)
        oracle = {}
        survey_references = set(survey_info.get("referenced_works", []))
        for paper_id, selection_score in scored[: self.num_oracle_papers]:
            paper = dict(expanded[paper_id])
            paper["oracle_selection_score"] = selection_score
            paper["is_referenced_by_survey"] = paper_id in survey_references
            oracle[paper_id] = paper

        local_citation_count, pagerank = self._local_citation_and_pagerank(oracle)
        max_citation = max((self._citation_count_by_eval_date(paper) for paper in oracle.values()), default=1)
        max_local_citation = max(local_citation_count.values(), default=1)
        for paper_id, paper in oracle.items():
            citation_count = self._citation_count_by_eval_date(paper)
            f1 = math.log1p(citation_count) / math.log1p(max_citation or 1)
            paper["features"] = [
                similarity.get(paper_id, 0.0),
                f1,
                math.log1p(local_citation_count.get(paper_id, 0)) / math.log1p(max_local_citation or 1),
                pagerank.get(paper_id, 0.0) * 100,
                f1 / math.log1p(self._paper_age(paper) + 1),
            ]
        return oracle


def iter_dataset(dataset_path: Path):
    with dataset_path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip(): continue
            yield index, json.loads(line)


def sanitize_filename(text: str, max_length: int = 100) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    return cleaned[:max_length].strip("_") or "untitled_survey"


def output_path_for(index: int, title: str) -> Path:
    return OUTPUT_DIR / f"{index:04d}_{sanitize_filename(title)}.json"


async def collect_single_survey(base_config: ToolConfig, index: int, item: Dict[str, Any], overwrite: bool = False):
    title = item["title"]
    output_path = output_path_for(index, title)
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "index": index, "title": title, "output": str(output_path)}

    eval_date = datetime.strptime(item["publication_date"], "%Y-%m-%d")
    config = replace(base_config, evaluation_date=eval_date)
    survey_fetcher = SurveyInfoFetcher(eval_date)
    query_expand = StrictQueryExpand(config)
    oracle_collector = OracleFeatureCollector(config)
    stage = "survey_fetcher"
    RateLimit.reset_openalex_count()

    try:
        print(f"[{index}] survey_fetcher:start")
        survey_info = await survey_fetcher(title, item["publication_date"])
        print(f"[{index}] survey_fetcher:done")
        stage = "query_expand"
        print(f"[{index}] query_expand:start")
        query_data = await query_expand(title)
        print(f"[{index}] query_expand:done")
        stage = "oracle_collector"
        print(f"[{index}] oracle_collector:start")
        oracle = await oracle_collector(title, query_data["library"], survey_info)
        print(f"[{index}] oracle_collector:done")
    except OpenAlexBudgetExceeded as exc:
        request_count = RateLimit.get_openalex_count()
        print(f"[{index}] openalex_requests={request_count}")
        raise
    except Exception as exc:
        request_count = RateLimit.get_openalex_count()
        print(f"[{index}] openalex_requests={request_count}")
        return {
            "status": "network_error",
            "index": index,
            "title": title,
            "stage": stage,
            "openalex_requests": request_count,
            "error": f"{type(exc).__name__}: {exc}",
        }

    request_count = RateLimit.get_openalex_count()
    print(f"[{index}] openalex_requests={request_count}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(oracle, f, ensure_ascii=False, indent=2)
    return {
        "status": "ok",
        "index": index,
        "title": title,
        "output": str(output_path),
        "library_size": len(query_data["library"]),
        "oracle_size": len(oracle),
        "openalex_requests": request_count,
    }


async def main():
    START, LIMIT = 133, 170
    base_config = ToolConfig()
    RateLimit.configure_openalex(
        requests_per_second=base_config.openalex_requests_per_second,
        enabled=base_config.openalex_rate_limit_enabled,
        max_concurrency=base_config.openalex_max_concurrency,
    )
    print(
        "OpenAlex throttle: "
        f"enabled={base_config.openalex_rate_limit_enabled} "
        f"rps={base_config.openalex_requests_per_second} "
        f"concurrency={base_config.openalex_max_concurrency}"
    )
    await SessionManager.init()
    try:
        for index, item in iter_dataset(DATASET_PATH):
            if index < START: continue
            if index >= LIMIT: break
            output_path = output_path_for(index, item['title'])
            if output_path.exists(): continue
            try:
                result = await collect_single_survey(base_config, index, item)
                print(json.dumps(result, ensure_ascii=False))
            except OpenAlexBudgetExceeded as exc:
                payload = exc.payload or {}
                print(json.dumps({
                    "status": "openalex_budget_exceeded",
                    "index": index,
                    "title": item["title"],
                    "retryAfter": payload.get("retryAfter"),
                    "message": payload.get("message", str(exc)),
                }, ensure_ascii=False))
                return
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    import os
    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
    asyncio.run(main())
