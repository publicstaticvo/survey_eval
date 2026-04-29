import asyncio
import json
import math
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.tools.utility.openalex import OPENALEX_SELECT, get_openalex_client
from agent.tools.utility.request_utils import OpenAlexBudgetExceeded, RateLimit, SessionManager
from agent.tools.utility.tool_config import ToolConfig
from agent.tools.preprocess.utils import cosine_similarity_matrix
from train_letor.gt import OPENALEX_KEYS
from train_letor.main import OracleFeatureCollector, StrictQueryExpand


SURVEYS_PATH = Path(__file__).resolve().parent / "surveys.jsonl"
SURVEYS_WITH_QUERY_PATH = Path(__file__).resolve().parent / "surveys_with_query.jsonl"
SOURCE_DATASET_PATH = Path(__file__).resolve().parent / "surge.jsonl"
OUTPUTS_DIRECT_DIR = Path(__file__).resolve().parent / "outputs_direct"
OUTPUTS_UNLIMITED_DIR = Path(__file__).resolve().parent / "outputs_unlimited"
OUTPUTS_UNHIT_DIR = Path(__file__).resolve().parent / "outputs_unhit"
ENUMERATE_START, START, LIMIT = 205, 205, 215
base_config = ToolConfig(openalex_api_keys=OPENALEX_KEYS)


def sanitize_filename(text: str, max_length: int = 100) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    return cleaned[:max_length].strip("_") or "untitled_survey"


def output_path_for(directory: Path, index: int | str, title: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{int(index):04d}_{sanitize_filename(title)}.json"


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def load_item_by_index(path: Path, index: int | str) -> dict[str, Any] | None:
    target = str(index)
    for item in iter_jsonl(path):
        if str(item.get("index")) == target:
            return item
    return None


def load_source_dataset_map(path: Path) -> dict[str, dict[str, Any]]:
    dataset = {}
    for index, item in enumerate(iter_jsonl(path)):
        original_index = str(item.get("index", index))
        dataset[original_index] = item
    return dataset


def get_eval_config(source_item: dict[str, Any] | None) -> ToolConfig:
    publication_date = ""
    if source_item:
        publication_date = source_item.get("publication_date", "") or ""
    if publication_date:
        eval_date = datetime.strptime(publication_date, "%Y-%m-%d")
        return replace(base_config, evaluation_date=eval_date)
    return base_config


async def build_direct_oracle_for_item(
    survey_item: dict[str, Any],
    source_item: dict[str, Any] | None = None,
    output_dir: Path = OUTPUTS_DIRECT_DIR,
):
    config = get_eval_config(source_item)
    query_expand = StrictQueryExpand(config)
    query = (survey_item.get("query") or "").strip()
    if not query:
        raise ValueError("survey_item has no query")

    papers = await query_expand._request_for_papers(
        query,
        uplimit=config.num_oracle_papers,
        select=f"{OPENALEX_SELECT},relevance_score,abstract_inverted_index",
    )
    oracle = {}
    for paper in papers:
        paper_id = paper.get("id")
        if paper_id:
            oracle[paper_id] = paper | {"query": query}

    output_path = output_path_for(output_dir, survey_item["index"], survey_item["title"])
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(oracle, f, ensure_ascii=False, indent=2)
    return oracle


async def build_unlimited_oracle_for_item(
    survey_item: dict[str, Any],
    source_item: dict[str, Any] | None = None,
    output_dir: Path = OUTPUTS_UNLIMITED_DIR,
):
    config = get_eval_config(source_item)
    oracle_limit = config.num_oracle_papers
    query_expand = StrictQueryExpand(config)
    oracle_helper = OracleFeatureCollector(config)
    query = (survey_item.get("query") or "").strip()
    if not query:
        raise ValueError("survey_item has no query")

    seed_papers = await query_expand._request_for_papers(
        query,
        uplimit=100,
        select=f"{OPENALEX_SELECT},relevance_score,abstract_inverted_index",
    )
    seed_library = {paper["id"]: paper | {"query": query} for paper in seed_papers if paper.get("id")}
    seed_ids = set(seed_library)

    openalex = get_openalex_client(config)
    neighbor_papers: dict[str, dict[str, Any]] = {}
    cites_seed_sources: dict[str, set[str]] = {}
    cited_by_seed_sources: dict[str, set[str]] = {}

    async def _fetch_neighbors(seed_id: str):
        cited_by_seed = {}
        cites_seed = {}

        results = await openalex.search_works(
            "works",
            filter={"cited_by": seed_id, "to_publication_date": config.evaluation_date.strftime("%Y-%m-%d")},
            per_page=200,
            select=f"{OPENALEX_SELECT},abstract_inverted_index",
            sort="cited_by_count:desc",
        )
        count = results['count']
        for paper in results.get("results", []):
            if paper.get("id"):
                cited_by_seed[paper["id"]] = paper
        
        if count > 200:
            for i in range(2, (count - 1) // 200 + 2):
                results = await openalex.search_works(
                    "works",
                    filter={"cited_by": seed_id, "to_publication_date": config.evaluation_date.strftime("%Y-%m-%d")},
                    per_page=200, page=i,
                    select=f"{OPENALEX_SELECT},abstract_inverted_index",                    
                    sort="cited_by_count:desc",
                )
                count = results['count']
                for paper in results.get("results", []):
                    if paper.get("id"):
                        cited_by_seed[paper["id"]] = paper

        # results = await openalex.search_works(
        #     "works",
        #     filter={"cites": seed_id, "to_publication_date": config.evaluation_date.strftime("%Y-%m-%d")},
        #     per_page=200,
        #     select=f"{OPENALEX_SELECT},abstract_inverted_index",
        #     sort="cited_by_count:desc",
        # )
        # for paper in results.get("results", []):
        #     if paper.get("id"):
        #         cites_seed[paper["id"]] = paper

        return seed_id, cited_by_seed, cites_seed

    tasks = [asyncio.create_task(_fetch_neighbors(seed_id)) for seed_id in seed_library]
    for task in asyncio.as_completed(tasks):
        seed_id, cited_by_seed, cites_seed = await task
        for paper_id, paper in cited_by_seed.items():
            if paper_id in seed_ids: continue
            neighbor_papers[paper_id] = paper
            cited_by_seed_sources.setdefault(paper_id, set()).add(seed_id)
        for paper_id, paper in cites_seed.items():
            if paper_id in seed_ids: continue
            neighbor_papers[paper_id] = paper
            cites_seed_sources.setdefault(paper_id, set()).add(seed_id)

    co_neighbors, long_tail = {}, {}
    for paper_id, paper in neighbor_papers.items():
        connected_seed_ids = cites_seed_sources.get(paper_id, set()) | cited_by_seed_sources.get(paper_id, set())
        paper["connected_seed_count"] = len(connected_seed_ids)
        paper["cited_by_seed_count"] = len(cited_by_seed_sources.get(paper_id, set()))
        paper["cites_seed_count"] = len(cites_seed_sources.get(paper_id, set()))
        if paper["connected_seed_count"] >= 2:
            co_neighbors[paper_id] = paper
        else:
            long_tail[paper_id] = paper

    selected_count, selected_reason, selected_included = 0, {}, {}
    for paper_id in seed_library:
        selected_count += 1
        selected_reason[paper_id] = "seed"
        selected_included[paper_id] = selected_count <= oracle_limit
    print(f"{selected_count} seeds", end=',')

    ranked_co_neighbors = sorted(
        co_neighbors.items(),
        key=lambda item: item[1]["connected_seed_count"] * 100 + math.log1p(max(0, oracle_helper._citation_count_by_eval_date(item[1]))),
        reverse=True,
    )
    for paper_id, _ in ranked_co_neighbors:
        selected_count += 1
        selected_reason[paper_id] = "neighbors"
        selected_included[paper_id] = selected_count <= oracle_limit

    print(f"{selected_count} co-neighbors", end=',')
    if long_tail:
        similarity = oracle_helper._score_similarity(query, long_tail)
        ranked_long_tail = sorted(
            long_tail.items(),
            key=lambda item: similarity.get(item[0], 0.0) * 10 + math.log1p(max(0, oracle_helper._citation_count_by_eval_date(item[1]))),
            reverse=True,
        )
        for paper_id, paper in ranked_long_tail:
            paper["query_similarity"] = similarity.get(paper_id, 0.0)
            selected_count += 1
            selected_reason[paper_id] = "neighbors"
            selected_included[paper_id] = selected_count <= oracle_limit
        print(f"{selected_count} long-tails", end=',')
    print()

    unlimited = {}
    for paper_id, paper in seed_library.items():
        paper_copy = dict(paper)
        paper_copy["oracle_included"] = selected_included.get(paper_id, False)
        paper_copy["oracle_reason"] = "seed"
        paper_copy["oracle_limit"] = oracle_limit
        paper_copy["oracle_1000_included"] = paper_copy["oracle_included"]
        paper_copy["oracle_1000_reason"] = paper_copy["oracle_reason"]
        paper_copy["cited_by_seed_count"] = sum(1 for other_paper in seed_library.values() if paper_id in (other_paper.get("referenced_works") or []))
        paper_copy["cites_seed_count"] = sum(1 for ref_id in paper_copy.get("referenced_works", []) if ref_id in seed_ids)
        unlimited[paper_id] = paper_copy

    for paper_id, paper in neighbor_papers.items():
        paper_copy = dict(paper)
        paper_copy["oracle_included"] = selected_included.get(paper_id, False)
        paper_copy["oracle_reason"] = selected_reason.get(paper_id, "")
        paper_copy["oracle_limit"] = oracle_limit
        paper_copy["oracle_1000_included"] = paper_copy["oracle_included"]
        paper_copy["oracle_1000_reason"] = paper_copy["oracle_reason"]
        unlimited[paper_id] = paper_copy

    output_path = output_path_for(output_dir, survey_item["index"], survey_item["title"])
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(unlimited, f, ensure_ascii=False, indent=2)
    return unlimited


async def build_reference_cites_dict(
    survey_item: dict[str, Any],
    source_item: dict[str, Any] | None = None,
    output_dir: Path = OUTPUTS_UNHIT_DIR,
) -> dict[str, Any]:
    config = get_eval_config(source_item)
    openalex = get_openalex_client(config)
    oracle_helper = OracleFeatureCollector(config)
    references = survey_item.get("references") or []
    query = (survey_item.get("query") or "").strip()
    publication_date = ""
    if source_item:
        publication_date = source_item.get("publication_date", "") or ""
    if not publication_date:
        publication_date = config.evaluation_date.strftime("%Y-%m-%d")

    async def _single(ref_id: str):
        work = await openalex.get_entity(
            ref_id,
            entity_type="works",
            select=f"{OPENALEX_SELECT},authorships,sources,primary_location,concepts,topics",
        )
        return ref_id, work

    tasks = [asyncio.create_task(_single(ref_id)) for ref_id in references if ref_id]
    works = {}
    for task in asyncio.as_completed(tasks):
        try:
            ref_id, work = await task
            if work:
                works[ref_id] = work
        except OpenAlexBudgetExceeded:
            for other_task in tasks:
                if not other_task.done():
                    other_task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        except Exception:
            continue

    work_ids = list(works)
    sentences = [
        f"{works[work_id].get('title', '')}. {works[work_id].get('abstract', '')}".strip()
        for work_id in work_ids
    ]
    if query and sentences:
        embeddings = oracle_helper.sentence_transformer.embed([query, *sentences])
        similarities = cosine_similarity_matrix(embeddings[:1], embeddings[1:])[0].tolist()
    else:
        similarities = [0.0 for _ in work_ids]

    cites = {}
    for work_id, similarity in zip(work_ids, similarities):
        work = works[work_id]
        cites[work_id] = {
            **work, 
            "paper_age": oracle_helper._paper_age(work),
            "semantic_similarity": float(similarity),
            "cited_by_count_by_eval_date": oracle_helper._citation_count_by_eval_date(work),
        }

    result = {
        "title": survey_item.get("title", ""),
        "query": survey_item.get("query", ""),
        "publication_date": publication_date,
        "cites": cites,
    }

    output_path = output_path_for(output_dir, survey_item["index"], survey_item["title"])
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


async def main():
    RateLimit.configure_openalex(
        requests_per_second=base_config.openalex_requests_per_second,
        enabled=base_config.openalex_rate_limit_enabled,
        max_concurrency=base_config.openalex_max_concurrency,
    )
    source_map = load_source_dataset_map(SOURCE_DATASET_PATH)

    await SessionManager.init()
    try:
        for index, survey_item in enumerate(iter_jsonl(SURVEYS_WITH_QUERY_PATH), ENUMERATE_START):
            original_index = int(survey_item.get("index", index))
            if original_index < START: continue
            if original_index >= LIMIT: break
            output_path = output_path_for(OUTPUTS_UNHIT_DIR, original_index, survey_item["title"])
            if output_path.exists(): continue

            source_item = source_map.get(str(survey_item.get("index", original_index)))
            print(f"[{original_index}] start: {survey_item.get('title', '')}")
            try:
                # await build_direct_oracle_for_item(survey_item, source_item)
                # await build_unlimited_oracle_for_item(survey_item, source_item)
                await build_reference_cites_dict(survey_item, source_item)
                print(f"[{original_index}] done")
            except OpenAlexBudgetExceeded as exc:
                payload = exc.payload or {}
                print(json.dumps({
                    "status": "openalex_budget_exceeded",
                    "index": original_index,
                    "title": survey_item.get("title", ""),
                    "retryAfter": payload.get("retryAfter"),
                    "message": payload.get("message", str(exc)),
                }, ensure_ascii=False))
                return
            except Exception as exc:
                print(json.dumps({
                    "status": "failed",
                    "index": original_index,
                    "title": survey_item.get("title", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }, ensure_ascii=False))
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    asyncio.run(main())
