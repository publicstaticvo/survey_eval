import asyncio
import json
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.tools.preprocess.build_sources import DirectSeedGraphSource
from agent.tools.preprocess.dynamic_candidate_pool import DynamicCandidatePool, LetorCandidateScorer
from agent.tools.utility.openalex import OPENALEX_SELECT, get_openalex_client
from agent.tools.utility.request_utils import OpenAlexBudgetExceeded, RateLimit, SessionManager
from agent.tools.utility.tool_config import ToolConfig
from agent.tools.preprocess.utils import cosine_similarity_matrix
from train_letor.gt import OPENALEX_KEYS


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
    seed_source = DirectSeedGraphSource(config)
    query = (survey_item.get("query") or "").strip()
    if not query:
        raise ValueError("survey_item has no query")

    oracle = await seed_source.search_seed_papers(query, uplimit=config.num_oracle_papers)

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
    dynamic_pool = DynamicCandidatePool(config)
    query = (survey_item.get("query") or "").strip()
    if not query:
        raise ValueError("survey_item has no query")

    pool_data = await dynamic_pool(query, download_anchor_surveys=False, calc_rank=True)
    scored_pool = pool_data["oracle_papers"]
    included_ids = {
        paper_id
        for paper_id, _ in sorted(
            scored_pool.items(),
            key=lambda item: item[1].get("rank", 0.0),
            reverse=True,
        )[:oracle_limit]
    }
    unlimited = {}
    for paper_id, paper in scored_pool.items():
        paper_copy = dict(paper)
        paper_copy["oracle_included"] = paper_id in included_ids
        paper_copy["oracle_reason"] = paper_copy.get("candidate_source", "")
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
    oracle_helper = LetorCandidateScorer(config)
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
