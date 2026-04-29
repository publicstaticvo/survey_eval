import asyncio
import json
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.tools.preprocess.anchor_surveys import AnchorSurveyFetch
from agent.tools.preprocess.dynamic_candidate_pool import LetorCitationScorer, LiteratureCandidatePoolBuilder
from agent.tools.utility.openalex import OPENALEX_SELECT, get_openalex_client
from agent.tools.utility.request_utils import OpenAlexBudgetExceeded, RateLimit, SessionManager
from agent.tools.utility.tool_config import ToolConfig
from agent.tools.utility.utils import valid_check


DATASET_PATH = Path(__file__).resolve().parent / "surveys_with_query.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SURVEY_SELECT = f"{OPENALEX_SELECT},best_oa_location,locations,relevance_score"
OPENALEX_KEYS = [
    'NXd77zSxqdt2XLfu14Npp2', 
    'v8Fl7dmrRk2ERkT3npPapC',
    'xnaKKdDHuqcXQPY1Crplwu',
    'OKsOaFG3SbaxrRoYSIUBfx',
    'YFl8EWRMHmmZvEd9cljGXt'
]
ENUMERATE_START, START, LIMIT = 205, 205, 215


class SurveyInfoFetcher:
    def __init__(self, config: ToolConfig):
        self.openalex = get_openalex_client(config)

    def _normalize_title(self, title: str) -> str:
        title = title.replace("\\\\", " ")
        return re.sub(r"\s+", " ", re.sub(r"[:,.!?&]", " ", title or "")).strip()

    async def __call__(self, title: str) -> Dict[str, Any]:
        normalized_title = self._normalize_title(title)
        paper = await self.openalex.find_work_by_title(normalized_title, select=SURVEY_SELECT)
        if paper and valid_check(normalized_title, paper.get("title", "")):
            return paper
        return {}


class StrictQueryExpand:
    def __init__(self, config: ToolConfig):
        self.eval_date = config.evaluation_date
        self.openalex = get_openalex_client(config)

    async def _request_for_papers(self, query: str, uplimit: int = 100, select: str = f"{OPENALEX_SELECT},relevance_score") -> List[Dict[str, Any]]:
        search = query.strip()
        if not search:
            return []

        papers = []
        total = uplimit
        for page in range(1, (uplimit - 1) // 200 + 2):
            results = await self.openalex.search_works(
                "works",
                search=search,
                per_page=min(200, uplimit),
                select=select,
                page=page,
                filter={"to_publication_date": self.eval_date.strftime("%Y-%m-%d")},
            )
            total = min(total, results.get("count", 0) or total)
            papers.extend(results.get("results", []))
            if len(papers) >= total:
                break
        return papers[:uplimit]

    async def __call__(self, query: str, num_seed_papers: int = 100):
        search_query = (query or "").strip()
        print(f"The search query is: {search_query}")

        papers = await self._request_for_papers(search_query, num_seed_papers)
        library = {}
        for paper in papers:
            if paper.get("id"):
                library[paper["id"]] = paper | {"query": search_query}
        return {
            "queries": [search_query] if library else [],
            "library": library,
        }


def iter_dataset(dataset_path: Path, start: int = 0):
    with dataset_path.open(encoding="utf-8") as f:
        for index, line in enumerate(f, start):
            if not line.strip(): continue
            yield index, json.loads(line)


def sanitize_filename(text: str, max_length: int = 100) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    return cleaned[:max_length].strip("_") or "untitled_survey"


def output_path_for(index: int, title: str) -> Path:
    return OUTPUT_DIR / f"{index:04d}_{sanitize_filename(title)}.json"


async def collect_single_survey(base_config: ToolConfig, index: int, item: Dict[str, Any], overwrite: bool = False):
    original_index = int(item.get("index", index))
    title = item["title"]
    query = (item.get("query") or "").strip()
    if not query:
        raise ValueError(f"No query found for survey index={original_index}")

    output_path = output_path_for(original_index, title)
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "index": original_index, "title": title, "output": str(output_path)}

    openalex = get_openalex_client(base_config)    
    openalex.reset_request_count()

    try:
        stage = "survey_fetcher"
        print(f"[{original_index}] title: {title}")
        survey_fetcher = SurveyInfoFetcher(base_config)
        print(f"[{original_index}] survey_fetcher:start")
        survey_info = await survey_fetcher(title)
        print(f"[{original_index}] survey_fetcher:done")
        if not survey_info: raise ValueError("No survey info")
        stage = "anchor_surveys"
        publication_date = item.get("publication_date") or survey_info.get("publication_date")
        eval_date = datetime.strptime(publication_date, "%Y-%m-%d")
        config = replace(base_config, evaluation_date=eval_date)
        anchor_fetcher = AnchorSurveyFetch(config)
        print(f"[{original_index}] anchor_surveys:start")
        anchor_data = await anchor_fetcher(query)
        print(f"[{original_index}] anchor_surveys:done")
        stage = "candidate_pool"
        candidate_builder = LiteratureCandidatePoolBuilder(config)
        scorer = LetorCitationScorer(config)
        print(f"[{original_index}] candidate_pool:start")
        candidate_data = await candidate_builder(query, anchor_data=anchor_data)
        print(f"[{original_index}] candidate_pool:done")
        stage = "letor"
        print(f"[{original_index}] letor:start")
        oracle = scorer(query, candidate_data["candidate_pool"])
        print(f"[{original_index}] letor:done")
    except OpenAlexBudgetExceeded as exc:
        request_count = openalex.get_request_count()
        print(f"[{original_index}] openalex_requests={request_count}")
        raise
    except Exception as exc:
        request_count = openalex.get_request_count()
        print(f"[{original_index}] openalex_requests={request_count}")
        return {
            "status": "network_error",
            "index": original_index,
            "title": title,
            "stage": stage,
            "openalex_requests": request_count,
            "error": f"{type(exc).__name__}: {exc}",
        }

    # with open("surveys.jsonl", "a+", encoding='utf-8') as f:
    #     f.write(json.dumps({"index": index, "metadata": survey_info}, ensure_ascii=False) + "\n")
    request_count = openalex.get_request_count()
    print(f"[{original_index}] openalex_requests={request_count}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(oracle, f, ensure_ascii=False, indent=2)
    return {
        "status": "ok",
        "index": original_index,
        "title": title,
        "output": str(output_path),
        "library_size": len(candidate_data.get("library", {})),
        "candidate_pool_size": len(candidate_data.get("candidate_pool", {})),
        "oracle_size": len(oracle),
        "openalex_requests": request_count,
    }


async def main():
    base_config = ToolConfig(openalex_api_keys=OPENALEX_KEYS)
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
        for index, item in iter_dataset(DATASET_PATH, ENUMERATE_START):
            original_index = int(item.get("index", index))
            if original_index < START:
                continue
            if original_index >= LIMIT:
                break
            output_path = output_path_for(original_index, item["title"])
            if output_path.exists(): continue
            try:
                result = await collect_single_survey(base_config, index, item)
                print(json.dumps(result, ensure_ascii=False))
            except OpenAlexBudgetExceeded as exc:
                payload = exc.payload or {}
                print(json.dumps({
                    "status": "openalex_budget_exceeded",
                    "index": original_index,
                    "title": item["title"],
                    "retryAfter": payload.get("retryAfter"),
                    "message": payload.get("message", str(exc)),
                }, ensure_ascii=False))
                return
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    asyncio.run(main())
