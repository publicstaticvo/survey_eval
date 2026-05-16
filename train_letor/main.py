import asyncio
import json
import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from agent.tools.utility.openalex import OPENALEX_SELECT, get_openalex_client
from agent.tools.utility.request_utils import OpenAlexBudgetExceeded, RateLimit, SessionManager
from agent.tools.utility.tool_config import ToolConfig
from agent.tools.utility.utils import valid_check
from train_letor.topic_report import SurveyTopicReportBuilder, OVERWRITE


DATASET_PATH = Path(__file__).resolve().parent / "surveys_with_query.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SURVEY_SELECT = f"{OPENALEX_SELECT},best_oa_location,locations,relevance_score"
OPENALEX_KEYS = [
    "NXd77zSxqdt2XLfu14Npp2",
    "v8Fl7dmrRk2ERkT3npPapC",
    "xnaKKdDHuqcXQPY1Crplwu",
    "OKsOaFG3SbaxrRoYSIUBfx",
    "YFl8EWRMHmmZvEd9cljGXt",
]
ENUMERATE_START, START, LIMIT = 0, 6, 10


def iter_dataset(dataset_path: Path, start: int = 0):
    with dataset_path.open(encoding="utf-8") as f:
        for index, line in enumerate(f, start):
            if line.strip():
                yield index, json.loads(line)


def sanitize_filename(text: str, max_length: int = 100) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    return cleaned[:max_length].strip("_") or "untitled_survey"


def output_path_for(index: int, title: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{index:04d}_{sanitize_filename(title)}.json"


class SurveyInfoFetcher:
    def __init__(self, config: ToolConfig):
        self.openalex = get_openalex_client(config)

    def _normalize_title(self, title: str) -> str:
        title = title.replace("\\\\", " ")
        return re.sub(r"\s+", " ", re.sub(r"[:,.!?&]", " ", title or "")).strip()

    async def __call__(self, title: str) -> dict:
        normalized_title = self._normalize_title(title)
        paper = await self.openalex.find_work_by_title(normalized_title, select=SURVEY_SELECT)
        if paper and valid_check(normalized_title, paper.get("title", "")):
            return paper
        return {}


async def collect_single_survey(config: ToolConfig, index: int, item: dict, overwrite: bool = False) -> dict:
    original_index = int(item.get("index", index))
    title = item.get("title", "")
    output_path = output_path_for(original_index, title)
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "index": original_index, "title": title, "output": str(output_path)}

    openalex = get_openalex_client(config)
    openalex.reset_request_count()
    stage = "survey_lookup"
    try:
        survey_info = await SurveyInfoFetcher(config)(title)
        if not survey_info or not survey_info.get("id"):
            raise ValueError("No OpenAlex survey info")

        stage = "topic_report"
        builder = SurveyTopicReportBuilder(config)
        report = await builder.build_report_for_item(
            {
                **item,
                "id": survey_info["id"],
                "openalex_id": survey_info["id"],
                "original_index": original_index,
                "publication_date": item.get("publication_date") or survey_info.get("publication_date"),
            }
        )
    except OpenAlexBudgetExceeded:
        raise
    except Exception as exc:
        return {
            "status": "failed",
            "index": original_index,
            "title": title,
            "stage": stage,
            "openalex_requests": openalex.get_request_count(),
            "error": f"{type(exc).__name__}: {exc}",
        }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return {
        "status": "ok",
        "index": original_index,
        "title": title,
        "output": str(output_path),
        "openalex_requests": openalex.get_request_count(),
    }


async def main():
    base_config = ToolConfig(openalex_api_keys=OPENALEX_KEYS)
    await SessionManager.init()
    try:
        for index, item in iter_dataset(DATASET_PATH, ENUMERATE_START):
            original_index = int(item.get("index", index))
            if original_index < START: continue
            if original_index >= LIMIT: break
            if not OVERWRITE and output_path_for(original_index, item.get("title", "")).exists(): continue
            try:
                result = await collect_single_survey(base_config, index, item, OVERWRITE)
                print(json.dumps(result, ensure_ascii=False))
            except OpenAlexBudgetExceeded as exc:
                payload = exc.payload or {}
                print(
                    json.dumps(
                        {
                            "status": "openalex_budget_exceeded",
                            "index": original_index,
                            "title": item.get("title", ""),
                            "retryAfter": payload.get("retryAfter"),
                            "message": payload.get("message", str(exc)),
                        },
                        ensure_ascii=False,
                    )
                )
                return
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    asyncio.run(main())
