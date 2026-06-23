import asyncio
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from agent.tools.preprocess.citation_parser import CitationParser
from agent.tools.utility.openalex import get_openalex_client
from agent.tools.utility.request_utils import OpenAlexBudgetExceeded, SessionManager
from agent.tools.utility.tool_config import ToolConfig


DATASET_PATH = Path(__file__).resolve().parent / "surveys_with_query.jsonl"
OUTPUT_DIR = ROOT_DIR / "agent" / "golden"
OPENALEX_KEYS = [
    "NXd77zSxqdt2XLfu14Npp2",
    "v8Fl7dmrRk2ERkT3npPapC",
    "xnaKKdDHuqcXQPY1Crplwu",
    "OKsOaFG3SbaxrRoYSIUBfx",
    "YFl8EWRMHmmZvEd9cljGXt",
]
ENUMERATE_START, START, LIMIT = 0, 0, None


def iter_dataset(dataset_path: Path, start: int = 0):
    with dataset_path.open(encoding="utf-8") as f:
        for index, line in enumerate(f, start):
            if line.strip():
                yield index, json.loads(line)


def output_path_for(index: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{index}.json"


def first_paragraph_has_environment_type(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as f:
            full_content = (json.load(f).get("full_content") or {})
    except (OSError, json.JSONDecodeError):
        return False

    def iter_paragraphs(skeleton: object):
        if not isinstance(skeleton, dict):
            return
        yield from skeleton.get("paragraphs") or []
        for section in skeleton.get("sections") or []:
            yield from iter_paragraphs(section)

    for paragraph in iter_paragraphs(full_content):
        if isinstance(paragraph, list) and paragraph:
            first_sentence = paragraph[0]
            return isinstance(first_sentence, dict) and "environment_type" in first_sentence
        if isinstance(paragraph, dict):
            return "environment_type" in paragraph
    return False


class SurveyFulltextFetcher:
    def __init__(self, config: ToolConfig):
        self.parser = CitationParser(config)

    def _has_fulltext(self, full_content: object) -> bool:
        return (
            isinstance(full_content, dict)
            and bool(full_content.get("paragraphs") or full_content.get("sections"))
        )

    async def __call__(self, title: str) -> dict:
        info = await self.parser._search_paper_from_api(title)
        metadata = info.get("metadata") or {}
        openalex_info = metadata.get("openalex") or {}
        semantic_scholar_info = metadata.get("semantic scholar") or {}

        if not openalex_info and not semantic_scholar_info:
            raise ValueError("OpenAlex and Semantic Scholar both failed to find matching metadata")
        if not self._has_fulltext(info.get("full_content")):
            sources = ", ".join(metadata.keys()) or "none"
            raise ValueError(f"Full text download or parsing failed after metadata lookup; metadata_sources={sources}")

        return {
            "title": title,
            "abstract": info.get("abstract", ""),
            "source": info.get("source", ""),
            "full_content": info["full_content"],
            "openalex": openalex_info,
            "semantic_scholar": semantic_scholar_info,
        }


async def collect_single_survey(
    config: ToolConfig,
    fulltext_fetcher: SurveyFulltextFetcher,
    index: int,
    item: dict,
) -> dict:
    original_index = int(item.get("index", index))
    title = item.get("title", "")
    output_path = output_path_for(original_index)
    stage = "fulltext_lookup"
    try:
        fulltext_data = await fulltext_fetcher(title)
        output = {
            **item,
            "original_index": original_index,
            **fulltext_data,
        }
    except OpenAlexBudgetExceeded:
        raise
    except Exception as exc:
        return {
            "status": "failed",
            "index": original_index,
            "title": title,
            "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
        }

    del output['references']
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return {
        "status": "ok",
        "index": original_index,
        "title": title,
        "output": str(output_path),
    }


async def main():
    base_config = ToolConfig(openalex_api_keys=OPENALEX_KEYS, grobid_parse_mode="strict")
    fulltext_fetcher = SurveyFulltextFetcher(base_config)
    await SessionManager.init()
    try:
        for index, item in iter_dataset(DATASET_PATH, ENUMERATE_START):
            original_index = int(item.get("index", index))
            if original_index < START: continue
            if LIMIT is not None and original_index >= LIMIT: break
            output_path = OUTPUT_DIR / f"{original_index}.json"
            if output_path.exists() and first_paragraph_has_environment_type(output_path): continue
            try:
                result = await collect_single_survey(base_config, fulltext_fetcher, index, item)
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
