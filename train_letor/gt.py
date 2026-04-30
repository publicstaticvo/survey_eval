import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import aiohttp


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.tools.utility.llmclient import AsyncChat
from agent.tools.utility.openalex import OpenAlex, get_openalex_client
from agent.tools.utility.request_utils import HEADERS, OpenAlexBudgetExceeded, RateLimit, SessionManager
from agent.tools.utility.tool_config import ToolConfig
from agent.tools.preprocess.utils import extract_json
from agent.tools.utility.utils import valid_check
from main import SURVEY_SELECT


DATASET_PATH = Path(__file__).resolve().parent / "surveygen.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parent / "surveys.jsonl"
SURVEYS_WITH_QUERY_PATH = Path(__file__).resolve().parent / "surveys_with_query.jsonl"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
OPENALEX_KEYS = [
    'NXd77zSxqdt2XLfu14Npp2', 
    'v8Fl7dmrRk2ERkT3npPapC',
    'xnaKKdDHuqcXQPY1Crplwu',
    'OKsOaFG3SbaxrRoYSIUBfx',
    'YFl8EWRMHmmZvEd9cljGXt'
]
ENUMERATE_START, START, LIMIT = 205, 205, 370

TITLE_TO_TOPIC_QUERY_PROMPT = """
You are given the title of a survey paper. Rewrite it into a compact literature search query that keeps only the topic-bearing terms.

Requirements:
1. Remove survey-oriented words and phrases such as: survey, review, overview, summary, taxonomy, tutorial, perspective, open problems, future directions, challenges, opportunities, progress, advances.
2. Remove lightweight connective words that do not affect the research topic, such as: toward, towards, on, for, of, in, with, and, an, a, the, from.
3. Remove broad framing or organizational words unless they are truly part of the technical topic, such as: methodologies, methods, method, models, model, approaches, comparison, comparisons, benchmark, benchmarking, evaluation, recent advancements, recent advances, applications, trends.
4. Keep only the core technical topic terms, including important task names, model families, domains, and abbreviations.
5. Prefer the smallest self-contained topic phrase that would retrieve papers on the same research area. Usually 2 to 6 words is best.
6. Do not add any new concepts that are not stated in the title.
7. Output one short search query only. Prefer noun phrases over full sentences.
8. If the title already mainly consists of topic words, keep it close to the original.

Good example:
Title: A Review of Publicly Available Automatic Brain Segmentation Methodologies, Machine Learning Models, Recent Advancements, and Their Comparison
Output: {{"query": "automatic brain segmentation"}}

Return JSON only:
{{"query": "<cleaned topic query>"}}

Title: {query}
"""


class SearchKeywordLLMClient(AsyncChat):
    PROMPT = TITLE_TO_TOPIC_QUERY_PROMPT

    def _availability(self, response, context):
        payload = extract_json(response)
        query = (payload.get("query", "") if isinstance(payload, dict) else "").strip()
        if not query:
            raise ValueError("Empty search keyword query")
        return query


class SemanticScholarGroundTruthBuilder:
    def __init__(self, config: ToolConfig):
        self.config = config
        self.openalex: OpenAlex = get_openalex_client(config)

    def _as_dict(self, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _normalize_title(self, title: str) -> str:
        title = title.replace("\\\\", " ")
        return re.sub(r"\s+", " ", re.sub(r"[:,.!?&]", " ", title or "")).strip()

    async def _semantic_scholar_get(self, endpoint: str, params: dict[str, Any]) -> dict:
        """Call Semantic Scholar and retry forever on HTTP 429."""
        session = SessionManager.get()
        url = f"{SEMANTIC_SCHOLAR_API}{endpoint}"

        while True:
            try:
                async with session.get(
                    url,
                    headers=HEADERS,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientResponseError:
                raise
            except KeyboardInterrupt:
                raise

    async def _autocomplete_paper_id(self, title: str, require_valid: bool = False, original_title: str = "") -> str:
        """Find a Semantic Scholar paper id from title autocomplete results."""
        payload = await self._semantic_scholar_get("/paper/autocomplete", {"query": title})
        matches = payload.get("matches") or []
        if not matches:
            return ""
        if not require_valid:
            return str(matches[0].get("id", "") or "")

        target_title = original_title or title
        for match in matches:
            match = self._as_dict(match)
            candidate_title = (
                match.get("title")
                or self._as_dict(match.get("paper")).get("title")
                or match.get("name")
                or ""
            )
            if candidate_title and valid_check(target_title, candidate_title):
                return str(match.get("id", "") or "")
        return ""

    async def _fetch_references(self, paper_id: str) -> list[dict[str, Any]]:
        """Fetch up to 1000 references from Semantic Scholar."""
        payload = await self._semantic_scholar_get(
            f"/paper/{paper_id}/references",
            {"limit": 1000, "fields": "title,year"},
        )
        if not payload.get("data"): return []
        references = []
        for item in payload.get("data"):
            cited_paper = item.get("citedPaper") or {}
            title = (cited_paper.get("title") or "").strip()
            if not title: continue
            references.append(
                {
                    "title": title,
                    "year": cited_paper.get("year"),
                }
            )
        return references

    async def _try_fetch_references(self, semantic_paper_id: str) -> list[dict[str, Any]] | None:
        """Return None when an identifier fallback simply does not resolve."""
        if not semantic_paper_id:
            return None
        try:
            return await self._fetch_references(semantic_paper_id)
        except aiohttp.ClientResponseError as exc:
            if exc.status in {400, 404}:
                return None
            raise

    def _extract_arxiv_id_from_text(self, text: str) -> str:
        """Extract arXiv identifier from url or free text."""
        if not text:
            return ""
        match = re.search(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", text, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"\barxiv:([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)\b", text, flags=re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).removesuffix(".pdf")

    def _extract_arxiv_id_from_work(self, work: dict[str, Any]) -> str:
        """Extract arXiv identifier from OpenAlex work metadata."""
        texts = []
        for field in ["primary_location", "best_oa_location"]:
            location = self._as_dict(work.get(field))
            source = self._as_dict(location.get("source"))
            texts.extend(
                [
                    location.get("landing_page_url", ""),
                    location.get("pdf_url", ""),
                    location.get("id", ""),
                    source.get("display_name", ""),
                ]
            )
        for location in work.get("locations", []) or []:
            location = self._as_dict(location)
            source = self._as_dict(location.get("source"))
            texts.extend(
                [
                    location.get("landing_page_url", ""),
                    location.get("pdf_url", ""),
                    location.get("id", ""),
                    source.get("display_name", ""),
                ]
            )
        for text in texts:
            arxiv_id = self._extract_arxiv_id_from_text(text)
            if arxiv_id:
                return arxiv_id
        return ""

    def _normalize_doi(self, value: str) -> str:
        """Normalize DOI url or identifier to bare DOI."""
        value = (value or "").strip()
        if not value: return ""
        value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
        value = value.removeprefix("doi:").strip()
        return value

    async def _lookup_openalex_identifiers(self, title: str) -> dict[str, str]:
        """Use OpenAlex autocomplete plus work lookup to find arXiv and DOI."""
        payload = await self.openalex.autocomplete("works", title)
        results = payload.get("results") or []
        if not results:
            return {"arxiv_id": "", "doi": ""}

        for item in results:
            item = self._as_dict(item)
            if not valid_check(title, item.get("display_name")): continue
            work_id = str(item.get("id", "") or "").strip()
            if not work_id: continue
            work = await self.openalex.get_entity(
                work_id,
                entity_type="works",
                select="id,title,doi,locations,best_oa_location,primary_location,indexed_in",
            )
            if not work: continue
            arxiv_id = self._extract_arxiv_id_from_work(work)
            doi = self._normalize_doi(work.get("doi", "") or item.get("external_id", ""))
            if "arxiv." in doi: arxiv_id = doi[doi.index('arxiv.') + 6:]
            if arxiv_id or doi:
                return {"arxiv_id": arxiv_id, "doi": doi}
        return {"arxiv_id": "", "doi": ""}

    def _truncate_title_for_semantic(self, title: str) -> str:
        """Trim title before the first non-connector punctuation mark."""
        connector_chars = set("-_/+&")
        for idx, ch in enumerate(title or ""):
            if ch.isalnum() or ch.isspace() or ch in connector_chars:
                continue
            return (title or "")[:idx].strip()
        return (title or "").strip()

    def _match_openalex_ids(self, target_title: str, results: list[dict[str, Any]]) -> list[str]:
        """Keep every OpenAlex autocomplete result whose title passes valid_check."""
        matched_ids = []
        for item in results:
            item = self._as_dict(item)
            candidate_title = item.get("display_name") or item.get("name") or item.get("title") or ""
            if not isinstance(candidate_title, str):
                candidate_title = ""
            if not valid_check(target_title, candidate_title):
                continue
            paper_id = str(item.get("id", "") or "").strip()
            if paper_id:
                matched_ids.append(paper_id)
        return matched_ids

    async def _resolve_reference_openalex_ids(self, reference: dict[str, Any]) -> list[str]:
        """Resolve one reference title to all valid OpenAlex ids returned by autocomplete."""
        title = reference.get("title", "")
        if not title:
            return []
        payload = await self.openalex.autocomplete("works", title)
        return self._match_openalex_ids(title, payload.get("results") or [])

    async def build_references(self, survey_title: str, semantic_paper_id: str = "") -> list[str]:
        """Build the OpenAlex reference id list for one survey title."""
        if semantic_paper_id:
            references = await self._try_fetch_references(semantic_paper_id)
        else:
            survey_title = self._normalize_title(survey_title)
            identifiers = await self._lookup_openalex_identifiers(survey_title)
            semantic_paper_ids: list[str] = []
            references = None

            arxiv_id = identifiers.get("arxiv_id", "")
            doi = identifiers.get("doi", "")
            if arxiv_id:
                semantic_paper_ids.append(f"ARXIV:{arxiv_id}")
            if doi:
                semantic_paper_ids.append(f"DOI:{doi}")

            seen_semantic_ids = set()
            for candidate_paper_id in semantic_paper_ids:
                if not candidate_paper_id or candidate_paper_id in seen_semantic_ids: continue
                seen_semantic_ids.add(candidate_paper_id)
                references = await self._try_fetch_references(candidate_paper_id)
                if references is not None: break

            if references is None:
                semantic_paper_id = await self._autocomplete_paper_id(survey_title)
                if semantic_paper_id:
                    references = await self._try_fetch_references(semantic_paper_id)
                else:
                    truncated_title = self._truncate_title_for_semantic(survey_title)
                    if truncated_title and truncated_title != survey_title:
                        truncated_paper_id = await self._autocomplete_paper_id(
                            truncated_title,
                            require_valid=True,
                            original_title=survey_title,
                        )
                        if truncated_paper_id:
                            references = await self._try_fetch_references(truncated_paper_id)

        if not references: return []

        collected_ids: list[str] = []
        seen_ids: set[str] = set()
        tasks = [asyncio.create_task(self._resolve_reference_openalex_ids(reference)) for reference in references]

        for task in asyncio.as_completed(tasks):
            try:
                matched_ids = await task
            except OpenAlexBudgetExceeded:
                for other_task in tasks:
                    if not other_task.done():
                        other_task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            except Exception:
                continue

            for paper_id in matched_ids:
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)
                collected_ids.append(paper_id)

        return collected_ids


def iter_dataset(dataset_path: Path, start: int = 0):
    with dataset_path.open(encoding="utf-8") as f:
        for index, line in enumerate(f, start):
            if not line.strip():
                continue
            yield index, json.loads(line)


def load_completed_indexes(output_path: Path) -> set[str]:
    completed = set()
    if not output_path.exists():
        return completed
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if "index" in payload:
                completed.add(str(payload["index"]))
    return completed


def fallback_keyword_query(query: str) -> str:
    lowered = re.sub(r"[\-_:,.;!?(){}\[\]/\\]+", " ", query or "")
    lowered = re.sub(
        r"\b(survey|review|overview|summary|taxonomy|tutorial|perspective|"
        r"open problems?|future directions?|challenges?|opportunities|"
        r"progress|advances?)\b",
        " ",
        lowered,
        flags=re.IGNORECASE,
    )
    lowered = re.sub(
        r"\b(methodologies|methodology|methods?|models?|approaches?|"
        r"comparison|comparisons|benchmark(?:ing)?|evaluation|"
        r"applications?|trends|recent advancements?|recent advances?)\b",
        " ",
        lowered,
        flags=re.IGNORECASE,
    )
    lowered = re.sub(
        r"\b(toward|towards|on|for|of|in|with|and|an|a|the|from)\b",
        " ",
        lowered,
        flags=re.IGNORECASE,
    )
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered or (query or "").strip()


async def build_surveys_with_query(
    config: ToolConfig,
    input_path: Path = OUTPUT_PATH,
    output_path: Path = SURVEYS_WITH_QUERY_PATH,
):
    keyword_llm = SearchKeywordLLMClient(config.llm_server_info, config.sampling_params)
    items = list(iter_dataset(input_path))

    async def _single(index: int, item: dict[str, Any]):
        title = item.get("title", "")
        try:
            query = await keyword_llm.call(inputs={"query": title})
        except Exception:
            query = fallback_keyword_query(title)
        return item | {"index": item.get("index", index), "query": query}

    tasks = [asyncio.create_task(_single(index, item)) for index, item in items]
    results = [await task for task in asyncio.as_completed(tasks)]
    results.sort(key=lambda item: int(item.get("index", 0)))

    with output_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


async def process_single_item(
    builder: SemanticScholarGroundTruthBuilder,
    index: int,
    item: dict[str, Any],
) -> dict[str, Any]:
    original_index = item.get("index", index)
    title = item.get("title", "")
    if s2id := item.get('s2id', ""):
        references = await builder.build_references(title, s2id)
    else:
        references = await builder.build_references(title)
    return {
        "index": original_index,
        "title": title,
        "references": references,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DATASET_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--start", type=int, default=START)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--openalex-rps", type=float, default=4.0)
    parser.add_argument("--openalex-concurrency", type=int, default=3)
    parser.add_argument("--openalex-api-key", action="append", dest="openalex_api_keys", default=OPENALEX_KEYS)
    args = parser.parse_args()

    dataset_path = Path(args.input)
    output_path = Path(args.output)
    completed_indexes = load_completed_indexes(output_path)

    config = ToolConfig(
        openalex_requests_per_second=args.openalex_rps,
        openalex_max_concurrency=args.openalex_concurrency,
        openalex_api_keys=args.openalex_api_keys,
    )
    RateLimit.configure_openalex(
        requests_per_second=config.openalex_requests_per_second,
        enabled=config.openalex_rate_limit_enabled,
        max_concurrency=config.openalex_max_concurrency,
    )

    builder = SemanticScholarGroundTruthBuilder(config)
    await SessionManager.init()
    try:
        with output_path.open("a", encoding="utf-8") as fout:
            for index, item in iter_dataset(dataset_path, ENUMERATE_START):
                original_index = str(item.get("index", index))
                if index < args.start: continue
                if args.limit >= 0 and index >= args.limit: break
                if original_index in completed_indexes: continue

                title = item.get("title", "")
                print(f"[{index}] start: {title}")
                try:
                    result = await process_single_item(builder, index, item)
                except OpenAlexBudgetExceeded as exc:
                    payload = exc.payload or {}
                    print(
                        json.dumps(
                            {
                                "status": "openalex_budget_exceeded",
                                "index": index,
                                "title": title,
                                "retryAfter": payload.get("retryAfter"),
                                "message": payload.get("message", str(exc)),
                            },
                            ensure_ascii=False,
                        )
                    )
                    return
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "status": "failed",
                                "index": index,
                                "title": title,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue

                if not result["references"]:
                    print(f"[{index}] done: 0 references, skipped")
                    continue

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{index}] done: {len(result['references'])} references")
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    asyncio.run(main())
