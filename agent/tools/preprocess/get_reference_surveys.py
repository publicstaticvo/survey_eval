import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import jsonschema

from ..prompts import REFERENCE_SURVEY_SCHEMA, REFERENCE_SURVEY_SELECT
from ..utils import extract_json
from ..utility.academic_engine import get_academic_engine
from ..utility.llmclient import AsyncChat
from ..utility.openalex import OPENALEX_SELECT, get_openalex_client
from ..utility.paper_download import (
    PaperDownload,
    SemanticScholarPaperDownload,
    yield_location,
)
from ..utility.s2 import get_semantic_scholar_client
from ..utility.tool_config import ToolConfig
from .extract_scope import ScopeClaimExtract


ORACLE_SELECT = f"{OPENALEX_SELECT},locations,best_oa_location,relevance_score"
SURVEY_TITLE_KEYWORDS = ("survey", "summary", "review", "overview", "comprehensive study")


class ReferenceSurveySelect(AsyncChat):
    PROMPT: str = REFERENCE_SURVEY_SELECT

    def _availability(self, response: str, context: dict):
        results = extract_json(response)
        jsonschema.validate(results, REFERENCE_SURVEY_SCHEMA)
        title_to_scope = context["title_to_scope"]
        selected = []
        for item in results["surveys"]:
            title = item["title"]
            assert title in title_to_scope
            allowed_subtopics = set(title_to_scope[title]["section_map"].values())
            allowed_subtopics.update(title_to_scope[title]["aspect_list"])
            for subtopic in item["subtopics_covered"]:
                assert subtopic in allowed_subtopics
            selected.append({**title_to_scope[title], **item})
        return selected

    def _organize_inputs(self, inputs):
        scopes = inputs["surveys"]
        title_to_scope = {item["title"]: item for item in scopes}
        prompt = self.PROMPT.format(
            query=inputs["query"],
            candidates=json.dumps(scopes, ensure_ascii=False, indent=2),
        )
        return prompt, {"title_to_scope": title_to_scope}


class GetReferenceSurveys:
    SELECT = f"{OPENALEX_SELECT},best_oa_location,locations"

    def __init__(self, config: ToolConfig):
        self.config = config
        self.eval_date = config.evaluation_date
        self.openalex_downloader = PaperDownload(config)
        self.semantic_scholar_downloader = SemanticScholarPaperDownload(config)
        self.scope_extractor = ScopeClaimExtract(config)
        self.survey_select = ReferenceSurveySelect(config.llm_server_info, config.sampling_params)
        self.openalex = get_openalex_client(config)
        self.semantic_scholar = get_semantic_scholar_client(config)
        self.academic_engine = get_academic_engine(config)
        self.min_citations = config.minimum_reference_survey_citations
        self.debug_dir: Path | None = None

    async def _search_surveys(self, query: str, limit: int = 50):
        search_query = f'{query} + (survey | summary | review | overview | "comprehensive study")'
        print(f"Search keywords: {search_query}")
        payload = await self.academic_engine.search_works(
            search=search_query,
            filter={"to_publication_date": self.eval_date.strftime("%Y-%m-%d")},
            per_page=limit,
            select=self.SELECT,
        )
        candidates = [item for item in payload.get("results", []) if (item.get("title") or "").strip()]
        if len(candidates) > limit:
            print(f"referenceSurveySearch clipped {len(candidates)} candidates to limit={limit}")
        return candidates[:limit]

    def _is_review_like(self, paper: dict) -> bool:
        title = (paper.get("title") or "").lower()
        publication_types = {str(item).lower() for item in paper.get("publicationTypes", []) or []}
        publication_types |= {str(item).lower() for item in paper.get("publication_types", []) or []}
        return any(keyword in title for keyword in SURVEY_TITLE_KEYWORDS) or "review" in publication_types

    async def _resolve_openalex(self, survey: dict) -> dict:
        if self.config.default_academic_search_engine == "openalex" and survey.get("id"):
            return survey
        try:
            return await self.openalex.find_work_by_title(survey["title"], select=self.SELECT) or {}
        except Exception as exc:
            print(f"referenceSurveyOpenAlex {survey['title']} {exc}")
            return {}

    async def _resolve_semantic_scholar(self, survey: dict) -> dict:
        if self.config.default_academic_search_engine == "semantic scholar" and survey.get("id"):
            return survey
        try:
            return await self.semantic_scholar.find_work_by_title(survey["title"]) or {}
        except Exception as exc:
            print(f"referenceSurveyS2 {survey['title']} {exc}")
            return {}

    async def _download_openalex_paper(self, metadata: dict) -> tuple[dict | None, set[str]]:
        if not metadata:
            return None, set()
        attempted_urls = set(yield_location(metadata))
        try:
            result = await self.openalex_downloader.download_single_paper(
                metadata,
                openalex_id=metadata.get("id", ""),
            )
            return result, attempted_urls
        except Exception as exc:
            print(f"referenceSurveyOpenAlexDownload {metadata.get('title', '')} {exc}")
            return None, attempted_urls

    async def _download_semantic_scholar_paper(self, metadata: dict, excluded_urls: set[str]) -> dict | None:
        if not metadata:
            return None
        try:
            return await self.semantic_scholar_downloader.download_single_paper(
                metadata,
                excluded_urls=excluded_urls,
            )
        except Exception as exc:
            print(f"referenceSurveyS2Download {metadata.get('title', '')} {exc}")
            return None

    async def _download_full_content(self, item: dict) -> dict | None:
        openalex_result, attempted_urls = await self._download_openalex_paper(item.get("openalex") or {})
        if openalex_result:
            return openalex_result
        return await self._download_semantic_scholar_paper(item.get("semantic_scholar") or {}, attempted_urls)

    def _survey_title(self, item: dict) -> str:
        return (
            (item.get("semantic_scholar") or {}).get("title")
            or (item.get("openalex") or {}).get("title")
            or ""
        )

    def _survey_file_name(self, item: dict, index: int) -> str:
        title = self._survey_title(item) or f"survey_{index}"
        safe_title = re.sub(r"[^0-9A-Za-z._-]+", "_", title).strip("._")[:90] or f"survey_{index}"
        digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
        return f"{index:03d}_{safe_title}_{digest}.json"

    def _save_downloaded_surveys(self, surveys: list[dict]):
        if self.debug_dir is None:
            return
        output_dir = Path(self.debug_dir) / "surveys"
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(surveys, 1):
            payload = {
                "title": self._survey_title(item),
                "metadata": {
                    "openalex": item.get("openalex") or {},
                    "semantic_scholar": item.get("semantic_scholar") or {},
                },
                "paper": item.get("full_content") or {},
            }
            path = output_dir / self._survey_file_name(item, index)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"referenceSurveyDebug: saved {len(surveys)} surveys to {output_dir}")

    def _load_downloaded_surveys(self) -> list[dict]:
        if self.debug_dir is None:
            return []
        input_dir = Path(self.debug_dir) / "surveys"
        if not input_dir.exists():
            return []
        surveys = []
        for path in sorted(input_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"referenceSurveyDebugLoad {path.name} {exc}")
                continue
            metadata = payload.get("metadata") or {}
            paper = payload.get("paper") or {}
            if not paper:
                continue
            surveys.append(
                {
                    "openalex": metadata.get("openalex") or {},
                    "semantic_scholar": metadata.get("semantic_scholar") or {},
                    "full_content": paper,
                }
            )
        if surveys:
            print(f"referenceSurveyDebug: loaded {len(surveys)} surveys from {input_dir}")
        return surveys

    async def _filter_surveys(self, surveys: list[dict]) -> list[dict]:
        review_like = [paper for paper in surveys if self._is_review_like(paper)]
        print(f"referenceSurveyRuleFilter: {len(review_like)} review-like surveys")

        async def _single(survey: dict):
            openalex_task = asyncio.create_task(self._resolve_openalex(survey))
            semantic_task = asyncio.create_task(self._resolve_semantic_scholar(survey))
            openalex_meta, semantic_meta = await asyncio.gather(openalex_task, semantic_task)
            cited_by_count = int((semantic_meta or survey).get("cited_by_count", 0) or 0)
            if cited_by_count < self.min_citations:
                return None
            item = {"openalex": openalex_meta or {}, "semantic_scholar": semantic_meta or survey}
            full_content = await self._download_full_content(item)
            if not full_content:
                return None
            item["full_content"] = full_content
            return item

        tasks = [asyncio.create_task(_single(survey)) for survey in review_like]
        resolved = []
        for task in asyncio.as_completed(tasks):
            item = await task
            if item:
                resolved.append(item)
        print(f"referenceSurveyRuleFilter: {len(resolved)} downloaded surveys")
        return resolved

    async def _extract_scope_declarations(self, surveys: list[dict]) -> list[dict]:
        scopes = []

        async def _single(item: dict):
            paper = item.get("full_content", {}).get("full_content", {})
            title = (
                (item.get("semantic_scholar") or {}).get("title")
                or (item.get("openalex") or {}).get("title")
                or ""
            )
            try:
                self_scope = await self.scope_extractor(
                    paper,
                    candidate_types=["introduction", "conclusion"],
                )
            except Exception as exc:
                print(f"referenceSurveyScope {title} {exc}")
                return None
            if len(self_scope["section_map"]) + len(self_scope["aspect_list"]) < 3:
                return None
            self_scope["evidence_records"] = [x["evidence"] for x in self_scope["evidence_records"]]
            del self_scope["errors"]
            return {"title": title, **self_scope, "_survey_record": item}

        tasks = [asyncio.create_task(_single(item)) for item in surveys]
        for task in asyncio.as_completed(tasks):
            item = await task
            if item:
                scopes.append(item)
        print(f"referenceSurveyScope: {len(scopes)} surveys with >=3 scope items")
        return scopes

    async def _download_surveys(self, papers: list[dict]):
        downloaded = {}
        for paper in papers:
            meta = paper.get("openalex") or paper.get("semantic_scholar") or {}
            skeleton = (paper.get("full_content") or {}).get("full_content") or {}
            if skeleton:
                downloaded[meta["title"]] = {
                    "skeleton": skeleton,
                    "abstract": (paper.get("full_content") or {}).get("abstract", ""),
                    "meta": meta,
                    "metadata": {
                        "openalex": paper.get("openalex") or {},
                        "semantic_scholar": paper.get("semantic_scholar") or {},
                    },
                }
        return downloaded

    async def _fetch_cited_by_neighbors(self, survey_id: str) -> dict[str, dict[str, Any]]:
        papers = {}
        results = await self.academic_engine.get_references(
            survey_id,
            offset=0,
            limit=9999,
            fields=ORACLE_SELECT,
            to_publication_date=self.eval_date.strftime("%Y-%m-%d"),
        )
        for paper in results.get("results", [])[:9999]:
            if paper.get("id"):
                papers[paper["id"]] = paper
        return papers

    async def _collect_reference_papers(self, surveys: list[dict]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        paper_meta, citation_counter = {}, {}

        async def _single(survey: dict):
            meta = survey.get("openalex") or survey.get("semantic_scholar") or {}
            survey_id = meta.get("id")
            if not survey_id:
                return survey_id, {}
            try:
                return survey_id, await self._fetch_cited_by_neighbors(survey_id)
            except Exception as exc:
                print(f"referenceSurveyRefs {meta.get('title', survey_id)} {exc}")
                return survey_id, {}

        tasks = [asyncio.create_task(_single(survey)) for survey in surveys]
        for task in asyncio.as_completed(tasks):
            _, references = await task
            for paper_id, paper in references.items():
                paper_meta.setdefault(paper_id, paper)
                citation_counter[paper_id] = citation_counter.get(paper_id, 0) + 1

        return paper_meta, citation_counter

    async def __call__(self, query: str):
        surveys = self._load_downloaded_surveys()
        if not surveys:
            try:
                surveys_raw = await self._search_surveys(query)
            except Exception as exc:
                print(f"referenceSurveySearch {exc}")
                surveys_raw = []
            print(f"referenceSurveySource: {len(surveys_raw)} candidate surveys")

            surveys = await self._filter_surveys(surveys_raw)
            print(f"referenceSurveySource: {len(surveys)} rule-filtered surveys")
            self._save_downloaded_surveys(surveys)
        else:
            print(f"referenceSurveySource: reuse {len(surveys)} downloaded surveys")

        scopes = await self._extract_scope_declarations(surveys)
        try:
            selected_scopes = await self.survey_select.call(inputs={"query": query, "surveys": scopes})
        except Exception as exc:
            print(f"referenceSurveySelect {exc}")
            selected_scopes = []

        selected = [item["_survey_record"] for item in selected_scopes]
        prints = "\n".join([f'- {x["title"]}' for x in selected_scopes]) if selected_scopes else "0"
        print(f"Selected referenceSurvey =\n{prints}")
        if not selected:
            return {"reference_papers": {}, "reference_surveys": {}}

        golden_references_meta, citation_counter = await self._collect_reference_papers(selected)
        print(f"referenceSurveySource: {len(citation_counter)} cited ids")
        for paper_id, metadata in golden_references_meta.items():
            metadata["survey_cited_by_count"] = citation_counter[paper_id]
            metadata["candidate_source"] = "high_consensus" if citation_counter[paper_id] >= 2 else "single_reference"

        for paper in selected:
            meta = paper.get("openalex") or paper.get("semantic_scholar") or {}
            survey_paper = dict(meta)
            survey_paper["candidate_source"] = "reference_survey"
            if survey_paper.get("id"):
                golden_references_meta[survey_paper["id"]] = survey_paper
        print(f"referenceSurveySource: {len(golden_references_meta)} reference metas")

        downloaded = await self._download_surveys(selected)
        print(f"source 1 done with {len(downloaded)} downloaded surveys and {len(golden_references_meta)}")

        return {
            "reference_papers": golden_references_meta,
            "reference_surveys": downloaded,
        }
