import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..prompts import REFERENCE_SURVEY_SELECT
from ..utils import extract_json
from ..utility.academic_engine import get_academic_engine
from ..utility.llmclient import AsyncChat
from ..utility.openalex import OPENALEX_SELECT, get_openalex_client
from ..utility.paper_download import (
    PaperDownload,
    S2PaperDownload,
    yield_location,
)
from ..utility.s2 import get_semantic_scholar_client
from ..utility.tool_config import ToolConfig
from ..utility.latex_parser import LatexPaperParser


ORACLE_SELECT = f"{OPENALEX_SELECT},locations,best_oa_location,relevance_score"
SURVEY_TITLE_KEYWORDS = ("survey", "summary", "review", "overview", "comprehensive study")
S2_ENGINE_NAMES = {"semantic scholar", "semantic_scholar", "semanticscholar", "s2"}


class SurveyDownload(PaperDownload):
    """Download a reference survey and attach ordered section headings."""

    def _post_hook(self, xml_content: str) -> dict:
        titles = self.paper_parser.get_titles(xml_content)
        result = super()._post_hook(xml_content)
        result["titles"] = titles
        print(f"This survey has {len(titles)} titles")
        return result

    def _latex_post_hook(self, paper, latex_content: str = "") -> dict:
        titles = LatexPaperParser(latex_content).get_titles()
        result = super()._latex_post_hook(paper, latex_content)
        result["titles"] = titles
        print(f"This TeX survey has {len(titles)} titles")
        return result


class SurveyS2Download(S2PaperDownload):
    """Download a Semantic Scholar reference survey and attach ordered section headings."""

    def _post_hook(self, xml_content: str) -> dict:
        titles = self.paper_parser.get_titles(xml_content)
        result = super()._post_hook(xml_content)
        result["titles"] = titles
        print(f"This survey has {len(titles)} titles")
        return result

    def _latex_post_hook(self, paper, latex_content: str = "") -> dict:
        titles = LatexPaperParser(latex_content).get_titles()
        result = super()._latex_post_hook(paper, latex_content)
        result["titles"] = titles
        print(f"This TeX survey has {len(titles)} titles")
        return result


class ReferenceSurveySelect(AsyncChat):
    PROMPT: str = REFERENCE_SURVEY_SELECT

    def _availability(self, response: str, context: dict):
        results = extract_json(response)
        titles = [item["title"] for item in results["surveys"]]
        title_to_paper = {item["title"]: item for item in context["surveys"]}
        # 若title不在title_to_paper中，会弹出KeyError给tenacity捕捉并重试。
        return [title_to_paper[title] for title in titles]
    
    def _get_abstract_part(self, abstract: str):
        if not abstract: return ""
        return abstract.split(".")[0] + "."

    def _organize_inputs(self, inputs):
        candidates = "\n".join(f"{i + 1:02d}. Title: {paper['title']}\n    Abstract: {paper['abstract']}" for i, paper in enumerate(inputs["surveys"]))
        prompt = self.PROMPT.format(query=inputs["query"], candidates=candidates)
        print(prompt)
        return prompt, {"surveys": inputs["surveys"]}


class GetReferenceSurveys:
    SELECT = f"{OPENALEX_SELECT},best_oa_location,locations"

    def __init__(self, config: ToolConfig):
        self.config = config
        self.eval_date = config.evaluation_date
        self.openalex_downloader = SurveyDownload(config)
        self.semantic_scholar_downloader = SurveyS2Download(config)
        self.survey_select = ReferenceSurveySelect(config.llm_server_info, config.sampling_params)
        self.openalex = get_openalex_client(config)
        self.semantic_scholar = get_semantic_scholar_client(config)
        self.academic_engine = get_academic_engine(config)
        self.academic_engine_type = config.default_academic_search_engine

    def _uses_semantic_scholar_engine(self) -> bool:
        return (self.academic_engine_type or "").strip().lower() in S2_ENGINE_NAMES

    async def _search_surveys(self, query: str, limit: int = 50):
        if self.academic_engine_type == "openalex": 
            payload = await self.academic_engine.search_works(
                search=query,
                filter={
                    "to_publication_date": self.eval_date.strftime("%Y-%m-%d"), 
                    "title.search": 'survey|summary|review|overview|"comprehensive study"'
                },
                per_page=limit,
                select=self.SELECT,
            )
        else:
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
        if any(keyword in title for keyword in SURVEY_TITLE_KEYWORDS): return paper["cited_by_count"] > 10
        if self.academic_engine_type != "openalex":
            publication_types = {str(item).lower() for item in paper.get("publicationTypes", []) or []}
            publication_types |= {str(item).lower() for item in paper.get("publication_types", []) or []}
            if "review" in publication_types: return paper["cited_by_count"] > 10
        return False

    def _is_openalex_record(self, paper: dict) -> bool:
        paper_id = str(paper.get("id") or "").strip()
        return paper_id.startswith("W")

    def _is_semantic_scholar_record(self, paper: dict) -> bool:
        return bool(
            paper.get("paperId")
            or paper.get("corpusId")
            or (paper.get("externalIds") or {}).get("CorpusId")
            or (paper.get("external_ids") or {}).get("CorpusId")
        )

    async def _resolve_openalex(self, survey: dict) -> dict:
        if self._is_openalex_record(survey):
            return survey
        try:
            return await self.openalex.find_work_by_title(survey["title"], select=self.SELECT) or {}
        except Exception as exc:
            print(f"referenceSurveyOpenAlex {survey['title']} {exc}")
            return {}

    async def _resolve_semantic_scholar(self, survey: dict) -> dict:
        if self._is_semantic_scholar_record(survey):
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

    def _survey_title(self, item: dict) -> str:
        return (
            (item.get("semantic_scholar") or {}).get("title")
            or (item.get("openalex") or {}).get("title")
            or ""
        )

    def _survey_openalex_id(self, item: dict) -> str:
        return str((item.get("openalex") or {}).get("id") or "").strip()

    def _survey_file_name(self, item: dict, index: int) -> str:
        title = self._survey_openalex_id(item) or self._survey_title(item) or f"survey_{index}"
        safe_title = re.sub(r"[^0-9A-Za-z._-]+", "_", title).strip("._")[:90] or f"survey_{index}"
        digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
        return f"{index:03d}_{safe_title}_{digest}.json"

    async def _download_selected_surveys(self, surveys: list[dict]) -> list[dict]:
        async def _single(survey: dict):
            openalex_meta = await self._resolve_openalex(survey)
            item = {"openalex": openalex_meta or {}, "semantic_scholar": {}}
            full_content = None
            if openalex_meta:
                full_content, attempted_urls = await self._download_openalex_paper(openalex_meta)
            else:
                attempted_urls = set()
            if not full_content:
                semantic_meta = await self._resolve_semantic_scholar(survey)
                item["semantic_scholar"] = semantic_meta or {}
                full_content = await self._download_semantic_scholar_paper(item["semantic_scholar"], attempted_urls)
            if not full_content: return None
            item["full_content"] = full_content
            return item

        tasks = [asyncio.create_task(_single(survey)) for survey in surveys]
        resolved = []
        for task in asyncio.as_completed(tasks):
            item = await task
            if item:
                resolved.append(item)
        print(f"referenceSurveyDownload: {len(resolved)} downloaded surveys")
        return resolved

    async def _download_surveys(self, papers: list[dict]):
        downloaded = {}
        for paper in papers:
            meta = paper.get("openalex") or paper.get("semantic_scholar") or {}
            skeleton = (paper.get("full_content") or {}).get("full_content") or {}
            if skeleton:
                downloaded[meta["title"]] = {
                    "titles": (paper.get("full_content") or {}).get("titles", []),
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
        use_s2 = self._uses_semantic_scholar_engine()

        async def _single(survey: dict):
            meta = survey.get("semantic_scholar") if use_s2 else survey.get("openalex")
            meta = meta or {}
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
        try:
            surveys_raw = await self._search_surveys(query)
        except Exception as exc:
            print(f"referenceSurveySearch {exc}")
            surveys_raw = []
        print(f"referenceSurveySource: {len(surveys_raw)} candidate surveys")

        review_like = [x for x in surveys_raw if self._is_review_like(x)]
        print(f"referenceSurveySource: {len(review_like)} rule-filtered surveys")
        if not review_like:
            return {"reference_papers": {}, "reference_surveys": {}}
        try:
            selected_candidates = await self.survey_select.call(inputs={"query": query, "surveys": review_like})
        except Exception as exc:
            print(f"referenceSurveySelect {exc}")
            selected_candidates = []
        prints = "\n".join([f'- {x["title"]}' for x in selected_candidates]) if selected_candidates else "0"
        print(f"Selected referenceSurvey = {prints}")
        if not selected_candidates:
            return {"reference_papers": {}, "reference_surveys": {}}
        selected = await self._download_selected_surveys(selected_candidates)
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
        print(f"source 1 done with {len(selected)} downloaded surveys and {len(golden_references_meta)}")

        return {
            "reference_papers": golden_references_meta,
            "reference_surveys": selected,
        }
