import asyncio
import math
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from ..utility.utils import valid_check
from ..prompts import REFERENCE_SURVEY_SELECT
from ..utils import months_between, parse_date, extract_json
from ..utility.academic_engine import get_academic_engine
from ..utility.llmclient import AsyncChat
from ..utility.openalex import OPENALEX_SELECT, get_openalex_client
from ..utility.paper_download import PaperDownload, download_bytes_to_memory
from ..utility.tool_config import ToolConfig
from ..utility.latex_parser import LatexPaperParser


ORACLE_SELECT = f"{OPENALEX_SELECT},locations,best_oa_location,relevance_score"
SURVEY_TITLE_KEYWORDS = ("survey", "summary", "review", "overview", "comprehensive study")


class SurveyDownload(PaperDownload):
    """Only extract ordered section headings from reference surveys."""

    def _post_hook(self, xml_content: str):
        try:
            titles = self.paper_parser.get_titles(xml_content)
            print(f"This survey has {len(titles)} titles")
            return titles, None
        except Exception as exc:
            print(f"Fatal: no survey title parser {exc}")
            return [], None

    def _latex_post_hook(self, paper, latex_content: str = ""):
        try:
            titles = LatexPaperParser(latex_content).get_titles()
            print(f"This TeX survey has {len(titles)} titles")
            return titles, None
        except Exception as exc:
            print(f"Fatal: no TeX survey title parser {exc}")
            return [], None

    async def _try_arxiv_source(self, arxiv_id: str) -> dict:
        src_url = f"https://arxiv.org/src/{arxiv_id}"
        try:
            buffer = await download_bytes_to_memory(src_url)
            if not buffer:
                return {"result": None, "download_error": True, "parse_error": False}
            if buffer[:5] == b"%PDF-":
                print(f"{src_url} returned PDF instead of TeX source")
                return {"result": None, "download_error": False, "parse_error": True}
            print(f"Downloaded TeX source from {src_url}")
            tmp_parent = Path("C:/tmp") if Path("C:/tmp").exists() else None
            with tempfile.TemporaryDirectory(
                prefix=f"arxiv_title_{re.sub(r'[^0-9A-Za-z]+', '_', arxiv_id)}_",
                dir=str(tmp_parent) if tmp_parent else None,
                ignore_cleanup_errors=True,
            ) as tmp:
                source_dir = Path(tmp)
                try:
                    self._safe_extract_tar(buffer, source_dir)
                except tarfile.TarError:
                    tex_path = source_dir / "source.tex"
                    tex_path.write_bytes(buffer)
                main_tex = self._find_main_tex(source_dir)
                if not main_tex:
                    print(f"{src_url} No TeX file")
                    return {"result": None, "download_error": False, "parse_error": True}
                result = self._latex_post_hook(None, self._read_text_file(main_tex))
                if result and result[0]:
                    return {"result": result, "download_error": False, "parse_error": False}
                return {"result": None, "download_error": False, "parse_error": True}
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"arXiv title source failed for {arxiv_id}: {exc}")
            return {"result": None, "download_error": False, "parse_error": True}


class ReferenceSurveySelect(AsyncChat):
    PROMPT: str = REFERENCE_SURVEY_SELECT

    def _availability(self, response: str, context: dict):
        results = extract_json(response)
        titles = [item["title"] for item in results.get("surveys", [])]
        title_to_paper = {item["title"]: item for item in context["surveys"]}
        return [title_to_paper[title] for title in titles if title in title_to_paper]

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(
            query=inputs["query"],
            titles="\n".join(f"- {paper['title']}" for paper in inputs["surveys"]),
        )
        return prompt, {"surveys": inputs["surveys"]}


class GetReferenceSurveys:
    SELECT = f"{OPENALEX_SELECT},best_oa_location,locations"

    def __init__(self, config: ToolConfig):
        self.config = config
        self.eval_date = config.evaluation_date
        self.survey_download = SurveyDownload(config)
        self.survey_select = ReferenceSurveySelect(config.llm_server_info, config.sampling_params)
        self.openalex = get_openalex_client(config)
        self.academic_engine = get_academic_engine(config)
        self.keep_ratio = config.citation_velocity_keep_ratio

    async def _search_surveys(self, query: str, limit: int = 100):
        search_query = f'{query} + (survey | summary | review | overview | "comprehensive study")'
        print(f"Search keywords: {search_query}")
        payload = await self.academic_engine.search_works(
            "works",
            search=search_query,
            filter={"to_publication_date": self.eval_date.strftime("%Y-%m-%d")},
            per_page=limit,
            select=self.SELECT,
        )
        return [item for item in payload.get("results", []) if (item.get("title") or "").strip()]

    def _is_review_like(self, paper: dict) -> bool:
        title = (paper.get("title") or "").lower()
        publication_types = {str(item).lower() for item in paper.get("publicationTypes", []) or []}
        publication_types |= {str(item).lower() for item in paper.get("publication_types", []) or []}
        return any(keyword in title for keyword in SURVEY_TITLE_KEYWORDS) or "review" in publication_types

    def _citation_count_by_eval_date(self, paper: dict) -> int:
        citation_count = int(paper.get("cited_by_count", paper.get("citationCount", 0)) or 0)
        counts_by_year = paper.get("counts_by_year", []) or []
        if not counts_by_year:
            return citation_count
        eval_year = self.eval_date.year
        if citation_count:
            for item in counts_by_year:
                if int(item.get("year", 0) or 0) > eval_year:
                    citation_count -= int(item.get("cited_by_count", 0) or 0)
            return max(0, citation_count)
        return sum(
            int(item.get("cited_by_count", 0) or 0)
            for item in counts_by_year
            if int(item.get("year", 0) or 0) <= eval_year
        )

    def _survey_impact_score(self, paper: dict) -> float:
        publication = parse_date(paper.get("publication_date"))
        months = months_between(publication, self.eval_date) if publication else 1.0
        return self._citation_count_by_eval_date(paper) / math.log(1 + max(months, 1.0))

    async def _filter_surveys(self, surveys: list[dict]) -> list[dict]:
        review_like = [paper for paper in surveys if self._is_review_like(paper)]
        print(f"referenceSurveyRuleFilter: {len(review_like)} review-like surveys")
        if not review_like:
            return []

        resolved = []

        async def _single(survey: dict):
            if self.config.default_academic_search_engine == "openalex" and survey.get("id"):
                merged = dict(survey)
                merged["openalex_paper"] = survey
                merged["impact_score"] = self._survey_impact_score(survey)
                return merged
            try:
                paper = await self.openalex.find_work_by_title(survey["title"], select=self.SELECT)
            except Exception as exc:
                print(f"referenceSurveyOpenAlex {survey['title']} {exc}")
                return None
            if not paper or not valid_check(survey["title"], paper.get("title", "")):
                return None
            merged = dict(survey)
            merged["openalex_paper"] = paper
            merged["impact_score"] = self._survey_impact_score(paper)
            return merged

        tasks = [asyncio.create_task(_single(survey)) for survey in review_like]
        for task in asyncio.as_completed(tasks):
            item = await task
            if item:
                resolved.append(item)

        if not resolved:
            return []
        resolved.sort(key=lambda item: item.get("impact_score", 0.0), reverse=True)
        keep_count = max(1, math.ceil(len(resolved) * self.keep_ratio))
        selected = resolved[:keep_count]
        print(f"referenceSurveyRuleFilter: {len(selected)}/{len(resolved)} kept by impact top {self.keep_ratio:.0%}")
        return selected

    async def _download_surveys(self, papers: list[dict]):
        downloaded = {}
        for paper in papers:
            try:
                survey = await self.survey_download.download_single_paper(paper_meta=paper, title=paper.get("title", ""))
            except Exception as exc:
                print(f"SurveyDownload {paper.get('title', '')} {exc}")
                survey = None
            if isinstance(survey, tuple):
                titles, _ = survey
                if titles:
                    downloaded[paper["title"]] = {"titles": titles, "meta": paper}
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
            survey_id = survey.get("id")
            if not survey_id:
                return survey_id, {}
            try:
                return survey_id, await self._fetch_cited_by_neighbors(survey_id)
            except Exception as exc:
                print(f"referenceSurveyRefs {survey.get('title', survey_id)} {exc}")
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

        surveys = await self._filter_surveys(surveys_raw)
        print(f"referenceSurveySource: {len(surveys)} rule-filtered surveys")

        try:
            selected = await self.survey_select.call(inputs={"query": query, "surveys": surveys})
        except Exception as exc:
            print(f"referenceSurveySelect {exc}")
            selected = []
        print(f"referenceSurveySource: {len(selected)} selected surveys")

        openalex_selected = [item["openalex_paper"] for item in selected if item.get("openalex_paper")]
        print(f"referenceSurveySource: {len(openalex_selected)} real surveys")
        if not openalex_selected:
            return {"reference_papers": {}, "reference_surveys": {}}

        golden_references_meta, citation_counter = await self._collect_reference_papers(openalex_selected)
        print(f"referenceSurveySource: {len(citation_counter)} cited ids")
        for paper_id, metadata in golden_references_meta.items():
            metadata["survey_cited_by_count"] = citation_counter[paper_id]
            metadata["candidate_source"] = "high_consensus" if citation_counter[paper_id] >= 2 else "single_reference"

        for paper in openalex_selected:
            survey_paper = dict(paper)
            survey_paper["candidate_source"] = "reference_survey"
            golden_references_meta[survey_paper["id"]] = survey_paper
        print(f"referenceSurveySource: {len(golden_references_meta)} reference metas")

        downloaded = await self._download_surveys(openalex_selected[:5])
        print(f"source 1 done with {len(downloaded)} downloaded surveys and {len(golden_references_meta)}")

        return {
            "reference_papers": golden_references_meta,
            "reference_surveys": downloaded,
        }
