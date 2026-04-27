import asyncio
import math
import random
import aiohttp
from datetime import timedelta

from .llmclient import AsyncChat
from .openalex import OPENALEX_SELECT, get_openalex_client
from .paper_download import PaperDownload
from .prompts import ANCHOR_SURVEY_SELECT
from .request_utils import HEADERS, SessionManager
from .tool_config import ToolConfig
from .utils import extract_json, valid_check


class SurveyDownload(PaperDownload):
    STRUCTURE_TITLES = ["introduction", "background", "conclusion", "discussion", "experiment", "result", "method", "limitation"]

    def _check_title(self, title: str):
        for x in self.STRUCTURE_TITLES:
            title = title.replace(x, "")
        return len(title) >= 10

    def _flatten_title_paths(self, paper_skeleton: dict):
        titles = []

        def _walk(section: dict):
            title = (section.get("title") or "").strip()
            if title and self._check_title(title.lower()):
                titles.append(title)
            for child in section.get("sections", []):
                _walk(child)

        for section in paper_skeleton.get("sections", []):
            _walk(section)
        return titles

    def _post_hook(self, xml_content: str):
        try:
            paper = self.paper_parser.parse(xml_content, mode="strict")
            paper_skeleton = paper.get_skeleton()
            titles = list(dict.fromkeys(self._flatten_title_paths(paper_skeleton)))
            print(f"This survey has {len(titles)} titles")
            return titles, paper_skeleton
        except Exception as e:
            print(f"Fatal: no survey parser {e}")
            return [], None


class AnchorSurveySelect(AsyncChat):
    """Select the most relevant surveys from candidate surveys."""

    PROMPT: str = ANCHOR_SURVEY_SELECT

    def _availability(self, response: str, context: dict):
        results = extract_json(response)
        titles = [x["title"] for x in results["surveys"]]
        title_to_paper = {x["title"]: x for x in context["surveys"]}
        return [title_to_paper[x] for x in titles if x in title_to_paper]

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(query=inputs["query"], titles="\n".join(f"- {t['title']}" for t in inputs["surveys"]))
        return prompt, {"surveys": inputs["surveys"]}


class AnchorSurveyFetch:
    SELECT = f"{OPENALEX_SELECT},best_oa_location,locations"
    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, config: ToolConfig):
        self.eval_date = config.evaluation_date
        self.survey_download = SurveyDownload(config)
        self.survey_select = AnchorSurveySelect(config.llm_server_info, config.sampling_params)
        self.openalex = get_openalex_client(config)

    async def _semantic_scholar_get(self, endpoint: str, params: dict):
        """Call Semantic Scholar and retry on 429."""
        session = SessionManager.get()
        url = f"{self.SEMANTIC_SCHOLAR_API}{endpoint}"
        while True:
            async with session.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                    continue
                resp.raise_for_status()
                return await resp.json()

    async def _search_semantic_scholar_surveys(self, query: str, limit: int = 20):
        """Search survey-like papers from Semantic Scholar."""
        search_query = f"{query} + (survey | summary | review | overview | \"comprehensive study\")"
        from_date = (self.eval_date - timedelta(days=1096)).strftime("%Y-%m-%d")
        payload = await self._semantic_scholar_get(
            "/paper/search/bulk",
            {"query": search_query, "limit": limit, "fields": "title,abstract", 
             'sort': 'citationCount', 'minCitationCount': 40, 
             'publicationDateOrYear': f'{from_date}:{self.eval_date.strftime("%Y-%m-%d")}'},
        )
        surveys = []
        for item in payload.get("data") or []:
            title = (item.get("title") or "").strip()
            if title:
                surveys.append({"title": title, "abstract": item.get("abstract")})
        return surveys

    async def _resolve_openalex_surveys(self, surveys: list[dict]):
        """Resolve Semantic Scholar survey titles to OpenAlex works."""
        resolved = []

        async def _single(survey: dict):
            try:
                paper = await self.openalex.find_work_by_title(survey["title"], select=self.SELECT)
            except Exception as e:
                print(f"AnchorSurveyOpenAlex {survey['title']} {e}")
                return None
            if not paper or not valid_check(survey["title"], paper.get("title", "")):
                return None
            return paper

        tasks = [asyncio.create_task(_single(survey)) for survey in surveys]
        for task in asyncio.as_completed(tasks):
            paper = await task
            if paper and paper.get("id"):
                resolved.append(paper)
        return resolved

    def _citation_count_by_eval_date(self, paper: dict):
        eval_year = int(self.eval_date.year)
        citation_count = paper.get("cited_by_count", 0) or 0
        if citation_count:
            for item in paper.get("counts_by_year", []):
                if item["year"] > eval_year:
                    citation_count -= item["cited_by_count"]
            return citation_count
        return sum(item["cited_by_count"] for item in paper.get("counts_by_year", []) if item["year"] <= eval_year)

    async def _download_surveys(self, papers: list[dict]):
        downloaded = {}
        for paper in papers:
            try:
                survey = await self.survey_download.download_single_paper(paper_meta=paper, title=paper.get("title", ""))
            except Exception as e:
                print(f"SurveyDownload {paper.get('title', '')} {e}")
                survey = None
            if isinstance(survey, tuple):
                titles, paper_skeleton = survey
                if titles and paper_skeleton:
                    downloaded[paper["title"]] = {"titles": titles, "skeleton": paper_skeleton, "paper": paper}
        return downloaded

    async def _get_paper_meta_by_id(self, papers: list[str]):
        paper_meta = {}

        async def _single(paper_id: str):
            try:
                return await self.openalex.get_entity(paper_id, entity_type="works")
            except Exception as e:
                print(f"AnchorPaperMeta {paper_id} {e}")
                return None

        tasks = [asyncio.create_task(_single(paper_id)) for paper_id in papers if paper_id]
        for task in asyncio.as_completed(tasks):
            paper = await task
            if paper and paper.get("id"):
                paper_meta[paper["id"]] = paper
        return paper_meta

    async def __call__(self, query: str):
        try:
            semantic_surveys = await self._search_semantic_scholar_surveys(query)
        except Exception as e:
            print(f"AnchorSurveySemantic {e}")
            semantic_surveys = []
        print(f"AnchorSurveyFetch: {len(semantic_surveys)} semantic scholar surveys")

        real_surveys = await self._resolve_openalex_surveys(semantic_surveys)
        print(f"AnchorSurveyFetch: {len(real_surveys)} real surveys")
        if not real_surveys:
            return {"anchor_papers": {}, "anchor_surveys": {}}

        try:
            selected = await self.survey_select.call(inputs={"query": query, "surveys": real_surveys})
        except Exception as e:
            print(f"AnchorSurveySelect {e}")
            selected = []
        print(f"AnchorSurveyFetch: {len(selected)} selected surveys")

        downloaded = await self._download_surveys(selected[:5])
        print(f"AnchorSurveyFetch: {len(downloaded)} downloaded surveys")

        selected_titles = set(downloaded)
        selected = [paper for paper in selected if paper["title"] in selected_titles]

        citation_counter = {}
        for paper in selected:
            for ref in paper.get("referenced_works", []):
                citation_counter[ref] = citation_counter.get(ref, 0) + 1

        threshold = max(2, math.ceil(0.6 * max(1, len(selected))))
        anchor_ids = [paper_id for paper_id, count in citation_counter.items() if count >= threshold]
        print(f"AnchorSurveyFetch: {len(anchor_ids)} anchor ids")

        anchor_meta = await self._get_paper_meta_by_id(anchor_ids)
        for paper_id, metadata in anchor_meta.items():
            metadata["survey_cited_by_count"] = citation_counter[paper_id]
        print(f"AnchorSurveyFetch: {len(anchor_meta)} anchor metas")

        return {
            "anchor_papers": anchor_meta,
            "anchor_surveys": downloaded,
        }
