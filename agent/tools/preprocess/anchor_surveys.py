import asyncio
import random
import aiohttp
from datetime import timedelta
from typing import Dict, Any, List, Optional

from .query_expand import QueryExpand
from ..prompts import ANCHOR_SURVEY_SELECT
from ..utility.llmclient import AsyncChat
from ..utility.openalex import OPENALEX_SELECT, get_openalex_client
from ..utility.paper_download import PaperDownload
from ..utility.request_utils import HEADERS, SessionManager
from ..utility.tool_config import ToolConfig
from .utils import extract_json, valid_check

ORACLE_SELECT = f"{OPENALEX_SELECT},locations,best_oa_location,relevance_score"
STRUCTURE_TITLES = ["introduction", "background", "conclusion", "discussion", "experiment", "result", "method", "limitation"]


class SurveyDownload(PaperDownload):    

    def _check_title(self, title: str):
        for keyword in STRUCTURE_TITLES:
            title = title.replace(keyword, "")
        return len(title) >= 10

    def _flatten_title_paths(self, paper_skeleton: dict):
        titles = []

        def _walk(section: dict):
            title = (section.get("title") or "").strip()
            if title and self._check_title(title.lower()):
                titles.append(title)
            for child in section.get("sections", []):
                _walk(child)

        for section in paper_skeleton.get("sections", []): _walk(section)
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
    PROMPT: str = ANCHOR_SURVEY_SELECT

    def _availability(self, response: str, context: dict):
        results = extract_json(response)
        titles = [item["title"] for item in results["surveys"]]
        title_to_paper = {item["title"]: item for item in context["surveys"]}
        return [title_to_paper[title] for title in titles if title in title_to_paper]

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(query=inputs["query"], titles="\n".join(f"- {paper['title']}" for paper in inputs["surveys"]))
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
        search_query = f'{query} + (survey | summary | review | overview | "comprehensive study")'
        from_date = (self.eval_date - timedelta(days=1096)).strftime("%Y-%m-%d")
        payload = await self._semantic_scholar_get(
            "/paper/search/bulk",
            {
                "query": search_query,
                "limit": limit,
                "fields": "title,abstract",
                "sort": "citationCount",
                "minCitationCount": 40,
                "publicationDateOrYear": f'{from_date}:{self.eval_date.strftime("%Y-%m-%d")}',
            },
        )
        surveys = []
        for item in payload.get("data") or []:
            title = (item.get("title") or "").strip()
            if title:
                surveys.append({"title": title, "abstract": item.get("abstract")})
        return surveys

    async def _resolve_openalex_surveys(self, surveys: list[dict]):
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

        try:
            selected = await self.survey_select.call(inputs={"query": query, "surveys": semantic_surveys})
        except Exception as e:
            print(f"AnchorSurveySelect {e}")
            selected = []
        print(f"AnchorSurveyFetch: {len(selected)} selected surveys")

        # selected is already a list of Semantic Scholar survey records.
        openalex_selected = await self._resolve_openalex_surveys(selected)
        print(f"AnchorSurveyFetch: {len(openalex_selected)} real surveys")
        if not openalex_selected:
            return {"anchor_papers": {}, "anchor_surveys": {}}

        citation_counter = {}
        for paper in openalex_selected:
            for ref in paper.get("referenced_works", []):
                citation_counter[ref] = citation_counter.get(ref, 0) + 1

        cited_ids = list(citation_counter)
        print(f"AnchorSurveyFetch: {len(cited_ids)} cited ids")
        anchor_meta = await self._get_paper_meta_by_id(cited_ids)
        for paper_id, metadata in anchor_meta.items():
            metadata["survey_cited_by_count"] = citation_counter[paper_id]
            metadata["candidate_source"] = "high_consensus" if citation_counter[paper_id] >= 2 else "single_anchor"

        for paper in openalex_selected:
            survey_paper = dict(paper)
            survey_paper["survey_cited_by_count"] = len(openalex_selected)
            survey_paper["candidate_source"] = "anchor_survey"
            anchor_meta[survey_paper["id"]] = survey_paper
        print(f"AnchorSurveyFetch: {len(anchor_meta)} anchor metas")

        downloaded = await self._download_surveys(openalex_selected[:5])
        print(f"AnchorSurveyFetch: {len(downloaded)} downloaded surveys")

        return {
            "anchor_papers": anchor_meta,
            "anchor_surveys": downloaded,
        }


class DirectSeedGraphSource:
    def __init__(self, config: ToolConfig):
        self.eval_date = config.evaluation_date
        self.openalex = get_openalex_client(config)

    async def search_seed_papers(self, query: str, uplimit: int = 100) -> dict[str, dict[str, Any]]:
        papers = {}
        total = uplimit
        for page in range(1, (uplimit - 1) // 200 + 2):
            results = await self.openalex.search_works(
                "works",
                search=query.strip(),
                per_page=min(200, uplimit),
                page=page,
                select=ORACLE_SELECT,
                filter={"to_publication_date": self.eval_date.strftime("%Y-%m-%d")},
            )
            total = min(total, results.get("count", 0) or total)
            for paper in results.get("results", []):
                if paper.get("id"):
                    item = dict(paper)
                    item["candidate_source"] = "seed"
                    item["candidate_sources"] = ["seed"]
                    papers[item["id"]] = item
            if len(papers) >= total:
                break
        return dict(list(papers.items())[:uplimit])

    async def _fetch_cited_by_neighbors(self, seed_id: str) -> dict[str, dict[str, Any]]:
        neighbors = {}
        page, total = 1, 1
        while (page - 1) * 200 < total:
            results = await self.openalex.search_works(
                "works",
                filter={"cited_by": seed_id, "to_publication_date": self.eval_date.strftime("%Y-%m-%d")},
                per_page=200,
                page=page,
                select=ORACLE_SELECT,
                sort="cited_by_count:desc",
            )
            total = results.get("count", 0) or 0
            for paper in results.get("results", []):
                if paper.get("id"):
                    neighbors[paper["id"]] = paper
            page += 1
        return neighbors

    async def graph_expansion(self, seed_library: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        papers = {}
        neighbor_seed_map: dict[str, set[str]] = {}

        async def _single(seed_id: str):
            try:
                return seed_id, await self._fetch_cited_by_neighbors(seed_id)
            except Exception as exc:
                print(f"CandidatePool cited_by {seed_id} {exc}")
                return seed_id, {}

        tasks = [asyncio.create_task(_single(seed_id)) for seed_id in seed_library]
        for task in asyncio.as_completed(tasks):
            seed_id, neighbors = await task
            for paper_id, paper in neighbors.items():
                if paper_id in seed_library:
                    continue
                papers[paper_id] = paper
                neighbor_seed_map.setdefault(paper_id, set()).add(seed_id)
                # If needed later, cites-neighbor collection can be added here as a separate source.

        for paper_id, paper in papers.items():
            paper["candidate_source"] = "neighbor"
            paper["candidate_sources"] = ["neighbor"]
            paper["neighbor_seed_count"] = len(neighbor_seed_map.get(paper_id, set()))
            paper["neighbor_seed_ids"] = sorted(neighbor_seed_map.get(paper_id, set()))
        return papers

    async def __call__(self, query: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        seed_library = await self.search_seed_papers(query, uplimit=100)
        neighbors = await self.graph_expansion(seed_library)
        return seed_library, neighbors


class RecentSemanticSource:
    def __init__(self, config: ToolConfig):
        self.query_expand = QueryExpand(config)

    async def __call__(self, query: str) -> dict[str, Any]:
        query_data = await self.query_expand(query)
        papers = {}
        for paper_id, paper in query_data.get("library", {}).items():
            item = dict(paper)
            item["candidate_source"] = "new_papers"
            item["candidate_sources"] = ["new_papers"]
            papers[paper_id] = item
        return {"queries": query_data.get("queries", []), "new_papers": papers}
