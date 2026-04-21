from typing import List, Dict, Any
from datetime import datetime
import jsonschema

from .utils import extract_json
from .llmclient import AsyncChat
from .tool_config import ToolConfig
from .prompts import QUERY_EXPANSION_PROMPT, QUERY_SCHEMA, SURVEY_SPECIFIED_QUERY_EXPANSION
from .openalex import OPENALEX_SELECT, get_openalex_client, to_openalex

SURVEY_KEYWORDS = "survey|summary|review|overview|synthesis|taxonomy|study"
    

class QueryExpansionLLMClient(AsyncChat):
    PROMPT: str = QUERY_EXPANSION_PROMPT
    def _availability(self, response, context):
        queries = extract_json(response)
        jsonschema.validate(queries, QUERY_SCHEMA)
        return queries
    

class SurveySpecifiedLLMClient(AsyncChat):
    PROMPT: str = SURVEY_SPECIFIED_QUERY_EXPANSION
    def _availability(self, response, context):
        return extract_json(response)['query']


class QueryExpand:
    def __init__(self, config: ToolConfig):
        self.llm = QueryExpansionLLMClient(config.llm_server_info, config.sampling_params)
        self.survey_llm = SurveySpecifiedLLMClient(config.llm_server_info, config.sampling_params)
        self.eval_date = config.evaluation_date
        self.openalex = get_openalex_client(config)

    async def _request_for_papers(self, query: str, uplimit: int, select=f"{OPENALEX_SELECT},relevance_score") -> List[Dict[str, Any]]:
        queries = to_openalex(query)
        search = [*[("default.search", q) for q in queries], ("to_publication_date", self.eval_date.strftime("%Y-%m-%d"))]
        papers = []
        total = uplimit
        for page in range(1, (uplimit - 1) // 200 + 2):
            results = await self.openalex.search_works("works", filter=search, per_page=min(200, uplimit), select=select, page=page)
            total = min(total, results.get("count", 0) or total)
            papers.extend(results.get("results", []))
            if len(papers) >= total:
                break
        return papers[:uplimit]

    async def _request_for_surveys(self, query: str, uplimit: int, select=f"{OPENALEX_SELECT},relevance_score") -> List[Dict[str, Any]]:
        queries = to_openalex(query)
        search = [
            *[("title.search", q) for q in queries],
            ("title.search", SURVEY_KEYWORDS),
            ("to_publication_date", self.eval_date.strftime("%Y-%m-%d")),
        ]
        papers = []
        total = uplimit
        for page in range(1, (uplimit - 1) // 200 + 2):
            results = await self.openalex.search_works("works", filter=search, per_page=min(200, uplimit), select=select, page=page)
            total = min(total, results.get("count", 0) or total)
            papers.extend(results.get("results", []))
            if len(papers) >= total:
                break
        return papers[:uplimit]

    async def __call__(self, query: str, papers_for_each_query: int = 50):
        try:
            queries = await self.llm.call(inputs={"query": query})
        except Exception:
            queries = {
                "strategy": "fallback",
                "core_anchor": [query, query],
                "theoretical_bridge": query,
                "methodological_bridge": query,
            }
        try:
            survey_query = await self.survey_llm.call(inputs={"query": query})
        except Exception:
            survey_query = query

        expanded_queries = [
            *(queries.get("core_anchor") or [])[:2],
            queries.get("theoretical_bridge") or query,
            queries.get("methodological_bridge") or query,
        ]
        library = {}
        valid_queries = []
        for expanded_query in expanded_queries:
            try:
                papers = await self._request_for_papers(expanded_query, papers_for_each_query)
            except Exception:
                continue
            if not papers:
                continue
            valid_queries.append(expanded_query)
            for paper in papers:
                current = library.get(paper["id"])
                if current is None or paper.get("relevance_score", 0) > current.get("relevance_score", 0):
                    library[paper["id"]] = paper | {"query": expanded_query}
        try:
            survey_candidates = await self._request_for_surveys(
                survey_query,
                papers_for_each_query,
                f"{OPENALEX_SELECT},best_oa_location,locations,relevance_score",
            )
        except Exception:
            survey_candidates = []
        return {
            "queries": valid_queries,
            "survey_query": survey_query,
            "core": survey_candidates,
            "library": library,
        }
