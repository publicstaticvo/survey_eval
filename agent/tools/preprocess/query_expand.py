import asyncio
from typing import Any, Dict, List
from datetime import timedelta

import jsonschema

from ..prompts import QUERY_EXPANSION_PROMPT, QUERY_SCHEMA
from ..utility.llmclient import AsyncChat
from ..utility.openalex import OPENALEX_SELECT, get_openalex_client
from ..utility.tool_config import ToolConfig
from .utils import extract_json

SURVEY_KEYWORDS = "survey|summary|review|overview|synthesis|taxonomy|study"


class QueryExpansionLLMClient(AsyncChat):
    PROMPT: str = QUERY_EXPANSION_PROMPT
    def _availability(self, response, context):
        queries = extract_json(response)
        jsonschema.validate(queries, QUERY_SCHEMA)
        return queries


class QueryExpand:
    def __init__(self, config: ToolConfig):
        self.eval_date = config.evaluation_date
        self.llm = QueryExpansionLLMClient(config.llm_server_info, config.sampling_params)
        self.openalex = get_openalex_client(config)

    async def _request_for_papers(
        self,
        query: str,
        uplimit: int = 100,
        select: str = f"{OPENALEX_SELECT},relevance_score",
    ) -> List[Dict[str, Any]]:
        search = query.strip()
        if not search: return []

        papers = []
        total = uplimit
        for page in range(1, (uplimit - 1) // 200 + 2):
            results = await self.openalex.search_works(
                "works",
                search=search,
                per_page=min(200, uplimit),
                select=select,
                page=page,
                filter={
                    "from_publication_date": (self.eval_date - timedelta(days=731)).strftime("%Y-%m-%d"),
                    "to_publication_date": (self.eval_date - timedelta(days=90)).strftime("%Y-%m-%d"),
                },
            )
            total = min(total, results.get("count", 0) or total)
            papers.extend(results.get("results", []))
            if len(papers) >= total:
                break
        return papers[:uplimit]

    async def __call__(self, query: str, num_seed_papers: int = 50):
        library = {}
        try:
            queries = await self.llm.call(inputs={"query": query})
            expanded_queries = [
                *(queries.get("core_anchor") or [])[:2],
                queries.get("theoretical_bridge") or query,
                queries.get("methodological_bridge") or query,
            ]
        except Exception:
            expanded_queries = [query]

        expanded_queries = [item for item in expanded_queries if item]
        try:
            paper_groups = await asyncio.gather(*[
                self._request_for_papers(expanded_query, num_seed_papers)
                for expanded_query in expanded_queries[:4]
            ], return_exceptions=True)
        except Exception:
            paper_groups = []

        for expanded_query, papers in zip(expanded_queries, paper_groups):
            if isinstance(papers, Exception):
                continue
            for paper in papers:
                if paper.get("id"):
                    library[paper["id"]] = paper | {"query": expanded_query}

        return {
            "queries": expanded_queries[:4] if library else [],
            "library": library,
        }
