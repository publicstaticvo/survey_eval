from typing import Any, Dict, List

from .openalex import OPENALEX_SELECT, get_openalex_client
from .tool_config import ToolConfig

SURVEY_KEYWORDS = "survey|summary|review|overview|synthesis|taxonomy|study"


class QueryExpand:
    def __init__(self, config: ToolConfig):
        self.eval_date = config.evaluation_date
        self.openalex = get_openalex_client(config)

    async def _request_for_papers(
        self,
        query: str,
        uplimit: int = 100,
        select: str = f"{OPENALEX_SELECT},relevance_score",
    ) -> List[Dict[str, Any]]:
        search = query.strip()
        if not search:
            return []

        papers = []
        total = uplimit
        for page in range(1, (uplimit - 1) // 200 + 2):
            results = await self.openalex.search_works(
                "works",
                search=search,
                per_page=min(200, uplimit),
                select=select,
                page=page,
                filter={"to_publication_date": self.eval_date.strftime("%Y-%m-%d")},
            )
            total = min(total, results.get("count", 0) or total)
            papers.extend(results.get("results", []))
            if len(papers) >= total:
                break
        return papers[:uplimit]

    async def __call__(self, query: str, num_seed_papers: int = 100):
        library = {}
        try:
            papers = await self._request_for_papers(query, num_seed_papers)
        except Exception:
            papers = []

        for paper in papers:
            if paper.get("id"):
                library[paper["id"]] = paper | {"query": query}

        return {
            "queries": [query] if library else [],
            "survey_query": query,
            "core": [],
            "library": library,
        }
