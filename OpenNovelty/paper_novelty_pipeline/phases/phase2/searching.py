"""Phase 2: OpenAlex paper retrieval for the local OpenNovelty checkout."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from paper_novelty_pipeline.models import ExtractedContent
from survey_eval.agent.tools.utility.openalex import get_openalex_client
from survey_eval.agent.tools.utility.tool_config import ToolConfig
from survey_eval.agent.tools.utility.request_utils import SessionManager


class PaperSearcher:
    """Search OpenAlex for the Phase-1 core task and contribution query variants."""

    SELECT = "id,title,abstract_inverted_index,authorships,publication_date,cited_by_count,referenced_works,doi,primary_location"

    def __init__(self, concurrency: Optional[int] = None, config: ToolConfig | None = None):
        self.concurrency = max(1, concurrency or 4)
        self.config = config or ToolConfig.from_yaml(Path(__file__).resolve().parents[4] / "agent.yaml")
        self.openalex = get_openalex_client(self.config)

    def _queries(self, extracted: ExtractedContent) -> list[tuple[str, str]]:
        queries: list[tuple[str, str]] = []
        for query in [extracted.core_task.text, *(extracted.core_task.query_variants or [])]:
            if query and query not in [value for _scope, value in queries]:
                queries.append(("core_task", query))
        for contribution in extracted.contributions:
            for query in [contribution.prior_work_query, *(contribution.query_variants or [])]:
                if query and query not in [value for _scope, value in queries]:
                    queries.append((f"contribution:{contribution.id}", query))
        return queries

    async def _search_one(self, scope: str, query: str) -> dict[str, Any]:
        payload = await self.openalex.search_works(search=query, per_page=200, select=self.SELECT)
        return {"scope": scope, "query": query, "count": payload["count"], "results": payload["results"]}

    def search_all(self, extracted: ExtractedContent, out_dir: Path) -> Dict[str, Any]:
        """Retrieve and persist raw OpenAlex result sets for every Phase-1 query."""
        phase2_dir = Path(out_dir) / "phase2"
        raw_dir = phase2_dir / "raw_responses"
        raw_dir.mkdir(parents=True, exist_ok=True)
        queries = self._queries(extracted)

        async def run():
            semaphore = asyncio.Semaphore(self.concurrency)
            async def limited(scope: str, query: str):
                async with semaphore:
                    return await self._search_one(scope, query)
            return await asyncio.gather(*(limited(scope, query) for scope, query in queries))

        async def run_with_session():
            await SessionManager.init()
            try:
                return await run()
            finally:
                await SessionManager.close()

        results = asyncio.run(run_with_session())
        records = []
        for index, result in enumerate(results):
            path = raw_dir / f"openalex_{index:02d}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append({"scope": result["scope"], "query": result["query"], "count": result["count"], "path": str(path)})
        manifest = {"backend": "openalex", "queries": records}
        (phase2_dir / "openalex_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


def run_phase2_search(extracted: ExtractedContent, out_dir: Path, concurrency: Optional[int] = None) -> Dict[str, Any]:
    """Run OpenAlex-backed Phase 2 retrieval."""
    return PaperSearcher(concurrency=concurrency).search_all(extracted, out_dir)
