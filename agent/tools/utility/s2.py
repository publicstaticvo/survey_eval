import asyncio
import random
from typing import Any

import aiohttp

from .request_utils import HEADERS, SessionManager
from .tool_config import ToolConfig


SEMANTIC_SCHOLAR_GRAPH_API = "https://api.semanticscholar.org/graph/v1"
S2_DEFAULT_FIELDS = (
    "paperId,title,abstract,year,publicationDate,citationCount,referenceCount,"
    "authors,externalIds,openAccessPdf,url,venue,fieldsOfStudy,publicationTypes"
)


class SemanticScholar:
    def __init__(self, config: ToolConfig):
        self.config = config
        self.api_key = (config.semantic_scholar_api_key or "").strip()

    def _headers(self) -> dict[str, str]:
        headers = dict(HEADERS)
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def _request_json(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict:
        session = SessionManager.get()
        url = f"{SEMANTIC_SCHOLAR_GRAPH_API}{endpoint}"
        params = {key: value for key, value in (params or {}).items() if value not in (None, "", [], {})}

        while True:
            async with session.request(
                method.upper(),
                url,
                headers=self._headers(),
                params=params,
                json=json_body,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                if resp.status == 429 and not self.api_key:
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                    continue
                resp.raise_for_status()
                return await resp.json()

    def _normalize_paper(self, paper: dict | None) -> dict:
        paper = dict(paper or {})
        paper_id = paper.get("paperId") or paper.get("id") or ""
        if not paper_id:
            return {}

        authors = []
        for author in paper.get("authors") or []:
            if isinstance(author, str):
                name = author
            elif isinstance(author, dict):
                name = author.get("name") or ""
            else:
                name = ""
            if name:
                authors.append(name)

        publication_date = paper.get("publicationDate")
        if not publication_date and paper.get("year"):
            publication_date = f"{paper['year']}-01-01"

        normalized = dict(paper)
        normalized["id"] = paper_id
        normalized["title"] = paper.get("title") or ""
        normalized["abstract"] = paper.get("abstract")
        normalized["publication_date"] = publication_date
        normalized["cited_by_count"] = paper.get("citationCount", 0) or 0
        normalized["reference_count"] = paper.get("referenceCount", 0) or 0
        normalized["authorships"] = list(dict.fromkeys(authors))
        normalized["external_ids"] = paper.get("externalIds") or {}
        return normalized

    def _wrap_results(self, payload: dict, paper_key: str | None = None) -> dict:
        data = payload.get("data")
        if data is None:
            data = payload.get("results") or []

        results = []
        for item in data or []:
            if paper_key:
                paper = item.get(paper_key) or {}
            else:
                paper = item
            normalized = self._normalize_paper(paper)
            if normalized:
                results.append(normalized)

        total = payload.get("total")
        if total is None:
            total = len(results)
        return {
            "count": total,
            "results": results,
            "next": payload.get("next"),
        }

    def _format_filter(self, filter: list[tuple] | dict | None = None) -> dict:
        if not filter:
            return {}
        items = dict(filter) if isinstance(filter, dict) else {key: value for key, value in filter}
        mapped = {}
        from_date = items.pop("from_publication_date", None)
        to_date = items.pop("to_publication_date", None)
        if from_date or to_date:
            mapped["publicationDateOrYear"] = f"{from_date or ''}:{to_date or ''}"
        if "cited_by" in items:
            mapped["cites"] = items.pop("cited_by")
        if "cites" in items:
            mapped["citedPaperId"] = items.pop("cites")
        mapped.update(items)
        return mapped

    def _normalize_fields(self, fields: str | None) -> str | None:
        if fields is None:
            return None
        openalex_only = {
            "id",
            "cited_by_count",
            "counts_by_year",
            "referenced_works",
            "publication_date",
            "created_date",
            "abstract_inverted_index",
            "authorships",
            "best_oa_location",
            "locations",
            "relevance_score",
        }
        requested = {field.strip() for field in fields.split(",") if field.strip()}
        if requested & openalex_only:
            return S2_DEFAULT_FIELDS
        return fields

    async def autocomplete(self, entity_type: str = "paper", title: str = "", **request_kwargs) -> dict:
        payload = await self._request_json(
            "GET",
            f"/{entity_type}/autocomplete",
            {"query": title, **request_kwargs},
        )
        matches = payload.get("matches") or []
        results = []
        for item in matches:
            item = dict(item or {})
            if item.get("id") and not item.get("paperId"):
                item["paperId"] = item["id"]
            normalized = self._normalize_paper(item)
            if normalized:
                normalized["display_name"] = item.get("title") or item.get("name") or normalized.get("title", "")
                results.append(normalized)
        return {"count": len(results), "results": results}

    async def search(
        self,
        query: str = "",
        offset: int = 0,
        limit: int = 100,
        fields: str | None = S2_DEFAULT_FIELDS,
        **request_kwargs,
    ) -> dict:
        params = dict(request_kwargs)
        params.update({"query": query, "offset": offset, "limit": limit})
        fields = self._normalize_fields(fields)
        if fields:
            params["fields"] = fields
        payload = await self._request_json("GET", "/paper/search", params)
        return self._wrap_results(payload)

    async def filter(
        self,
        query: str = "",
        filter: list[tuple] | dict | None = None,
        offset: int = 0,
        limit: int = 100,
        fields: str | None = S2_DEFAULT_FIELDS,
        **request_kwargs,
    ) -> dict:
        params = self._format_filter(filter)
        params.update(request_kwargs)
        params.update({"query": query, "offset": offset, "limit": limit})
        fields = self._normalize_fields(fields)
        if fields:
            params["fields"] = fields
        payload = await self._request_json("GET", "/paper/search/bulk", params)
        return self._wrap_results(payload)

    async def search_works(
        self,
        entity_type: str = "paper",
        search: str = "",
        filter: list[tuple] | dict | None = None,
        per_page: int = 100,
        select: str | None = S2_DEFAULT_FIELDS,
        offset: int = 0,
        **request_kwargs,
    ) -> dict:
        if filter:
            return await self.filter(search, filter, offset=offset, limit=per_page, fields=select, **request_kwargs)
        return await self.search(search, offset=offset, limit=per_page, fields=select, **request_kwargs)

    async def get_work_by_id(
        self,
        paper_id: str,
        fields: str | None = S2_DEFAULT_FIELDS,
        **request_kwargs,
    ) -> dict:
        params = dict(request_kwargs)
        fields = self._normalize_fields(fields)
        if fields:
            params["fields"] = fields
        payload = await self._request_json("GET", f"/paper/{paper_id}", params)
        return self._normalize_paper(payload)

    async def get_entity(
        self,
        entity_id: str,
        entity_type: str = "paper",
        select: str | None = S2_DEFAULT_FIELDS,
        **request_kwargs,
    ) -> dict:
        if entity_type != "paper":
            raise ValueError("SemanticScholar only supports paper entities")
        return await self.get_work_by_id(entity_id, fields=select, **request_kwargs)

    async def get_citations(
        self,
        paper_id: str,
        offset: int = 0,
        limit: int = 100,
        fields: str | None = S2_DEFAULT_FIELDS,
        **request_kwargs,
    ) -> dict:
        params = dict(request_kwargs)
        params.update({"offset": offset, "limit": limit})
        fields = self._normalize_fields(fields)
        if fields:
            params["fields"] = fields
        payload = await self._request_json("GET", f"/paper/{paper_id}/citations", params)
        return self._wrap_results(payload, paper_key="citingPaper")

    async def get_references(
        self,
        paper_id: str,
        offset: int = 0,
        limit: int = 100,
        fields: str | None = S2_DEFAULT_FIELDS,
        **request_kwargs,
    ) -> dict:
        params = dict(request_kwargs)
        params.update({"offset": offset, "limit": limit})
        fields = self._normalize_fields(fields)
        if fields:
            params["fields"] = fields
        payload = await self._request_json("GET", f"/paper/{paper_id}/references", params)
        return self._wrap_results(payload, paper_key="citedPaper")

    async def get_works_batch(
        self,
        paper_ids: list[str],
        fields: str | None = S2_DEFAULT_FIELDS,
        **request_kwargs,
    ) -> dict:
        params = dict(request_kwargs)
        fields = self._normalize_fields(fields)
        if fields:
            params["fields"] = fields
        payload = await self._request_json(
            "POST",
            "/paper/batch",
            params=params,
            json_body={"ids": list(dict.fromkeys(paper_ids))},
        )
        results = [paper for item in payload or [] if (paper := self._normalize_paper(item))]
        return {"count": len(results), "results": results}


_S2_CLIENT: SemanticScholar | None = None


def get_semantic_scholar_client(config: ToolConfig | None = None) -> SemanticScholar:
    global _S2_CLIENT
    if _S2_CLIENT is None:
        _S2_CLIENT = SemanticScholar(config or ToolConfig())
    elif config is not None and _S2_CLIENT.config.semantic_scholar_api_key != config.semantic_scholar_api_key:
        _S2_CLIENT = SemanticScholar(config)
    return _S2_CLIENT
