import asyncio
import hashlib
import random
import re
from typing import Any

import aiohttp
import Levenshtein

from .request_utils import HEADERS, SessionManager
from .tool_config import ToolConfig
from .utils import normalize_text, valid_check


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
        normalized["ids"] = list(dict.fromkeys([paper_id, *(paper.get("ids") or [])]))
        return normalized

    def _paper_year(self, paper: dict) -> int | None:
        publication_date = paper.get("publication_date") or paper.get("publicationDate") or ""
        if isinstance(publication_date, str) and len(publication_date) >= 4 and publication_date[:4].isdigit():
            return int(publication_date[:4])
        year = paper.get("year")
        return int(year) if isinstance(year, int) else None

    def _external_id_pairs(self, paper: dict) -> set[tuple[str, str]]:
        pairs = set()
        for key, value in (paper.get("external_ids") or paper.get("externalIds") or {}).items():
            if value:
                pairs.add((normalize_text(str(key)), normalize_text(str(value))))
        return pairs

    def _same_paper(self, left: dict, right: dict) -> bool:
        left_external_ids = self._external_id_pairs(left)
        right_external_ids = self._external_id_pairs(right)
        if left_external_ids and right_external_ids and left_external_ids & right_external_ids:
            return True

        left_title = normalize_text(left.get("title", ""))
        right_title = normalize_text(right.get("title", ""))
        if not left_title or not right_title:
            return False
        max_title_len = max(len(left_title), len(right_title), 1)
        if Levenshtein.distance(left_title, right_title) > max(1, int(0.1 * max_title_len)):
            return False

        left_authors = {normalize_text(author) for author in left.get("authorships", []) if normalize_text(author)}
        right_authors = {normalize_text(author) for author in right.get("authorships", []) if normalize_text(author)}
        if left_authors and right_authors:
            overlap = len(left_authors & right_authors) / max(1, min(len(left_authors), len(right_authors)))
            if overlap < 0.8:
                return False

        left_year = self._paper_year(left)
        right_year = self._paper_year(right)
        if left_year is not None and right_year is not None and abs(left_year - right_year) > 1:
            return False
        return True

    def _minhash_buckets(self, title: str, num_hashes: int = 24, band_size: int = 4) -> list[tuple[int, tuple[int, ...]]]:
        tokens = set(re.findall(r"[a-z0-9]+", normalize_text(title)))
        if not tokens:
            tokens = {normalize_text(title)}
        signature = []
        for seed in range(num_hashes):
            values = []
            for token in tokens:
                digest = hashlib.blake2b(f"{seed}:{token}".encode("utf-8"), digest_size=8).hexdigest()
                values.append(int(digest, 16))
            signature.append(min(values) if values else 0)
        return [
            (band_idx, tuple(signature[band_idx: band_idx + band_size]))
            for band_idx in range(0, num_hashes, band_size)
        ]

    def merge_duplicate_papers(self, papers: list[dict], original_title: str = "") -> dict | None:
        merged = self.deduplicate_papers(papers, original_title=original_title)
        if not merged:
            return None
        if len(merged) == 1:
            return merged[0]
        return self._merge_paper_cluster(merged, original_title)

    def deduplicate_papers(self, papers: list[dict], original_title: str = "") -> list[dict]:
        papers = [dict(paper) for paper in papers if paper and paper.get("title")]
        if not papers:
            return []

        parent = list(range(len(papers)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(left: int, right: int):
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        buckets = {}
        external_index = {}
        for idx, paper in enumerate(papers):
            for bucket in self._minhash_buckets(paper.get("title", "")):
                buckets.setdefault(bucket, []).append(idx)
            for pair in self._external_id_pairs(paper):
                external_index.setdefault(pair, []).append(idx)

        for indexes in list(buckets.values()) + list(external_index.values()):
            for pos, left in enumerate(indexes):
                for right in indexes[pos + 1:]:
                    if self._same_paper(papers[left], papers[right]):
                        union(left, right)

        clusters = {}
        for idx in range(len(papers)):
            clusters.setdefault(find(idx), []).append(papers[idx])
        return [self._merge_paper_cluster(cluster, original_title) for cluster in clusters.values()]

    def _merge_paper_cluster(self, papers: list[dict], original_title: str = "") -> dict:
        base = max(papers, key=lambda paper: paper.get("cited_by_count", 0) or paper.get("citationCount", 0) or 0)
        merged = dict(base)
        base_id = base.get("id")
        ids = []
        authors = []
        external_ids = {}
        for paper in papers:
            ids.extend(paper.get("ids") or ([paper["id"]] if paper.get("id") else []))
            authors.extend(paper.get("authorships", []) or [])
            for key, value in (paper.get("external_ids") or {}).items():
                if value and key not in external_ids:
                    external_ids[key] = value

        ids = list(dict.fromkeys(([base_id] if base_id else []) + ids))
        if ids:
            merged["id"] = ids[0]
            merged["paperId"] = ids[0]
        merged["ids"] = ids
        merged["authorships"] = list(dict.fromkeys(authors))
        merged["external_ids"] = external_ids
        merged["externalIds"] = external_ids

        dates = [paper.get("publication_date") for paper in papers if paper.get("publication_date")]
        if dates:
            merged["publication_date"] = min(dates)
            merged["publicationDate"] = min(dates)

        title_anchor = original_title or papers[0].get("title", "")
        merged["title"] = min(
            (paper.get("title", "") for paper in papers if paper.get("title")),
            key=lambda value: Levenshtein.distance(normalize_text(title_anchor), normalize_text(value)),
        )
        cited_source = max(papers, key=lambda paper: paper.get("cited_by_count", 0) or paper.get("citationCount", 0) or 0)
        citation_count = cited_source.get("cited_by_count", 0) or cited_source.get("citationCount", 0) or 0
        merged["cited_by_count"] = citation_count
        merged["citationCount"] = citation_count
        reference_count = max((paper.get("reference_count", 0) or paper.get("referenceCount", 0) or 0) for paper in papers)
        merged["reference_count"] = reference_count
        merged["referenceCount"] = reference_count

        referenced_works = []
        for paper in papers:
            referenced_works.extend(paper.get("referenced_works", []) or [])
        if referenced_works:
            merged["referenced_works"] = list(dict.fromkeys(referenced_works))
        return merged

    def _wrap_results(self, payload: dict, paper_key: str | None = None) -> dict:
        data = payload.get("data")
        if data is None:
            data = payload.get("results") or []
        raw_result_count = len(data or [])

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
        results = self.deduplicate_papers(results)
        return {
            "count": total,
            "results": results,
            "next": payload.get("next"),
            "_raw_result_count": raw_result_count,
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
        results = self.deduplicate_papers(results, original_title=title)
        return {"count": len(results), "results": results}

    async def find_work_by_title(self, title: str, fields: str | None = S2_DEFAULT_FIELDS) -> dict | None:
        results = await self.autocomplete("paper", title)
        if not results.get("results"):
            return None
        paper_ids = []
        for item in results.get("results", []):
            candidate_title = item.get("display_name") or item.get("title") or item.get("name") or ""
            if not valid_check(title, candidate_title):
                continue
            paper_id = item.get("id") or item.get("paperId")
            if paper_id:
                paper_ids.append(paper_id)
        if not paper_ids:
            return None

        async def _single(paper_id: str):
            try:
                return await self.get_work_by_id(paper_id, fields=fields)
            except Exception:
                return None

        tasks = [asyncio.create_task(_single(paper_id)) for paper_id in dict.fromkeys(paper_ids)]
        papers = [paper for paper in await asyncio.gather(*tasks) if paper]
        return self.merge_duplicate_papers(papers, original_title=title)

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
        limit: int = 9999,
        fields: str | None = S2_DEFAULT_FIELDS,
        **request_kwargs,
    ) -> dict:
        per_page = min(max(1, limit), 1000)
        current_offset = offset
        results, total, next_token = [], None, None
        while len(results) < limit:
            params = dict(request_kwargs)
            params.update({"offset": current_offset, "limit": per_page})
            fields = self._normalize_fields(fields)
            if fields:
                params["fields"] = fields
            payload = await self._request_json("GET", f"/paper/{paper_id}/citations", params)
            wrapped = self._wrap_results(payload, paper_key="citingPaper")
            if total is None:
                total = int(wrapped.get("count", 0) or 0)
            batch = wrapped.get("results", []) or []
            raw_batch_count = int(wrapped.get("_raw_result_count", len(batch)) or 0)
            results.extend(batch)
            next_token = wrapped.get("next")
            current_offset += per_page
            if raw_batch_count == 0 or raw_batch_count < per_page or current_offset >= total:
                break
        results = self.deduplicate_papers(results)
        return {"count": total or len(results), "results": results[:limit], "next": next_token}

    async def get_references(
        self,
        paper_id: str,
        offset: int = 0,
        limit: int = 9999,
        fields: str | None = S2_DEFAULT_FIELDS,
        **request_kwargs,
    ) -> dict:
        per_page = min(max(1, limit), 1000)
        current_offset = offset
        results, total, next_token = [], None, None
        while len(results) < limit:
            params = dict(request_kwargs)
            params.update({"offset": current_offset, "limit": per_page})
            fields = self._normalize_fields(fields)
            if fields:
                params["fields"] = fields
            payload = await self._request_json("GET", f"/paper/{paper_id}/references", params)
            wrapped = self._wrap_results(payload, paper_key="citedPaper")
            if total is None:
                total = int(wrapped.get("count", 0) or 0)
            batch = wrapped.get("results", []) or []
            raw_batch_count = int(wrapped.get("_raw_result_count", len(batch)) or 0)
            results.extend(batch)
            next_token = wrapped.get("next")
            current_offset += per_page
            if raw_batch_count == 0 or raw_batch_count < per_page or current_offset >= total:
                break
        results = self.deduplicate_papers(results)
        return {"count": total or len(results), "results": results[:limit], "next": next_token}

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
        results = self.deduplicate_papers(results)
        return {"count": len(results), "results": results}


_S2_CLIENT: SemanticScholar | None = None


def get_semantic_scholar_client(config: ToolConfig | None = None) -> SemanticScholar:
    global _S2_CLIENT
    if _S2_CLIENT is None:
        _S2_CLIENT = SemanticScholar(config or ToolConfig())
    elif config is not None and _S2_CLIENT.config.semantic_scholar_api_key != config.semantic_scholar_api_key:
        _S2_CLIENT = SemanticScholar(config)
    return _S2_CLIENT
