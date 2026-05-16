import re
import tqdm
import asyncio
from typing import Dict, Any

from .websearch import WebSearchFallback
from ..utility.openalex import OPENALEX_SELECT, get_openalex_client
from ..utility.paper_download import PaperDownload, SemanticScholarPaperDownload, yield_location
from ..utility.request_utils import RateLimit
from ..utility.s2 import get_semantic_scholar_client
from ..utility.tool_config import ToolConfig
from .utils import valid_check


class CitationParser:
    SELECT = f"{OPENALEX_SELECT},best_oa_location,locations"

    def __init__(self, config: ToolConfig):
        self.paper_downloader = PaperDownload(config)
        self.semantic_scholar_downloader = SemanticScholarPaperDownload(config)
        self.websearch = WebSearchFallback(config)
        self.openalex = get_openalex_client(config)
        self.semantic_scholar = get_semantic_scholar_client(config)

    def _empty_info(self, title: str) -> Dict[str, Any]:
        return {
            "metadata": {},
            "title": title,
            "abstract": "",
            "full_content": {},
            "status": 3,
            "source": "unresolved",
        }

    def _normalize_title(self, title: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[:,.!?&]", " ", title or "")).strip()

    def _finalize_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        full_content = info.get("full_content")
        if isinstance(full_content, dict) and (full_content.get("paragraphs") or full_content.get("sections")):
            info["status"] = 0
        elif info.get("abstract"):
            info["status"] = 1
            if not info.get("full_content"):
                info["full_content"] = info["abstract"]
        elif info.get("metadata"):
            info["status"] = 2
        else:
            info["status"] = 3
            info["abstract"] = ""
            info["full_content"] = {}
        return info

    async def _download_openalex_paper(self, info: Dict[str, Any], metadata: dict) -> Dict[str, Any]:
        matched_metadata = dict(metadata or {})
        downloaded = None
        attempted_urls = set(yield_location(matched_metadata))
        try:
            async with RateLimit.CITATION_DOWNLOAD_SEMAPHORE:
                downloaded = await self.paper_downloader.download_single_paper(
                    matched_metadata,
                    openalex_id=matched_metadata.get("id", ""),
                )
        except Exception as exc:
            print(f"CitationParser openalex download failed: {matched_metadata.get('title', '')} {exc}")
        info["_attempted_openalex_urls"] = list(attempted_urls)
        if downloaded:
            info["full_content"] = downloaded.get("full_content", {})
            info["abstract"] = downloaded.get("abstract", "") or matched_metadata.get("abstract", "") or ""
        else:
            info["abstract"] = matched_metadata.get("abstract", "") or ""
        return self._finalize_info(info)

    async def _download_semantic_scholar_paper(self, info: Dict[str, Any], metadata: dict) -> Dict[str, Any]:
        matched_metadata = dict(metadata or {})
        downloaded = None
        excluded_urls = set(info.get("_attempted_openalex_urls", []) or [])
        try:
            async with RateLimit.CITATION_DOWNLOAD_SEMAPHORE:
                downloaded = await self.semantic_scholar_downloader.download_single_paper(
                    matched_metadata,
                    excluded_urls=excluded_urls,
                )
        except Exception as exc:
            print(f"CitationParser semantic scholar download failed: {matched_metadata.get('title', '')} {exc}")
        if downloaded:
            info["full_content"] = downloaded.get("full_content", {})
            info["abstract"] = downloaded.get("abstract", "") or matched_metadata.get("abstract", "") or ""
        else:
            info["abstract"] = info.get("abstract") or matched_metadata.get("abstract", "") or ""
        return self._finalize_info(info)

    async def _search_paper(self, title: str, engine, source_name: str, **find_kwargs) -> dict | None:
        paper_title = self._normalize_title(title)
        if not paper_title: return None
        try:
            paper_info = await engine.find_work_by_title(paper_title, **find_kwargs)
        except Exception:
            paper_info = None

        if paper_info and valid_check(paper_title, paper_info.get("title", "")):
            return paper_info
        return None

    async def _search_paper_from_api(self, citation_info: str | Dict[str, Any]) -> Dict[str, Any]:
        title = citation_info["title"] if isinstance(citation_info, dict) else str(citation_info or "")
        info = self._empty_info(title)
        openalex_task = asyncio.create_task(
            self._search_paper(title, self.openalex, "openalex", select=self.SELECT)
        )
        semantic_task = asyncio.create_task(
            self._search_paper(title, self.semantic_scholar, "semantic scholar")
        )
        openalex_meta, semantic_meta = await asyncio.gather(openalex_task, semantic_task)
        if openalex_meta:
            openalex_meta = dict(openalex_meta)
            info["metadata"]["openalex"] = openalex_meta
            info["abstract"] = openalex_meta.get("abstract", "") or info.get("abstract", "")
        if semantic_meta:
            semantic_meta = dict(semantic_meta)
            info["metadata"]["semantic scholar"] = semantic_meta
            info["abstract"] = info.get("abstract") or semantic_meta.get("abstract", "") or ""

        if openalex_meta:
            info = await self._download_openalex_paper(info, openalex_meta)
        if info["status"] > 0 and semantic_meta:
            info = await self._download_semantic_scholar_paper(info, semantic_meta)
        info.pop("_attempted_openalex_urls", None)
        info["source"] = "+".join(info["metadata"].keys()) if info["metadata"] else "unresolved"
        return self._finalize_info(info)

    async def _fallback_websearch(self, title: str, info: Dict[str, Any]) -> Dict[str, Any]:
        if link := info.get('link', None):
            try:
                fallback = await self.websearch.extract_content_from_url(link)
            except Exception: return info
        try:
            fallback = await self.websearch.search_title(title)
        except Exception: return info
        if not fallback.get("exist"): return info
        updated = dict(info)
        updated["source"] = "websearch"
        updated["metadata"] = updated.get("metadata") or {"websearch": fallback.get("metadata")}
        updated["abstract"] = fallback.get("abstract", updated.get("abstract", "")) or ""
        updated["full_content"] = fallback.get("full_content", updated.get("full_content", {}))
        return self._finalize_info(updated)

    async def _parse_single(self, citation_key: str, citation_info: Any):
        info = await self._search_paper_from_api(citation_info)
        if info["status"] == 3:
            title = citation_info["title"] if isinstance(citation_info, dict) else str(citation_info or "")
            info = await self._fallback_websearch(title, info)
        return citation_key, info

    # async def _parse_single_with_timeout(self, citation_key: str, citation_info: Any):
    #     try:
    #         return await asyncio.wait_for(
    #             self._parse_single(citation_key, citation_info),
    #             timeout=self.CITATION_TIMEOUT_SECONDS,
    #         )
    #     except asyncio.TimeoutError:
    #         title = citation_info["title"] if isinstance(citation_info, dict) else str(citation_info or "")
    #         print(f"CitationParser citation timeout: {citation_key} {title}")
    #         return citation_key, self._empty_info(title)

    async def __call__(self, citations: Dict[str, Any]) -> Dict[str, Any]:
        tasks = [
            asyncio.create_task(self._parse_single(citation_key, citation_info))
            for citation_key, citation_info in citations.items()
        ]
        paper_content_map = {}
        for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                citation_key, info = await task
                paper_content_map[citation_key] = info
            except Exception as e:
                print(f"CitationParser {e}")
        return {"paper_content_map": paper_content_map}
