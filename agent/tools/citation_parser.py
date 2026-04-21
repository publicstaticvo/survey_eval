import re
import json
import tqdm
import asyncio
from typing import Dict, Any

from .tool_config import ToolConfig
from .websearch import WebSearchFallback
from .paper_download import PaperDownload
from .utils import valid_check, index_to_abstract
from .openalex import OPENALEX_SELECT, get_openalex_client


class CitationParser:
    SELECT = f"{OPENALEX_SELECT},best_oa_location,locations"

    def __init__(self, config: ToolConfig):
        self.paper_downloader = PaperDownload(config)
        self.websearch = WebSearchFallback(config)
        self.openalex = get_openalex_client(config)

    def _empty_info(self, title: str) -> Dict[str, Any]:
        return {
            "metadata": None,
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

    async def _search_paper_from_api(self, citation_info: str | Dict[str, Any]) -> Dict[str, Any]:
        title = citation_info["title"] if isinstance(citation_info, dict) else str(citation_info or "")
        info = self._empty_info(title)
        paper_title = self._normalize_title(title)
        if not paper_title:
            return info

        try:
            paper_info = await self.openalex.find_work_by_title(paper_title, select=self.SELECT)
        except Exception:
            paper_info = None

        if paper_info and valid_check(paper_title, paper_info.get("title", "")):
            matched_metadata = dict(paper_info)
            downloaded = await self.paper_downloader.download_single_paper(matched_metadata, openalex_id=matched_metadata.get("id"))
            if downloaded:
                info["full_content"] = downloaded.get("full_content", {})
                info["abstract"] = downloaded.get("abstract", "") or matched_metadata.get("abstract", "") or ""
            else:
                info["abstract"] = matched_metadata.get("abstract", "") or ""
            matched_metadata.pop("locations", None)
            matched_metadata.pop("best_oa_location", None)
            info["metadata"] = matched_metadata
            info["source"] = "openalex"
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
        updated["metadata"] = updated.get("metadata") or fallback.get("metadata")
        updated["abstract"] = fallback.get("abstract", updated.get("abstract", "")) or ""
        updated["full_content"] = fallback.get("full_content", updated.get("full_content", {}))
        return self._finalize_info(updated)

    async def _parse_single(self, citation_key: str, citation_info: Any):
        info = await self._search_paper_from_api(citation_info)
        if info["status"] == 3:
            title = citation_info["title"] if isinstance(citation_info, dict) else str(citation_info or "")
            info = await self._fallback_websearch(title, info)
        return citation_key, info

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
