"""
citation_parser.py
闁兼儳鍢茶ぐ鍥礂閵娿倗绉肩€殿喗娲橀弸鍍瞖tadata闁告粌鑻崣蹇涘棘閸ワ缚绻嗛柟?
濞达綀娉曢弫銈夋晬?
闁?闁挎稑顦遍弫顦昿enalex闁靛棔璁?闁靛棔璁rper闁兼儳鍢茶ぐ鍥棘閸モ晝褰垮ǎ鍥ｅ墲娴?
闁?闁挎稑顦伴悧鎾箲椤旇姤鐎紒鏃傚С娣囧﹪骞侀娆戠憮閺夌偠妫勯崣蹇涘棘?
闁?闁挎稑顦遍弫顥痭foLLMClient闁告帒妫欓悗浠嬪嫉椤掍焦鐎柟缁樺姇閸ゎ厽绂嶉崱鏇犵焼濞?
闁告瑯鍨堕埀顒€顦伴弫濂稿礉椤帞绐?
闁?闁挎稑顦·鍐礉閻曠磧xiv api
闁?闁挎稑顢爀bSearch闁瑰瓨鐗為鎴﹀棘閸ワ妇鐟撻弶鐐存灮缁辨繄绱掑鏄縠nreview闁告娲滅€氼厾鎷嬮幑鎰靛悁濞戞挴鍋撳┑鍌涱殔椤︹晠鎮堕崱妯荤厵婵℃鐗勯埀?
"""
import re
import tqdm
import asyncio
import logging
from typing import Dict, Any

from .websearch import WebSearchFallback
from ..utility.openalex import OPENALEX_SELECT, get_openalex_client
from ..utility.paper_download import PaperDownload, S2PaperDownload, yield_location
from ..utility.request_utils import RateLimit
from ..utility.s2 import get_semantic_scholar_client
from ..utility.tool_config import ToolConfig
from ..utility.llmclient import AsyncChat
from .utils import valid_check, extract_json
from ..prompts import EXTRACT_TITLE


class InfoLLMClient(AsyncChat):
    def _availability(self, response, context):
        response = extract_json(response)
        assert response['title'] in context['info']
        return response['title']
    
    def _organize_inputs(self, inputs):
        return [{"role": 'user', 'content': EXTRACT_TITLE.format(**inputs)}], inputs



class CitationParser:
    SELECT = f"{OPENALEX_SELECT},best_oa_location,locations"

    def __init__(self, config: ToolConfig):
        self.paper_downloader = PaperDownload(config)
        self.use_semantic_scholar = config.use_semantic_scholar()
        self.semantic_scholar_downloader = S2PaperDownload(config) if self.use_semantic_scholar else None
        self.websearch = WebSearchFallback(config)
        self.openalex = get_openalex_client(config)
        self.info_llm = InfoLLMClient(config.llm_server_info)
        self.semantic_scholar = get_semantic_scholar_client(config) if self.use_semantic_scholar else None

    def _clean_title(self, title: str) -> str:
        title = re.sub(r"[{}]", "", title or "")
        return re.sub(r"\s+", " ", title).strip()

    def _empty_info(self, title: str) -> Dict[str, Any]:
        return {
            "metadata": {},
            "title": self._clean_title(title),
            "abstract": "",
            "full_content": {},
            "status": 3,
            "source": "unresolved",
        }

    def _normalize_title(self, title: str) -> str:
        return self._clean_title(title)
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
        if self.semantic_scholar_downloader is None:
            return self._finalize_info(info)
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
        if "title" not in citation_info:
            citation_info['title'] = await self.info_llm.call(inputs={"info": citation_info['info']})
        
        title = self._clean_title(citation_info["title"] if isinstance(citation_info, dict) else str(citation_info or ""))
        info = self._empty_info(title)
        openalex_task = asyncio.create_task(
            self._search_paper(title, self.openalex, "openalex", select=self.SELECT)
        )
        if self.use_semantic_scholar:
            semantic_task = asyncio.create_task(
                self._search_paper(title, self.semantic_scholar, "semantic scholar")
            )
            openalex_meta, semantic_meta = await asyncio.gather(openalex_task, semantic_task)
        else:
            openalex_meta = await openalex_task
            semantic_meta = None
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
            title = self._clean_title(citation_info["title"] if isinstance(citation_info, dict) else str(citation_info or ""))
            info = await self._fallback_websearch(title, info)
        return citation_key, info

    async def refresh_status3(self, citations: Dict[str, Any], cached_data: Dict[str, Any]) -> Dict[str, Any]:
        paper_content_map = dict((cached_data or {}).get("paper_content_map", {}) or {})
        unresolved_keys = [
            citation_key
            for citation_key, info in paper_content_map.items()
            if isinstance(info, dict) and info.get("status") == 3 and citation_key in citations
        ]
        if not unresolved_keys:
            return cached_data

        logging.info("Retrying %d unresolved status=3 citations", len(unresolved_keys))
        tasks = [
            asyncio.create_task(self._parse_single(citation_key, citations[citation_key]))
            for citation_key in unresolved_keys
        ]
        for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Citation Reparse"):
            try:
                citation_key, info = await task
                paper_content_map[citation_key] = info
            except Exception as e:
                print(f"CitationParser retry {e}")
        refreshed = dict(cached_data or {})
        refreshed["paper_content_map"] = paper_content_map
        return refreshed
    
    async def __call__(self, citations: Dict[str, Any]) -> Dict[str, Any]:
        logging.info(f"This paper has {len(citations)} citations")
        tasks = [
            asyncio.create_task(self._parse_single(citation_key, citation_info))
            for citation_key, citation_info in citations.items()
        ]
        paper_content_map = {}
        for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Citation Parse"):
            try:
                citation_key, info = await task
                paper_content_map[citation_key] = info
            except Exception as e:
                print(f"CitationParser {e}")
        return {"paper_content_map": paper_content_map}

