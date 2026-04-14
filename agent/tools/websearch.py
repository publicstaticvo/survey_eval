import asyncio, aiohttp
from typing import List, Dict, Any
from urllib.parse import urlparse
from trafilatura import extract
from .tool_config import ToolConfig
from .llmclient import AsyncChat
from .prompts import WEBSEARCH_FILTER_PROMPT
from .grobidpdf.paper_parser import PaperParser
from .sbert_client import SentenceTransformerClient
from .utils import cosine_similarity_pair, extract_json
from .paper_download import parse_with_grobid, download_paper_to_memory
from .request_utils import RateLimit, async_request_template, HEADERS, SessionManager


def llm_should_retry(exception: BaseException) -> bool:
    if isinstance(exception, KeyboardInterrupt): return False
    if isinstance(exception, NotImplementedError): return False
    return True


class WebSearchFilterLLM(AsyncChat):
    PROMPT = WEBSEARCH_FILTER_PROMPT

    def _availability(self, response, context):
        data = extract_json(response)
        matched = data.get("matched_indices", [])
        return [context["candidates"][idx - 1] for idx in matched if isinstance(idx, int) and 1 <= idx <= len(context["candidates"])]

    def _organize_inputs(self, inputs):
        candidates_str = "\n".join(
            f"{i}. title={candidate.get('title', '')}\n   url={candidate.get('link', '')}\n   snippet={candidate.get('snippet', '')}"
            for i, candidate in enumerate(inputs["candidates"], 1)
        )
        return self.PROMPT.format(title=inputs["title"], candidates=candidates_str), {"candidates": inputs["candidates"]}


class WebSearchFallback:
    ACADEMIC_HOST_HINTS = {
        "doi.org",
        "arxiv.org",
        "openreview.net",
        "aclanthology.org",
        "dblp.org",
        "semanticscholar.org",
        "springer.com",
        "ieeexplore.ieee.org",
        "sciencedirect.com",
        "nature.com",
        "researchgate.net",
    }

    def __init__(self, config: ToolConfig):
        self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)
        self.filter_llm = WebSearchFilterLLM(config.llm_server_info, config.sampling_params)
        self.url = config.websearch_url
        self.key = config.websearch_apikey
        self.grobid = config.grobid_url
        self.paper_parser = PaperParser()

    def _title_similarity(self, query: str, result: str) -> float:
        embeddings = self.sentence_transformer.embed([query or "", result or ""])
        return cosine_similarity_pair(embeddings[0], embeddings[1])

    def _is_parseable_url(self, url: str) -> bool:
        parsed = urlparse(url or "")
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _prefer_candidate(self, candidate: Dict[str, Any]) -> tuple[int, int]:
        host = urlparse(candidate.get("link", "")).netloc.lower()
        academic_bonus = int(any(host.endswith(hint) for hint in self.ACADEMIC_HOST_HINTS))
        pdf_bonus = int(candidate.get("link", "").lower().endswith(".pdf"))
        return academic_bonus, pdf_bonus

    def _text_to_pseudo_skeleton(self, text: str) -> Dict[str, Any]:
        paragraphs = []
        raw_paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        if not raw_paragraphs:
            raw_paragraphs = [text.strip()] if text and text.strip() else []
        for paragraph in raw_paragraphs:
            paragraphs.append([{"text": paragraph, "citations": []}])
        return {
            "title": "",
            "author": "",
            "abstract": [],
            "paragraphs": paragraphs,
            "sections": [],
            "citations": {},
        }

    async def extract_content_from_url(self, url: str):
        async with RateLimit.WEBSEARCH_SEMAPHORE:
            session = SessionManager.get()
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "").split(";")[0].lower()
                if content_type == "application/pdf":
                    pdf_buffer = await download_paper_to_memory(url)
                    if not pdf_buffer: return {}
                    xml_content = await parse_with_grobid(self.grobid, pdf_buffer)
                    if not xml_content: return {}
                    paper = self.paper_parser.parse(xml_content)
                    skeleton = paper.get_skeleton()
                    abstract = "\n\n".join(" ".join(s["text"] for s in p) for p in skeleton.get("abstract", []))
                    return {"full_content": skeleton, "abstract": abstract}
                if content_type in {"text/html", "text/plain"}:
                    content = await resp.text()
                    extracted = extract(content) or ""
                    if not extracted: return {}
                    pseudo_skeleton = self._text_to_pseudo_skeleton(extracted)
                    abstract = raw_abstract = "\n\n".join(paragraph[0]["text"] for paragraph in pseudo_skeleton["paragraphs"][:3])
                    return {"full_content": pseudo_skeleton, "abstract": raw_abstract}
                return {}

    async def _request_search(self, title: str) -> Dict[str, Any]:
        headers = {"Content-type": "application/json", "X-API-KEY": self.key} | HEADERS
        payload = {"q": title}
        async with RateLimit.WEBSEARCH_SEMAPHORE:
            return await async_request_template("post", self.url, headers, payload)

    async def _filter_candidates(self, title: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        try:
            filtered = await self.filter_llm.call(inputs={"title": title, "candidates": candidates})
        except Exception:
            filtered = []
        results = filtered or candidates[:3]
        results.sort(key=lambda candidate: (self._prefer_candidate(candidate), -candidate.get("position", 999)), reverse=True)
        return results

    async def search_title(self, title: str) -> Dict[str, Any]:
        if not title: return {"exist": False}
        results = await self._request_search(title)
        candidates = []
        knowledge = results.get("knowledgeGraph") or {}
        knowledge_title = knowledge.get("title", "")
        knowledge_link = knowledge.get("descriptionLink", "")
        if knowledge_title and knowledge_link and self._is_parseable_url(knowledge_link):
            if self._title_similarity(title, knowledge_title) >= 0.8:
                candidates.append(
                    {
                        "title": knowledge_title,
                        "link": knowledge_link,
                        "snippet": knowledge.get("description", ""),
                        "position": 0,
                    }
                )
        for result in results.get("organic", []):
            link = result.get("link", "")
            if not self._is_parseable_url(link):
                continue
            if self._title_similarity(title, result.get("title", "")) >= 0.8:
                candidates.append(result)
        filtered_candidates = await self._filter_candidates(title, candidates)
        for candidate in filtered_candidates:
            try:
                content = await self.extract_content_from_url(candidate["link"])
            except Exception:
                content = {}
            if content:
                return {
                    "exist": True,
                    "title": candidate.get("title", title),
                    "metadata": {"title": candidate.get("title", title), "url": candidate.get("link", "")},
                    **content,
                }
        if filtered_candidates:
            best = filtered_candidates[0]
            snippet = best.get("snippet", "")
            return {
                "exist": True,
                "title": best.get("title", title),
                "metadata": {"title": best.get("title", title), "url": best.get("link", "")},
                "abstract": snippet,
                "full_content": self._text_to_pseudo_skeleton(snippet) if snippet else None,
            }
        return {"exist": False}

    async def __call__(self, refs_to_search: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tasks = [asyncio.create_task(self.search_title(item.get("title", ""))) for item in refs_to_search]
        return await asyncio.gather(*tasks, return_exceptions=True)
