from __future__ import annotations

import asyncio
import re
from datetime import timedelta
from typing import Any

from ..prompts import QUERY_EXPAND
from ..utility.citation_utils import citation_keys as normalize_citation_keys
from ..utility.llmclient import AsyncChat, AsyncRerank
from ..utility.openalex import OPENALEX_SELECT, get_openalex_client
from ..utility.paper_download import PaperDownload, S2PaperDownload
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json
from .fact_check_single import SingleFactCorrectness


class QueryExpandClient(AsyncChat):
    PROMPT = QUERY_EXPAND

    def _tokens(self, text: str) -> set[str]:
        return {token.casefold() for token in re.findall(r"[\w-]+", text or "")}

    def _availability(self, response, context):
        result = extract_json(response)
        assert isinstance(result["searchable"], bool)
        assert isinstance(result["query"], str)
        assert isinstance(result["key_entities"], list)
        if not result["searchable"]:
            return result
        af_tokens = self._tokens(context["af_text"])
        query_tokens = self._tokens(result["query"])
        assert query_tokens and query_tokens <= af_tokens
        query_text = result["query"].casefold()
        for entity in result["key_entities"]:
            assert isinstance(entity, str) and entity.strip()
            assert entity.casefold() in query_text
        return result

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(af_text=inputs["af_text"])
        return prompt, {"af_text": inputs["af_text"]}


class UncitedClaimVerifier:
    def __init__(self, config: ToolConfig):
        self.config = config
        self.fact_check = SingleFactCorrectness(config)
        self.query_expand = QueryExpandClient(config.llm_server_info, config.sampling_params)
        self.rerank = AsyncRerank(config.rerank_server_info)
        self.openalex = get_openalex_client(config)
        self.openalex_downloader = PaperDownload(config)
        self.semantic_scholar_downloader = S2PaperDownload(config)

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _claim_text(self, claim: dict[str, Any]) -> str:
        return str(claim.get("claim") or claim.get("text") or "").strip()

    def _claim_citation_keys(self, claim: dict[str, Any]) -> list[str]:
        return normalize_citation_keys(claim.get("citations") or claim.get("citation_keys") or [])

    def _paragraph_citation_keys(self, claim: dict[str, Any]) -> list[str]:
        return [str(key) for key in claim.get("_paragraph_citation_keys", []) if str(key).strip()]

    def _paragraph_entity_names(self, claim: dict[str, Any]) -> list[str]:
        names = []
        for name in claim.get("_paragraph_entities", []) or []:
            name = str(name).strip()
            if name:
                names.append(name)
        return list(dict.fromkeys(names))

    def _metadata_sources(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        metadata = info.get("metadata") or {}
        if isinstance(metadata, dict) and ("openalex" in metadata or "semantic scholar" in metadata):
            return [paper for paper in metadata.values() if isinstance(paper, dict)]
        return [metadata] if isinstance(metadata, dict) and metadata else []

    def _paper_sources_for_keys(self, keys: list[str], paper_content_map: dict[str, Any] | None) -> list[dict[str, Any]]:
        papers = []
        paper_content_map = paper_content_map or {}
        for key in keys:
            info = paper_content_map.get(key)
            if isinstance(info, dict):
                papers.extend(self._metadata_sources(info))
        return [paper for paper in papers if paper.get("title")]

    def _claim_entity_sources(self, claim: dict[str, Any], entity_data: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(entity_data, dict):
            return []
        claim_text = self._normalize_text(self._claim_text(claim)).casefold()
        claim_entities = {self._normalize_text(name).casefold() for name in self._paragraph_entity_names(claim) if name}
        sources = []
        for item in entity_data.get("uncited_entities", []) or []:
            if not isinstance(item, dict):
                continue
            entity_names = [item.get("entity", ""), *(item.get("alternative_names", []) or [])]
            normalized = [self._normalize_text(name).casefold() for name in entity_names if self._normalize_text(name)]
            if not normalized:
                continue
            matched = any(name in claim_text or claim_text in name or name in claim_entities for name in normalized)
            if matched:
                for paper in item.get("matched_papers", []) or []:
                    if isinstance(paper, dict):
                        sources.append(paper)
        return [paper for paper in sources if paper.get("title")]

    def _paper_ids(self, paper: dict[str, Any]) -> set[str]:
        ids = set()
        for key in ("id", "paperId", "corpusId"):
            if paper.get(key):
                ids.add(str(paper[key]).replace("https://openalex.org/", ""))
        raw_ids = paper.get("ids")
        if isinstance(raw_ids, dict):
            ids.update(str(value).replace("https://openalex.org/", "") for value in raw_ids.values() if value)
        elif isinstance(raw_ids, (list, tuple, set)):
            ids.update(str(value).replace("https://openalex.org/", "") for value in raw_ids if value)
        return {item for item in ids if item}

    def _paper_title(self, paper: dict[str, Any]) -> str:
        return self._normalize_text(paper.get("title", "")).casefold()

    def _paper_key(self, paper: dict[str, Any]) -> str:
        ids = sorted(self._paper_ids(paper))
        if ids:
            return f"id:{ids[0]}"
        title = self._paper_title(paper)
        return f"title:{title}" if title else ""

    def _unique_papers(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique = {}
        for paper in papers:
            if not isinstance(paper, dict) or not paper.get("title"):
                continue
            key = self._paper_key(paper)
            if key and key not in unique:
                unique[key] = paper
        return list(unique.values())

    def _paper_contains_entities(self, paper: dict[str, Any], key_entities: list[str]) -> bool:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".casefold()
        return all(entity.casefold() in text for entity in key_entities)

    async def _download_paper(self, paper: dict[str, Any]) -> dict[str, Any]:
        downloaded = None
        try:
            if bool(paper.get("paperId") or paper.get("externalIds") or paper.get("openAccessPdf")):
                downloaded = await self.semantic_scholar_downloader.download_single_paper(paper)
            else:
                downloaded = await self.openalex_downloader.download_single_paper(
                    paper,
                    openalex_id=str(paper.get("id", "")).replace("https://openalex.org/", ""),
                )
        except Exception as exc:
            print(f"uncitedClaimDownload {paper.get('title', '')} {exc}")
        content = dict(paper)
        if downloaded:
            content["full_content"] = downloaded.get("full_content", {})
            content["abstract"] = downloaded.get("abstract", content.get("abstract", ""))
        else:
            content["full_content"] = content.get("abstract", "")
        return content

    async def _download_papers(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tasks = [asyncio.create_task(self._download_paper(paper)) for paper in papers]
        return [item for item in await asyncio.gather(*tasks, return_exceptions=True) if isinstance(item, dict)]

    async def _search_academic_engine(self, claim: str) -> list[dict[str, Any]]:
        query_data = await self.query_expand.call(inputs={"af_text": claim})
        if not query_data["searchable"] or not query_data["key_entities"]:
            return []
        payload = await self.openalex.search_works(
            search=" AND ".join(query_data["key_entities"]),
            filter={"to_publication_date": (self.config.evaluation_date - timedelta(days=90)).strftime("%Y-%m-%d")},
            per_page=10,
            select=OPENALEX_SELECT,
        )
        return [paper for paper in (payload.get("results", []) or []) if self._paper_contains_entities(paper, query_data["key_entities"])]

    async def _fact_check_claim(self, claim: str, papers: list[dict[str, Any]]) -> dict[str, Any]:
        if not papers:
            return {"claim": claim, "judgment": "NEUTRAL", "evidence": "", "reason": "insufficient information", "score": 0.0, "material": "title_abstract", "sources": []}
        return await self.fact_check._fact_check_claim(claim, papers)

    async def _process_claim(self, claim: dict[str, Any], paper_content_map: dict[str, Any], entity_data: dict[str, Any] | None) -> dict[str, Any]:
        claim_text = self._claim_text(claim)
        priority_papers = self._paper_sources_for_keys(self._paragraph_citation_keys(claim), paper_content_map)
        priority_papers.extend(self._claim_entity_sources(claim, entity_data))
        priority_papers = self._unique_papers(priority_papers)

        try:
            result = await self._fact_check_claim(claim_text, priority_papers)
        except Exception as exc:
            result = {"claim": claim_text, "judgment": "ERROR", "evidence": "", "reason": f"priority fact check error: {type(exc).__name__}: {exc}", "score": 0.0, "material": "title_abstract", "sources": []}

        if result.get("judgment") == "NEUTRAL":
            try:
                candidates = await self._search_academic_engine(claim_text)
            except Exception as exc:
                print(f"uncitedClaimSearch {claim_text[:80]} {exc}")
                candidates = []
            downloaded = await self._download_papers(candidates)
            try:
                search_result = await self._fact_check_claim(claim_text, downloaded)
            except Exception as exc:
                search_result = {"claim": claim_text, "judgment": "ERROR", "evidence": "", "reason": f"search fact check error: {type(exc).__name__}: {exc}", "score": 0.0, "material": "title_abstract", "sources": []}
            if self.fact_check._is_better_result(
                search_result.get("judgment", "NEUTRAL"),
                search_result.get("reason", ""),
                float(search_result.get("score", 0.0)),
                result,
            ):
                result = search_result

        return {
            "claim": claim_text,
            "label": claim.get("label", ""),
            "claim_type": claim.get("claim_type", ""),
            "citation_keys": self._claim_citation_keys(claim),
            "kind": "uncited_fact_check",
            **result,
        }

    async def __call__(
        self,
        claims: list[dict[str, Any]],
        paper_content_map: dict[str, Any],
        entity_data: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        tasks = [asyncio.create_task(self._process_claim(claim, paper_content_map, entity_data)) for claim in claims]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        checked = []
        for claim, result in zip(claims, results):
            if isinstance(result, dict):
                checked.append(result)
                continue
            claim_text = self._claim_text(claim)
            checked.append({
                "claim": claim_text,
                "label": claim.get("label", ""),
                "claim_type": claim.get("claim_type", ""),
                "citation_keys": self._claim_citation_keys(claim),
                "kind": "uncited_fact_check",
                "judgment": "ERROR",
                "reason": f"{type(result).__name__}: {result}",
            })
        return checked
