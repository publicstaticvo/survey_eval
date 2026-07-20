from __future__ import annotations

import asyncio
import re
from datetime import timedelta
from typing import Any

from ..fact.fact_check import FactualCorrectnessCritic
from ..utility.citation_utils import has_citations
from ..prompts import QUERY_EXPAND
from ..utility.openalex import OPENALEX_SELECT, get_openalex_client
from ..utility.llmclient import AsyncChat, AsyncRerank
from ..utility.paper_download import PaperDownload, S2PaperDownload
from ..utility.tool_config import ToolConfig
from .utils import extract_json


TARGET_CLAIM_LABELS = {"GAP", "SYNTHESIS", "EVALUATION"}


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
        return self.PROMPT.format(af_text=inputs["af_text"]), {"af_text": inputs["af_text"]}

class UncitedProspective:
    def __init__(self, config: ToolConfig):
        self.config = config
        self.fact_check = FactualCorrectnessCritic(config)
        self.query_expand = QueryExpandClient(config.llm_server_info, config.sampling_params)
        self.rerank = AsyncRerank(config.rerank_server_info)
        self.openalex = get_openalex_client(config)
        self.openalex_downloader = PaperDownload(config)
        self.semantic_scholar_downloader = S2PaperDownload(config)

    def _iter_sentences(self, paper: dict[str, Any]):
        def walk(node: Any):
            if isinstance(node, dict):
                if "sentences" in node:
                    yield from walk(node.get("sentences", []) or [])
                    return
                for paragraph in node.get("paragraphs", []) or []:
                    yield from walk(paragraph)
                for section in node.get("sections", []) or []:
                    yield from walk(section)
            elif isinstance(node, list):
                for sentence in node:
                    if isinstance(sentence, dict) and sentence.get("text"):
                        yield sentence

        yield from walk(paper)

    def _claims(self, paper: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"text": sentence["text"], "label": sentence.get("label", "")}
            for sentence in self._iter_sentences(paper)
            if sentence.get("label") in TARGET_CLAIM_LABELS and not has_citations(sentence.get("citations"))
        ]

    def _normalize_extra_claim(self, claim: dict[str, Any] | str) -> dict[str, Any] | None:
        if isinstance(claim, str):
            text = claim.strip()
            label = ""
        elif isinstance(claim, dict):
            text = str(claim.get("text", "") or claim.get("claim", "")).strip()
            label = str(claim.get("label", "") or "").strip()
        else:
            return None
        if not text:
            return None
        return {"text": text, "label": label}

    def _merge_claims(
        self,
        claims: list[dict[str, Any]],
        extra_claims: list[dict[str, Any] | str] | None = None,
    ) -> list[dict[str, Any]]:
        merged = []
        seen = set()
        for claim in [*claims, *(extra_claims or [])]:
            normalized = self._normalize_extra_claim(claim)
            if not normalized:
                continue
            key = " ".join(normalized["text"].lower().split())
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
        return merged

    def _is_semantic_scholar_paper(self, paper: dict[str, Any]) -> bool:
        return bool(paper.get("paperId") or paper.get("externalIds") or paper.get("openAccessPdf"))

    async def _download_paper(self, paper: dict[str, Any]) -> dict[str, Any]:
        downloaded = None
        try:
            if self._is_semantic_scholar_paper(paper):
                downloaded = await self.semantic_scholar_downloader.download_single_paper(paper)
            else:
                downloaded = await self.openalex_downloader.download_single_paper(
                    paper,
                    openalex_id=str(paper.get("id", "")).replace("https://openalex.org/", ""),
                )
        except Exception as exc:
            print(f"uncitedProspectiveDownload {paper.get('title', '')} {exc}")
        content = dict(paper)
        if downloaded:
            content["full_content"] = downloaded.get("full_content", {})
            content["abstract"] = downloaded.get("abstract", content.get("abstract", ""))
        else:
            content["full_content"] = content.get("abstract", "")
        return content

    async def _download_papers(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tasks = [asyncio.create_task(self._download_paper(paper)) for paper in papers]
        return [
            item
            for item in await asyncio.gather(*tasks, return_exceptions=True)
            if isinstance(item, dict)
        ]

    def _candidate_source(self, paper: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": paper.get("id") or paper.get("paperId", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "url": paper.get("url", ""),
        }

    async def _select_multi_paper_candidates(
        self,
        claim: str,
        papers: list[dict[str, Any]],
        top_n: int = 10,
    ) -> list[tuple[dict[str, Any], str, str]]:
        candidates = []
        for paper in papers:
            for material, text in self.fact_check._content_candidates(paper):
                candidates.append((paper, material, text))
        if len(candidates) <= top_n:
            return candidates
        documents = [text for _, _, text in candidates]
        selected_texts = await self.rerank.call(claim, documents, top_n=top_n)
        remaining = list(candidates)
        selected = []
        for text in selected_texts:
            for index, candidate in enumerate(remaining):
                if candidate[2] == text:
                    selected.append(candidate)
                    remaining.pop(index)
                    break
        return selected or candidates[:top_n]

    async def _fact_check_claim(self, claim: str, papers: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = await self._select_multi_paper_candidates(claim, papers, top_n=10)
        best = {
            "claim": claim,
            "judgment": "NEUTRAL",
            "evidence": "",
            "reason": "insufficient information",
            "score": 0.0,
            "material": "title_abstract",
            "sources": [],
        }
        by_judgment = {"SUPPORTED": [], "REFUTED": []}
        for paper, material, text in candidates:
            judgment, evidence, score = await self.fact_check._judge(claim, material, text)
            if judgment in by_judgment:
                by_judgment[judgment].append({
                    "evidence": evidence,
                    "score": score,
                    "material": material,
                    "paper": self._candidate_source(paper),
                })
            if score > best["score"] or judgment != "NEUTRAL":
                best = {
                    "claim": claim,
                    "judgment": judgment,
                    "evidence": evidence,
                    "reason": "" if judgment != "NEUTRAL" else "insufficient information",
                    "score": score,
                    "material": material,
                    "sources": [self._candidate_source(paper)] if judgment in {"SUPPORTED", "REFUTED"} else [],
                }
        if by_judgment["SUPPORTED"] and by_judgment["REFUTED"]:
            best["evidence_sources"] = by_judgment
        elif best["judgment"] in by_judgment:
            best["evidence_sources"] = {best["judgment"]: by_judgment[best["judgment"]]}
        return best

    def _paper_contains_entities(self, paper: dict[str, Any], key_entities: list[str]) -> bool:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".casefold()
        return all(entity.casefold() in text for entity in key_entities)

    async def _process_claim(self, claim: dict[str, Any]) -> dict[str, Any]:
        query_data = {"searchable": False, "query": "", "key_entities": []}
        try:
            query_data = await self.query_expand.call(inputs={"af_text": claim["text"]})
            if query_data["searchable"] and query_data["key_entities"]:
                payload = await self.openalex.search_works(
                    search=" AND ".join(query_data["key_entities"]),
                    filter={
                        "to_publication_date": (self.config.evaluation_date - timedelta(days=90)).strftime("%Y-%m-%d"),
                    },
                    per_page=10,
                    select=OPENALEX_SELECT,
                )
                selected = [
                    paper for paper in (payload.get("results", []) or [])
                    if self._paper_contains_entities(paper, query_data["key_entities"])
                ]
            else:
                selected = []
        except Exception as exc:
            print(f"uncitedProspectiveSearch {claim['text'][:80]} {exc}")
            selected = []
        downloaded = await self._download_papers(selected)
        fact_check = await self._fact_check_claim(claim["text"], downloaded)
        return {"claim": claim["text"], "label": claim["label"], "query_expand": query_data, "fact_check": fact_check}
    async def __call__(
        self,
        paper: dict[str, Any],
        extra_claims: list[dict[str, Any] | str] | None = None,
    ):
        claims = self._merge_claims(self._claims(paper), extra_claims)
        tasks = [
            asyncio.create_task(self._process_claim(claim))
            for claim in claims
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        grouped: dict[str, list[dict[str, Any]]] = {}
        details: dict[str, dict[str, Any]] = {}
        for result in results:
            if not isinstance(result, dict):
                continue
            claim_text = result.get("claim", "")
            fact_check = result.get("fact_check", {}) or {}
            if fact_check.get("judgment") != "REFUTED":
                continue
            papers = []
            sources = list(fact_check.get("sources") or [])
            for entry in (fact_check.get("evidence_sources", {}) or {}).get("REFUTED", []) or []:
                source_paper = entry.get("paper")
                if isinstance(source_paper, dict):
                    sources.append(source_paper)
            seen = set()
            for source in sources:
                if not isinstance(source, dict) or not source.get("title"):
                    continue
                key = source.get("id") or source.get("paperId") or source.get("title")
                if key in seen:
                    continue
                seen.add(key)
                papers.append(source)
            if papers:
                grouped[claim_text] = papers
                details[claim_text] = result
        return {"uncited_prospective": grouped, "uncited_prospective_details": details}
