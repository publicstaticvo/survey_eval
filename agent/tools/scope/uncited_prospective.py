from __future__ import annotations

import asyncio
from typing import Any

from ..fact.fact_check import FactualCorrectnessCritic
from ..utility.llmclient import AsyncRerank
from ..utility.paper_download import PaperDownload, S2PaperDownload
from ..utility.sbert_client import SentenceTransformerClient
from ..utility.tool_config import ToolConfig


TARGET_CLAIM_LABELS = {"GAP", "SYNTHESIS", "EVALUATION"}


class UncitedProspective:
    def __init__(self, config: ToolConfig):
        self.config = config
        self.sbert = SentenceTransformerClient(config.sbert_server_url)
        self.rerank = AsyncRerank(config.rerank_server_info)
        self.fact_check = FactualCorrectnessCritic(config)
        self.openalex_downloader = PaperDownload(config)
        self.semantic_scholar_downloader = S2PaperDownload(config)

    def _iter_sentences(self, paper: dict[str, Any]):
        def walk(node: Any):
            if isinstance(node, dict):
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
            if sentence.get("label") in TARGET_CLAIM_LABELS and not sentence.get("citations")
        ]

    def _pool_items(self, literature_pool: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        raw_items = literature_pool.get("literature_pool", literature_pool) if isinstance(literature_pool, dict) else literature_pool
        if isinstance(raw_items, dict):
            values = raw_items.values()
        else:
            values = raw_items or []
        papers = []
        for item in values:
            paper = item.get("paper") if isinstance(item, dict) and item.get("paper") else item
            if isinstance(paper, dict) and paper.get("title"):
                papers.append(paper)
        return papers

    def _paper_text(self, paper: dict[str, Any]) -> str:
        return f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')}".strip()

    def _top_similar_papers(self, claim: str, papers: list[dict[str, Any]], top_n: int = 200) -> list[dict[str, Any]]:
        if not papers:
            return []
        texts = [claim, *[self._paper_text(paper) for paper in papers]]
        embeddings = self.sbert.embed(texts)
        claim_vec, paper_vecs = embeddings[:1], embeddings[1:]
        left_norm = (claim_vec ** 2).sum(axis=1, keepdims=True) ** 0.5
        right_norm = (paper_vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
        left_norm[left_norm == 0] = 1.0
        right_norm[right_norm == 0] = 1.0
        scores = ((claim_vec / left_norm) @ (paper_vecs / right_norm).T)[0].tolist()
        ranked = sorted(zip(papers, scores), key=lambda item: item[1], reverse=True)
        return [paper for paper, _ in ranked[:top_n]]

    async def _rerank_papers(self, claim: str, papers: list[dict[str, Any]], top_n: int = 10) -> list[dict[str, Any]]:
        if len(papers) <= top_n:
            return papers
        documents = [self._paper_text(paper) for paper in papers]
        selected_texts = await self.rerank.call(claim, documents, top_n=top_n)
        remaining = list(papers)
        selected = []
        for text in selected_texts:
            for index, paper in enumerate(remaining):
                if self._paper_text(paper) == text:
                    selected.append(paper)
                    remaining.pop(index)
                    break
        return selected or papers[:top_n]

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

    async def _process_claim(self, claim: dict[str, Any], pool_papers: list[dict[str, Any]]) -> dict[str, Any]:
        similar = self._top_similar_papers(claim["text"], pool_papers, top_n=200)
        selected = await self._rerank_papers(claim["text"], similar, top_n=10)
        downloaded = await self._download_papers(selected)
        fact_check = await self._fact_check_claim(claim["text"], downloaded)
        return {"claim": claim["text"], "label": claim["label"], "fact_check": fact_check}

    async def __call__(self, paper: dict[str, Any], literature_pool: dict[str, Any] | list[dict[str, Any]]):
        claims = self._claims(paper)
        pool_papers = self._pool_items(literature_pool)
        tasks = [
            asyncio.create_task(self._process_claim(claim, pool_papers))
            for claim in claims
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            "uncited_prospective": [
                result
                for result in results
                if isinstance(result, dict)
            ]
        }
