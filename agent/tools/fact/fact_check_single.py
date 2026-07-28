from __future__ import annotations

from typing import Any, Dict

from ..utility.evidence_check import EvidenceCheck
from ..prompts import FACTUAL_CORRECTNESS_PROMPT
from ..utility.llmclient import AsyncChat, AsyncRerank
from ..utility.tool_config import ToolConfig
from ..utility.paper_elements import Paper
from ..utility.content_walk import iter_paragraphs, paragraph_to_text
from ..utility.utils import extract_json


class FactCheckLLMClient(AsyncChat):
    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        data = extract_json(response)
        judgment = str(data.get("judgment", "NEUTRAL")).upper()
        evidence = data.get("evidence", "")
        contradiction_reason = ""
        if judgment == "REFUTED":
            contradiction_reason = str(data.get("contradiction_reason", "")).strip()
            assert contradiction_reason, "REFUTED judgment must include a contradiction_reason"
        if judgment in {"SUPPORTED", "REFUTED"} and evidence:
            evidence_list = evidence if isinstance(evidence, list) else [evidence]
            verified, score = self.check.verify(evidence_list, context["text"])
            if not verified:
                return "NEUTRAL", "", score, "evidence verify error"
            return judgment, evidence, score, contradiction_reason
        return "NEUTRAL", "", 0.0, ""

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(**inputs)
        return prompt, {"text": inputs["text"]}


class SingleFactCorrectness:
    def __init__(self, config: ToolConfig):
        self.llm = FactCheckLLMClient(config)
        self.rerank = AsyncRerank(config.rerank_server_info)
        self.max_passages = max(1, config.rerank_n_documents)
        self.chunk_char_limit = 10000
        self.chunk_paragraph_limit = 5

    def _group_paragraphs(self, paragraph_texts: list[str]) -> list[str]:
        # Group nearby paragraphs so rerank sees a bounded number of passages per paper.
        chunks, current, current_len = [], [], 0
        for paragraph in paragraph_texts:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            projected_len = current_len + len(paragraph) + (2 if current else 0)
            if current and (len(current) >= self.chunk_paragraph_limit or projected_len > self.chunk_char_limit):
                chunks.append("\n\n".join(current))
                current = [paragraph]
                current_len = len(paragraph)
            else:
                current.append(paragraph)
                current_len = projected_len
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def _paper_paragraph_texts(self, paper: Paper) -> list[str]:
        return [
            text
            for paragraph in iter_paragraphs(paper, include_appendix=True)
            if (text := paragraph_to_text(paragraph))
        ]

    def _content_candidates(self, cited_paper: Dict[str, Any]) -> list[tuple[str, str]]:
        title = cited_paper.get("title", "")
        abstract = cited_paper.get("abstract", "")
        candidates = []
        full_content = cited_paper.get("full_content")
        if isinstance(full_content, Paper):
            paragraph_texts = self._paper_paragraph_texts(full_content)
            for chunk in self._group_paragraphs(paragraph_texts):
                candidates.append(("full_text", f"Title: {title}\nAbstract: {abstract}\n\n{chunk}".strip()))
        if not candidates and title and abstract:
            candidates.append(("title_abstract", f"Title: {title}\nAbstract: {abstract}".strip()))
        elif not candidates and title:
            candidates.append(("title_abstract", f"Title: {title}".strip()))
        return candidates

    async def _select_candidates(self, claim: str, candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if len(candidates) <= self.max_passages:
            return candidates
        documents = [text for _, text in candidates]
        selected_texts = await self.rerank.call(claim, documents, top_n=self.max_passages)
        remaining = list(candidates)
        selected = []
        for text in selected_texts:
            for index, candidate in enumerate(remaining):
                if candidate[1] == text:
                    selected.append(candidate)
                    remaining.pop(index)
                    break
        return selected or candidates[:self.max_passages]

    async def _judge(self, claim: str, material: str, text: str) -> tuple[str, str, float, str]:
        content_type = "title and abstract" if material == "title_abstract" else "title, abstract, and full text"
        return await self.llm.call(inputs={"claim": claim, "text": text, "content_type": content_type})

    def _error_result(self, claim: str, reason: str, material: str = "") -> Dict[str, Any]:
        return {
            "fact_check": {
                "claim": claim,
                "judgment": "ERROR",
                "evidence": "",
                "reason": reason,
                "score": 0.0,
                "material": material,
            }
        }

    def _neutral_reason_rank(self, reason: str) -> int:
        if reason == "evidence verify error":
            return 3
        if reason == "":
            return 2
        if reason == "insufficient information":
            return 1
        return 0

    def _is_better_result(self, judgment: str, reason: str, score: float, best_result: Dict[str, Any]) -> bool:
        best_judgment = best_result["judgment"]
        if judgment == "SUPPORTED":
            return True
        if best_judgment == "SUPPORTED":
            return False
        if judgment == "REFUTED":
            return True
        if best_judgment == "REFUTED":
            return False
        if judgment == "NEUTRAL" and best_judgment == "NEUTRAL":
            reason_rank = self._neutral_reason_rank(reason)
            best_reason_rank = self._neutral_reason_rank(best_result.get("reason", ""))
            return reason_rank > best_reason_rank or (reason_rank == best_reason_rank and score > best_result["score"])
        return False

    def _candidate_source(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": paper.get("id") or paper.get("paperId", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "url": paper.get("url", ""),
        }

    async def _select_multi_paper_candidates(
        self,
        claim: str,
        papers: list[Dict[str, Any]],
        top_n: int | None = None,
    ) -> list[tuple[Dict[str, Any], str, str]]:
        candidates = []
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            for material, text in self._content_candidates(paper):
                candidates.append((paper, material, text))
        top_n = top_n or self.max_passages
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

    async def _fact_check_claim(self, claim: str, papers: list[Dict[str, Any]]) -> Dict[str, Any]:
        candidates = await self._select_multi_paper_candidates(claim, papers, top_n=self.max_passages)
        has_full_text = any(material == "full_text" for _, material, _ in candidates)
        best_result = {
            "claim": claim,
            "judgment": "NEUTRAL",
            "evidence": "",
            "reason": "" if has_full_text else "insufficient information",
            "score": 0.0,
            "material": "title_abstract",
            "sources": [],
        }
        by_judgment: dict[str, list[dict[str, Any]]] = {"SUPPORTED": [], "REFUTED": []}
        for paper, material, text in candidates:
            try:
                judgment, evidence, score, reason = await self._judge(claim, material, text)
            except Exception as exc:
                return {
                    "claim": claim,
                    "judgment": "ERROR",
                    "evidence": "",
                    "reason": f"LLM call error: {type(exc).__name__}: {exc}",
                    "score": 0.0,
                    "material": material,
                    "sources": [],
                }
            if judgment == "NEUTRAL" and reason == "" and not has_full_text:
                reason = "insufficient information"
            if judgment in by_judgment:
                by_judgment[judgment].append({
                    "evidence": evidence,
                    "score": score,
                    "material": material,
                    "paper": self._candidate_source(paper),
                })
            if self._is_better_result(judgment, reason, score, best_result):
                best_result = {
                    "claim": claim,
                    "judgment": judgment,
                    "evidence": evidence,
                    "reason": reason,
                    "score": score,
                    "material": material,
                    "sources": [self._candidate_source(paper)] if judgment in {"SUPPORTED", "REFUTED"} else [],
                }
            if judgment == "SUPPORTED":
                break
        if by_judgment["SUPPORTED"] and by_judgment["REFUTED"]:
            best_result["evidence_sources"] = by_judgment
        elif best_result["judgment"] in by_judgment:
            best_result["evidence_sources"] = {best_result["judgment"]: by_judgment[best_result["judgment"]]}
        return best_result

    async def __call__(self, claim: str, cited_paper: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return {"fact_check": await self._fact_check_claim(claim, [cited_paper])}
        except Exception as exc:
            return self._error_result(claim, f"fact check preparation error: {type(exc).__name__}: {exc}")
