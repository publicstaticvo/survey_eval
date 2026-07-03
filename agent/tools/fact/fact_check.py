from __future__ import annotations

import tqdm
import asyncio
from typing import Any, Dict

from ..utility.evidence_check import EvidenceCheck
from ..prompts import FACTUAL_CORRECTNESS_PROMPT
from ..utility.llmclient import AsyncChat, AsyncRerank
from ..utility.citation_utils import citation_keys as normalize_citation_keys
from ..utility.tool_config import ToolConfig
from .utils import extract_json, split_content_to_paragraph, paragraph_to_text


class FactCheckLLMClient(AsyncChat):
    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        data = extract_json(response)
        judgment = str(data.get("judgment", "NEUTRAL")).upper()
        evidence = data.get("evidence", "")
        if judgment in {"SUPPORTED", "REFUTED"} and evidence:
            evidence_list = evidence if isinstance(evidence, list) else [evidence]
            verified, score = self.check.verify(evidence_list, context["text"])
            if not verified:
                return "NEUTRAL", "", score, "evidence verify error"
            return judgment, evidence, score, ""
        return "NEUTRAL", "", 0.0, ""

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(**inputs)
        return prompt, {"text": inputs["text"]}


class FactualCorrectnessCritic:
    def __init__(self, config: ToolConfig):
        self.llm = FactCheckLLMClient(config)
        self.rerank = AsyncRerank(config.rerank_server_info)
        self.max_passages = max(1, config.rerank_n_documents)
        self.chunk_char_limit = 10000
        self.chunk_paragraph_limit = 5

    def _group_paragraphs(self, paragraph_texts: list[str]) -> list[str]:
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

    def _content_candidates(self, cited_paper: Dict[str, Any]) -> list[tuple[str, str]]:
        title = cited_paper.get("title", "")
        abstract = cited_paper.get("abstract", "")
        candidates = []
        if title and abstract:
            candidates.append(("title_abstract", f"Title: {title}\nAbstract: {abstract}".strip()))
        full_content = cited_paper.get("full_content")
        if isinstance(full_content, dict):
            paragraph_texts = []
            for paragraph in split_content_to_paragraph(full_content):
                paragraph_text = paragraph_to_text(paragraph)
                if paragraph_text:
                    paragraph_texts.append(paragraph_text)
            for chunk in self._group_paragraphs(paragraph_texts):
                candidates.append(("full_text", f"Title: {title}\nAbstract: {abstract}\n\n{chunk}".strip()))
        elif isinstance(full_content, str) and full_content.strip():
            paragraph_texts = [p.strip() for p in full_content.split("\n\n") if p.strip()]
            if not paragraph_texts:
                paragraph_texts = [full_content.strip()]
            for chunk in self._group_paragraphs(paragraph_texts):
                candidates.append(("full_text", f"Title: {title}\nAbstract: {abstract}\n\n{chunk}".strip()))
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
    async def __call__(self, claim: str, cited_paper: Dict[str, Any]) -> Dict[str, Any]:
        try:
            candidates = self._content_candidates(cited_paper)
            selected = await self._select_candidates(claim, candidates)
            has_full_text = any(material == "full_text" for material, _ in candidates)
        except Exception as exc:
            return self._error_result(claim, f"fact check preparation error: {type(exc).__name__}: {exc}")

        best_result = {
            "claim": claim,
            "judgment": "NEUTRAL",
            "evidence": "",
            "reason": "" if has_full_text else "insufficient information",
            "score": 0.0,
            "material": "title_abstract",
        }
        for material, text in selected:
            try:
                judgment, evidence, score, reason = await self._judge(claim, material, text)
            except Exception as exc:
                return self._error_result(claim, f"LLM call error: {type(exc).__name__}: {exc}", material)
            if judgment == "NEUTRAL" and reason == "" and not has_full_text:
                reason = "insufficient information"
            if self._is_better_result(judgment, reason, score, best_result):
                best_result = {
                    "claim": claim,
                    "judgment": judgment,
                    "evidence": evidence,
                    "reason": reason,
                    "score": score,
                    "material": material,
                }
            if judgment == "SUPPORTED":
                break
        return {"fact_check": best_result}


class CitedClaimVerifier:
    def __init__(self, config: ToolConfig):
        self.fact_check = FactualCorrectnessCritic(config)

    def _citation_keys(self, claim: dict[str, Any]) -> list[str]:
        return normalize_citation_keys(claim.get("citations"))

    def _aggregate_fact_results(self, results: list[dict[str, Any]]) -> str:
        judgments = [item.get("fact_check", {}).get("judgment", "NEUTRAL") for item in results]
        if "SUPPORTED" in judgments:
            return "SUPPORTED"
        if "REFUTED" in judgments:
            return "REFUTED"
        if "ERROR" in judgments:
            return "ERROR"
        return "NEUTRAL"

    def _aggregate_neutral_reason(self, results: list[dict[str, Any]]) -> str:
        reasons = [item.get("fact_check", {}).get("reason", "") for item in results]
        if "evidence verify error" in reasons:
            return "evidence verify error"
        if "" in reasons:
            return ""
        if "insufficient information" in reasons:
            return "insufficient information"
        return reasons[0] if reasons else "insufficient information"
    async def _fact_verification(self, claim: dict[str, Any], paper_content_map: dict[str, Any]) -> dict[str, Any]:
        citation_keys = self._citation_keys(claim)
        async def _single(key: str):
            citation_data = paper_content_map.get(key, {})
            if not citation_data or citation_data.get("status", 3) >= 3:
                return {"fact_check": {"claim": claim["claim"], "judgment": "NEUTRAL", "reason": "citation unresolved", "citation_key": key}}
            result = await self.fact_check(claim["claim"], citation_data)
            result["fact_check"]["citation_key"] = key
            return result

        tasks = [asyncio.create_task(_single(key)) for key in citation_keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        checked = []
        for key, result in zip(citation_keys, results):
            if isinstance(result, dict):
                checked.append(result)
            else:
                checked.append({"fact_check": {"claim": claim["claim"], "judgment": "ERROR", "reason": f"{type(result).__name__}: {result}", "citation_key": key}})
        judgment = self._aggregate_fact_results(checked)
        result = {"judgment": judgment, "references": [item["fact_check"] for item in checked]}
        if judgment == "NEUTRAL":
            result["reason"] = self._aggregate_neutral_reason(checked)
        return result

    async def _verify_claim(self, claim: dict[str, Any], paper_content_map: dict[str, Any]) -> dict[str, Any]:
        citation_keys = self._citation_keys(claim)
        verification = await self._fact_verification(claim, paper_content_map)
        return {
            "claim": claim["claim"],
            "claim_type": claim.get("claim_type", ""),
            "citation_keys": citation_keys,
            "kind": "fact_check",
            **verification,
        }

    async def __call__(self, claims: list[dict[str, Any]], paper_content_map: dict[str, Any]) -> dict[str, Any]:
        tasks = [asyncio.create_task(self._verify_claim(claim, paper_content_map)) for claim in claims]
        results = []
        for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="CitedClaimVerifier"):
            results.append(await task)
        print(f"We check {len(results)} claims")
        return {"fact_checks": results, "checked_count": len(results)}