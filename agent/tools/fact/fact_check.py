from __future__ import annotations

import asyncio
from typing import Any, Dict

from ..utility.evidence_check import EvidenceCheck
from ..prompts import FACTUAL_CORRECTNESS_PROMPT
from ..utility.llmclient import AsyncChat, AsyncRerank
from ..utility.sbert_client import SentenceTransformerClient
from ..utility.tool_config import ToolConfig
from .utils import extract_json, split_content_to_paragraph, paragraph_to_text, cosine_similarity_matrix


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
                return "NEUTRAL", "", score
            return judgment, evidence, score
        return "NEUTRAL", "", 0.0

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

    async def _judge(self, claim: str, material: str, text: str) -> tuple[str, str, float]:
        content_type = "title and abstract" if material == "title_abstract" else "title, abstract, and full text"
        return await self.llm.call(inputs={"claim": claim, "text": text, "content_type": content_type})

    async def __call__(self, claim: str, cited_paper: Dict[str, Any]) -> Dict[str, Any]:
        candidates = self._content_candidates(cited_paper)
        selected = await self._select_candidates(claim, candidates)
        best_result = {
            "claim": claim,
            "judgment": "NEUTRAL",
            "evidence": "",
            "reason": "insufficient information",
            "score": 0.0,
            "material": "title_abstract",
        }
        for material, text in selected:
            judgment, evidence, score = await self._judge(claim, material, text)
            if score > best_result["score"] or judgment != "NEUTRAL":
                best_result = {
                    "claim": claim,
                    "judgment": judgment,
                    "evidence": evidence,
                    "reason": "" if judgment != "NEUTRAL" else "insufficient information",
                    "score": score,
                    "material": material,
                }
            if judgment in {"SUPPORTED", "REFUTED"}:
                break
        return {"fact_check": best_result}


class CitedClaimVerifier:
    TARGET_LABELS = {"BACKGROUND", "SUMMARY", "SYNTHESIS", "EVALUATION"}
    FACT_LABELS = {"SUMMARY", "SYNTHESIS", "EVALUATION"}

    def __init__(self, config: ToolConfig):
        self.fact_check = FactualCorrectnessCritic(config)
        self.sbert = SentenceTransformerClient(config.sbert_server_url)
        self.background_threshold = config.background_reference_similarity_threshold

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

    def _citation_keys(self, sentence: dict[str, Any]) -> list[str]:
        keys = []
        for citation in sentence.get("citations", []) or []:
            key = citation.get("key") or citation.get("ref_text") if isinstance(citation, dict) else citation
            if key:
                keys.append(str(key))
        return list(dict.fromkeys(keys))

    def _metadata_papers(self, citation_data: dict[str, Any]) -> list[dict[str, Any]]:
        metadata = citation_data.get("metadata") or {}
        if isinstance(metadata, dict) and ("openalex" in metadata or "semantic scholar" in metadata):
            return [paper for paper in metadata.values() if isinstance(paper, dict)]
        return [metadata] if isinstance(metadata, dict) and metadata else []

    def _reference_text(self, citation_data: dict[str, Any]) -> str:
        papers = self._metadata_papers(citation_data)
        if papers:
            paper = papers[0]
            return f"{paper.get('title', '')}\n{paper.get('abstract', '')}".strip()
        return f"{citation_data.get('title', '')}\n{citation_data.get('abstract', '')}".strip()

    def _background_relevance(self, sentence: dict[str, Any], citation_keys: list[str], paper_content_map: dict[str, Any]):
        references = [
            {"citation_key": key, "text": self._reference_text(paper_content_map.get(key, {}))}
            for key in citation_keys
        ]
        references = [item for item in references if item["text"]]
        if not references:
            return {"judgment": "NEUTRAL", "reason": "no reference metadata", "references": []}
        texts = [sentence["text"], *[item["text"] for item in references]]
        embeddings = self.sbert.embed(texts)
        scores = cosine_similarity_matrix(embeddings[:1], embeddings[1:])[0].tolist()
        reference_results = []
        for item, score in zip(references, scores):
            reference_results.append({
                "citation_key": item["citation_key"],
                "similarity": float(score),
                "relevant": float(score) >= self.background_threshold,
            })
        judgment = "SUPPORTED" if all(item["relevant"] for item in reference_results) else "REFUTED"
        return {"judgment": judgment, "threshold": self.background_threshold, "references": reference_results}

    def _aggregate_fact_results(self, results: list[dict[str, Any]]) -> str:
        judgments = [item.get("fact_check", {}).get("judgment", "NEUTRAL") for item in results]
        if "REFUTED" in judgments:
            return "REFUTED"
        if "SUPPORTED" in judgments:
            return "SUPPORTED"
        return "NEUTRAL"

    async def _verify_fact_sentence(self, sentence: dict[str, Any], citation_keys: list[str], paper_content_map: dict[str, Any]):
        async def _single(key: str):
            citation_data = paper_content_map.get(key, {})
            if not citation_data or citation_data.get("status", 3) >= 3:
                return {
                    "fact_check": {
                        "claim": sentence["text"],
                        "judgment": "NEUTRAL",
                        "reason": "citation unresolved",
                        "citation_key": key,
                    }
                }
            result = await self.fact_check(sentence["text"], citation_data)
            result["fact_check"]["citation_key"] = key
            return result

        tasks = [asyncio.create_task(_single(key)) for key in citation_keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        checked = []
        for key, result in zip(citation_keys, results):
            if isinstance(result, dict):
                checked.append(result)
            else:
                checked.append({
                    "fact_check": {
                        "claim": sentence["text"],
                        "judgment": "NEUTRAL",
                        "reason": str(result),
                        "citation_key": key,
                    }
                })
        return {"judgment": self._aggregate_fact_results(checked), "references": [item["fact_check"] for item in checked]}

    async def __call__(self, paper: dict[str, Any], paper_content_map: dict[str, Any]) -> dict[str, Any]:
        targets = [
            sentence
            for sentence in self._iter_sentences(paper)
            if sentence.get("label") in self.TARGET_LABELS and self._citation_keys(sentence)
        ]
        results, neutral_opinions = [], []
        for sentence in targets:
            citation_keys = self._citation_keys(sentence)
            if sentence.get("label") == "BACKGROUND":
                verification = self._background_relevance(sentence, citation_keys, paper_content_map)
                kind = "background_relevance"
            else:
                verification = await self._verify_fact_sentence(sentence, citation_keys, paper_content_map)
                kind = "fact_check"
                if verification["judgment"] == "NEUTRAL" and sentence.get("label") in {"SYNTHESIS", "EVALUATION"}:
                    sentence["needs_uncited_prospective"] = True
                    neutral_opinions.append({"text": sentence["text"], "label": sentence.get("label", "")})
            item = {
                "sentence": sentence["text"],
                "label": sentence.get("label", ""),
                "citation_keys": citation_keys,
                "kind": kind,
                **verification,
            }
            sentence["citation_verification"] = item
            results.append(item)
        return {"fact_checks": results, "neutral_opinion_claims": neutral_opinions, "checked_count": len(results)}