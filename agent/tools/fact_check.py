from typing import List, Dict, Any

from .tool_config import ToolConfig
from .llmclient import AsyncChat
from .evidence_check import EvidenceCheck
from .prompts import FACTUAL_CORRECTNESS_PROMPT
from .utils import extract_json, split_content_to_paragraph, paragraph_to_text, cosine_similarity_matrix
from .sbert_client import SentenceTransformerClient


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
        self.sbert = SentenceTransformerClient(config.sbert_server_url)
        self.max_passages = max(1, config.rerank_n_documents)
        self.chunk_char_limit = 10000
        self.chunk_paragraph_limit = 5

    def _group_paragraphs(self, paragraph_texts: list[str]) -> list[str]:
        chunks, current, current_len = [], [], 0
        for paragraph in paragraph_texts:
            paragraph = paragraph.strip()
            if not paragraph: continue
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
                if paragraph_text: paragraph_texts.append(paragraph_text)
            for chunk in self._group_paragraphs(paragraph_texts):
                candidates.append(("full_text", f"Title: {title}\nAbstract: {abstract}\n\n{chunk}".strip()))
        elif isinstance(full_content, str) and full_content.strip():
            paragraph_texts = [p.strip() for p in full_content.split("\n\n") if p.strip()]
            if not paragraph_texts:
                paragraph_texts = [full_content.strip()]
            for chunk in self._group_paragraphs(paragraph_texts):
                candidates.append(("full_text", f"Title: {title}\nAbstract: {abstract}\n\n{chunk}".strip()))
        return candidates

    def _select_candidates(self, claim: str, candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if len(candidates) <= self.max_passages: return candidates
        texts = [claim, *[text for _, text in candidates]]
        embeddings = self.sbert.embed(texts)
        scores = cosine_similarity_matrix(embeddings[:1], embeddings[1:])[0].tolist()
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        selected = [ranked[0][0]]
        selected.extend(candidate for (candidate, _) in ranked[1 : self.max_passages])
        return selected

    async def _judge(self, claim: str, material: str, text: str) -> tuple[str, str, float]:
        content_type = "title and abstract" if material == "title_abstract" else "title, abstract, and full text"
        return await self.llm.call(inputs={"claim": claim, "text": text, "content_type": content_type})

    async def __call__(self, claim: str, cited_paper: Dict[str, Any]) -> Dict[str, Any]:
        candidates = self._content_candidates(cited_paper)
        selected = self._select_candidates(claim, candidates)
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
            if judgment in {"SUPPORTED", "REFUTED"}: break
        return {"fact_check": best_result}
