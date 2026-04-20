import re
import asyncio
import jsonschema
from typing import Any, List, Dict

from .prompts import CLAIM_CLASSIFICATION_PROMPT
from .utils import split_content_to_paragraph, extract_json
from .sbert_client import SentenceTransformerClient
from .tool_config import ToolConfig
from .llmclient import AsyncChat


def range_check(claim: str, paragraph: List[Dict[str, Any]], anchor_id: int) -> bool:
    """
    check if the claim's tokens are 
    - 100% from the paragraph and 
    - 90% from the anchor sentence and its adjacent sentences.
    - allow claim starting with "This paper ..."
    """

    def _tokenize(sequence: str) -> list[str]:
        raw = re.split(r"[\s]+", sequence)
        tokens = []
        for t in raw:        
            if t := re.sub(r"^[^\w]+|[^\w]+$", "", t.lower()): tokens.append(t)
        return tokens
    
    claim = claim.lower().replace("this paper", "").strip()
    claim_tokens = _tokenize(claim)
    paragraph_tokens = [_tokenize(s['text']) for s in paragraph]
    if not all(token in sum(paragraph_tokens, []) for token in claim_tokens): return False
    anchor_range = sum(paragraph_tokens[max(0, anchor_id - 1): anchor_id + 2], [])
    # print(sum(1 for token in claim_tokens if token in anchor_range), len(claim_tokens), anchor_range)
    return sum(1 for token in claim_tokens if token in anchor_range) / len(claim_tokens) >= 0.9


class ClaimSegmentationLLMClient(AsyncChat):
    PROMPT: str = CLAIM_CLASSIFICATION_PROMPT

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        return bool(result.get("is_verifiable_performance_claim"))

    def _organize_inputs(self, inputs):
        paragraph_text = "\n".join(sentence.get("text", "") for sentence in inputs["context_window"])
        prompt = self.PROMPT.format(text=inputs["text"], range=paragraph_text, keys=inputs["citation_keys"])
        return prompt, {}


class ClaimSegmentation:
    def __init__(self, config: ToolConfig):
        self.llm = ClaimSegmentationLLMClient(config.llm_server_info, config.sampling_params)

    def _normalize_citations(self, citations: Any) -> list[str]:
        normalized = []
        for citation in citations or []:
            if isinstance(citation, dict):
                key = citation.get("key") or citation.get("ref_text")
            else:
                key = citation
            if key:
                normalized.append(str(key))
        return normalized

    async def _is_verifiable(self, sentence: Dict[str, Any], paragraph: List[Dict[str, Any]], sentence_id: int) -> bool:
        inputs = {
            "text": sentence.get("text", ""),
            "citation_keys": self._normalize_citations(sentence.get("citations", [])),
            "context_window": paragraph[max(0, sentence_id - 1) : sentence_id + 2],
        }
        try:
            return await self.llm.call(inputs=inputs)
        except Exception:
            return False

    async def __call__(self, paper_content: Dict[str, Any]):
        paragraphs = split_content_to_paragraph(paper_content)
        claims = []
        errors = []
        for paragraph_id, paragraph in enumerate(paragraphs):
            for sentence_id, sentence in enumerate(paragraph):
                citation_keys = self._normalize_citations(sentence.get("citations", []))
                if len(citation_keys) != 1: continue
                try:
                    if not await self._is_verifiable(sentence, paragraph, sentence_id):
                        continue
                    claims.append(
                        {
                            "claim_text": sentence.get("text", ""),
                            "citation_key": citation_keys[0],
                            "paragraph_id": paragraph_id,
                            "sentence_id": sentence_id,
                        }
                    )
                except Exception as exc:
                    errors.append({"paragraph_id": paragraph_id, "sentence_id": sentence_id, "error": str(exc)})
        return {"claims": claims, "errors": errors}
