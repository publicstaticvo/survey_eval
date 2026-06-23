import re
import asyncio
import jsonschema
from typing import Any, List, Dict

from ..prompts import CLAIM_CLASSIFICATION, CLAIM_SEGMENTATION
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig
from ..fact.utils import split_content_to_paragraph, extract_json

TARGET_LABELS = {"BACKGROUND", "SUMMARY", "COMPARISON", "EVALUATION"}


class ClaimClassificationClient(AsyncChat):
    PROMPT: str = CLAIM_CLASSIFICATION

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        return bool(result.get("is_verifiable_performance_claim"))

    def _organize_inputs(self, inputs):
        paragraph_text = "\n".join(sentence.get("text", "") for sentence in inputs["context_window"])
        prompt = self.PROMPT.format(text=inputs["text"], range=paragraph_text, keys=inputs["citation_keys"])
        return prompt, {}


class ClaimSegmentationLLMClient(AsyncChat):
    PROMPT: str = CLAIM_SEGMENTATION

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        return bool(result.get("is_verifiable_performance_claim"))

    def _organize_inputs(self, inputs):
        paragraph_text = "\n".join(sentence.get("text", "") for sentence in inputs["context_window"])
        prompt = self.PROMPT.format(text=inputs["text"], range=paragraph_text, keys=inputs["citation_keys"])
        return prompt, {}


class ClaimSegmentation:
    def __init__(self, config: ToolConfig):
        self.classify_llm = ClaimClassificationClient(config.llm_server_info, config.sampling_params)
        self.segment_llm = ...

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
            return await self.classify_llm.call(inputs=inputs)
        except Exception:
            return False

    async def __call__(self, paper_content: Dict[str, Any]):
        paragraphs = split_content_to_paragraph(paper_content)
        claims, errors = [], []
        for paragraph_id, paragraph in enumerate(paragraphs):
            for sentence_id, sentence in enumerate(paragraph):
                if sentence.get("label", "") not in TARGET_LABELS: continue
                citation_keys = self._normalize_citations(sentence.get("citations", []))
                if len(citation_keys) != 1: continue
                try:
                    if not await self._is_verifiable(sentence, paragraph, sentence_id): continue
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