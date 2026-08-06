from __future__ import annotations

import json
from typing import Any

import jsonschema

from ..prompts import NOVELTY_COMPARISON, NOVELTY_COMPARISON_SCHEMA
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json
from .adequacy_common import iter_sentences, top_comments


class NoveltyComparisonClient(AsyncChat):
    PROMPT = NOVELTY_COMPARISON

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, NOVELTY_COMPARISON_SCHEMA)
        if result["target_quote"]:
            verified, _ = self.check.verify([result["target_quote"]], context["contribution_text"], min_char_len=8)
            assert verified, "Novelty target evidence must be copied verbatim from contribution text"
        if result["prior_quote"]:
            verified, _ = self.check.verify([result["prior_quote"]], context["reference_text"], min_char_len=8)
            assert verified, "Novelty prior-survey evidence must be copied verbatim"
        if result["differentiated"]:
            assert result["target_quote"], "A differentiated judgment requires an explicit target-survey contribution"
        return result

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(**inputs), {
            "contribution_text": inputs["contribution_text"],
            "reference_text": inputs["reference_surveys"],
        }


class ContributionAdequacy:
    """Check whether a survey explicitly positions its contribution against prior surveys."""

    def __init__(self, config: ToolConfig, top_k: int = 3):
        self.top_k = top_k
        self.llm = NoveltyComparisonClient(config)

    def _reference_metadata(self, reference_surveys: Any) -> list[dict[str, str]]:
        value = reference_surveys.get("reference_surveys", reference_surveys) if isinstance(reference_surveys, dict) else reference_surveys
        values = value.values() if isinstance(value, dict) else (value or [])
        records = []
        for item in values:
            if not isinstance(item, dict):
                continue
            metadata = next(
                (
                    item.get(key)
                    for key in ("openalex", "semantic_scholar", "paper", "metadata")
                    if isinstance(item.get(key), dict) and item[key].get("title")
                ),
                item,
            )
            if metadata.get("title"):
                records.append({
                    "title": str(metadata.get("title", "")),
                    "abstract": str(metadata.get("abstract", "")),
                })
        return records

    async def __call__(
        self,
        paper: Paper,
        contribution_evals: dict[str, Any] | None = None,
        reference_surveys: Any = None,
        query: str = "",
        *_args,
        **_kwargs,
    ) -> dict[str, Any]:
        contribution_sentences = [
            sentence["text"]
            for sentence in iter_sentences(paper)
            if sentence.get("label") in {"CONTRIBUTION", "CONTRIBUTION+SCOPE"}
        ]
        references = self._reference_metadata(reference_surveys)
        existence = 1.0 if contribution_sentences else 0.0
        if not references:
            return {
                "comments": [],
                "metrics": {
                    "contribution_novelty_existence": existence,
                    "novelty_comparison_available": 0.0,
                },
            }

        contribution_text = "\n".join(contribution_sentences)
        reference_text = json.dumps(references, ensure_ascii=False)
        decision = await self.llm.call(inputs={
            "query": query or paper.title,
            "contribution_text": contribution_text or "None",
            "reference_surveys": reference_text,
        })
        comments = []
        if not decision["differentiated"]:
            comments.append({
                "issue_type": "contribution_novelty_unpositioned",
                "issue": "The survey does not state a substantive difference from prior surveys on the same target topic.",
                "risk": 1.0,
                **decision,
            })
        return {
            "comments": top_comments(comments, self.top_k),
            "metrics": {
                "contribution_novelty_existence": float(decision["differentiated"]),
                "novelty_comparison_available": 1.0,
            },
        }
