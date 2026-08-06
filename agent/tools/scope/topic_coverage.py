from __future__ import annotations

import asyncio
from typing import Any

import jsonschema

from ..prompts import MISSING_TOPIC_CLAIM
from ..utility.content_walk import iter_sentences
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json
from .missing_topic_detection import MissingTopicDetector


MISSING_TOPIC_CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "has_claim": {"type": "boolean"},
        "evidence": {"type": "string"},
    },
    "required": ["has_claim", "evidence"],
    "additionalProperties": False,
}


class MissingTopicClaimClient(AsyncChat):
    PROMPT = MISSING_TOPIC_CLAIM

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, MISSING_TOPIC_CLAIM_SCHEMA)
        if result["has_claim"]:
            verified, _ = self.check.verify([result["evidence"]], context["text"], min_char_len=8)
            assert verified, "Scope-exclusion evidence must be copied verbatim from the survey"
        else:
            assert not result["evidence"].strip()
        return result

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(topic=inputs["topic"], text=inputs["text"]), {"text": inputs["text"]}


class TopicCoverage:
    """Generate ranked missing-topic comments and remove explicitly excluded topics."""

    def __init__(self, config: ToolConfig):
        self.missing_topic_detector = MissingTopicDetector(config)
        self.scope_claim = MissingTopicClaimClient(config)

    def _scope_text(self, paper: Paper) -> str:
        return "\n".join(
            sentence.text
            for sentence in iter_sentences(paper, include_abstract=True, include_appendix=True)
            if sentence.label in {"SCOPE", "CONTRIBUTION+SCOPE"}
        )

    async def _scope_decision(self, item: dict[str, Any], scope_text: str):
        if not scope_text:
            return item, {"has_claim": False, "evidence": ""}
        decision = await self.scope_claim.call(inputs={
            "topic": item["community_name"],
            "text": scope_text,
        })
        return item, decision

    async def __call__(
        self,
        queries: list[str],
        paper: Paper,
        reference_surveys: Any = None,
        literature_pool: dict[str, Any] | list[dict[str, Any]] | None = None,
        citation_graph: dict[str, Any] | None = None,
        missing_topics: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        detected, mixed, discarded, metrics = await self.missing_topic_detector.detect(
            queries, paper, literature_pool, citation_graph
        )
        decisions = await asyncio.gather(
            *(self._scope_decision(item, self._scope_text(paper)) for item in detected),
            return_exceptions=True,
        )
        comments = []
        excluded = []
        for result in decisions:
            if isinstance(result, Exception):
                continue
            item, scope = result
            enriched = {**item, "scope_exclusion": scope}
            if scope["has_claim"]:
                excluded.append(enriched)
            else:
                comments.append({
                    "type": "missing_topic",
                    "issue": f"The literature pool contains a query-relevant community not covered by the survey: {item['community_name']}.",
                    **enriched,
                    "risk": max(0.0, float(item["missing_score"])),
                })
        metrics = {
            **metrics,
            "missing_topic_risk": sum(item["risk"] for item in comments),
            "missing_topic_candidate_count": len(comments),
            "scope_excluded_community_count": len(excluded),
        }
        return {
            "comments": comments,
            "topic_evals": {
                "comments": comments,
                "missing_topics": comments,
                "scope_excluded_topics": excluded,
                "mixed_communities": mixed,
                "discarded_communities": discarded,
                **metrics,
            },
        }
