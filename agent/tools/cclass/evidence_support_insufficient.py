from __future__ import annotations

import re
from typing import Any

import jsonschema

from ..prompts import EVIDENCE_SUPPORT_INSUFFICIENT_PROMPT, EVIDENCE_SUPPORT_INSUFFICIENT_SCHEMA
from ..utility.content_walk import paragraph_to_text, paragraphs_to_text
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper, Section
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json


class EvidenceSupportClient(AsyncChat):
    PROMPT = EVIDENCE_SUPPORT_INSUFFICIENT_PROMPT

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        data = extract_json(response)
        jsonschema.validate(data, EVIDENCE_SUPPORT_INSUFFICIENT_SCHEMA)
        assert data["reasoning"].strip()
        if data["problem_type"] == "ARGUMENT_SUPPORT_FAILURE":
            claim_quote = data["claim_quote"].strip()
            assert claim_quote, "claim_quote is required for an argument-support failure"
            verified, score = self.check.verify([claim_quote], context["claim"], min_char_len=8)
            assert verified, "claim_quote is not copied verbatim from the claim"
            assert not data["support_quote"].strip(), "failure must not invent a support quote"
            data["claim_quote_evidence_score"] = score
        else:
            support_quote = data["support_quote"].strip()
            assert support_quote, "support_quote is required when an argument chain is found"
            verified, score = self.check.verify([support_quote], context["support_span"], min_char_len=8)
            assert verified, "support_quote is not copied verbatim from the inspected span"
            assert not data["claim_quote"].strip(), "non-finding must not emit a claim quote"
            data["support_quote_evidence_score"] = score
        return data

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(**inputs)
        return prompt, {"claim": inputs["claim"], "support_span": inputs["support_span"]}


class EvidenceSupportDetector:
    """Check local argumentative support, expanding from a paragraph to its section only when needed."""

    def __init__(self, config: ToolConfig):
        self.llm = EvidenceSupportClient(config)
        self.max_candidates = 40

    def _walk_sections(self, paper: Paper) -> list[Section]:
        sections: list[Section] = []
        def walk(section: Section):
            sections.append(section)
            for child in section.children:
                walk(child)
        for section in paper.children:
            walk(section)
        return sections

    def _is_strong(self, text: str) -> bool:
        return bool(re.search(
            r"\b(always|never|all|none|guarantee[sd]?|prove[sn]?|must|only|entirely|significantly|dramatically|clearly|demonstrate[sd]?|establish(?:es|ed)?|show(?:s|ed)?)\b|\b\d+(?:\.\d+)?%\b",
            text.casefold(),
        ))

    def _candidate_claims(self, paper: Paper) -> list[dict[str, Any]]:
        candidates = []
        for section in self._walk_sections(paper):
            section_text = paragraphs_to_text(section.paragraphs, False)
            for paragraph in section.paragraphs:
                paragraph_text = paragraph_to_text(paragraph, False)
                for sentence in paragraph.sentences:
                    claim = (sentence.caption or sentence.text or "").strip()
                    if len(claim) < 35 or not (self._is_strong(claim) or sentence.label == "SYNTHESIS"):
                        continue
                    candidates.append({
                        "claim": claim,
                        "paragraph": paragraph_text,
                        "section": section_text,
                        "section_name": section.name,
                    })
                    if len(candidates) >= self.max_candidates:
                        return candidates
        return candidates

    async def _has_support(self, topic: str, claim: str, scope: str, span: str) -> dict[str, Any]:
        return await self.llm.call(inputs={
            "topic": topic,
            "claim": claim,
            "search_scope": scope,
            "support_span": span,
        })

    async def __call__(self, paper: Paper, topic: str) -> dict[str, Any]:
        findings: list[dict[str, Any]] = []
        checked = 0
        supported = 0
        for candidate in self._candidate_claims(paper):
            checked += 1
            paragraph_result = await self._has_support(topic, candidate["claim"], "current paragraph", candidate["paragraph"])
            if paragraph_result["problem_type"] == "NO_COMMENT":
                supported += 1
                continue
            section_result = await self._has_support(topic, candidate["claim"], "enclosing section", candidate["section"])
            if section_result["problem_type"] == "NO_COMMENT":
                supported += 1
                continue
            findings.append({
                "module": "cclass.evidence_support_insufficient",
                "report_role": "Weakness",
                "section": candidate["section_name"],
                "paragraph_result": paragraph_result,
                **section_result,
            })
        return {
            "comments": findings,
            "metrics": {
                "argument_support_coverage": supported / checked if checked else 1.0,
                "argument_support_checked_count": checked,
                "argument_support_failure_count": len(findings),
            },
        }
