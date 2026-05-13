from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import jsonschema

from ..utils import paragraphs_to_text, section_text, is_generic_heading
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig
from ..prompts import SCOPE_CLAIM_EXTRACT, SCOPE_CLAIM_SCHEMA


class ScopeClaimExtractClient(AsyncChat):
    PROMPT = SCOPE_CLAIM_EXTRACT

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        data = json.loads(response)
        jsonschema.validate(data, SCOPE_CLAIM_SCHEMA)
        verified, _ = self.check.verify(data["evidence"], context["text"])
        if data["section_map"] or data["aspect_list"]: assert verified
        evidence_text = "\n".join(data["evidence"])
        for key, value in data["section_map"].items():
            assert not is_generic_heading(value)
            assert self._claim_grounded_in_evidence(key, value, evidence_text)
        for aspect in data["aspect_list"]:
            assert not is_generic_heading(aspect)
            assert self._claim_grounded_in_evidence("", aspect, evidence_text)
        return {
            "source_name": context["source_name"],
            "section_map": data["section_map"],
            "aspect_list": data["aspect_list"]
        }

    def _claim_grounded_in_evidence(self, key: str, value: str, evidence_text: str) -> bool:
        tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9]+", value or "") if len(token) > 2]
        if key and key not in evidence_text: return False
        if not tokens: return False
        evidence_lower = evidence_text.lower()
        hits = sum(1 for token in set(tokens) if token in evidence_lower)
        return hits / max(1, len(set(tokens))) >= 0.5

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(source_name=inputs["source_name"], text=inputs["text"]), {
            "source_name": inputs["source_name"],
            "text": inputs["text"],
        }


class ScopeClaimExtract:
    def __init__(self, config: ToolConfig):
        self.llm = ScopeClaimExtractClient(config)

    def _candidate_blocks(self, paper: dict[str, Any]) -> list[dict[str, str]]:
        """从abstract+introduction，以及每个section和第一个子section之间抽取scope声明。"""
        abstract = paragraphs_to_text(paper['abstract'])
        introduction = section_text(paper['sections'][0])
        blocks = [{"source_name": 1, "text": f"Abstract: {abstract}\n\n1 Introduction\n\n{introduction}"}]
        for i, s in enumerate(paper['sections'][1:], 2):
            if not s['sections']: continue
            text = paragraphs_to_text(s['paragraphs'])
            if text: blocks.append({"source_name": i, "text": text})
        return blocks

    async def __call__(self, paper: dict[str, Any]) -> dict[str, Any]:
        section_map, aspect_list, evidence_records, errors = {}, [], [], 0
        blocks = self._candidate_blocks(paper)
        tasks = [asyncio.create_task(self.llm.call(inputs=block)) for block in blocks]
        for task in asyncio.as_completed(tasks):
            try:
                validated = await task
            except Exception as e:
                print(f"ScopeClaimExtract {e}")
                errors += 1
                continue
            evidence_records.append(validated)
            section_map.update(validated.get("section_map") or {})
            for aspect in validated.get("aspect_list") or []:
                if aspect not in aspect_list:
                    aspect_list.append(aspect)
        print(f"{len(evidence_records)} scope claims {errors} errors")
        return {
            "section_map": section_map,
            "aspect_list": aspect_list,
            "evidence_records": evidence_records,
            "errors": errors
        }
