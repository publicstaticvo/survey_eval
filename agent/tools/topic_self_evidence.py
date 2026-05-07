from __future__ import annotations

import json
import re
from typing import Any

from .topic_utils import paragraphs_to_text, top_level_section_blocks
from .utility.evidence_check import EvidenceCheck
from .utility.llmclient import AsyncChat
from .utility.tool_config import ToolConfig


STRUCTURE_SCOPE_EXTRACTION_PROMPT = """You are extracting scope and organization evidence from a survey paper.

Text source:
{source_name}

Text:
{text}

Task:
- Extract only sentences that explicitly state the paper's scope, covered aspects, or section-by-section organization.
- Evidence must be copied verbatim from the text.
- If the text says "Section 2 discusses X", return a dict entry mapping the exact section number to X.
- If the text says "Section 3.1 discusses X" and the source section is a top-level section, keep the full subsection number.
- If the text only says the survey covers several aspects without section numbers, return those aspects as a list.
- If no such content exists, return empty objects/lists.

Return JSON only:
```json
{{
  "section_map": {{"2": "topic or aspect"}},
  "aspect_list": ["aspect"],
  "evidence": ["verbatim sentence"]
}}
```
"""


class ScopeEvidenceLLMClient(AsyncChat):
    PROMPT = STRUCTURE_SCOPE_EXTRACTION_PROMPT

    def _availability(self, response, context):
        try:
            data = json.loads(response)
        except Exception:
            match = re.search(r"\{.*\}", response or "", re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
        section_map = data.get("section_map") if isinstance(data.get("section_map"), dict) else {}
        aspect_list = data.get("aspect_list") if isinstance(data.get("aspect_list"), list) else []
        evidence = data.get("evidence") if isinstance(data.get("evidence"), list) else []
        return {
            "source_name": context["source_name"],
            "section_map": {str(k): str(v).strip() for k, v in section_map.items() if str(v).strip()},
            "aspect_list": [str(item).strip() for item in aspect_list if str(item).strip()],
            "evidence": [str(item).strip() for item in evidence if str(item).strip()],
        }

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(source_name=inputs["source_name"], text=inputs["text"]), {
            "source_name": inputs["source_name"],
        }


class SelfScopeEvidenceExtractor:
    def __init__(self, config: ToolConfig):
        self.llm = ScopeEvidenceLLMClient(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _first_section_text(self, paper: dict[str, Any]) -> tuple[str, str]:
        sections = paper.get("sections", []) or []
        if not sections:
            return "first_section", ""
        first = sections[0]
        source_name = f"Section {first.get('section_id', '1')} {first.get('title', '')}".strip()
        return source_name, paragraphs_to_text(first.get("paragraphs", []))

    def _candidate_blocks(self, paper: dict[str, Any]) -> list[dict[str, str]]:
        blocks = []
        source_name, text = self._first_section_text(paper)
        if text:
            blocks.append({"source_name": source_name, "text": text})
        for block in top_level_section_blocks(paper):
            text = block.get("pre_subsection_text") or ""
            if not text:
                continue
            source_name = f"Section {block.get('section_id')} {block.get('section_title')}".strip()
            if not any(existing["source_name"] == source_name for existing in blocks):
                blocks.append({"source_name": source_name, "text": text})
        return blocks

    def _validate_evidence(self, result: dict[str, Any], source_text: str) -> dict[str, Any]:
        evidence = result.get("evidence") or []
        verified, confidence = self.check.verify(evidence, source_text) if evidence else (False, 0.0)
        if not verified:
            return {"source_name": result.get("source_name", ""), "section_map": {}, "aspect_list": [], "evidence": [], "verified": False, "confidence": confidence}
        evidence_text = "\n".join(evidence)
        section_map = {
            key: value
            for key, value in (result.get("section_map") or {}).items()
            if self._claim_grounded_in_evidence(key, value, evidence_text)
        }
        aspect_list = [
            aspect
            for aspect in result.get("aspect_list") or []
            if self._claim_grounded_in_evidence("", aspect, evidence_text)
        ]
        return {
            **result,
            "section_map": section_map,
            "aspect_list": aspect_list,
            "verified": bool(section_map or aspect_list),
            "confidence": confidence,
        }

    def _claim_grounded_in_evidence(self, key: str, value: str, evidence_text: str) -> bool:
        tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9]+", value or "") if len(token) > 2]
        if key and key not in evidence_text:
            return False
        if not tokens:
            return False
        evidence_lower = evidence_text.lower()
        hits = sum(1 for token in set(tokens) if token in evidence_lower)
        return hits / max(1, len(set(tokens))) >= 0.5

    async def __call__(self, paper: dict[str, Any]) -> dict[str, Any]:
        section_map: dict[str, str] = {}
        aspect_list: list[str] = []
        evidence_records = []
        for block in self._candidate_blocks(paper):
            try:
                raw = await self.llm.call(inputs=block)
            except Exception as exc:
                evidence_records.append({"source_name": block["source_name"], "error": str(exc)})
                continue
            validated = self._validate_evidence(raw, block["text"])
            evidence_records.append(validated)
            section_map.update(validated.get("section_map") or {})
            for aspect in validated.get("aspect_list") or []:
                if aspect not in aspect_list:
                    aspect_list.append(aspect)
        evidence_type = "dict" if section_map else "list" if aspect_list else "empty"
        return {
            "type": evidence_type,
            "section_map": section_map,
            "aspect_list": aspect_list,
            "evidence_records": evidence_records,
        }
