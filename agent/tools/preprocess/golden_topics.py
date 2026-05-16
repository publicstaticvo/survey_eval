from __future__ import annotations

import re
import asyncio
import json, jsonschema
from typing import Any

from .get_reference_surveys import GetReferenceSurveys
from .utils import extract_json
from ..prompts import TOPIC_CLUSTER_PROMPT, TOPIC_CLUSTER_SCHEMA, SCOPE_CLAIM_EXTRACT, SCOPE_CLAIM_SCHEMA
from ..utils import flatten_section_titles, is_generic_heading, paragraphs_to_text, section_text
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig
from ..utility.evidence_check import EvidenceCheck


class TopicClusterClient(AsyncChat):
    """Cluster anchor survey headings into auditable topics."""

    PROMPT = TOPIC_CLUSTER_PROMPT

    def _normalize_survey_titles(self, survey: dict[str, Any]) -> list[str]:
        headings = []
        for item in survey.get("titles", []) or []:
            if isinstance(item, dict):
                title = item.get("section_name") or item.get("section_title") or item.get("title") or ""
            else:
                title = str(item or "")
            title = title.strip()
            if title and not is_generic_heading(title):
                headings.append(title)
        skeleton = survey.get("skeleton") or survey.get("paper") or {}
        if not headings and skeleton:
            for item in flatten_section_titles(skeleton):
                title = (item.get("section_title") or item.get("title") or "").strip()
                if title and not is_generic_heading(title):
                    headings.append(title)
        return list(dict.fromkeys(headings))

    def _organize_inputs(self, inputs):
        title_lookup: dict[str, set[str]] = {}
        surveys = []
        for idx, (survey_key, survey) in enumerate((inputs["anchor_surveys"] or {}).items(), 1):
            survey_id = f"S{idx}"
            meta = survey.get("meta") or survey.get("paper") or {}
            headings = self._normalize_survey_titles(survey)
            title_lookup[survey_id] = set(headings)
            surveys.append(
                {
                    "survey_id": survey_id,
                    "survey_title": meta.get("title") or survey_key,
                    "section_headings": headings,
                }
            )
        prompt = self.PROMPT.format(
            query=inputs["query"],
            headings=json.dumps(surveys, ensure_ascii=False, indent=2),
        )
        return prompt, {"title_lookup": title_lookup}

    def _availability(self, response, context):
        data = extract_json(response)
        jsonschema.validate(data, TOPIC_CLUSTER_SCHEMA)
        title_lookup = context["title_lookup"]
        verified_topics = []
        for item in data["topics"]:
            sources = []
            for source in item["sources"]:
                survey_id = source["survey_id"]
                section_title = source["section_title"]
                assert section_title in title_lookup[survey_id]
                sources.append({"survey_id": survey_id, "section_title": section_title})
            verified_topics.append(
                {
                    "topic": item["topic"],
                    "topic_name": item["topic"],
                    "source": "anchor_surveys",
                    "sources": sources,
                    "anchor_survey_coverage": len({source["survey_id"] for source in sources}),
                }
            )
        return {"topics": verified_topics}


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


class GoldenTopicGenerator:
    def __init__(self, config: ToolConfig):
        self.source = GetReferenceSurveys(config)
        self.llm = TopicClusterClient(config.llm_server_info, config.sampling_params)
        self.self_scope_llm = ScopeClaimExtractClient(config)

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
    
    async def _self_scope(self, paper: dict[str, Any]) -> dict[str, Any]:
        section_map, aspect_list, evidences, errors = {}, set(), [], 0
        blocks = self._candidate_blocks(paper)
        tasks = [asyncio.create_task(self.self_scope_llm.call(inputs=block)) for block in blocks]
        for task in asyncio.as_completed(tasks):
            try:
                validated = await task
            except Exception as e:
                print(f"ScopeClaimExtract {e}")
                errors += 1
                continue
            section_map.update(validated['section_map'])
            aspect_list.update(validated["aspect_list"])
            evidences.append(validated)
        print(f"{len(section_map)} section maps {len(aspect_list)} aspects {errors} errors")
        return {
            "section_map": section_map,
            "aspect_list": list(aspect_list),
            "llm_output_with_evidences": evidences,
            "errors": errors
        }

    async def __call__(self, query: str, review_paper: dict[str, Any]) -> dict[str, Any]:
        reference_data = await self.source(query)
        reference_surveys = reference_data.get("reference_surveys", {}) or {}

        reference_topics = []
        if len(reference_surveys) >= 2 and sum(len(x.get("titles", [])) for x in reference_surveys.values()) > 0:
            raw_topics = await self.llm.call(inputs={"query": query, "anchor_surveys": reference_surveys})
            reference_topics = raw_topics["topics"]

        return {
            "reference_data": reference_data,
            "reference_topics": reference_topics,
            "self_topics": await self._self_scope(review_paper),
        }
