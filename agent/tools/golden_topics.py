from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .preprocess.build_sources import AnchorSurveySource
from .topic_self_evidence import SelfScopeEvidenceExtractor
from .topic_utils import flatten_section_titles, is_generic_heading
from .utility.llmclient import AsyncChat
from .utility.tool_config import ToolConfig


@dataclass
class TopicRecord:
    topic_name: str
    source: str
    representative_papers: list[str]
    evidence_titles: list[str]
    metadata: dict[str, Any]


ANCHOR_HEADING_TOPIC_PROMPT = """You are deriving auditable subtopics for evaluating a target survey.

Target query:
{query}

Anchor survey section headings:
{headings}

Instructions:
- Group semantically equivalent or closely related non-generic headings into subtopics.
- Every subtopic must include one or more sources.
- Each source must copy the original section title exactly as given.
- Do not invent section titles.
- Prefer concise human-readable topic names.
- Return an empty topics list if the headings are insufficient.

Return JSON only:
```json
{{
  "topics": [
    {{
      "topic": "topic_name",
      "sources": [
        {{"survey_id": "S1", "section_title": "exact original heading"}}
      ]
    }}
  ]
}}
```
"""


class AnchorHeadingTopicLLMClient(AsyncChat):
    PROMPT = ANCHOR_HEADING_TOPIC_PROMPT

    def _availability(self, response, context):
        try:
            data = json.loads(response)
        except Exception:
            match = re.search(r"\{.*\}", response or "", re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
        topics = data.get("topics") if isinstance(data.get("topics"), list) else []
        return {"topics": topics, "title_lookup": context["title_lookup"]}

    def _organize_inputs(self, inputs):
        lines = []
        for item in inputs["heading_records"]:
            lines.append(f'- survey_id: {item["survey_id"]}; section_title: {item["section_title"]}')
        return self.PROMPT.format(query=inputs["query"], headings="\n".join(lines)), {
            "title_lookup": inputs["title_lookup"],
        }


class GoldenTopicGenerator:
    def __init__(self, config: ToolConfig):
        self.anchor_source = AnchorSurveySource(config)
        self.anchor_topic_llm = AnchorHeadingTopicLLMClient(config.llm_server_info, config.sampling_params)
        self.self_scope = SelfScopeEvidenceExtractor(config)

    def _anchor_heading_records(self, anchor_surveys: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
        records = []
        title_lookup: dict[str, set[str]] = defaultdict(set)
        for idx, (survey_key, survey) in enumerate(anchor_surveys.items(), 1):
            survey_id = f"S{idx}"
            paper = survey.get("paper") or {}
            real_id = paper.get("id") or survey_key
            skeleton = survey.get("skeleton") or {}
            titles = flatten_section_titles(skeleton)
            if not titles:
                titles = [{"section_title": title, "section_id": ""} for title in survey.get("titles", [])]
            for item in titles:
                title = (item.get("section_title") or item.get("title") or "").strip()
                if not title or is_generic_heading(title):
                    continue
                records.append(
                    {
                        "survey_id": survey_id,
                        "real_survey_id": real_id,
                        "survey_key": survey_key,
                        "section_title": title,
                        "section_id": str(item.get("section_id") or ""),
                    }
                )
                title_lookup[survey_id].add(title)
        return records, title_lookup

    def _verify_anchor_topics(self, raw_topics: dict[str, Any], title_lookup: dict[str, set[str]]) -> list[dict[str, Any]]:
        verified = []
        for item in raw_topics.get("topics") or []:
            topic_name = str(item.get("topic") or "").strip()
            if not topic_name:
                continue
            sources = []
            for source in item.get("sources") or []:
                survey_id = str(source.get("survey_id") or "").strip()
                section_title = str(source.get("section_title") or "").strip()
                if section_title and section_title in title_lookup.get(survey_id, set()):
                    sources.append({"survey_id": survey_id, "section_title": section_title})
            if sources:
                verified.append(
                    {
                        "topic": topic_name,
                        "topic_name": topic_name,
                        "source": "anchor_surveys",
                        "sources": sources,
                        "anchor_survey_coverage": len({s["survey_id"] for s in sources}),
                    }
                )
        return verified

    def _self_topics(self, self_evidence: dict[str, Any]) -> list[dict[str, Any]]:
        topics = []
        for section_id, description in (self_evidence.get("section_map") or {}).items():
            topics.append(
                {
                    "topic": description,
                    "topic_name": description,
                    "source": "survey_self_claim",
                    "sources": [{"section_id": section_id, "evidence_type": "section_map"}],
                }
            )
        for aspect in self_evidence.get("aspect_list") or []:
            topics.append(
                {
                    "topic": aspect,
                    "topic_name": aspect,
                    "source": "survey_self_claim",
                    "sources": [{"evidence_type": "aspect_list"}],
                }
            )
        return topics

    async def __call__(self, query: str, review_paper: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        anchor_data = await self.anchor_source(query, download=True)
        anchor_surveys = anchor_data.get("anchor_surveys", {}) or {}
        heading_records, title_lookup = self._anchor_heading_records(anchor_surveys)

        anchor_topics = []
        if len(anchor_surveys) > 2 and heading_records:
            raw_topics = await self.anchor_topic_llm.call(
                inputs={"query": query, "heading_records": heading_records, "title_lookup": title_lookup}
            )
            anchor_topics = self._verify_anchor_topics(raw_topics, title_lookup)

        self_evidence = await self.self_scope(review_paper or {}) if review_paper else {
            "type": "empty",
            "section_map": {},
            "aspect_list": [],
            "evidence_records": [],
        }
        self_topics = self._self_topics(self_evidence)
        topics = [*anchor_topics, *self_topics]
        return {
            "query": query,
            "golden_topics": topics,
            "anchor_topics": anchor_topics,
            "self_topics": self_topics,
            "self_evidence": self_evidence,
            "anchor_surveys": anchor_surveys,
            "anchor_papers": anchor_data.get("anchor_papers", {}),
            "metadata": {
                "anchor_surveys_count": len(anchor_surveys),
                "downloaded_anchor_surveys_count": len(anchor_surveys),
                "anchor_heading_count": len(heading_records),
                "self_evidence_type": self_evidence.get("type"),
            },
        }
