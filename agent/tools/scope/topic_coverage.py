from __future__ import annotations

from typing import Any

from ..prompts import CONTENT_TAGS, MISSING_TOPIC_CLAIM, SECTION_LABELS
from ..utility.academic_engine import get_academic_engine
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.openalex import OPENALEX_SELECT
from ..utility.s2 import S2_DEFAULT_FIELDS
from ..utility.tool_config import ToolConfig
from .utils import extract_json


NON_METHOD_CONTENT_TAGS = sorted(CONTENT_TAGS - {"METHOD", "GENERAL"})
CONTENT_TAG_KEYWORDS = {
    "DATASET": ["dataset", "corpus"],
    "BENCHMARK": ["benchmark", "evaluate"],
    "ETHICS_AND_SAFETY": [
        "ethics",
        "fairness",
        "bias",
        "safety",
        "robustness",
        "explainability",
        "privacy",
        "read-team",
        "jailbreak",
    ],
    "TOOLKIT": ["toolkit", "software", "implementation"],
    "APPLICATION": ["application", "deployment", "industrial", "real-world"],
}


class MissingTopicClient(AsyncChat):
    PROMPT = MISSING_TOPIC_CLAIM

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        data = extract_json(response)
        if data["has_claim"]:
            verified, _ = self.check.verify(data["evidence"], context["text"])
            assert verified
        return data

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(topic=inputs["topic"], text=inputs["text"]), {"text": inputs["text"]}


class TopicCoverage:
    """Use academic search and reference surveys to identify missing non-method content topics."""

    def __init__(self, config: ToolConfig):
        self.config = config
        self.engine = get_academic_engine(config)
        self.engine_name = (config.default_academic_search_engine or "openalex").strip().lower()
        self.missing_topic_client = MissingTopicClient(config)
        self.search_limit = config.topic_coverage_search_limit

    def _uses_semantic_scholar(self) -> bool:
        return self.engine_name in {"semantic_scholar", "semanticscholar", "semantic scholar", "s2"}

    def _select_fields(self) -> str:
        return S2_DEFAULT_FIELDS if self._uses_semantic_scholar() else OPENALEX_SELECT

    def _iter_sections(self, paper: dict[str, Any]):
        def walk(section: dict[str, Any], depth: int):
            yield section, depth
            for child in section.get("sections", []) or []:
                if isinstance(child, dict):
                    yield from walk(child, depth + 1)

        for section in paper.get("sections", []) or []:
            if isinstance(section, dict):
                yield from walk(section, 1)

    def _iter_sentences(self, paper: dict[str, Any]):
        def walk(node: Any):
            if isinstance(node, dict):
                for paragraph in node.get("paragraphs", []) or []:
                    yield from walk(paragraph)
                for section in node.get("sections", []) or []:
                    yield from walk(section)
            elif isinstance(node, list):
                for sentence in node:
                    if isinstance(sentence, dict) and sentence.get("text"):
                        yield sentence

        yield from walk(paper)

    def _section_labels(self, paper: dict[str, Any]) -> tuple[set[str], set[str], dict[str, float]]:
        functional_types, content_tags = set(), set()
        missing_by_depth = {"section": [0, 0], "subsection": [0, 0], "subsubsection": [0, 0]}
        for section, depth in self._iter_sections(paper):
            bucket = "section" if depth == 1 else "subsection" if depth == 2 else "subsubsection"
            missing_by_depth[bucket][1] += 1
            if not section.get("functional_type"):
                missing_by_depth[bucket][0] += 1
            else:
                functional_types.add(section["functional_type"])
            for tag in section.get("content_tags", []) or []:
                if tag:
                    content_tags.add(tag)
        missing_rates = {
            key: (counts[0] / counts[1] if counts[1] else 0.0)
            for key, counts in missing_by_depth.items()
        }
        return functional_types, content_tags, missing_rates

    def _limitation_text(self, paper: dict[str, Any]) -> str:
        return "\n".join(
            sentence["text"]
            for sentence in self._iter_sentences(paper)
            if sentence.get("label") == "LIMITATION"
        )

    async def _excluded_by_limitation(self, query: str, tag: str, limitation_text: str) -> dict[str, Any]:
        if not limitation_text:
            return {"has_claim": False, "evidence": ""}
        return await self.missing_topic_client.call(inputs={
            "topic": f"{tag} of {query}",
            "text": limitation_text,
        })

    def _reference_survey_contents(self, reference_surveys: Any) -> list[dict[str, Any]]:
        if isinstance(reference_surveys, dict):
            reference_surveys = reference_surveys.get("reference_surveys", reference_surveys)
        values = reference_surveys.values() if isinstance(reference_surveys, dict) else (reference_surveys or [])
        contents = []
        for item in values:
            if not isinstance(item, dict):
                continue
            content = (item.get("full_content") or {}).get("full_content") or item.get("full_content")
            if isinstance(content, dict):
                contents.append(content)
        return contents

    def _reference_has_tag(self, reference_surveys: Any, tag: str) -> bool:
        for content in self._reference_survey_contents(reference_surveys):
            for section, _ in self._iter_sections(content):
                if tag in (section.get("content_tags", []) or []):
                    return True
        return False

    def _paper_text(self, paper: dict[str, Any]) -> str:
        return f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

    def _tag_count(self, tag: str, text: str) -> int:
        if tag == "ETHICS_AND_SAFETY":
            return text.count("ethics") + text.count("safety")
        return text.count(tag.lower())

    def _filter_search_results(self, tag: str, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            paper
            for paper in papers
            if self._tag_count(tag, self._paper_text(paper)) >= 2
        ]

    async def _search_tag(self, query: str, tag: str) -> list[dict[str, Any]]:
        keywords = CONTENT_TAG_KEYWORDS[tag]
        keyword_query = " OR ".join(keywords)
        payload = await self.engine.search_works(
            search=f'"{query}" AND ({keyword_query})',
            per_page=self.search_limit,
            select=self._select_fields(),
        )
        return self._filter_search_results(tag, payload.get("results", []) or [])

    async def __call__(
        self,
        query: str | dict[str, Any],
        paper: dict[str, Any],
        reference_surveys: Any = None,
    ) -> dict[str, Any]:
        if isinstance(query, dict):
            topic_data = query
            reference_data = topic_data.get("reference_data", {}) or {}
            reference_surveys = reference_surveys or reference_data.get("reference_surveys")
            query = topic_data.get("query") or paper.get("title", "")
        functional_types, content_tags, missing_label_rates = self._section_labels(paper)
        missing_functional_types = sorted(SECTION_LABELS - functional_types)
        missing_content_tags = sorted(CONTENT_TAGS - content_tags - {"GENERAL"})
        limitation_text = self._limitation_text(paper)

        excluded = {}
        remaining_tags = []
        for tag in missing_content_tags:
            if tag == "METHOD":
                continue
            try:
                exclusion = await self._excluded_by_limitation(query, tag, limitation_text)
            except Exception as exc:
                print(f"TopicCoverage limitation check {tag} {exc}")
                exclusion = {"has_claim": False, "evidence": ""}
            if exclusion["has_claim"]:
                excluded[tag] = exclusion
            else:
                remaining_tags.append(tag)

        missing_tag_reports = []
        for tag in remaining_tags:
            if reference_surveys:
                if self._reference_has_tag(reference_surveys, tag):
                    missing_tag_reports.append({
                        "content_tag": tag,
                        "evidence_source": "reference_surveys",
                        "evidence": [],
                    })
                continue
            if tag not in CONTENT_TAG_KEYWORDS:
                continue
            try:
                papers = await self._search_tag(query, tag)
            except Exception as exc:
                print(f"TopicCoverage search {tag} {exc}")
                papers = []
            if papers:
                missing_tag_reports.append({
                    "content_tag": tag,
                    "evidence_source": "academic_search",
                    "evidence": papers,
                })

        return {
            "topic_evals": {
                "missing_functional_types": [
                    {"functional_type": functional_type, "reason": "absent_section_type"}
                    for functional_type in missing_functional_types
                ],
                "missing_content_tags": missing_tag_reports,
                "excluded_content_tags": excluded,
                "section_label_missing_rates": missing_label_rates,
            }
        }


MissingTopicLLMClient = MissingTopicClient
TopicCoverageCritic = TopicCoverage
