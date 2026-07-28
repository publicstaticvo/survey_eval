from __future__ import annotations

import logging
from typing import Any

from ..prompts import CONTENT_TAGS, MISSING_TOPIC_CLAIM, SECTION_LABELS, SENTENCE_LABELS
from ..utility.academic_engine import get_academic_engine
from ..utility.content_walk import iter_sentences
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.openalex import OPENALEX_SELECT
from ..utility.paper_elements import Paper, Section
from ..utility.s2 import S2_DEFAULT_FIELDS
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json
from ..preprocess.section_classify import SectionClassification
from .missing_topic_detection import MissingTopicDetector


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
        "red-team",
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
            verified, _ = self.check.verify([data["evidence"]], context["text"])
            assert verified, "MissingTopic: Evidence not valid"
        return data

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(topic=inputs["topic"], text=inputs["text"])
        return prompt, {"text": inputs["text"]}


class TopicCoverage:
    """Evaluate missing survey-level content, method-family topics, and scope exclusions."""

    def __init__(self, config: ToolConfig):
        self.config = config
        self.engine = get_academic_engine(config)
        self.engine_name = (config.default_academic_search_engine or "openalex").strip().lower()
        self.missing_topic_client = MissingTopicClient(config)
        self.search_limit = config.topic_coverage_search_limit
        self.section_classification = SectionClassification(config)
        self.missing_topic_detector = MissingTopicDetector(config)

    def _uses_semantic_scholar(self) -> bool:
        return self.engine_name in {"semantic_scholar", "semanticscholar", "semantic scholar", "s2"}

    def _select_fields(self) -> str:
        return S2_DEFAULT_FIELDS if self._uses_semantic_scholar() else OPENALEX_SELECT

    def _iter_sections(self, paper: Paper):
        def walk(section: Section, depth: int):
            yield section, depth
            for child in section.children:
                yield from walk(child, depth + 1)

        for section in paper.children:
            yield from walk(section, 1)

    def _logical_sentence_labels(self, label: str) -> set[str]:
        if label == "CONTRIBUTION+SCOPE":
            return {"CONTRIBUTION", "SCOPE"}
        return {label} if label else set()

    def _sentence_labels(self, paper: Paper) -> set[str]:
        labels = set()
        for sentence in iter_sentences(paper, include_abstract=True, include_appendix=True):
            labels.update(self._logical_sentence_labels(sentence.label))
        return labels

    def _section_sentence_labels(self, section: Section) -> set[str]:
        labels = set()
        for sentence in iter_sentences(section, include_abstract=False, include_appendix=False):
            labels.update(self._logical_sentence_labels(sentence.label))
        return labels

    def _research_object_count(self, section: Section) -> int:
        names = set()
        for current, _ in self._iter_section_subtree(section):
            parsed = current.parsed_contents if isinstance(current.parsed_contents, dict) else {}
            for obj in parsed.get("objects", []) or []:
                if isinstance(obj, dict) and str(obj.get("name", "")).strip():
                    names.add(str(obj["name"]).strip().casefold())
        return len(names)

    def _iter_section_subtree(self, section: Section):
        def walk(current: Section, depth: int):
            yield current, depth
            for child in current.children:
                yield from walk(child, depth + 1)
        yield from walk(section, 0)

    def _content_sections(self, paper: Paper) -> list[Section]:
        return [section for section, _ in self._iter_sections(paper) if section.functional_type == "CONTENT"]

    def _top_level_content_sections(self, paper: Paper) -> list[tuple[str, Section]]:
        return [
            (str(index + 1), section)
            for index, section in enumerate(paper.children)
            if str(index + 1).isdigit() and section.functional_type == "CONTENT"
        ]

    def _missing_sentence_label_reports(self, paper: Paper) -> list[dict[str, Any]]:
        reports = []
        global_labels = self._sentence_labels(paper)
        for label in sorted((set(SENTENCE_LABELS) - {"CONTRIBUTION+SCOPE", "CONTRAST", "SYNTHESIS"}) - global_labels):
            reports.append({"sentence_label": label, "reason": "absent_sentence_label"})
        content_sections = self._top_level_content_sections(paper)
        if not content_sections:
            for label in ["CONTRAST", "SYNTHESIS"]:
                if label not in global_labels:
                    reports.append({"sentence_label": label, "reason": "absent_sentence_label"})
            return reports
        missing_synthesis_sections = []
        missing_contrast_sections = []
        for section_id, section in content_sections:
            labels = self._section_sentence_labels(section)
            if "SYNTHESIS" not in labels:
                missing_synthesis_sections.append(section_id)
            if self._research_object_count(section) > 1 and "CONTRAST" not in labels:
                missing_contrast_sections.append(section_id)
        if missing_synthesis_sections:
            reports.append({
                "sentence_label": "SYNTHESIS",
                "reason": "top_level_content_sections_missing_synthesis",
                "sections": missing_synthesis_sections,
            })
        if missing_contrast_sections:
            reports.append({
                "sentence_label": "CONTRAST",
                "reason": "top_level_content_sections_missing_contrast",
                "sections": missing_contrast_sections,
            })
        return reports

    def _section_labels(self, paper: Paper) -> tuple[set[str], set[str], dict[str, float]]:
        functional_types, content_tags = set(), set()
        missing_by_depth = {"section": [0, 0], "subsection": [0, 0], "subsubsection": [0, 0]}
        for section, depth in self._iter_sections(paper):
            bucket = "section" if depth == 1 else "subsection" if depth == 2 else "subsubsection"
            missing_by_depth[bucket][1] += 1
            if not section.functional_type:
                missing_by_depth[bucket][0] += 1
            else:
                functional_types.add(section.functional_type)
            for tag in section.content_tags:
                if tag:
                    content_tags.add(tag)
        missing_rates = {
            key: (counts[0] / counts[1] if counts[1] else 0.0)
            for key, counts in missing_by_depth.items()
        }
        return functional_types, content_tags, missing_rates

    def _scope_text(self, paper: Paper) -> str:
        return "\n".join(
            sentence.text
            for sentence in iter_sentences(paper, include_abstract=True, include_appendix=True)
            if sentence.label in {"SCOPE", "CONTRIBUTION+SCOPE"}
        )

    async def _excluded_by_scope(self, query: str, tag: str, scope_text: str) -> dict[str, Any]:
        if not scope_text:
            return {"has_claim": False, "evidence": ""}
        return await self.missing_topic_client.call(inputs={"topic": f"{tag} of {query}", "text": scope_text})

    async def _scope_check_missing_topics(
        self,
        query_text: str,
        missing_topics: list[dict[str, Any]],
        scope_text: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        checked, excluded = [], []
        for item in missing_topics or []:
            topic_name = item["topic_name"]
            try:
                exclusion = await self._excluded_by_scope(query_text, topic_name, scope_text)
            except Exception as exc:
                print(f"TopicCoverage missing topic scope check {topic_name} {exc}")
                exclusion = {"has_claim": False, "evidence": ""}
            enriched = {**item, "scope_exclusion": exclusion}
            if exclusion.get("has_claim"):
                excluded.append(enriched)
            else:
                checked.append(enriched)
        return checked, excluded

    def _reference_survey_contents(self, reference_surveys: Any) -> list[Paper]:
        if isinstance(reference_surveys, dict):
            reference_surveys = reference_surveys.get("reference_surveys", reference_surveys)
        values = reference_surveys.values() if isinstance(reference_surveys, dict) else (reference_surveys or [])
        contents = []
        for item in values:
            if not isinstance(item, dict):
                continue
            full_content = item.get("full_content") or {}
            content = full_content.get("full_content") if isinstance(full_content, dict) else full_content
            if isinstance(content, Paper):
                contents.append(content)
            elif isinstance(content, dict):
                contents.append(Paper.from_skeleton(content))
        return contents

    def _needs_section_classify(self, content: Paper) -> bool:
        sections = list(self._iter_sections(content))
        return bool(sections) and any(
            not section.functional_type or not section.content_tags
            for section, _ in sections
        )

    async def _ensure_reference_sections_classified(self, reference_surveys: Any) -> list[Paper]:
        contents = self._reference_survey_contents(reference_surveys)
        targets = [item for item in contents if self._needs_section_classify(item)]
        if not targets:
            return contents
        for index, content in enumerate(targets):
            logging.info("Classify reference survey sections %d of %d", index + 1, len(targets))
            await self.section_classification(content)
        return contents

    async def _reference_has_tag(self, reference_surveys: Any, tag: str) -> bool:
        for content in await self._ensure_reference_sections_classified(reference_surveys):
            for section, _ in self._iter_sections(content):
                if tag in (section.content_tags or []): return True
        return False

    def _paper_text(self, paper: dict[str, Any]) -> str:
        return f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

    def _paper_matches_queries(self, paper: dict[str, Any], queries: list[str]) -> bool:
        text = self._paper_text(paper)
        return all(query.casefold() in text for query in queries if query)

    def _tag_count(self, tag: str, text: str) -> int:
        if tag == "ETHICS_AND_SAFETY":
            return text.count("ethics") + text.count("safety")
        return text.count(tag.lower())

    def _filter_search_results(self, tag: str, papers: list[dict[str, Any]], queries: list[str]) -> list[dict[str, Any]]:
        return [
            paper
            for paper in papers
            if self._paper_matches_queries(paper, queries) and self._tag_count(tag, self._paper_text(paper)) >= 2
        ]

    async def _search_tag(self, queries: list[str], tag: str) -> list[dict[str, Any]]:
        keywords = CONTENT_TAG_KEYWORDS[tag]
        keyword_query = " OR ".join(keywords)
        payload = await self.engine.search_works(
            search=f'{" AND ".join(queries)} AND ({keyword_query})',
            per_page=self.search_limit,
            select=self._select_fields(),
        )
        return self._filter_search_results(tag, payload.get("results", []) or [], queries)

    async def _missing_topic_reports(
        self,
        queries: list[str],
        paper: Paper,
        literature_pool: Any,
        citation_graph: dict[str, Any] | None,
        scope_text: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        if not literature_pool:
            return [], [], [], []
        detected, unresolved, discarded = await self.missing_topic_detector.detect(
            queries,
            paper,
            literature_pool,
            citation_graph,
        )
        query_text = " ".join(queries)
        scoped, excluded = await self._scope_check_missing_topics(query_text, detected, scope_text)
        ranked = sorted(
            scoped,
            key=lambda item: (item["content_tag_priority"], item["community"]),
        )
        return ranked, excluded, unresolved, discarded

    async def __call__(
        self,
        queries: list[str],
        paper: Paper,
        reference_surveys: Any = None,
        literature_pool: dict[str, Any] | list[dict[str, Any]] | None = None,
        citation_graph: dict[str, Any] | None = None,
        missing_topics: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        query_text = " ".join(queries)
        functional_types, content_tags, missing_label_rates = self._section_labels(paper)
        missing_functional_types = sorted(SECTION_LABELS - functional_types)
        missing_sentence_labels = self._missing_sentence_label_reports(paper)
        missing_content_tags = sorted(CONTENT_TAGS - content_tags - {"GENERAL"})
        scope_text = self._scope_text(paper)
        missing_topics, scope_excluded_topics, unresolved_topics, discarded_topics = await self._missing_topic_reports(
            queries,
            paper,
            literature_pool,
            citation_graph,
            scope_text,
        )
        logging.info("Detected missing functional types: %s", missing_functional_types)
        logging.info("Detected missing sentence labels: %s", missing_sentence_labels)
        logging.info("Detected missing content tags: %s", missing_content_tags)

        excluded = {}
        remaining_tags = []
        for tag in missing_content_tags:
            if tag == "METHOD":
                logging.warning("This survey does not have a METHOD tag")
            try:
                exclusion = await self._excluded_by_scope(query_text, tag, scope_text)
            except Exception as exc:
                print(f"TopicCoverage limitation check {tag} {exc}")
                exclusion = {"has_claim": False, "evidence": ""}
            if exclusion["has_claim"]:
                excluded[tag] = exclusion
            else:
                remaining_tags.append(tag)

        missing_tag_reports = []
        if reference_surveys:
            for tag in remaining_tags:
                if await self._reference_has_tag(reference_surveys, tag):
                    missing_tag_reports.append({
                        "content_tag": tag,
                        "evidence_source": "reference_surveys",
                        "evidence": [],
                    })
            #     continue
            # if tag not in CONTENT_TAG_KEYWORDS: continue
            # try:
            #     papers = await self._search_tag(queries, tag)
            # except Exception as exc:
            #     print(f"TopicCoverage search {tag} {exc}")
            #     papers = []
            # if papers:
            #     missing_tag_reports.append({
            #         "content_tag": tag,
            #         "evidence_source": "academic_search",
            #         "evidence": papers,
            #     })

        return {
            "topic_evals": {
                "missing_functional_types": [
                    {"functional_type": functional_type, "reason": "absent_section_type"}
                    for functional_type in missing_functional_types
                ],
                "missing_sentence_labels": missing_sentence_labels,
                "missing_content_tags": missing_tag_reports,
                "excluded_content_tags": excluded,
                "missing_topics": missing_topics,
                "scope_excluded_missing_topics": scope_excluded_topics,
                "unresolved_missing_topic_communities": unresolved_topics,
                "discarded_missing_topic_communities": discarded_topics,
                "section_label_missing_rates": missing_label_rates,
            }
        }
