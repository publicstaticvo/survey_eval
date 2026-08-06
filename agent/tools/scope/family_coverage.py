from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import jsonschema
import networkx as nx

from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.openalex import get_openalex_client
from ..utility.paper_elements import Paper, Section
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json, normalize_text


OPENALEX_FAMILY_SELECT = "id,title,abstract_inverted_index,cited_by_count,publication_date,referenced_works,primary_topic"
FAMILY_SCHEMA = {
    "type": "object",
    "properties": {
        "families": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "scope": {"type": "string", "minLength": 1},
                    "paper_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": ["name", "scope", "paper_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["families"],
    "additionalProperties": False,
}
RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "papers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "minLength": 1},
                    "relevant": {"type": "boolean"},
                    "verbatim_evidence": {"type": "string"},
                },
                "required": ["paper_id", "relevant", "verbatim_evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["papers"],
    "additionalProperties": False,
}


COVERAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "families": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "discussed": {"type": "boolean"},
                    "verbatim_evidence": {"type": "string"},
                },
                "required": ["name", "discussed", "verbatim_evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["families"],
    "additionalProperties": False,
}


class QueryRelevanceClient(AsyncChat):
    """Retain only works that address the query, with candidate-text provenance."""

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, RELEVANCE_SCHEMA)
        returned = {item["paper_id"]: item for item in result["papers"]}
        assert set(returned) == set(context["paper_ids"]), "Relevance response must decide every retrieved paper"
        for item in returned.values():
            if item["relevant"]:
                verified, _ = self.check.verify([item["verbatim_evidence"]], context["candidate_text"], min_char_len=10)
                assert verified, "Relevance evidence is not copied verbatim from the candidate title"
            else:
                assert item["verbatim_evidence"] == "", "Irrelevant candidates must not fabricate evidence"
        return result

    def _organize_inputs(self, inputs):
        prompt = (
            "### Task\n"
            "For each retrieved work, decide whether it directly studies the topic query or a recognizably in-scope methodological, theoretical, or application branch. Reject generic machine-learning papers, merely adjacent uses of ambiguous query words, and papers whose only connection is an application domain. For each retained work, copy its complete title verbatim into verbatim_evidence; for rejected works, use an empty evidence string. Return JSON only: {{\"papers\":[{{\"paper_id\":\"...\",\"relevant\":true,\"verbatim_evidence\":\"...\"}}]}}.\n\n"
            "Topic query: {query}\n\n"
            "Retrieved works:\n{papers}"
        ).format(**inputs)
        return prompt, {"paper_ids": inputs["paper_ids"], "candidate_text": inputs["candidate_text"]}


class FamilyFrontierClient(AsyncChat):
    """Construct a disjoint, paper-level family frontier from a fixed evaluation pool."""

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, FAMILY_SCHEMA)
        expected_ids = context["paper_ids"]
        assigned_ids = [paper_id for family in result["families"] for paper_id in family["paper_ids"]]
        assert Counter(assigned_ids) == Counter(expected_ids), "Family frontier must assign every retrieved paper exactly once"
        return result

    def _organize_inputs(self, inputs):
        prompt = (
            "### Task\n"
            "Construct a literature-family frontier for a survey evaluation pool. A family is a coherent subfield, method family, task family, or evidence family that can be treated as one non-overlapping coverage unit. Return JSON only: {{\"families\":[{{\"name\":\"...\",\"scope\":\"...\",\"paper_ids\":[\"...\"]}}]}}. Assign every supplied paper_id exactly once. Families must be mutually exclusive, collectively exhaustive, and at one common granularity; do not use an \"other\" bucket. Prefer 3--12 papers per family when the corpus permits. The scope must state the membership rule, not a vague summary.\n\n"
            "Topic query: {query}\n\n"
            "Evaluation-pool papers:\n{papers}"
        ).format(**inputs)
        return prompt, {"paper_ids": inputs["paper_ids"]}


class FamilyDiscussionClient(AsyncChat):
    """Determine whether each external literature family is substantively discussed in the survey."""

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, COVERAGE_SCHEMA)
        expected_names = context["family_names"]
        returned = {item["name"]: item for item in result["families"]}
        assert set(returned) == set(expected_names), "Coverage response must decide every family"
        for item in returned.values():
            if item["discussed"]:
                verified, _ = self.check.verify(
                    [item["verbatim_evidence"]],
                    context["survey_text"],
                    min_char_len=12,
                )
                assert verified, "Family discussion evidence is not copied verbatim from the survey"
            else:
                assert item["verbatim_evidence"] == "", "Undiscussed families must not fabricate evidence"
        return result

    def _organize_inputs(self, inputs):
        prompt = (
            "### Task\n"
            "Decide whether the survey substantively discusses each external literature family. Mark discussed=true only when the supplied survey text explains, compares, synthesizes, or otherwise treats that family as a topic; a passing mention or a generic statement does not count. For every discussed family, copy one supporting passage verbatim from the survey. For every undiscussed family, set verbatim_evidence to an empty string. Return JSON only: {{\"families\":[{{\"name\":\"...\",\"discussed\":true,\"verbatim_evidence\":\"...\"}}]}}.\n\n"
            "Families:\n{families}\n\n"
            "Survey text:\n{survey_text}"
        ).format(**inputs)
        return prompt, {"family_names": inputs["family_names"], "survey_text": inputs["survey_text"]}


@dataclass(frozen=True)
class FamilyCoverageResult:
    query: str
    pool_size: int
    retrieved_pool_size: int
    graph_nodes: int
    graph_edges: int
    citation_macro: float
    citation_micro: float
    topic_macro: float | None
    topic_micro: float | None
    families: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "pool_size": self.pool_size,
            "retrieved_pool_size": self.retrieved_pool_size,
            "graph": {"nodes": self.graph_nodes, "edges": self.graph_edges},
            "coverage": {
                "citation_macro": self.citation_macro,
                "citation_micro": self.citation_micro,
                "topic_macro": self.topic_macro,
                "topic_micro": self.topic_micro,
            },
            "families": self.families,
        }


class FamilyFrontierCoverage:
    """Estimate citation and topic coverage against an OpenAlex-derived family frontier.

    The retrieved evaluation pool is fixed before taxonomy construction. The LLM
    only partitions that pool; it cannot add or remove denominator papers.
    """

    def __init__(self, config: ToolConfig, pool_size: int = 200, assess_topics: bool = False):
        self.config = config
        self.pool_size = pool_size
        self.assess_topics = assess_topics
        self.openalex = get_openalex_client(config)
        self.relevance = QueryRelevanceClient(config)
        self.frontier = FamilyFrontierClient(config)
        self.discussion = FamilyDiscussionClient(config)

    def _survey_text(self, paper: Paper, max_chars: int = 90000) -> str:
        parts = [paper.title]
        if paper.abstract:
            parts.extend(sentence.text for sentence in paper.abstract.get_sentences() if sentence.text)

        def walk(section: Section):
            parts.append(f"\n## {section.name}")
            for paragraph in section.paragraphs:
                parts.extend(sentence.text for sentence in paragraph.sentences if sentence.text)
            for child in section.children:
                walk(child)

        for section in paper.children:
            walk(section)
        return "\n".join(item.strip() for item in parts if item and item.strip())[:max_chars]

    def _citation_titles(self, paper: Paper) -> set[str]:
        titles = set()
        for entry in paper.references.values():
            if isinstance(entry, dict) and entry.get("title"):
                normalized = normalize_text(str(entry["title"]))
                if normalized:
                    titles.add(normalized)
        return titles

    def _is_cited(self, candidate: dict[str, Any], cited_titles: set[str]) -> bool:
        candidate_title = normalize_text(str(candidate["title"]))
        if candidate_title in cited_titles:
            return True
        candidate_tokens = set(candidate_title.split())
        for cited_title in cited_titles:
            cited_tokens = set(cited_title.split())
            if len(candidate_tokens) >= 5 and len(cited_tokens) >= 5:
                overlap = len(candidate_tokens & cited_tokens) / max(len(candidate_tokens), len(cited_tokens))
                if overlap >= 0.9:
                    return True
        return False

    def _graph(self, candidates: list[dict[str, Any]]) -> nx.DiGraph:
        graph = nx.DiGraph()
        ids = {candidate["id"] for candidate in candidates}
        graph.add_nodes_from(ids)
        for candidate in candidates:
            for referenced in candidate.get("referenced_works", []) or []:
                target = str(referenced).replace("https://openalex.org/", "")
                if target in ids:
                    graph.add_edge(target, candidate["id"])
        return graph

    def _papers_for_prompt(self, candidates: list[dict[str, Any]]) -> str:
        lines = []
        for candidate in candidates:
            abstract = re.sub(r"\s+", " ", str(candidate.get("abstract") or "")).strip()[:500]
            lines.append(
                json.dumps(
                    {
                        "paper_id": candidate["id"],
                        "title": candidate["title"],
                        "abstract": abstract,
                        "year": str(candidate.get("publication_date") or "")[:4],
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(lines)

    def _openalex_frontier(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        families: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            primary_topic = candidate.get("primary_topic") or {}
            name = str(primary_topic.get("display_name") or "Unclassified OpenAlex topic").strip()
            family = families.setdefault(name, {
                "name": name,
                "scope": f"Works whose OpenAlex primary topic is {name}.",
                "paper_ids": [],
            })
            family["paper_ids"].append(candidate["id"])
        return sorted(families.values(), key=lambda family: (-len(family["paper_ids"]), family["name"]))

    def _family_prompt(self, families: list[dict[str, Any]], candidates_by_id: dict[str, dict[str, Any]]) -> str:
        records = []
        for family in families:
            representatives = [
                candidates_by_id[paper_id]["title"]
                for paper_id in family["paper_ids"][:3]
            ]
            records.append({
                "name": family["name"],
                "scope": family["scope"],
                "representative_titles": representatives,
            })
        return json.dumps(records, ensure_ascii=False, indent=2)

    async def __call__(self, query: str, paper: Paper) -> FamilyCoverageResult:
        payload = await self.openalex.search_works(
            search=query,
            per_page=self.pool_size,
            select=OPENALEX_FAMILY_SELECT,
        )
        retrieved_candidates = payload["results"]
        assert retrieved_candidates, f"OpenAlex returned no candidates for {query!r}"
        candidate_text = self._papers_for_prompt(retrieved_candidates)
        candidate_titles = "\n".join(candidate["title"] for candidate in retrieved_candidates)
        relevance = await self.relevance.call(inputs={
            "query": query,
            "papers": candidate_text,
            "paper_ids": [candidate["id"] for candidate in retrieved_candidates],
            "candidate_text": candidate_titles,
        })
        retained_ids = {item["paper_id"] for item in relevance["papers"] if item["relevant"]}
        candidates = [candidate for candidate in retrieved_candidates if candidate["id"] in retained_ids]
        assert len(candidates) >= 3, f"Too few query-relevant candidates for {query!r}"
        graph = self._graph(candidates)
        frontier = await self.frontier.call(inputs={
            "query": query,
            "papers": self._papers_for_prompt(candidates),
            "paper_ids": [candidate["id"] for candidate in candidates],
        })
        survey_text = self._survey_text(paper)
        by_id = {candidate["id"]: candidate for candidate in candidates}
        if self.assess_topics:
            discussion = await self.discussion.call(inputs={
                "families": self._family_prompt(frontier["families"], by_id),
                "survey_text": survey_text,
                "family_names": [family["name"] for family in frontier["families"]],
            })
            discussed = {item["name"]: item for item in discussion["families"]}
        else:
            discussed = {}
        cited_titles = self._citation_titles(paper)
        families = []
        for family in frontier["families"]:
            family_candidates = [by_id[paper_id] for paper_id in family["paper_ids"]]
            cited_count = sum(self._is_cited(candidate, cited_titles) for candidate in family_candidates)
            citation_fraction = cited_count / len(family_candidates)
            topic_covered = discussed[family["name"]]["discussed"] if self.assess_topics else None
            families.append({
                **family,
                "mass": len(family_candidates),
                "cited_paper_count": cited_count,
                "citation_fraction": citation_fraction,
                "citation_covered": citation_fraction > 0.0,
                "topic_covered": topic_covered,
                "topic_evidence": discussed[family["name"]]["verbatim_evidence"] if self.assess_topics else "",
            })
        total_mass = sum(family["mass"] for family in families)
        return FamilyCoverageResult(
            query=query,
            pool_size=len(candidates),
            retrieved_pool_size=len(retrieved_candidates),
            graph_nodes=graph.number_of_nodes(),
            graph_edges=graph.number_of_edges(),
            citation_macro=sum(family["citation_fraction"] for family in families) / len(families),
            citation_micro=sum(family["cited_paper_count"] for family in families) / total_mass,
            topic_macro=(sum(family["topic_covered"] for family in families) / len(families)) if self.assess_topics else None,
            topic_micro=(sum(family["mass"] for family in families if family["topic_covered"]) / total_mass) if self.assess_topics else None,
            families=families,
        )
