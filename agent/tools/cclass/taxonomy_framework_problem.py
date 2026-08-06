from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any

import jsonschema
import networkx as nx

from ..prompts import TAXONOMY_FRAMEWORK_PROBLEM_PROMPT, TAXONOMY_FRAMEWORK_PROBLEM_SCHEMA
from ..utility.content_walk import paragraphs_to_text
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper, Section
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json


PROBLEM_MAP = {
    "MISSING_CATEGORY_DEFINITION": "definition",
    "OVERLAPPING_CATEGORIES": "exclusivity",
    "MIXED_ORGANIZING_AXES": "axis",
    "NO_COMMENT": "none",
}


class TaxonomyFrameworkProblemClient(AsyncChat):
    PROMPT = TAXONOMY_FRAMEWORK_PROBLEM_PROMPT

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        data = extract_json(response)
        jsonschema.validate(data, TAXONOMY_FRAMEWORK_PROBLEM_SCHEMA)
        if data["problem_type"] == "NO_COMMENT":
            return data
        verified, score = self.check.verify([data["survey_quote"]], context["survey_text"], min_char_len=4)
        assert verified, "Taxonomy survey_quote must be copied verbatim from survey context"
        data["survey_quote_evidence_score"] = score
        retrieval_overlap = (
            data["problem_type"] == "OVERLAPPING_CATEGORIES"
            and context["candidate_kind"] == "sibling_partition_retrieval"
        )
        if retrieval_overlap:
            verified, score = self.check.verify(
                [data["external_quote"]], context["external_text"], min_char_len=8
            )
            assert verified, "Retrieved-overlap evidence must be copied verbatim"
            data["external_quote_evidence_score"] = score
        else:
            assert not data["external_quote"].strip()
        return data

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(**inputs), {
            "survey_text": inputs["context"],
            "external_text": inputs["external_evidence"],
            "candidate_kind": inputs["candidate_kind"],
        }


class TaxonomyFrameworkProblemDetector:
    """Audit category definitions and sibling partitions through separate evidence paths."""

    def __init__(self, config: ToolConfig):
        self.llm = TaxonomyFrameworkProblemClient(config)
        self.max_overlap_candidates = 8

    def _walk(self, paper: Paper):
        def visit(section: Section, path: list[str]):
            yield section, path
            for child in section.children:
                yield from visit(child, [*path, child.name])
        for section in paper.children:
            yield from visit(section, [section.name])

    def _context(self, section: Section) -> str:
        return f"Section heading: {section.name}\n{paragraphs_to_text(section.paragraphs, False)}"

    def _topic_records(self, paper: Paper) -> list[dict[str, Any]]:
        records = []
        seen = set()
        for section, path in self._walk(paper):
            parsed = section.parsed_contents if isinstance(section.parsed_contents, dict) else {}
            details = parsed.get("topic_details", []) or [
                {
                    "label": topic,
                    "anchor_type": "section_title" if str(topic).casefold() == section.name.casefold() else "inferred",
                    "evidence_span": None,
                }
                for topic in parsed.get("topics", []) or []
            ]
            objects = parsed.get("objects", []) or []
            for detail in details:
                label = str(detail["label"]).strip()
                if not label:
                    continue
                anchor_type = detail.get("anchor_type", "inferred")
                is_section_title = anchor_type == "section_title" or label.casefold() == section.name.casefold()
                key = (tuple(path[:-1]), label.casefold(), section.name.casefold())
                if key in seen:
                    continue
                seen.add(key)
                citations = set()
                object_evidence = []
                for obj in objects:
                    if label in (obj.get("topics", []) or []):
                        citations.update(str(citation) for citation in obj.get("citation_keys", []) or [])
                        if obj.get("evidence_span"):
                            object_evidence.append(str(obj["evidence_span"]))
                records.append({
                    "label": label,
                    "section": section,
                    "parent_path": tuple(path[:-1]) if len(path) > 1 else tuple(path),
                    "context": self._context(section),
                    "definition": detail.get("evidence_span") or (object_evidence[0] if object_evidence else ""),
                    "citation_keys": citations,
                    "is_section_title": is_section_title,
                })
        return records

    def _citation_key_map(self, pool: dict[str, Any]) -> dict[str, str]:
        return {
            str(citation_key): str(node)
            for node, item in pool.items()
            for citation_key in item.get("citation_keys", []) or []
        }

    def _graph(self, citation_graph: dict[str, Any]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(citation_graph.get("nodes", []) or [])
        graph.add_edges_from(
            (str(edge["source"]), str(edge["target"]))
            for edge in citation_graph.get("edges", []) or []
        )
        return graph

    def _paper_summary(self, item: dict[str, Any]) -> dict[str, str]:
        paper = item.get("paper", item)
        return {"title": str(paper.get("title", "")), "abstract": str(paper.get("abstract", ""))}

    def _common_neighbor_evidence(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        pool: dict[str, Any],
        key_map: dict[str, str],
        graph: nx.Graph,
    ) -> str:
        left_nodes = {key_map[key] for key in left["citation_keys"] if key in key_map}
        right_nodes = {key_map[key] for key in right["citation_keys"] if key in key_map}
        if not left_nodes or not right_nodes:
            return ""
        left_neighbors = set().union(*(set(graph.neighbors(node)) for node in left_nodes if node in graph))
        right_neighbors = set().union(*(set(graph.neighbors(node)) for node in right_nodes if node in graph))
        common = sorted(
            left_neighbors & right_neighbors,
            key=lambda node: (graph.degree(node), node),
            reverse=True,
        )
        evidence = [self._paper_summary(pool[node]) for node in common[:self.max_overlap_candidates] if node in pool]
        return json.dumps(evidence, ensure_ascii=False) if evidence else ""

    async def _judge(
        self,
        topic: str,
        candidate_kind: str,
        artifact: str,
        context: str,
        external_evidence: str,
    ) -> dict[str, Any]:
        return await self.llm.call(inputs={
            "topic": topic,
            "candidate_kind": candidate_kind,
            "artifact": artifact,
            "context": context,
            "external_evidence": external_evidence or "None",
        })

    def _metrics(self, findings: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {"definition": 0, "exclusivity": 0, "axis": 0}
        for finding in findings:
            mapped = PROBLEM_MAP[finding["problem_type"]]
            if mapped in counts:
                counts[mapped] += 1
        return {
            "taxonomy_structural_consistency": float(not findings),
            "taxonomy_problem_count": len(findings),
            "taxonomy_definition_count": counts["definition"],
            "taxonomy_exclusivity_count": counts["exclusivity"],
            "taxonomy_axis_count": counts["axis"],
        }

    async def __call__(
        self,
        paper: Paper,
        topic: str,
        literature_pool: Any = None,
        citation_graph: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        records = self._topic_records(paper)
        pool = literature_pool.get("literature_pool", literature_pool) if isinstance(literature_pool, dict) else {}
        graph = self._graph(citation_graph or {})
        key_map = self._citation_key_map(pool)

        definition_tasks = []
        definition_records = []
        for record in records:
            if record["is_section_title"] or record["definition"] or record["citation_keys"]:
                continue
            definition_records.append(record)
            definition_tasks.append(self._judge(
                topic, "undefined_topic", record["label"], record["context"], ""
            ))
        definition_results = await asyncio.gather(*definition_tasks, return_exceptions=True)
        definition_findings = []
        for record, data in zip(definition_records, definition_results):
            if isinstance(data, dict) and data["problem_type"] == "MISSING_CATEGORY_DEFINITION":
                definition_findings.append({
                    "module": "cclass.taxonomy_framework_problem",
                    "report_role": "Weakness",
                    "detection_path": "category_definition",
                    "section": record["section"].name,
                    **data,
                })

        by_parent: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if not record["is_section_title"]:
                by_parent[record["parent_path"]].append(record)

        pair_records = []
        direct_tasks = []
        retrieval_tasks = []
        retrieval_pairs = []
        for siblings in by_parent.values():
            for index, left in enumerate(siblings):
                for right in siblings[index + 1:]:
                    artifact = f"{left['label']} | {right['label']}"
                    context = f"{left['context']}\n\n{right['context']}"
                    pair_records.append((left, right, artifact, context))
                    direct_tasks.append(self._judge(
                        topic, "sibling_partition_direct", artifact, context, ""
                    ))
                    evidence = self._common_neighbor_evidence(left, right, pool, key_map, graph)
                    if evidence:
                        retrieval_pairs.append((left, right, artifact, context))
                        retrieval_tasks.append(self._judge(
                            topic, "sibling_partition_retrieval", artifact, context, evidence
                        ))

        direct_results, retrieval_results = await asyncio.gather(
            asyncio.gather(*direct_tasks, return_exceptions=True),
            asyncio.gather(*retrieval_tasks, return_exceptions=True),
        )
        direct_findings = []
        for (left, right, _artifact, _context), data in zip(pair_records, direct_results):
            if isinstance(data, dict) and data["problem_type"] != "NO_COMMENT":
                direct_findings.append({
                    "module": "cclass.taxonomy_framework_problem",
                    "report_role": "Weakness",
                    "detection_path": "direct_llm",
                    "section": left["section"].name,
                    **data,
                })
        retrieval_findings = []
        for (left, right, _artifact, _context), data in zip(retrieval_pairs, retrieval_results):
            if isinstance(data, dict) and data["problem_type"] == "OVERLAPPING_CATEGORIES":
                retrieval_findings.append({
                    "module": "cclass.taxonomy_framework_problem",
                    "report_role": "Weakness",
                    "detection_path": "retrieval_common_neighbor",
                    "section": left["section"].name,
                    **data,
                })

        combined = definition_findings + direct_findings + retrieval_findings
        return {
            "comments": combined,
            "category_definition_findings": definition_findings,
            "direct_overlap_findings": direct_findings,
            "retrieval_overlap_findings": retrieval_findings,
            "metrics": self._metrics(combined),
        }
