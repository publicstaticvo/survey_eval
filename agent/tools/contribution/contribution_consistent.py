import asyncio
import logging
import re
from typing import Any

import jsonschema

# from ..preprocess.contribution_classify import GRAPH_ENVIRONMENT_TYPES
from ..prompts import CONTRIBUTION_CONSISTENT, CONTRIBUTION_CONSISTENT_SCHEMA
from ..utility.content_walk import iter_sections_with_context
from ..utility.llmclient import AsyncChat, AsyncRerank
from ..utility.paper_elements import Paper, Section
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json

GRAPH_ENVIRONMENT_TYPES = {"figure", "figure*", "table", "table*", "tabular", "longtable"}
SECTION_RANGE_RE = re.compile(r"^(?:Section\s+)?(?P<start>\d+(?:\.\d+)*)\s*-\s*(?P<end>\d+(?:\.\d+)*)$")


class ContributionConsistentClient(AsyncChat):
    PROMPT: str = CONTRIBUTION_CONSISTENT

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        jsonschema.validate(result, CONTRIBUTION_CONSISTENT_SCHEMA)
        evidence_items = context["evidence_items"]
        for item in result["supporting_evidence"]:
            assert item["item"] in evidence_items[item["pool"]], f"Evidence {item['item']} not in pool {item['pool']}"
        return result

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(**inputs)
        return prompt, {
            "evidence_items": inputs["evidence_items"],
        }


class ContributionConsistency:
    def __init__(self, config: ToolConfig):
        self.config = config
        self.llm = ContributionConsistentClient(config.llm_server_info, config.sampling_params)
        self.rerank = AsyncRerank(config.rerank_server_info)
        self.rerank_top_k = max(1, config.rerank_n_documents)

    def _walk_sections(self, paper: Paper, groups=("sections", "limitation", "appendix")):
        yield from iter_sections_with_context(paper, groups=tuple(groups))

    def _root_sections(self, paper: Paper, groups=("sections", "limitation", "appendix")) -> list[tuple[Section, list[str], str]]:
        roots = []
        if "sections" in groups:
            roots.extend((section, [section.name], str(index + 1)) for index, section in enumerate(paper.children))
        if "limitation" in groups:
            roots.extend((section, [section.name], "Limitation") for section in paper.limitation)
        if "appendix" in groups:
            roots.extend((section, [section.name], "Appendix") for section in paper.appendix)
        return roots

    def _section_id_parts(self, section_id: str) -> tuple[int, ...] | None:
        if not re.match(r"^\d+(?:\.\d+)*$", section_id):
            return None
        return tuple(int(part) for part in section_id.split("."))

    def _expand_section_range(self, start: str, end: str, ordered_ids: list[str]) -> list[str]:
        id_set = set(ordered_ids)
        start_parts = self._section_id_parts(start)
        end_parts = self._section_id_parts(end)
        if start_parts and end_parts and len(start_parts) == len(end_parts) and start_parts[:-1] == end_parts[:-1]:
            step = 1 if start_parts[-1] <= end_parts[-1] else -1
            generated = [
                ".".join(str(part) for part in (*start_parts[:-1], value))
                for value in range(start_parts[-1], end_parts[-1] + step, step)
            ]
            if all(section_id in id_set for section_id in generated):
                return generated

        if start in id_set and end in id_set:
            start_index = ordered_ids.index(start)
            end_index = ordered_ids.index(end)
            if start_index <= end_index:
                return ordered_ids[start_index:end_index + 1]
            return ordered_ids[end_index:start_index + 1]
        return []

    def _section_range_ids(self, paper: Paper, key: str) -> list[str]:
        match = SECTION_RANGE_RE.match(key)
        if not match:
            return []
        ordered_ids = [section_id.strip() for _, _, section_id in self._walk_sections(paper) if section_id.strip()]
        return self._expand_section_range(match.group("start"), match.group("end"), ordered_ids)

    def _section_matches(self, section: Section, section_id: str, key: str) -> bool:
        return key in {section_id, section.name, f"Section {section_id}"}

    def _scope_sections(self, paper: Paper, key: str) -> list[tuple[Section, list[str], str]]:
        if key == "document":
            return self._root_sections(paper)
        if key in {"Limitation", "Appendix"}:
            group = paper.limitation if key == "Limitation" else paper.appendix
            scoped = [(section, [section.name or key], key) for section in group]
            if key == "Appendix":
                scoped.extend(
                    item for item in self._walk_sections(paper, groups=("sections",))
                    if item[0].name.strip().casefold() == "appendix"
                )
            deduped = []
            seen = set()
            for section, title_path, section_id in scoped:
                marker = id(section)
                if marker not in seen:
                    deduped.append((section, title_path, section_id))
                    seen.add(marker)
            return deduped
        range_ids = self._section_range_ids(paper, key)
        if range_ids:
            return [item for item in self._walk_sections(paper) if item[2].strip() in range_ids]
        return [item for item in self._walk_sections(paper) if self._section_matches(item[0], item[2], key)]

    def _section_location(self, section: Section, title_path: list[str], section_id: str) -> str:
        section_id = section_id.strip()
        title = " > ".join(part for part in title_path if part)
        return " ".join(part for part in [section_id, title] if part).strip() or "document"

    def _sentences_in_sections(self, sections: list[tuple[Section, list[str], str]]) -> list[dict[str, Any]]:
        sentences = []
        position = 0

        def collect(section: Section, title_path: list[str], section_id: str):
            nonlocal position
            location = self._section_location(section, title_path, section_id)
            for paragraph in section.paragraphs:
                for sentence in paragraph.sentences:
                    position += 1
                    sentences.append({
                        "sentence": sentence,
                        "text": sentence.caption or sentence.text,
                        "label": sentence.label,
                        "environment_type": sentence.environment_type,
                        "section_location": location,
                        "position": position,
                    })
            for index, child in enumerate(section.children):
                child_id = f"{section_id}.{index + 1}" if section_id else str(index + 1)
                collect(child, [*title_path, child.name], child_id)

        for section, title_path, section_id in sections:
            collect(section, title_path, section_id)
        return sentences
    def _all_sentence_items(self, paper: dict[str, Any]) -> list[dict[str, Any]]:
        return self._sentences_in_sections(self._root_sections(paper))

    def _scope_sentence_items(self, paper: dict[str, Any], key: str) -> list[dict[str, Any]]:
        if not key.startswith(("Figure ", "Table ")):
            return self._sentences_in_sections(self._scope_sections(paper, key))
        target_kind, raw_index = key.split(maxsplit=1)
        wanted = int(raw_index) if raw_index.isdigit() else None
        if wanted is None:
            return []
        figure_index, table_index = 0, 0
        for item in self._all_sentence_items(paper):
            environment_type = item["environment_type"]
            if environment_type not in GRAPH_ENVIRONMENT_TYPES:
                continue
            if environment_type in {"table", "table*", "tabular", "longtable"}:
                table_index += 1
                if target_kind == "Table" and table_index == wanted:
                    return [item]
            elif environment_type in {"figure", "figure*"}:
                figure_index += 1
                if target_kind == "Figure" and figure_index == wanted:
                    return [item]
        return []

    def _after_original_sentence(self, items: list[dict[str, Any]], original_sentence: str) -> list[dict[str, Any]]:
        original_sentence = str(original_sentence or "").strip()
        if not original_sentence:
            return items
        for index, item in enumerate(items):
            if item["sentence"].text.strip() == original_sentence:
                return items[index + 1:]
        return []

    def _query_strings(self, query: str | list[str] | None) -> list[str]:
        if isinstance(query, list):
            return [str(item or "").casefold() for item in query if str(item or "").strip()]
        return [str(query or "").casefold()] if str(query or "").strip() else []

    def _object_is_query_only(self, name: str, queries: list[str]) -> bool:
        words = re.findall(r"[A-Za-z0-9]+", name.casefold())
        return bool(words and queries and all(any(word in query for query in queries) for word in words))

    def _section_outline_label(self, section: Section, section_id: str) -> str:
        section_id = section_id.strip()
        prefix = f"Section {section_id}" if re.match(r"^\d+(?:\.\d+)*$", section_id) else section_id
        return " ".join(part for part in [prefix, section.name.strip()] if part).strip() or "document"

    def _section_topic_object_outline(
        self,
        paper: Paper,
        query: str | list[str] | None = None,
    ) -> tuple[str, dict[str, set[str]]]:
        queries = self._query_strings(query)
        lines = []
        evidence_items: dict[str, set[str]] = {"section": set(), "topic": set(), "object": set()}
        for section, _title_path, section_id in self._walk_sections(paper):
            section_label = self._section_outline_label(section, section_id)
            lines.append(f"- {section_label}")
            evidence_items["section"].add(section_label)
            parsed = section.parsed_contents or {}
            if not isinstance(parsed, dict):
                continue
            topics = list(dict.fromkeys(str(topic).strip() for topic in parsed.get("topics", []) or [] if str(topic).strip()))
            objects: list[tuple[str, list[str]]] = []
            for obj in parsed.get("objects", []) or []:
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("name", "") or "").strip()
                if not name or self._object_is_query_only(name, queries):
                    continue
                obj_topics = [str(topic).strip() for topic in obj.get("topics", []) or [] if str(topic).strip()]
                objects.append((name, obj_topics))
            deduped_objects = []
            seen_object_names = set()
            for name, obj_topics in objects:
                if name in seen_object_names:
                    continue
                deduped_objects.append((name, obj_topics))
                seen_object_names.add(name)
            objects = deduped_objects
            assigned_objects = set()
            for topic in topics:
                lines.append(f"  - Topic: {topic}")
                evidence_items["topic"].add(topic)
                for name, obj_topics in objects:
                    if topic in obj_topics:
                        lines.append(f"    - Object: {name}")
                        evidence_items["object"].add(name)
                        assigned_objects.add(name)
            for name, obj_topics in objects:
                if name in assigned_objects:
                    continue
                lines.append(f"  - Object: {name}")
                evidence_items["object"].add(name)
        return "\n".join(lines) if lines else "- None", evidence_items

    def _format_bullets(self, items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- None"

    def _candidate_labels(self, claim_type: str) -> set[str]:
        label = claim_type.split(":", 1)[1]
        labels = {label}
        if label == "SYNTHESIS":
            labels.add("SUMMARY")
        return labels

    def _sentence_candidates(self, claim_type: str, scoped_items: list[dict[str, Any]]) -> list[str]:
        labels = self._candidate_labels(claim_type)
        candidates = []
        for item in scoped_items:
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            if item.get("label") in labels:
                candidates.append(f"[{item['section_location']}] {text}")
            elif "COMPARISON" in labels and item.get("environment_type") in GRAPH_ENVIRONMENT_TYPES:
                candidates.append(f"[{item['section_location']}] {text}")
        return candidates

    async def _reranked_sentence_candidates(
        self,
        claim_text: str,
        claim_type: str,
        scoped_items: list[dict[str, Any]],
    ) -> list[str]:
        candidates = self._sentence_candidates(claim_type, scoped_items)
        if len(candidates) <= self.rerank_top_k:
            return candidates
        selected = await self.rerank.call(claim_text, candidates, top_n=self.rerank_top_k)
        return selected or candidates[:self.rerank_top_k]

    async def _check_claim(
        self,
        paper: Paper,
        section_key: str,
        claim: dict[str, Any],
        section_outline: str,
        evidence_items: dict[str, set[str]],
    ) -> dict[str, Any]:
        claim_text = claim["target"]
        original_sentence = claim.get("original_contribution_sentence", "")
        if str(section_key).startswith("Fig") or str(section_key).startswith("Tab") :
            return {
                "section": section_key,
                "type": claim["type"],
                "target": claim_text,
                "original_contribution_sentence": original_sentence,
                "verdict": "SKIPPED",
                "consistent": True,
                "skipped": True,
                "skip_reason": "figure_claim_not_supported",
                "supporting_evidence": [],
                "reasoning": "Figure-scoped contribution claims are skipped because the current pipeline does not inspect visual image content reliably.",
                "candidate_sentences": [],
            }
        scoped_items = self._scope_sentence_items(paper, section_key)
        scoped_items = self._after_original_sentence(scoped_items, original_sentence)
        sentence_candidates = []
        if claim["type"].startswith("sentence:"):
            sentence_candidates = await self._reranked_sentence_candidates(claim_text, claim["type"], scoped_items)
        inputs = {
            "claim_text": claim_text,
            "original_contribution_sentence": original_sentence,
            "full_list_of_section_subtopics_and_research_objects": section_outline,
            "list_of_reranked_candidate_sentences_with_section_location": self._format_bullets(sentence_candidates),
            "K": self.rerank_top_k,
            "evidence_items": {
                **evidence_items,
                "sentence": set(sentence_candidates),
            },
        }
        result = await self.llm.call(inputs=inputs)
        return {
            "section": section_key,
            "type": claim["type"],
            "target": claim_text,
            "original_contribution_sentence": original_sentence,
            "verdict": result["verdict"],
            "consistent": result["verdict"] == "FULFILLED",
            "supporting_evidence": result["supporting_evidence"],
            "reasoning": result["reasoning"],
            "candidate_sentences": sentence_candidates,
        }

    async def _check_claim_or_error(
        self,
        paper: Paper,
        section_key: str,
        claim: dict[str, Any],
        section_outline: str,
        evidence_items: dict[str, set[str]],
    ) -> dict[str, Any]:
        try:
            return await self._check_claim(paper, section_key, claim, section_outline, evidence_items)
        except Exception as exc:
            logging.exception("Contribution consistency check failed: section=%s target=%s", section_key, claim.get("target"))
            return {
                "section": section_key,
                "type": claim.get("type", ""),
                "target": claim.get("target", ""),
                "original_contribution_sentence": claim.get("original_contribution_sentence", ""),
                "consistent": True,
                "check_failed": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "supporting_evidence": [],
                "reasoning": "",
                "candidate_sentences": [],
            }

    async def __call__(self, paper: Paper, query: str | list[str] | None = None):
        section_outline, evidence_items = self._section_topic_object_outline(paper, query)
        tasks = []
        for key, claims in paper.contribution_claims.items():
            for claim in claims:
                tasks.append(asyncio.create_task(
                    self._check_claim_or_error(paper, key, claim, section_outline, evidence_items)
                ))
        checks = await asyncio.gather(*tasks)
        error_count = sum(1 for check in checks if check.get("check_failed"))
        measured_checks = [check for check in checks if not check.get("skipped") and not check.get("check_failed")]
        fulfilled_count = sum(1 for check in measured_checks if check.get("consistent"))
        contribution_consistency_rate = fulfilled_count / len(measured_checks) if measured_checks else 1.0
        return {
            "checks": checks,
            "consistent": error_count == 0 and all(check["consistent"] for check in checks),
            "error_count": error_count,
            "measured_count": len(measured_checks),
            "fulfilled_count": fulfilled_count,
            "contribution_consistency_rate": contribution_consistency_rate,
        }

