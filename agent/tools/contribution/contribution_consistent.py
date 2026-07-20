"""
contribution_consistent.py
闂備礁鎲＄敮鍥磹閺嶎厼钃熼柛銉墮濡﹢鏌涢妷銏℃珖鐟滄澘娼￠弻锝夊Ω閵夈儺浠鹃梺浼欒吂閸撴繄绮欐径鎰劦妞ゆ帊鑳堕埢鏃堟煟濮橆収鍔峚im闂備礁鎼€氱兘宕规导鏉戠畾濞撴埃鍋撻柡灞界墦閹稿﹥寰勫畝鈧粻鎺楁⒑閸涘﹤娴い锝忓閼洪亶鍨鹃幇浣哄弳闂傚嫬娲畷妯荤節濮橆儵?濠电偞鍨堕幐鎼佀囬鈧弻灞筋煥閸繄锛欏┑鐐叉閹告挳宕戦幘鏉戠窞閻庯綆鍓氬娲煟鎼淬垻鈯曢柛鏂炲懏顫?- section & subsection titles
- parsed content topics
- relative sentences
"""
import asyncio
import re
from typing import Any

import jsonschema

from ..preprocess.contribution_classify import GRAPH_ENVIRONMENT_TYPES
from ..preprocess.utils import extract_json
from ..prompts import CONTRIBUTION_CONSISTENT, CONTRIBUTION_CONSISTENT_SCHEMA
from ..utility.llmclient import AsyncChat, AsyncRerank
from ..utility.tool_config import ToolConfig


SECTION_RANGE_RE = re.compile(r"^(?:Section\s+)?(?P<start>\d+(?:\.\d+)*)\s*-\s*(?P<end>\d+(?:\.\d+)*)$")


class ContributionConsistentClient(AsyncChat):
    PROMPT: str = CONTRIBUTION_CONSISTENT

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        jsonschema.validate(result, CONTRIBUTION_CONSISTENT_SCHEMA)
        evidence_items = context["evidence_items"]
        for item in result["supporting_evidence"]:
            assert item["item"] in evidence_items[item["pool"]]
        return result

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(**inputs), {
            "evidence_items": inputs["evidence_items"],
        }


class ContributionConsistency:
    def __init__(self, config: ToolConfig):
        self.config = config
        self.llm = ContributionConsistentClient(config.llm_server_info, config.sampling_params)
        self.rerank = AsyncRerank(config.rerank_server_info)
        self.rerank_top_k = max(1, config.rerank_n_documents)

    def _walk_sections(self, paper: dict[str, Any], groups=("sections", "limitation", "appendix")):
        def walk(section: dict[str, Any], title_path: list[str]):
            current_path = [*title_path, section.get("title", "")]
            yield section, current_path
            for child in section.get("sections", []) or []:
                if isinstance(child, dict):
                    yield from walk(child, current_path)

        for group_name in groups:
            group = paper.get(group_name, [])
            if isinstance(group, dict):
                group = [group]
            for section in group or []:
                if isinstance(section, dict):
                    yield from walk(section, [])

    def _root_sections(self, paper: dict[str, Any], groups=("sections", "limitation", "appendix")) -> list[tuple[dict[str, Any], list[str]]]:
        roots = []
        for group_name in groups:
            group = paper.get(group_name, [])
            if isinstance(group, dict):
                group = [group]
            for section in group or []:
                if isinstance(section, dict):
                    roots.append((section, [section.get("title", "")]))
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

    def _section_range_ids(self, paper: dict[str, Any], key: str) -> list[str]:
        match = SECTION_RANGE_RE.match(key)
        if not match:
            return []
        ordered_ids = [
            str(section.get("section_id", "") or "").strip()
            for section, _ in self._walk_sections(paper)
            if str(section.get("section_id", "") or "").strip()
        ]
        return self._expand_section_range(match.group("start"), match.group("end"), ordered_ids)

    def _section_matches(self, section: dict[str, Any], key: str) -> bool:
        section_id = str(section.get("section_id", "") or "")
        title = str(section.get("title", "") or "")
        return key in {section_id, title, f"Section {section_id}"}

    def _scope_sections(self, paper: dict[str, Any], key: str) -> list[tuple[dict[str, Any], list[str]]]:
        if key == "document":
            return self._root_sections(paper)
        if key in {"Limitation", "Appendix"}:
            group_name = "limitation" if key == "Limitation" else "appendix"
            group = paper.get(group_name, [])
            group = group if isinstance(group, list) else [group]
            scoped = [(section, [section.get("title", "") or key]) for section in group if isinstance(section, dict)]
            if key == "Appendix":
                scoped.extend(
                    item for item in self._walk_sections(paper, groups=("sections",))
                    if str(item[0].get("title", "") or "").strip().casefold() == "appendix"
                )
            deduped = []
            seen = set()
            for section, title_path in scoped:
                marker = id(section)
                if marker not in seen:
                    deduped.append((section, title_path))
                    seen.add(marker)
            return deduped
        range_ids = self._section_range_ids(paper, key)
        if range_ids:
            return [item for item in self._walk_sections(paper) if str(item[0].get("section_id", "") or "").strip() in range_ids]
        return [item for item in self._walk_sections(paper) if self._section_matches(item[0], key)]

    def _paragraph_sentences(self, paragraph):
        if isinstance(paragraph, dict):
            return paragraph.get("sentences", [])
        return paragraph if isinstance(paragraph, list) else []

    def _section_location(self, section: dict[str, Any], title_path: list[str]) -> str:
        section_id = str(section.get("section_id", "") or "").strip()
        title = " > ".join(part for part in title_path if part)
        return " ".join(part for part in [section_id, title] if part).strip() or "document"

    def _sentences_in_sections(self, sections: list[tuple[dict[str, Any], list[str]]]) -> list[dict[str, Any]]:
        sentences = []
        position = 0

        def collect(section: dict[str, Any], title_path: list[str]):
            nonlocal position
            location = self._section_location(section, title_path)
            for paragraph in section.get("paragraphs", []) or []:
                for sentence in self._paragraph_sentences(paragraph):
                    if isinstance(sentence, dict):
                        position += 1
                        sentences.append({
                            "sentence": sentence,
                            "text": sentence.get("caption") or sentence.get("text", ""),
                            "label": sentence.get("label"),
                            "environment_type": sentence.get("environment_type", "text"),
                            "section_location": location,
                            "position": position,
                        })
            for child in section.get("sections", []) or []:
                if isinstance(child, dict):
                    collect(child, [*title_path, child.get("title", "")])

        for section, title_path in sections:
            collect(section, title_path)
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
            if item["sentence"].get("text", "").strip() == original_sentence:
                return items[index + 1:]
        return []

    def _title_items(self, paper: dict[str, Any]) -> list[str]:
        titles = []
        for section, _ in self._walk_sections(paper):
            title = str(section.get("title", "") or "").strip()
            if title:
                titles.append(title)
        return titles

    def _topic_items(self, paper: dict[str, Any]) -> list[str]:
        items = []
        for section, title_path in self._walk_sections(paper):
            location = self._section_location(section, title_path)
            parsed = section.get("parsed_contents") or {}
            if not isinstance(parsed, dict):
                continue
            for topic in parsed.get("topics", []) or []:
                if topic:
                    items.append(f"{location}: topic: {topic}")
            for obj in parsed.get("objects", []) or []:
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("name", "") or "").strip()
                obj_topics = [topic for topic in obj.get("topics", []) or [] if topic]
                if name and obj_topics:
                    items.append(f"{location}: object: {name}; topics: {', '.join(obj_topics)}")
                elif name:
                    items.append(f"{location}: object: {name}")
        return list(dict.fromkeys(items))

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
        paper: dict[str, Any],
        section_key: str,
        claim: dict[str, Any],
        titles: list[str],
        topics: list[str],
    ) -> dict[str, Any]:
        claim_text = claim["target"]
        original_sentence = claim.get("original_contribution_sentence", "")
        if str(section_key).startswith("Figure "):
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
            "full_list_of_titles_in_order": self._format_bullets(titles),
            "full_list_of_section_subtopics_and_research_objects": self._format_bullets(topics),
            "list_of_reranked_candidate_sentences_with_section_location": self._format_bullets(sentence_candidates),
            "K": self.rerank_top_k,
            "evidence_items": {
                "title": set(titles),
                "topic": set(topics),
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

    async def __call__(self, paper: dict[str, Any]):
        titles = self._title_items(paper)
        topics = self._topic_items(paper)
        tasks = []
        for key, claims in paper["contribution_claims"].items():
            for claim in claims:
                tasks.append(asyncio.create_task(self._check_claim(paper, key, claim, titles, topics)))
        checks = await asyncio.gather(*tasks)
        return {
            "checks": checks,
            "consistent": all(check["consistent"] for check in checks),
        }