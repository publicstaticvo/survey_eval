from typing import Any

import numpy as np

from ..preprocess.contribution_classify import GRAPH_ENVIRONMENT_TYPES
from ..utility.sbert_client import SentenceTransformerClient
from ..utility.tool_config import ToolConfig


class ContributionConsistency:
    def __init__(self, config: ToolConfig):
        self.config = config
        self.sbert = SentenceTransformerClient(config.sbert_server_url)
        self.similarity_threshold = config.contribution_similarity_threshold

    def _walk_sections(self, paper: dict[str, Any], groups=("sections", "limitation", "appendix")):
        def walk(section: dict[str, Any]):
            yield section
            for child in section.get("sections", []) or []:
                if isinstance(child, dict):
                    yield from walk(child)

        for group_name in groups:
            group = paper.get(group_name, [])
            if isinstance(group, dict):
                group = [group]
            for section in group or []:
                if isinstance(section, dict):
                    yield from walk(section)

    def _section_matches(self, section: dict[str, Any], key: str) -> bool:
        section_id = str(section.get("section_id", "") or "")
        title = str(section.get("title", "") or "")
        return key in {section_id, title, f"Section {section_id}"}

    def _scope_sections(self, paper: dict[str, Any], key: str) -> list[dict[str, Any]]:
        if key == "document":
            return list(self._walk_sections(paper))
        if key == "Limitation":
            group = paper.get("limitation", [])
            return group if isinstance(group, list) else [group]
        return [section for section in self._walk_sections(paper) if self._section_matches(section, key)]

    def _all_sentences(self, paper: dict[str, Any]) -> list[dict[str, Any]]:
        return self._sentences_in_sections(list(self._walk_sections(paper)))

    def _scope_sentences(self, paper: dict[str, Any], key: str) -> list[dict[str, Any]]:
        if not key.startswith(("Figure ", "Table ")):
            return self._sentences_in_sections(self._scope_sections(paper, key))
        target_kind, raw_index = key.split(maxsplit=1)
        wanted = int(raw_index) if raw_index.isdigit() else None
        if wanted is None:
            return []
        figure_index, table_index = 0, 0
        for sentence in self._all_sentences(paper):
            if sentence.get("environment_type") not in GRAPH_ENVIRONMENT_TYPES:
                continue
            if sentence["environment_type"] in {"table", "table*", "tabular", "longtable"}:
                table_index += 1
                if target_kind == "Table" and table_index == wanted:
                    return [sentence]
            elif sentence["environment_type"] in {"figure", "figure*"}:
                figure_index += 1
                if target_kind == "Figure" and figure_index == wanted:
                    return [sentence]
        return []

    def _paragraph_sentences(self, paragraph):
        if isinstance(paragraph, dict):
            return paragraph.get("sentences", [])
        return paragraph if isinstance(paragraph, list) else []

    def _sentences_in_sections(self, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sentences = []
        for section in sections:
            for paragraph in section.get("paragraphs", []) or []:
                sentences.extend(
                    sentence
                    for sentence in self._paragraph_sentences(paragraph)
                    if isinstance(sentence, dict)
                )
            sentences.extend(self._sentences_in_sections(section.get("sections", []) or []))
        return sentences

    def _section_candidates(self, sections: list[dict[str, Any]], functional_type: str) -> list[str]:
        candidates = []
        for section in sections:
            if section.get("functional_type") == functional_type:
                candidates.append(section.get("title", ""))
            candidates.extend(self._section_candidates(section.get("sections", []) or [], functional_type))
        return [candidate for candidate in candidates if candidate]

    def _tag_candidates(self, sections: list[dict[str, Any]], tag: str) -> list[str]:
        candidates = []
        for section in sections:
            if tag in (section.get("content_tags", []) or []):
                candidates.append(section.get("title", ""))
            candidates.extend(self._tag_candidates(section.get("sections", []) or [], tag))
        return [candidate for candidate in candidates if candidate]

    def _graph_caption_candidates(self, sentences: list[dict[str, Any]]) -> list[str]:
        return [
            sentence.get("caption") or sentence.get("text", "")
            for sentence in sentences
            if sentence.get("environment_type") in GRAPH_ENVIRONMENT_TYPES
            and (sentence.get("caption") or sentence.get("text"))
        ]

    def _candidates(
        self,
        sections: list[dict[str, Any]],
        claim_type: str,
        scoped_sentences: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        if claim_type.startswith("sentence:"):
            label = claim_type.split(":", 1)[1]
            sentences = scoped_sentences if scoped_sentences is not None else self._sentences_in_sections(sections)
            candidates = [
                sentence.get("text", "")
                for sentence in sentences
                if sentence.get("label") == label and sentence.get("text")
            ]
            if label == "COMPARISON":
                candidates.extend(self._graph_caption_candidates(sentences))
            return candidates
        if claim_type.startswith("section:"):
            return self._section_candidates(sections, claim_type.split(":", 1)[1])
        if claim_type.startswith("tag:"):
            return self._tag_candidates(sections, claim_type.split(":", 1)[1])
        if claim_type == "coverage":
            return [section.get("title", "") for section in sections if section.get("title")]
        return []

    def _semantic_match(self, target: str, candidates: list[str]) -> bool:
        targets = [part.strip() for part in str(target).split(";") if part.strip()] or [str(target)]
        if not targets or not candidates:
            return False
        embeddings = self.sbert.embed([*targets, *candidates])
        target_embeddings = embeddings[:len(targets)]
        candidate_embeddings = embeddings[len(targets):]
        sims = target_embeddings @ candidate_embeddings.T
        return bool(np.all(np.max(sims, axis=1) >= self.similarity_threshold))

    def _is_consistent(self, target: str, candidates: list[str]) -> bool:
        if not candidates:
            return False
        target_lower = str(target).casefold()
        if any(target_lower in candidate.casefold() for candidate in candidates):
            return True
        return self._semantic_match(str(target), candidates)

    def __call__(self, paper: dict[str, Any], contributions: dict[str, list[dict[str, Any]]]):
        checks = []
        for key, claims in contributions.items():
            sections = self._scope_sections(paper, key)
            scoped_sentences = self._scope_sentences(paper, key)
            for claim in claims:
                candidates = self._candidates(sections, claim["type"], scoped_sentences)
                consistent = self._is_consistent(claim["target"], candidates)
                checks.append({
                    "section": key,
                    "type": claim["type"],
                    "target": claim["target"],
                    "consistent": consistent,
                    "candidates": candidates,
                })
        return {
            "checks": checks,
            "consistent": all(check["consistent"] for check in checks),
        }
