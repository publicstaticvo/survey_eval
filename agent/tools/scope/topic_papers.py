from __future__ import annotations

import asyncio
import re
from typing import Any

from ..utility.citation_utils import citation_keys as normalize_citation_keys

import numpy as np

from ..prompts import QUERY_EXPAND
from ..utility.academic_engine import get_academic_engine
from ..utility.llmclient import AsyncChat
from ..utility.openalex import OPENALEX_SELECT
from ..utility.s2 import S2_DEFAULT_FIELDS
from ..utility.sbert_client import SentenceTransformerClient
from ..utility.tool_config import ToolConfig
from .utils import extract_json, cosine_similarity_matrix


TARGET_SECTION_TYPES = {"CONTENT", "TAXONOMY"}
TARGET_SENTENCE_LABELS = {"SUMMARY", "SYNTHESIS"}
OPENALEX_TOPIC_PAPER_SELECT = OPENALEX_SELECT
S2_TOPIC_PAPER_SELECT = S2_DEFAULT_FIELDS


class QueryExpand(AsyncChat):
    PROMPT: str = QUERY_EXPAND

    def _availability(self, response, context):
        result = extract_json(response)
        return result["query"]


class TopicSpecificPapers:
    """Find topic-specific potentially missing references from content/taxonomy sections."""

    def __init__(self, config: ToolConfig):
        self.config = config
        self.engine = get_academic_engine(config)
        self.engine_name = (config.default_academic_search_engine or "openalex").strip().lower()
        self.sbert = SentenceTransformerClient(config.sbert_server_url)
        self.query_expand = QueryExpand(config.llm_server_info, config.sampling_params)
        self.search_limit = config.topic_papers_search_limit

    def _uses_semantic_scholar(self) -> bool:
        return self.engine_name in {"semantic_scholar", "semanticscholar", "semantic scholar", "s2"}

    def _select_fields(self) -> str:
        return S2_TOPIC_PAPER_SELECT if self._uses_semantic_scholar() else OPENALEX_TOPIC_PAPER_SELECT

    def _section_title(self, section: dict[str, Any]) -> str:
        return str(section.get("title") or "").strip()

    def _target_leaf_sections(self, paper: dict[str, Any]) -> list[tuple[list[str], dict[str, Any]]]:
        """
        如何识别能用来搜索文章的section标题
        - 类别需要为CONTENT或TAXONOMY
        - 没有CONTENT或TAXONOMY的子节点（这一项可以删除）
        """
        targets = []

        # def has_target_child(section: dict[str, Any]) -> bool:
        #     for child in section.get("sections", []) or []:
        #         if not isinstance(child, dict):
        #             continue
        #         if child.get("functional_type") in TARGET_SECTION_TYPES:
        #             return True
        #         if has_target_child(child):
        #             return True
        #     return False

        def walk(section: dict[str, Any], path: list[str]):
            title = self._section_title(section)
            next_path = [*path, title] if title else path
            #  and not has_target_child(section)
            if section.get("functional_type") in TARGET_SECTION_TYPES:
                targets.append((next_path, section))
            for child in section.get("sections", []) or []:
                if isinstance(child, dict):
                    walk(child, next_path)

        for section in paper.get("sections", []) or []:
            if isinstance(section, dict):
                walk(section, [])
        return targets

    def _paragraph_entities(self, paragraph: Any) -> list[dict[str, Any]]:
        entities = paragraph.get("entities", []) if isinstance(paragraph, dict) else []
        return [entity for entity in entities if isinstance(entity, dict)]

    def _has_cited_entity(self, paragraph: Any) -> bool:
        return any(entity.get("sentence_has_citation") for entity in self._paragraph_entities(paragraph))

    def _citation_keys(self, sentence: dict[str, Any]) -> list[str]:
        return normalize_citation_keys(sentence.get("citations"))

    def _section_paragraphs(self, section: dict[str, Any]) -> list[Any]:
        paragraphs = []

        def walk(node: Any):
            if isinstance(node, dict):
                for paragraph in node.get("paragraphs", []) or []:
                    paragraphs.append(paragraph)
                for child in node.get("sections", []) or []:
                    walk(child)

        walk(section)
        return paragraphs

    def _paragraph_sentences(self, paragraph: Any) -> list[dict[str, Any]]:
        if isinstance(paragraph, dict):
            return [sentence for sentence in paragraph.get("sentences", []) or [] if isinstance(sentence, dict) and sentence.get("text")]
        if isinstance(paragraph, list):
            return [sentence for sentence in paragraph if isinstance(sentence, dict) and sentence.get("text")]
        return []

    def _section_evidence(self, section: dict[str, Any]) -> tuple[list[str], list[str]]:
        texts, citations = [], []
        for paragraph in self._section_paragraphs(section):
            if not self._has_cited_entity(paragraph):
                continue
            for sentence in self._paragraph_sentences(paragraph):
                if sentence.get("label") not in TARGET_SENTENCE_LABELS:
                    continue
                texts.append(sentence["text"])
                citations.extend(self._citation_keys(sentence))
        return texts, list(dict.fromkeys(citations))

    def _paper_sources(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        metadata = info.get("metadata") or {}
        if isinstance(metadata, dict) and ("openalex" in metadata or "semantic scholar" in metadata):
            return [paper for paper in metadata.values() if isinstance(paper, dict)]
        return [metadata] if isinstance(metadata, dict) and metadata else []

    def _cited_papers(self, citation_keys: list[str], paper_content_map: dict[str, Any]) -> list[dict[str, Any]]:
        papers = []
        for key in citation_keys:
            info = paper_content_map.get(key)
            if isinstance(info, dict):
                papers.extend(self._paper_sources(info))
        return [paper for paper in papers if paper.get("title")]

    def _paper_ids(self, paper: dict[str, Any]) -> set[str]:
        ids = set()
        for key in ("id", "paperId", "corpusId"):
            if paper.get(key):
                ids.add(str(paper[key]).replace("https://openalex.org/", ""))
        raw_ids = paper.get("ids")
        if isinstance(raw_ids, dict):
            ids.update(str(value).replace("https://openalex.org/", "") for value in raw_ids.values() if value)
        elif isinstance(raw_ids, (list, tuple, set)):
            ids.update(str(value).replace("https://openalex.org/", "") for value in raw_ids if value)
        return ids

    def _paper_title(self, paper: dict[str, Any]) -> str:
        return re.sub(r"\s+", " ", paper.get("title", "") or "").strip().lower()

    def _is_cited(self, candidate: dict[str, Any], cited_papers: list[dict[str, Any]]) -> bool:
        candidate_ids = self._paper_ids(candidate)
        candidate_title = self._paper_title(candidate)
        for cited in cited_papers:
            if candidate_ids and candidate_ids & self._paper_ids(cited):
                return True
            if candidate_title and candidate_title == self._paper_title(cited):
                return True
        return False

    def _paper_text(self, paper: dict[str, Any]) -> str:
        return f"{paper.get('title', '')}\n{paper.get('abstract', '')}".strip()

    def _reference_similarity(
        self,
        query_text: str,
        candidate_papers: list[dict[str, Any]],
        reference_papers: list[dict[str, Any]],
        insufficient_references: bool,
    ) -> list[float]:
        if not candidate_papers:
            return []
        reference_texts = [self._paper_text(paper) for paper in reference_papers]
        if insufficient_references:
            reference_texts.append(query_text)
        if not reference_texts:
            return [0.0 for _ in candidate_papers]
        texts = [*[self._paper_text(paper) for paper in candidate_papers], *reference_texts]
        embeddings = self.sbert.embed(texts)
        candidate_emb = embeddings[: len(candidate_papers)]
        reference_emb = embeddings[len(candidate_papers) :]
        matrix = cosine_similarity_matrix(candidate_emb, reference_emb)
        return [float(value) for value in matrix.max(axis=1)]

    async def _search(self, query: str) -> list[dict[str, Any]]:
        payload = await self.engine.search_works(
            search=query,
            per_page=self.search_limit,
            select=self._select_fields(),
        )
        return payload.get("results", []) or []

    async def _process_section(
        self,
        paper_title: str,
        path: list[str],
        section: dict[str, Any],
        paper_content_map: dict[str, Any],
    ) -> dict[str, Any] | None:
        summary_sentences, citation_keys = self._section_evidence(section)
        if not summary_sentences:
            return None
        section_title = " > ".join(path)
        query = await self.query_expand.call(inputs={
            "survey_title": paper_title,
            "section_title": section_title,
            "summary_sentences": "\n".join(f"- {text}" for text in summary_sentences),
        })
        section_cited_papers = self._cited_papers(citation_keys, paper_content_map)
        all_cited_papers = self._cited_papers(list(paper_content_map), paper_content_map)
        candidates = [
            paper
            for paper in await self._search(query)
            if not self._is_cited(paper, all_cited_papers)
        ]
        insufficient_references = len(section_cited_papers) < 2
        query_text = f"{paper_title} {section_title}".strip()
        similarities = self._reference_similarity(
            query_text,
            candidates,
            section_cited_papers,
            insufficient_references,
        )
        papers = []
        for paper, similarity in zip(candidates, similarities):
            item = dict(paper)
            item["similarity_to_section_references"] = similarity
            item["used_section_title_fallback"] = insufficient_references
            papers.append(item)
        return {
            "topic": section_title,
            "query": query,
            "summary_sentences": summary_sentences,
            "citation_keys": citation_keys,
            "insufficient_references": insufficient_references,
            "papers": papers,
        }

    async def __call__(self, paper: dict[str, Any], paper_content_map: dict[str, Any]) -> dict[str, Any]:
        paper_title = paper.get("title", "")
        tasks = [
            asyncio.create_task(self._process_section(paper_title, path, section, paper_content_map))
            for path, section in self._target_leaf_sections(paper)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            "topic_specific_papers": [
                result
                for result in results
                if isinstance(result, dict)
            ]
        }

