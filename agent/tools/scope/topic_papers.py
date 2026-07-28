from __future__ import annotations

import asyncio
import json
import re
from datetime import timedelta
from typing import Any

import jsonschema
import networkx as nx

from ..prompts import CITATION_WARRANT
from ..utility.academic_engine import get_academic_engine
from ..utility.llmclient import AsyncChat
from ..utility.openalex import OPENALEX_SELECT
from ..utility.s2 import S2_DEFAULT_FIELDS
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json


TARGET_SECTION_TYPES = {"CONTENT"}
WARRANT_LABELS = {
    "concept_symbol_or_landmark_warrant",
    "attribution_warrant",
    "taxonomy_or_scope_warrant",
    "claim_support_or_counterevidence_warrant",
    "benchmark_dataset_evaluation_warrant",
    "recency_update_warrant",
    "weak_related_work_suggestion",
    "no_obligation",
}
CITATION_WARRANT_SCHEMA = {
    "type": "object",
    "properties": {
        "warrant_label": {"type": "string", "enum": sorted(WARRANT_LABELS)},
        "citation_obligation": {"type": "boolean"},
        "evidence": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["warrant_label", "citation_obligation", "evidence", "reasoning"],
    "additionalProperties": True,
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "the", "to", "toward", "towards", "using", "via", "with",
}


class CitationWarrantClient(AsyncChat):
    PROMPT = CITATION_WARRANT

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, CITATION_WARRANT_SCHEMA)
        weak_labels = {"weak_related_work_suggestion", "no_obligation"}
        if result["warrant_label"] in weak_labels:
            assert result["citation_obligation"] is False
        if result["citation_obligation"]:
            assert result["warrant_label"] not in weak_labels
        assert result["reasoning"].strip()
        return result

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(
            query=inputs.get("query", ""),
            section_title=inputs.get("section_title", ""),
            topics=inputs.get("topics", "[]"),
            section_text=inputs.get("section_text", ""),
            cited_papers=inputs.get("cited_papers", "[]"),
            candidate_title=inputs.get("candidate_title", ""),
            candidate_abstract=inputs.get("candidate_abstract", ""),
            graph_evidence=inputs.get("graph_evidence", "{}"),
        )
        return prompt, {}


class TopicSpecificPapers:
    """Detect citation-warrant and outdated papers from already-covered survey content."""

    def __init__(self, config: ToolConfig):
        self.config = config
        self.eval_date = config.evaluation_date
        self.engine = get_academic_engine(config)
        self.engine_name = (config.default_academic_search_engine or "openalex").strip().lower()
        self.search_limit = config.topic_papers_search_limit
        self.citation_warrant = CitationWarrantClient(config.llm_server_info, config.sampling_params)

    def _uses_semantic_scholar(self) -> bool:
        return self.engine_name in {"semantic_scholar", "semanticscholar", "semantic scholar", "s2"}

    def _select_fields(self) -> str:
        return S2_DEFAULT_FIELDS if self._uses_semantic_scholar() else OPENALEX_SELECT

    def _pool(self, literature_pool: Any) -> dict[str, dict[str, Any]]:
        if isinstance(literature_pool, dict):
            return literature_pool.get("literature_pool", literature_pool) or {}
        return {}

    def _graph_data(self, literature_pool: Any, citation_graph: dict[str, Any] | None = None) -> dict[str, Any]:
        if citation_graph:
            return citation_graph
        if isinstance(literature_pool, dict):
            return literature_pool.get("citation_graph", {}) or {}
        return {}

    def _graph(self, literature_pool: Any, citation_graph: dict[str, Any] | None = None) -> nx.DiGraph:
        data = self._graph_data(literature_pool, citation_graph)
        graph = nx.DiGraph()
        graph.add_nodes_from(data.get("nodes", []) or self._pool(literature_pool).keys())
        graph.add_edges_from((edge["source"], edge["target"]) for edge in data.get("edges", []) or [])
        return graph

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
        return {item for item in ids if item}

    def _paper_title(self, paper: dict[str, Any]) -> str:
        return re.sub(r"\s+", " ", paper.get("title", "") or "").strip().lower()

    def _paper_key(self, paper: dict[str, Any]) -> str:
        ids = sorted(self._paper_ids(paper))
        if ids:
            return f"id:{ids[0]}"
        title = self._paper_title(paper)
        return f"title:{title}" if title else ""

    def _paper_text(self, paper: dict[str, Any]) -> str:
        return f"{paper.get('title', '')}\n{paper.get('abstract', '')}".casefold()

    def _text_contains_query(self, paper: dict[str, Any], query: str) -> bool:
        return query.casefold() in self._paper_text(paper)

    def _topic_chunks(self, topic: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", topic.casefold())
        chunks, current = [], []
        for token in tokens:
            if token in STOPWORDS:
                if current:
                    chunks.append(" ".join(current))
                    current = []
            else:
                current.append(token)
        if current:
            chunks.append(" ".join(current))
        return [chunk for chunk in chunks if chunk]

    def _text_contains_topic(self, text: str, topic: str) -> bool:
        topic = topic.strip().casefold()
        if not topic:
            return False
        if topic in text:
            return True
        chunks = self._topic_chunks(topic)
        return bool(chunks) and all(chunk in text for chunk in chunks)

    def _citation_key_to_pool_key(self, pool: dict[str, dict[str, Any]]) -> dict[str, str]:
        mapping = {}
        for key, item in pool.items():
            for citation_key in item.get("citation_keys", []) or []:
                mapping[str(citation_key)] = key
        return mapping

    def _parsed_contents(self, section: dict[str, Any]) -> list[dict[str, Any]]:
        contents = []
        parsed = section.get("parsed_contents")
        if isinstance(parsed, dict):
            contents.append(parsed)
        for child in section.get("sections", []) or []:
            if isinstance(child, dict):
                contents.extend(self._parsed_contents(child))
        return contents

    def _parsed_topics_and_citations(self, section: dict[str, Any]) -> tuple[list[str], list[str]]:
        topics, citation_keys = [], []
        for parsed in self._parsed_contents(section):
            topics.extend(parsed.get("topics", []) or [])
            for obj in parsed.get("objects", []) or []:
                topics.extend(obj.get("topics", []) or [])
                citation_keys.extend(str(key) for key in obj.get("citation_keys", []) or [])
        return list(dict.fromkeys(t for t in topics if t)), list(dict.fromkeys(k for k in citation_keys if k))

    def _content_sections(self, paper: dict[str, Any]) -> list[dict[str, Any]]:
        sections = []

        def walk(section: dict[str, Any]):
            if section.get("functional_type") in TARGET_SECTION_TYPES:
                sections.append(section)
            for child in section.get("sections", []) or []:
                if isinstance(child, dict):
                    walk(child)

        for section in paper.get("sections", []) or []:
            if isinstance(section, dict):
                walk(section)
        return sections

    def _filter_topic_papers(
        self,
        literature_pool: Any,
        query: str,
        topics: list[str],
        citation_graph: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], nx.DiGraph]:
        pool = self._pool(literature_pool)
        selected = {}
        cited_count = 0
        for key, item in pool.items():
            paper = item.get("paper", item)
            if not isinstance(paper, dict):
                continue
            if item.get("label") == "cited_papers":
                selected[key] = item
                cited_count += 1
                continue
            text = self._paper_text(paper)
            topic_ok = not topics or any(self._text_contains_topic(text, topic) for topic in topics)
            if self._text_contains_query(paper, query) and topic_ok:
                selected[key] = item
        graph = self._graph(literature_pool, citation_graph)
        subgraph = graph.subgraph(selected.keys()).copy()
        print(f"TopicSpecificPapers cited_papers={cited_count}, filtered_literature_pool={len(selected)}")
        return selected, subgraph

    def _clip_text(self, value: Any, limit: int = 6000) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text if len(text) <= limit else text[:limit].rstrip() + "..."

    def _sentence_text(self, sentence: Any) -> str:
        if isinstance(sentence, str):
            return sentence
        if isinstance(sentence, dict):
            return str(sentence.get("text") or sentence.get("sentence") or "")
        return ""

    def _paragraph_text(self, paragraph: Any) -> str:
        if isinstance(paragraph, str):
            return paragraph
        if isinstance(paragraph, list):
            return " ".join(self._sentence_text(sentence) for sentence in paragraph)
        if isinstance(paragraph, dict):
            pieces = [str(paragraph.get("text") or "")]
            for sentence in paragraph.get("sentences", []) or []:
                pieces.append(self._sentence_text(sentence))
            return " ".join(piece for piece in pieces if piece)
        return ""

    def _section_text(self, section: dict[str, Any], limit: int = 6000) -> str:
        pieces = []
        for paragraph in section.get("paragraphs", []) or []:
            pieces.append(self._paragraph_text(paragraph))
        for child in section.get("sections", []) or []:
            if isinstance(child, dict):
                pieces.append(self._section_text(child, limit))
        return self._clip_text(" ".join(piece for piece in pieces if piece), limit)

    def _paper_summary(self, item: dict[str, Any], key: str | None = None) -> dict[str, Any]:
        paper = item.get("paper", item)
        if not isinstance(paper, dict):
            paper = {}
        return {
            "node": key,
            "title": paper.get("title", ""),
            "publication_date": paper.get("publication_date") or paper.get("publicationDate") or paper.get("year") or "",
            "abstract": self._clip_text(paper.get("abstract", ""), 900),
        }

    def _cited_paper_summaries(self, pool: dict[str, dict[str, Any]], sources: list[str], limit: int = 12) -> str:
        summaries = [self._paper_summary(pool[source], source) for source in sources[:limit] if source in pool]
        return json.dumps(summaries, ensure_ascii=False)

    def _graph_evidence(self, graph: nx.DiGraph, key: str, sources: list[str], score: float, rank: int) -> str:
        evidence = {
            "directed_ppr_score": score,
            "directed_ppr_rank": rank,
            "candidate_in_degree": graph.in_degree(key) if key in graph else 0,
            "candidate_out_degree": graph.out_degree(key) if key in graph else 0,
            "cited_papers_that_cite_candidate": [source for source in sources if graph.has_edge(source, key)],
            "cited_papers_cited_by_candidate": [source for source in sources if graph.has_edge(key, source)],
        }
        return json.dumps(evidence, ensure_ascii=False)

    async def _classify_landmark(
        self,
        query: str,
        section_title: str,
        section_text: str,
        topics: list[str],
        cited_papers: str,
        graph_evidence: str,
        key: str,
        paper: dict[str, Any],
        score: float,
        rank: int,
    ):
        try:
            warrant = await self.citation_warrant.call(inputs={
                "query": query,
                "section_title": section_title,
                "topics": json.dumps(topics, ensure_ascii=False),
                "section_text": section_text,
                "cited_papers": cited_papers,
                "candidate_title": paper.get("title", ""),
                "candidate_abstract": self._clip_text(paper.get("abstract", ""), 3000),
                "graph_evidence": graph_evidence,
            })
        except Exception as exc:
            print(f"CitationWarrant {paper.get('title', '')} {exc}")
            return None
        if not warrant.get("citation_obligation"):
            return None
        return {
            "paper": paper,
            "landmark_type": warrant.get("warrant_label"),
            "warrant_label": warrant.get("warrant_label"),
            "citation_warrant": warrant,
            "pagerank": score,
            "rank": rank,
            "node": key,
        }

    async def _detect_landmarks(
        self,
        literature_pool: Any,
        query: str,
        section: dict[str, Any],
        citation_graph: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        topics, citation_keys = self._parsed_topics_and_citations(section)
        if not citation_keys:
            return []
        pool = self._pool(literature_pool)
        key_map = self._citation_key_to_pool_key(pool)
        sources = [key_map[key] for key in citation_keys if key in key_map]
        if not sources:
            return []
        filtered_pool, subgraph = self._filter_topic_papers(literature_pool, query, topics, citation_graph)
        sources = [source for source in sources if source in subgraph]
        if not sources:
            return []
        personalization = {node: 0.0 for node in subgraph.nodes()}
        for source in sources:
            personalization[source] = 1.0 / len(sources)
        try:
            scores = nx.pagerank(subgraph, personalization=personalization)
        except Exception:
            return []
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10]
        section_title = str(section.get("title", "") or "")
        section_text = self._section_text(section)
        cited_papers = self._cited_paper_summaries(pool, sources)
        candidates = [
            (key, score, rank)
            for rank, (key, score) in enumerate(ranked, 1)
            if filtered_pool.get(key, {}).get("label") != "cited_papers"
        ]
        tasks = [
            asyncio.create_task(self._classify_landmark(
                query,
                section_title,
                section_text,
                topics,
                cited_papers,
                self._graph_evidence(subgraph, key, sources, score, rank),
                key,
                filtered_pool[key].get("paper", filtered_pool[key]),
                score,
                rank,
            ))
            for key, score, rank in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [result for result in results if isinstance(result, dict)]

    def _publication_date(self, paper: dict[str, Any]):
        value = paper.get("publication_date") or paper.get("publicationDate") or ""
        if isinstance(value, str) and len(value) >= 10:
            try:
                from datetime import datetime
                return datetime.strptime(value[:10], "%Y-%m-%d")
            except ValueError:
                return None
        year = paper.get("year")
        if isinstance(year, int):
            from datetime import datetime
            return datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
        return None

    async def _search_outdated(self, query: str, section: dict[str, Any], latest_date, cited_nodes: set[str]):
        from_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
        to_date = (self.eval_date - timedelta(days=90)).strftime("%Y-%m-%d")
        search = f'"{query}" AND "{section.get("title", "")}"'
        limit = 50 if self._uses_semantic_scholar() else 10
        try:
            payload = await self.engine.search_works(
                search=search,
                filter={"from_publication_date": from_date, "to_publication_date": to_date, "cited_by_count": 10},
                per_page=limit,
                select=self._select_fields(),
                sort="cited_by_count:desc",
            )
        except Exception as exc:
            print(f"outdatedTopicSearch {section.get('title', '')} {exc}")
            return []
        title = str(section.get("title", "") or "")
        papers = []
        for paper in payload.get("results", []) or []:
            text = self._paper_text(paper)
            if query.casefold() not in text or title.casefold() not in text:
                continue
            key = self._paper_key(paper)
            if key and key in cited_nodes:
                continue
            papers.append(paper)
            if len(papers) >= 10:
                break
        return papers

    async def _detect_outdated_topics(
        self,
        literature_pool: Any,
        query: str,
        paper: dict[str, Any],
    ) -> list[dict[str, Any]]:
        pool = self._pool(literature_pool)
        key_map = self._citation_key_to_pool_key(pool)
        cited_nodes = {key for key, item in pool.items() if item.get("label") == "cited_papers"}
        tasks = []
        sections = []
        for section in self._content_sections(paper):
            _topics, citation_keys = self._parsed_topics_and_citations(section)
            dates = []
            for citation_key in citation_keys:
                pool_key = key_map.get(citation_key)
                if pool_key:
                    date = self._publication_date(pool[pool_key].get("paper", {}))
                    if date:
                        dates.append(date)
            if not dates:
                continue
            latest_date = max(dates)
            if self.eval_date - latest_date <= timedelta(days=365):
                continue
            sections.append((section, latest_date))
            tasks.append(asyncio.create_task(self._search_outdated(query, section, latest_date, cited_nodes)))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        reports = []
        for (section, latest_date), result in zip(sections, results):
            if isinstance(result, list) and result:
                reports.append({
                    "section": section.get("title", ""),
                    "latest_cited_publication_date": latest_date.strftime("%Y-%m-%d"),
                    "papers": result,
                })
        return reports

    async def __call__(
        self,
        query: str,
        paper: dict[str, Any],
        literature_pool: dict[str, Any] | list[dict[str, Any]] | None = None,
        citation_graph: dict[str, Any] | None = None,
        paper_content_map: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        literature_pool = literature_pool or {}
        whole_paper = dict(paper)
        whole_paper["title"] = ""
        sections = [whole_paper, *self._content_sections(paper)]
        landmark_tasks = [
            asyncio.create_task(self._detect_landmarks(literature_pool, query, section, citation_graph))
            for section in sections
        ]
        landmark_results = await asyncio.gather(*landmark_tasks, return_exceptions=True)
        landmarks = []
        for section, result in zip(sections, landmark_results):
            if isinstance(result, list) and result:
                landmarks.append({"section": section.get("title", ""), "papers": result})
        outdated = await self._detect_outdated_topics(literature_pool, query, paper)
        return {
            "topic_specific_papers": {
                "landmarks": landmarks,
                "outdated_topics": outdated,
            }
        }

