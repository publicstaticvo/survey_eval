from __future__ import annotations

import asyncio
import math
import re
from datetime import timedelta
from typing import Any

import numpy as np

from ..utils import months_between, paper_id_aliases, paper_text, parse_date
from ..utility.academic_engine import get_academic_engine
from ..utility.openalex import get_openalex_client
from ..utility.sbert_client import SentenceTransformerClient
from ..utility.tool_config import ToolConfig
from ..utility.utils import valid_check


def cosine_similarity_matrix(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_norm[left_norm == 0] = 1.0
    right_norm[right_norm == 0] = 1.0
    return (left / left_norm) @ (right / right_norm).T


OPENALEX_MISSING_SELECT = (
    "id,cited_by_count,counts_by_year,referenced_works,publication_date,"
    "created_date,abstract_inverted_index,title,authorships"
)


class MissingPaperCheck:
    """
    Missing citation detection contains exactly two checks.

    1. Reference-survey consensus papers:
       If more than two reference surveys are available, report papers that are cited by
       strictly more than half of those reference surveys but are not cited by the target
       survey. These are reported as weaknesses because the evidence comes from field-level
       survey consensus rather than from the target survey's own scope claims.

    2. Recent papers under self-declared topics:
       For topics declared by the target survey itself, search for recent papers and report
       high-velocity, non-redundant papers as comments. Topics that TopicCoverage already
       marks as missing are skipped here, because their omission is already handled by topic
       coverage and should not be double-counted as missing citations.
    """

    OPENALEX_COUNT_SELECT = "id,cited_by_count,counts_by_year,publication_date,title"

    def __init__(self, config: ToolConfig):
        self.config = config
        self.eval_date = config.evaluation_date
        self.sbert = SentenceTransformerClient(config.sbert_server_url)
        self.engine = get_academic_engine(config)
        self.openalex = get_openalex_client(config)
        self.engine_name = (config.default_academic_search_engine or "openalex").strip().lower()
        self.use_openalex_count_by_year = config.use_openalex_count_by_year
        self.new_topic_threshold = config.new_paper_topic_similarity_threshold
        self.new_reference_overlap_threshold = config.new_paper_reference_overlap_threshold

    def _citation_count_by_eval_date(self, paper: dict[str, Any]) -> int:
        """Return citation count truncated by evaluation date when OpenAlex yearly counts are available."""
        if self.engine_name in {"semantic_scholar", "semanticscholar", "s2"} and not self.use_openalex_count_by_year:
            return int(paper.get("citation_count", paper.get("citationCount", paper.get("cited_by_count", 0))) or 0)
        counts = paper.get("counts_by_year", []) or []
        if not counts:
            return int(paper.get("cited_by_count", 0) or 0)
        return sum(
            int(item.get("cited_by_count", 0) or 0)
            for item in counts
            if int(item.get("year", 0) or 0) <= self.eval_date.year
        )

    async def _attach_openalex_counts(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Optionally enrich Semantic Scholar results with OpenAlex counts_by_year for reproducibility."""
        if self.engine_name not in {"semantic_scholar", "semanticscholar", "s2"}:
            return papers
        if not self.use_openalex_count_by_year:
            return papers

        async def _single(paper: dict[str, Any]):
            title = paper.get("title", "")
            if not title:
                return paper
            try:
                openalex_paper = await self.openalex.find_work_by_title(title, select=self.OPENALEX_COUNT_SELECT)
            except Exception:
                openalex_paper = None
            if not openalex_paper:
                return paper
            merged = dict(paper)
            merged["openalex_count_metadata"] = openalex_paper
            merged["cited_by_count"] = openalex_paper.get("cited_by_count", merged.get("cited_by_count", 0))
            merged["counts_by_year"] = openalex_paper.get("counts_by_year", [])
            return merged

        tasks = [asyncio.create_task(_single(paper)) for paper in papers]
        return list(await asyncio.gather(*tasks))

    def _prepare_query(self, query: str, topic: str) -> str:
        query_tokens = query.split()
        query_lower = {token.lower() for token in query_tokens}
        topic_tokens = [token for token in topic.split() if token.lower() not in query_lower]
        merged = " ".join([query, " ".join(topic_tokens).strip()]).strip()
        return merged or topic

    async def _search_topic_papers(self, search_query: str, recent: bool) -> list[dict[str, Any]]:
        """Search topic candidates; recent=True restricts to the evaluation-date freshness window."""
        filters = {}
        if recent:
            filters = {
                "from_publication_date": (self.eval_date - timedelta(days=731)).strftime("%Y-%m-%d"),
                "to_publication_date": (self.eval_date - timedelta(days=90)).strftime("%Y-%m-%d"),
            }
        else:
            filters = {"to_publication_date": self.eval_date.strftime("%Y-%m-%d")}
        results = await self.engine.search_works(
            "works",
            search=search_query,
            filter=filters,
            per_page=50,
            select=OPENALEX_MISSING_SELECT,
        )
        return await self._attach_openalex_counts(results.get("results", []) or [])

    def _topic_similarity(self, topic: str, papers: list[dict[str, Any]]) -> list[float]:
        if not papers:
            return []
        embeddings = self.sbert.embed([topic, *[paper_text(paper) for paper in papers]])
        sims = cosine_similarity_matrix(embeddings[:1], embeddings[1:])[0]
        return [float(value) for value in sims]

    def _redundancy_scores(self, candidates: list[dict[str, Any]], cited_papers: list[dict[str, Any]]) -> list[float]:
        if not candidates or not cited_papers:
            return [0.0 for _ in candidates]
        texts = [paper_text(paper) for paper in [*candidates, *cited_papers]]
        embeddings = self.sbert.embed(texts)
        split = len(candidates)
        matrix = cosine_similarity_matrix(embeddings[:split], embeddings[split:])
        return [float(row.max()) for row in matrix]

    def _reference_overlap(self, paper: dict[str, Any], cited_ids: set[str]) -> float:
        refs = set(paper.get("referenced_works", []) or [])
        if not refs: return 0.0
        return len(refs & cited_ids) / len(refs)

    def _impact_cutoff(self, values: list[float], top_fraction: float) -> float:
        if not values:
            return float("inf")
        rank = max(0, int(len(values) * top_fraction) - 1)
        ordered = sorted(values, reverse=True)
        return ordered[min(rank, len(ordered) - 1)]

    def _paper_aliases(self, paper_id: str, paper: dict[str, Any]) -> set[str]:
        aliases = paper_id_aliases(paper)
        assert "https://openalex.org/" not in paper_id
        if paper_id: aliases.add(paper_id)
        return aliases

    def _source_name(self, value: str) -> str:
        value = (value or "").strip().lower().replace("_", " ")
        if value in {"semantic scholar", "semanticscholar", "s2"}:
            return "semantic scholar"
        if value == "openalex":
            return "openalex"
        if value == "websearch":
            return "websearch"
        return value

    def _infer_source(self, paper: dict[str, Any]) -> str:
        if paper.get("paperId") or paper.get("externalIds") or paper.get("citationCount") is not None:
            return "semantic scholar"
        if str(paper.get("id", "")).startswith("W") or paper.get("counts_by_year") is not None or paper.get("referenced_works") is not None:
            return "openalex"
        return self._source_name(self.engine_name)

    def _metadata_sources(self, info: dict[str, Any]) -> dict[str, dict[str, Any]]:
        metadata = info.get("metadata") or {}
        if not isinstance(metadata, dict):
            return {}
        if "openalex" in metadata or "semantic scholar" in metadata or "websearch" in metadata:
            return {
                self._source_name(source): paper
                for source, paper in metadata.items()
                if self._source_name(source) != "websearch" and isinstance(paper, dict)
            }
        source = self._source_name(info.get("source", ""))
        if source == "websearch":
            return {}
        return {source or self._infer_source(metadata): metadata}

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

    def _paper_doi(self, paper: dict[str, Any]) -> str:
        candidates = [paper.get("doi")]
        raw_ids = paper.get("ids")
        if isinstance(raw_ids, dict):
            candidates.append(raw_ids.get("doi"))
        external_ids = paper.get("external_ids") or paper.get("externalIds") or {}
        for key, value in external_ids.items():
            if str(key).lower() == "doi":
                candidates.append(value)
        for value in candidates:
            if not value:
                continue
            doi = str(value).strip().lower()
            doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
            doi = re.sub(r"^doi:", "", doi)
            if doi:
                return doi
        return ""

    def _same_source_match(self, cited: dict[str, Any], candidate: dict[str, Any]) -> bool:
        if self._paper_ids(cited) & self._paper_ids(candidate):
            return True
        cited_doi, candidate_doi = self._paper_doi(cited), self._paper_doi(candidate)
        if cited_doi and candidate_doi and cited_doi == candidate_doi:
            return True
        return valid_check(candidate.get("title", ""), cited.get("title", ""))

    def _cross_source_match(self, cited: dict[str, Any], candidate: dict[str, Any]) -> bool:
        cited_doi, candidate_doi = self._paper_doi(cited), self._paper_doi(candidate)
        if cited_doi and candidate_doi and cited_doi == candidate_doi:
            return True
        return valid_check(candidate.get("title", ""), cited.get("title", ""))

    def _is_cited(self, candidate: dict[str, Any], citations: dict[str, Any]) -> bool:
        candidate_source = self._infer_source(candidate)
        for info in citations.values():
            sources = self._metadata_sources(info)
            if candidate_source in sources and self._same_source_match(sources[candidate_source], candidate):
                return True
            for source, cited in sources.items():
                if source == candidate_source:
                    continue
                if self._cross_source_match(cited, candidate):
                    return True
        return False

    def _cited_id_set(self, citations: dict[str, Any]) -> set[str]:
        ids = set()
        for info in citations.values():
            for paper in self._metadata_sources(info).values():
                ids.update(self._paper_ids(paper))
        return ids

    def _cited_papers(self, citations: dict[str, Any]) -> list[dict[str, Any]]:
        papers = []
        for info in citations.values():
            papers.extend(self._metadata_sources(info).values())
        return papers

    async def _referenced_papers(self, citations: dict[str, Any], reference_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Check consensus papers from reference surveys, without doing another topical search.

        Input contract:
        - reference_data["reference_surveys"] stores downloaded/selected reference surveys.
        - reference_data["reference_papers"] stores papers cited by those surveys.
        - each reference paper should carry survey_cited_by_count from GetReferenceSurveys.

        A paper is a weakness iff:
        - number of reference surveys > 2;
        - it is cited by strictly more than half of reference surveys;
        - it is not already cited by the target survey.
        """
        reference_surveys = reference_data.get("reference_surveys", {}) or {}
        reference_papers = reference_data.get("reference_papers", {}) or {}
        survey_count = len(reference_surveys)
        if survey_count <= 2: return []

        results = []
        for paper_id, paper in reference_papers.items():
            if paper.get("candidate_source") == "reference_survey": continue
            support = int(paper.get("survey_cited_by_count", 0) or 0)
            if support <= survey_count / 2: continue
            if self._is_cited(paper, citations): continue
            results.append(
                {
                    "paper": paper,
                    "reference_survey_support": support,
                    "reference_survey_count": survey_count,
                    "severity": "weakness",
                    "missing_type": "reference_consensus",
                }
            )
        return results

    async def _new_papers(self, query: str, topic: str, citations: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Search recent papers for one self-declared topic.

        This logic is only called after __call__ filters out topics already reported as
        missing by TopicCoverage. New papers are always comments, not weaknesses.
        """
        papers = await self._search_topic_papers(self._prepare_query(query, topic), recent=True)
        cited_ids = self._cited_id_set(citations)
        cited_papers = self._cited_papers(citations)
        topic_sims = self._topic_similarity(topic, papers)
        redundancy = self._redundancy_scores(papers, cited_papers)

        velocities = []
        for paper in papers:
            publication = parse_date(paper.get("publication_date"))
            if publication is None:
                velocities.append(0.0)
                continue
            months = months_between(publication, self.eval_date)
            velocities.append(self._citation_count_by_eval_date(paper) / math.log1p(months))
        velocity_cutoff = self._impact_cutoff(velocities, 0.2)

        results = []
        for paper, topic_sim, redundant_sim, velocity in zip(papers, topic_sims, redundancy, velocities):
            if self._is_cited(paper, citations): continue
            if topic_sim < self.new_topic_threshold: continue
            if velocity < velocity_cutoff: continue
            reference_overlap = self._reference_overlap(paper, cited_ids)
            if reference_overlap < self.new_reference_overlap_threshold: continue
            if redundant_sim >= 0.9: continue
            results.append(
                {
                    "paper": paper,
                    "topic": topic,
                    "topic_similarity": topic_sim,
                    "citation_velocity": velocity,
                    "reference_overlap": reference_overlap,
                    "severity": "comment",
                    "missing_type": "new",
                }
            )
        return results

    def _topic_key(self, topic: Any) -> str:
        return str(topic or "").strip().lower()

    def _self_topic_names(self, topics: dict[str, Any]) -> list[str]:
        """Extract only target-survey self-declared topics from GoldenTopicGenerator output."""
        self_topics = topics.get("self_topics", {}) or {}
        names = []
        for value in (self_topics.get("section_map") or {}).values():
            if value: names.append(str(value).strip())
        for value in self_topics.get("aspect_list") or []:
            if value: names.append(str(value).strip())
        return list(dict.fromkeys(name for name in names if name))

    def _topic_coverage_missing_topics(self, topic_eval: dict[str, Any]) -> set[str]:
        """Collect topics already reported by TopicCoverage so citation checks do not double-report them."""
        missing = set()
        for item in topic_eval.get("topic_evals", {}).get("missing_topics", []) or []:
            if isinstance(item, dict):
                missing.add(self._topic_key(item.get("topic")))
            else:
                missing.add(self._topic_key(item))
        return missing

    async def __call__(
        self,
        query: str,
        citations: dict[str, Any],
        topics: dict[str, Any],
        topic_eval: dict[str, Any],
    ) -> dict[str, Any]:
        """
            topics = GoldenTopics的输出，形式为{
                "reference_data": {
                    "reference_papers": 被reference_surveys引用的文献list，含详细信息及reference_surveys引用计数, 
                    "reference_surveys": reference_surveys list
                },
                "reference_topics": 从reference_surveys中总结出来的topics，格式为[{"topic": "topic名称", "sources": [来源]}],
                "self_topics": {
                    "section_map": 指定了具体章节的scope声明 <dict>, 
                    "aspect_list": 未指定具体章节的scope声明<list>
                },
            }
            topic_eval = TopicCoverage的输出
            若topic没有被covered，则不报告该topic下的引用文献。也就是说只检查文章内部声明/实际覆盖的topic下有哪些文献。
        """
        missing_reference = await self._referenced_papers(citations, topics.get("reference_data", {}) or {})
        missing_topic_keys = self._topic_coverage_missing_topics(topic_eval)
        tasks = [asyncio.create_task(self._new_papers(query, topic, citations))
                 for topic in self._self_topic_names(topics) if self._topic_key(topic) not in missing_topic_keys]
        new_missing = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                if result: new_missing.extend(result)
            except Exception as e:
                print(f"NewMissing {e}")

        return {"source_evals": {"missing_reference_papers": missing_reference, "missing_new_papers": new_missing}}
