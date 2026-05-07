from __future__ import annotations

from datetime import timedelta
from typing import Any

from .topic_utils import citation_id_set, months_between, paper_id_aliases, paper_text, parse_date
from .utility.academic_engine import get_academic_engine
from .utility.sbert_client import SentenceTransformerClient
from .utility.tool_config import ToolConfig


def cosine_similarity_matrix(left, right):
    import numpy as np

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
    def __init__(self, config: ToolConfig):
        self.config = config
        self.eval_date = config.evaluation_date
        self.sbert = SentenceTransformerClient(config.sbert_server_url)
        self.engine = get_academic_engine(config)
        self.new_topic_threshold = config.new_paper_topic_similarity_threshold
        self.new_reference_overlap_threshold = config.new_paper_reference_overlap_threshold

    def _citation_count_by_eval_date(self, paper: dict[str, Any]) -> int:
        counts = paper.get("counts_by_year", []) or []
        if not counts:
            return int(paper.get("cited_by_count", 0) or 0)
        return sum(int(item.get("cited_by_count", 0) or 0) for item in counts if int(item.get("year", 0) or 0) <= self.eval_date.year)

    def _prepare_query(self, query: str, topic: str) -> str:
        query_tokens = query.split()
        query_lower = {token.lower() for token in query_tokens}
        topic_tokens = [token for token in topic.split() if token.lower() not in query_lower]
        merged = " ".join([query, " ".join(topic_tokens).strip()]).strip()
        return merged or topic

    async def _search_topic_papers(self, search_query: str, recent: bool) -> list[dict[str, Any]]:
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
        return results.get("results", []) or []

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
        refs = {str(item).replace("https://openalex.org/", "") for item in paper.get("referenced_works", []) or [] if item}
        if not refs:
            return 0.0
        return len(refs & cited_ids) / len(refs)

    def _impact_cutoff(self, papers: list[dict[str, Any]], values: list[float], top_fraction: float) -> float:
        if not papers or not values:
            return float("inf")
        rank = max(0, int(len(values) * top_fraction) - 1)
        ordered = sorted(values, reverse=True)
        return ordered[min(rank, len(ordered) - 1)]

    def _anchor_support(self, paper: dict[str, Any], anchor_surveys: dict[str, Any]) -> int:
        aliases = paper_id_aliases(paper)
        count = 0
        for survey in anchor_surveys.values():
            refs = set()
            meta = survey.get("paper") or {}
            refs.update(str(item).replace("https://openalex.org/", "") for item in meta.get("referenced_works", []) or [] if item)
            if aliases & refs:
                count += 1
        return count

    async def _old_missing_for_topic(
        self,
        query: str,
        topic: str,
        citations: dict[str, Any],
        anchor_surveys: dict[str, Any],
    ) -> list[dict[str, Any]]:
        papers = await self._search_topic_papers(self._prepare_query(query, topic), recent=False)
        topic_sims = self._topic_similarity(topic, papers)
        cited_ids = citation_id_set(citations)
        cited_papers = [info.get("metadata") or info for info in citations.values() if (info.get("metadata") or info)]
        redundancy = self._redundancy_scores(papers, cited_papers)
        impact_values = [float(self._citation_count_by_eval_date(paper)) for paper in papers]
        impact_cutoff = self._impact_cutoff(papers, impact_values, 0.2)
        results = []
        anchor_count = max(1, len(anchor_surveys))
        for paper, topic_sim, redundant_sim, impact in zip(papers, topic_sims, redundancy, impact_values):
            if topic_sim < self.config.topic_sim_threshold:
                continue
            if impact < impact_cutoff:
                continue
            if paper_id_aliases(paper) & cited_ids:
                continue
            if redundant_sim >= 0.9:
                continue
            support = self._anchor_support(paper, anchor_surveys)
            if support <= 0:
                continue
            results.append(
                {
                    "paper": paper,
                    "topic": topic,
                    "topic_similarity": topic_sim,
                    "impact": impact,
                    "anchor_support": support,
                    "severity": "weakness" if support / anchor_count >= 0.75 else "comment",
                    "missing_type": "old",
                }
            )
        return results

    async def _new_missing_for_topic(self, query: str, topic: str, citations: dict[str, Any]) -> list[dict[str, Any]]:
        papers = await self._search_topic_papers(self._prepare_query(query, topic), recent=True)
        cited_ids = citation_id_set(citations)
        cited_papers = [info.get("metadata") or info for info in citations.values() if (info.get("metadata") or info)]
        topic_sims = self._topic_similarity(topic, papers)
        redundancy = self._redundancy_scores(papers, cited_papers)

        velocities = []
        for paper in papers:
            publication = parse_date(paper.get("publication_date"))
            if publication is None:
                velocities.append(0.0)
                continue
            months = months_between(publication, self.eval_date)
            velocities.append(self._citation_count_by_eval_date(paper) / months)
        velocity_cutoff = self._impact_cutoff(papers, velocities, 0.2)

        results = []
        for paper, topic_sim, redundant_sim, velocity in zip(papers, topic_sims, redundancy, velocities):
            if topic_sim < self.new_topic_threshold:
                continue
            if velocity < velocity_cutoff:
                continue
            if self._reference_overlap(paper, cited_ids) < self.new_reference_overlap_threshold:
                continue
            if redundant_sim >= 0.9:
                continue
            results.append(
                {
                    "paper": paper,
                    "topic": topic,
                    "topic_similarity": topic_sim,
                    "citation_velocity": velocity,
                    "reference_overlap": self._reference_overlap(paper, cited_ids),
                    "severity": "comment",
                    "missing_type": "new",
                }
            )
        return results

    async def __call__(
        self,
        query: str,
        citations: dict[str, Any],
        topic_bundle: dict[str, Any],
        topic_eval: dict[str, Any],
    ) -> dict[str, Any]:
        anchor_surveys = topic_bundle.get("anchor_surveys", {}) or {}
        covered_topics = list(dict.fromkeys(topic_eval.get("topic_evals", {}).get("covered_topics", []) or []))
        old_missing, new_missing = [], []
        for topic in covered_topics:
            old_missing.extend(await self._old_missing_for_topic(query, topic, citations, anchor_surveys))
            new_missing.extend(await self._new_missing_for_topic(query, topic, citations))
        return {"source_evals": {"missing_old_papers": old_missing, "missing_new_papers": new_missing}}
