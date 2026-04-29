import math
from typing import Any, Dict

import lightgbm
import networkx as nx
import numpy as np

from ..utility.openalex import get_openalex_client
from ..utility.sbert_client import SentenceTransformerClient
from ..utility.tool_config import ToolConfig
from .utils import cosine_similarity_matrix


SOURCE_PRIORITY = {
    "anchor_survey": 0,
    "high_consensus": 1,
    "single_anchor": 2,
    "seed": 3,
    "neighbor": 4,
    "new_papers": 5,
}


class LiteratureCandidateDeduplicate:

    def _preferred_source(self, sources: list[str]) -> str:
        return min(sources, key=lambda source: SOURCE_PRIORITY.get(source, 999)) if sources else ""

    def _merge_sources_after_dedup(self, paper: dict, raw_pool: dict[str, dict[str, Any]]):
        ids = paper.get("ids") or ([paper["id"]] if paper.get("id") else [])
        sources = set(paper.get("candidate_sources", []) or [])
        anchor_count = paper.get("anchor_survey_count", 0) or 0
        neighbor_count = paper.get("neighbor_seed_count", 0) or 0
        neighbor_ids = set(paper.get("neighbor_seed_ids", []) or [])
        for original_id in ids:
            original = raw_pool.get(original_id)
            if not original:
                continue
            sources.update(original.get("candidate_sources", []) or [])
            if original.get("anchor_survey_count", 0):
                anchor_count = max(anchor_count, original.get("anchor_survey_count", 0))
            if original.get("neighbor_seed_count", 0):
                neighbor_count = max(neighbor_count, original.get("neighbor_seed_count", 0))
            neighbor_ids.update(original.get("neighbor_seed_ids", []) or [])
        paper["candidate_sources"] = sorted(sources, key=lambda source: SOURCE_PRIORITY.get(source, 999))
        paper["candidate_source"] = self._preferred_source(paper["candidate_sources"])
        if anchor_count:
            paper["anchor_survey_count"] = anchor_count
        if neighbor_count:
            paper["neighbor_seed_count"] = neighbor_count
            paper["neighbor_seed_ids"] = sorted(neighbor_ids)
        return paper

    async def __call__(
        self,
        query: str,
        source_1_anchor: dict[str, Any],
        source_2_seed: dict[str, Any],
        source_3_new: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        raw_pool = {}
        for source in [source_1_anchor, source_2_seed, source_3_new]:
            for paper_id, paper in source.items():
                raw_pool[paper_id] = paper
        deduped = self.openalex.deduplicate_works(list(raw_pool.values()), original_title=query)
        candidate_pool = {}
        for paper in deduped:
            paper = self._merge_sources_after_dedup(paper, raw_pool)
            if paper.get("id"):
                candidate_pool[paper["id"]] = paper
        return {"candidate_pool": candidate_pool}


class LetorCitationScorer:
    def __init__(self, config: ToolConfig):
        self.letor = lightgbm.Booster(model_file=config.letor_path)
        self.eval_date = config.evaluation_date
        self.num_oracle_papers = config.num_oracle_papers
        self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)

    def _citation_count_by_eval_date(self, paper: dict):
        eval_year = int(self.eval_date.year)
        citation_count = paper.get("cited_by_count", 0) or 0
        if citation_count:
            for item in paper.get("counts_by_year", []):
                if item["year"] > eval_year:
                    citation_count -= item["cited_by_count"]
            return citation_count
        return sum(item["cited_by_count"] for item in paper.get("counts_by_year", []) if item["year"] <= eval_year)

    def _paper_age(self, paper):
        publication_date = paper.get("publication_date") or self.eval_date.strftime("%Y-%m-%d")
        year = int(self.eval_date.strftime("%Y")) - int(publication_date[:4])
        return max(1, year + 1)

    def _score_similarity(self, query: str, papers: Dict[str, Any]):
        paper_ids = list(papers)
        sentences = [
            f"{papers[paper_id].get('title', '')}. {papers[paper_id].get('abstract', '')}".strip()
            for paper_id in paper_ids
        ]
        if not sentences:
            return {}
        embeddings = self.sentence_transformer.embed([query, *sentences])
        scores = cosine_similarity_matrix(embeddings[:1], embeddings[1:])[0].tolist()
        return {paper_id: score for paper_id, score in zip(paper_ids, scores)}

    def _local_citation_and_pagerank(self, candidate_pool: Dict[str, Any]):
        graph = nx.DiGraph()
        graph.add_nodes_from(candidate_pool)
        local_citation_count = {paper_id: 0 for paper_id in candidate_pool}
        for source_id, paper in candidate_pool.items():
            for target_id in paper.get("referenced_works", []):
                if target_id in candidate_pool:
                    graph.add_edge(source_id, target_id)
                    local_citation_count[target_id] += 1
        pagerank = nx.pagerank(graph, alpha=0.85) if graph.number_of_nodes() else {}
        return local_citation_count, pagerank

    def __call__(self, query: str, candidate_pool: Dict[str, Any]) -> dict[str, Any]:
        if not candidate_pool: return {}
        scored_pool = {paper_id: dict(paper) for paper_id, paper in candidate_pool.items()}
        similarity = self._score_similarity(query, scored_pool)
        local_citation_count, pagerank = self._local_citation_and_pagerank(scored_pool)
        max_citation = max((self._citation_count_by_eval_date(paper) for paper in scored_pool.values()), default=1)
        max_local_citation = max(local_citation_count.values(), default=1)

        paper_ids = list(scored_pool)
        features = []
        for paper_id in paper_ids:
            paper = scored_pool[paper_id]
            citation_count = self._citation_count_by_eval_date(paper)
            global_prestige = math.log1p(citation_count) / math.log1p(max_citation or 1)
            local_prestige = math.log1p(local_citation_count.get(paper_id, 0)) / math.log1p(max_local_citation or 1)
            emergence = global_prestige / math.log1p(self._paper_age(paper) + 1)
            paper["features"] = [
                similarity.get(paper_id, 0.0),
                global_prestige,
                local_prestige,
                pagerank.get(paper_id, 0.0) * (self.num_oracle_papers / 10),
                emergence,
            ]
            features.append(paper["features"])

        ranks = self.letor.predict(np.asarray(features)).tolist()
        for paper_id, rank in zip(paper_ids, ranks):
            scored_pool[paper_id]["rank"] = rank
        return scored_pool


class DynamicCandidatePool:
    def __init__(self, config: ToolConfig):
        self.pool_builder = LiteratureCandidateDeduplicate(config)
        self.scorer = LetorCitationScorer(config)

    async def __call__(
        self,
        query: str,
        anchor_data: dict[str, Any] | None = None,
    ) -> dict:
        candidate_data = await self.pool_builder(query, anchor_data=anchor_data)
        oracle_papers = self.scorer(query, candidate_data["candidate_pool"])
        return {
            "candidate_pool": candidate_data["candidate_pool"],
            "oracle_papers": oracle_papers,
            "queries": candidate_data.get("queries", []),
            "library": candidate_data.get("library", {}),
            "new_papers": candidate_data.get("new_papers", {}),
        }
