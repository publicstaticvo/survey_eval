import math
import asyncio
from typing import Any, Dict

import lightgbm
import networkx as nx
import numpy as np

from .build_sources import AnchorSurveySource, DirectSeedGraphSource, RecentSemanticSource
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
    def __init__(self, config: ToolConfig):
        self.openalex = get_openalex_client(config)

    def _preferred_source(self, sources: list[str]) -> str:
        return min(sources, key=lambda source: SOURCE_PRIORITY.get(source, 999)) if sources else ""

    def _source_set(self, paper: dict) -> set[str]:
        sources = set(paper.get("candidate_sources", []) or [])
        if paper.get("candidate_source"):
            sources.add(paper["candidate_source"])
        return sources

    def _merge_exact_duplicate(self, existing: dict, incoming: dict) -> dict:
        merged = dict(existing)
        for key, value in incoming.items():
            if key not in merged or merged[key] in (None, "", [], {}):
                merged[key] = value

        sources = self._source_set(existing) | self._source_set(incoming)
        merged["candidate_sources"] = sorted(sources, key=lambda source: SOURCE_PRIORITY.get(source, 999))
        merged["candidate_source"] = self._preferred_source(merged["candidate_sources"])
        merged["anchor_survey_count"] = max(existing.get("anchor_survey_count", 0) or 0, incoming.get("anchor_survey_count", 0) or 0)
        merged["survey_cited_by_count"] = max(existing.get("survey_cited_by_count", 0) or 0, incoming.get("survey_cited_by_count", 0) or 0)
        merged["neighbor_seed_count"] = max(existing.get("neighbor_seed_count", 0) or 0, incoming.get("neighbor_seed_count", 0) or 0)
        neighbor_ids = set(existing.get("neighbor_seed_ids", []) or []) | set(incoming.get("neighbor_seed_ids", []) or [])
        if neighbor_ids:
            merged["neighbor_seed_ids"] = sorted(neighbor_ids)
        return merged

    def _merge_sources_after_dedup(self, paper: dict, raw_pool: dict[str, dict[str, Any]]):
        ids = paper.get("ids") or ([paper["id"]] if paper.get("id") else [])
        sources = self._source_set(paper)
        anchor_count = paper.get("anchor_survey_count", 0) or 0
        neighbor_count = paper.get("neighbor_seed_count", 0) or 0
        neighbor_ids = set(paper.get("neighbor_seed_ids", []) or [])
        for original_id in ids:
            original = raw_pool.get(original_id)
            if not original:
                continue
            sources.update(self._source_set(original))
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
                raw_pool[paper_id] = (
                    self._merge_exact_duplicate(raw_pool[paper_id], paper)
                    if paper_id in raw_pool
                    else dict(paper)
                )
        deduped = self.openalex.deduplicate_works(list(raw_pool.values()), original_title=query)
        candidate_pool = {}
        for paper in deduped:
            paper = self._merge_sources_after_dedup(paper, raw_pool)
            if paper.get("id"):
                candidate_pool[paper["id"]] = paper
        return {"candidate_pool": candidate_pool}


class LetorCandidateScorer:
    def __init__(self, config: ToolConfig):
        self.letor = config.letor_path
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

    def _build_alias_to_canonical_id(self, candidate_pool: Dict[str, Any]) -> dict[str, str]:
        alias_to_canonical = {}
        for canonical_id, paper in candidate_pool.items():
            aliases = set(paper.get("ids") or [])
            if paper.get("id"):
                aliases.add(paper["id"])
            aliases.add(canonical_id)
            for alias in aliases:
                if alias and alias not in alias_to_canonical:
                    alias_to_canonical[alias] = canonical_id
        return alias_to_canonical

    def _local_citation_and_pagerank(self, candidate_pool: Dict[str, Any]):
        alias_to_canonical = self._build_alias_to_canonical_id(candidate_pool)
        graph = nx.DiGraph()
        graph.add_nodes_from(candidate_pool)
        local_citation_count = {paper_id: 0 for paper_id in candidate_pool}
        for source_id, paper in candidate_pool.items():
            canonical_references = set()
            for target_id in paper.get("referenced_works", []):
                canonical_target_id = alias_to_canonical.get(target_id)
                if canonical_target_id and canonical_target_id != source_id:
                    canonical_references.add(canonical_target_id)
            paper["canonical_referenced_works"] = sorted(canonical_references)
            for target_id in canonical_references:
                graph.add_edge(source_id, target_id)
                local_citation_count[target_id] += 1
        pagerank = nx.pagerank(graph, alpha=0.85) if graph.number_of_nodes() else {}
        return local_citation_count, pagerank

    def __call__(self, query: str, candidate_pool: Dict[str, Any], calc_rank: bool = True) -> dict[str, Any]:
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

        if calc_rank:
            if isinstance(self.letor, str): self.letor = lightgbm.Booster(model_file=self.letor)
            ranks = self.letor.predict(np.asarray(features)).tolist()
            for paper_id, rank in zip(paper_ids, ranks):
                scored_pool[paper_id]["rank"] = rank
            return scored_pool
        
        for paper_id, feature in zip(paper_ids, features):
            scored_pool[paper_id]['feature'] = feature
        return scored_pool


class DynamicCandidatePool:
    def __init__(self, config: ToolConfig):
        self.anchor_source = AnchorSurveySource(config)
        self.direct_seed_source = DirectSeedGraphSource(config)
        self.recent_semantic_source = RecentSemanticSource(config)
        self.deduplicator = LiteratureCandidateDeduplicate(config)
        self.scorer = LetorCandidateScorer(config)

    async def __call__(
        self,
        query: str,
        download_anchor_surveys: bool = True,
        calc_rank: bool = True,
    ) -> dict:
        # anchor_data = await self.anchor_source(query, download=download_anchor_surveys)
        # seed_library, neighbor_papers = await self.direct_seed_source(query)
        # recent_data = await self.recent_semantic_source(query)
        anchor_data, (seed_library, neighbor_papers), recent_data = await asyncio.gather(
            self.anchor_source(query, download=download_anchor_surveys),
            self.direct_seed_source(query),
            self.recent_semantic_source(query)
        )
        source_1_anchor = anchor_data.get("anchor_papers", {}) or {}
        source_2_seed_graph = {**neighbor_papers, **seed_library}
        source_3_new = recent_data.get("new_papers", {}) or {}

        candidate_data = await self.deduplicator(
            query,
            source_1_anchor=source_1_anchor,
            source_2_seed=source_2_seed_graph,
            source_3_new=source_3_new,
        )
        oracle_papers = self.scorer(query, candidate_data["candidate_pool"], calc_rank=calc_rank)
        return {"candidate_pool": oracle_papers, "anchor_surveys": anchor_data['anchor_surveys']}
