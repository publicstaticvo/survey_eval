import math
import tqdm
import logging
import asyncio
import lightgbm
import numpy as np
import networkx as nx
from datetime import datetime
from typing import List, Dict, Any

from .sbert_client import SentenceTransformerClient
from .openalex import get_openalex_client
from .tool_config import ToolConfig
from .utils import cosine_similarity_matrix

debug = False


class DynamicOracleGenerator:
    def __init__(self, config: ToolConfig):
        self.letor = lightgbm.Booster(model_file=config.letor_path)
        self.eval_date = config.evaluation_date
        self.num_oracle_papers = config.num_oracle_papers
        self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)
        self.openalex = get_openalex_client(config)

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

    async def _fetch_neighbors(self, paper_id: str):
        neighbors = {}
        filters = [
            {"cites": paper_id, "to_publication_date": self.eval_date.strftime("%Y-%m-%d")},
            {"cited_by": paper_id, "to_publication_date": self.eval_date.strftime("%Y-%m-%d")},
        ]
        for filter_kwargs in filters:
            try:
                results = await self.openalex.search_works("works", filter=filter_kwargs, per_page=200)
            except Exception:
                continue
            for paper in results.get("results", []):
                if paper.get("id"):
                    neighbors[paper["id"]] = paper
        return neighbors, paper_id

    async def _expand_library(self, query: str, library: Dict[str, Any]):
        expanded = {**library}
        if len(expanded) >= self.num_oracle_papers:
            print(f"{len(expanded)} oracle papers")
            return expanded

        neighbor_sources, neighbor_papers, co_neighbors, long_tail = {}, {}, {}, {}
        tasks = [asyncio.create_task(self._fetch_neighbors(paper_id)) for paper_id in expanded]
        for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            neighbors, paper_id = await task
            for neighbor_id, paper in neighbors.items():
                if neighbor_id in expanded: continue
                neighbor_papers[neighbor_id] = paper
                neighbor_sources.setdefault(neighbor_id, set()).add(paper_id)

        for paper_id, paper in neighbor_papers.items():
            sources = neighbor_sources.get(paper_id, set())
            paper["neighbors_count"] = len(sources)
            # paper["neighbor_seed_ids"] = sorted(sources)
            if len(sources) >= 2: co_neighbors[paper_id] = paper
            elif len(sources) == 1: long_tail[paper_id] = paper

        ranked_co_neighbors = sorted(
            co_neighbors.items(),
            key=lambda item: (
                item[1]["neighbors_count"] * 100 + math.log1p(max(0, self._citation_count_by_eval_date(item[1])))
            ),
            reverse=True,
        )
        for paper_id, paper in ranked_co_neighbors:
            expanded[paper_id] = paper
            if len(expanded) >= self.num_oracle_papers:
                print(f"{len(expanded)} oracle papers")
                return expanded

        if long_tail and len(expanded) < self.num_oracle_papers:
            similarity = self._score_similarity(query, long_tail)
            ranked_long_tail = sorted(
                long_tail.items(),
                key=lambda item: (
                    similarity.get(item[0], 0.0) * 10 + math.log1p(max(0, self._citation_count_by_eval_date(item[1])))
                ),
                reverse=True,
            )
            for paper_id, paper in ranked_long_tail:
                paper["query_similarity"] = similarity.get(paper_id, 0.0)
                expanded[paper_id] = paper
                if len(expanded) >= self.num_oracle_papers:
                    break
        print(f"{len(expanded)} oracle papers")
        return expanded

    def _score_similarity(self, query: str, papers: Dict[str, Any]):
        sentences = []
        paper_ids = list(papers)
        for paper_id in paper_ids:
            paper = papers[paper_id]
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            sentences.append(f"{title}. {abstract}".strip())
        embeddings = self.sentence_transformer.embed([query, *sentences])
        scores = cosine_similarity_matrix(embeddings[:1], embeddings[1:])[0].tolist()
        return {paper_id: score for paper_id, score in zip(paper_ids, scores)}

    def _local_citation_and_pagerank(self, oracle: Dict[str, Any]):
        graph = nx.DiGraph()
        graph.add_nodes_from(oracle)
        local_citation_count = {paper_id: 0 for paper_id in oracle}
        for source_id, paper in oracle.items():
            for target_id in paper.get("referenced_works", []):
                if target_id in oracle:
                    graph.add_edge(source_id, target_id)
                    local_citation_count[target_id] += 1
        pagerank = nx.pagerank(graph, alpha=0.85) if graph.number_of_nodes() else {}
        return local_citation_count, pagerank

    def _predict_ranks(self, oracle: Dict[str, Any]):
        paper_ids = list(oracle)
        features = np.array([oracle[paper_id]["features"] for paper_id in paper_ids])
        ranks = self.letor.predict(features).tolist()
        for paper_id, rank in zip(paper_ids, ranks):
            oracle[paper_id]["rank"] = rank

    async def __call__(self, query: str, library: Dict[str, Any] | None = None) -> dict:
        library = library or {}
        expanded = await self._expand_library(query, library)
        similarity = self._score_similarity(query, expanded)
        scored = []
        for paper_id, paper in expanded.items():
            sim = similarity.get(paper_id, 0.0)
            prestige = math.log1p(max(0, self._citation_count_by_eval_date(paper)))
            age = self._paper_age(paper)
            scored.append((paper_id, sim + 0.05 * prestige + 0.02 / age))
        scored.sort(key=lambda item: item[1], reverse=True)
        oracle = {paper_id: expanded[paper_id] for paper_id, _ in scored[: self.num_oracle_papers]}

        local_citation_count, pagerank = self._local_citation_and_pagerank(oracle)
        max_citation = max((self._citation_count_by_eval_date(paper) for paper in oracle.values()), default=1)
        max_local_citation = max(local_citation_count.values(), default=1)
        for paper_id, paper in oracle.items():
            citation_count = self._citation_count_by_eval_date(paper)
            # calculate features
            # feature 0: cosine similarity (-1~1)
            # feature 1: citation count == global prestige (regularized to 0~1)
            # feature 2: local citation count == local_prestige (regularized to 0~1)
            # feature 3: local pagerank (regularized to 0~1)
            # feature 4: citation velocity == emengence
            f1 = math.log1p(citation_count) / math.log1p(max_citation or 1)
            paper["features"] = [
                similarity.get(paper_id, 0.0), f1,
                math.log1p(local_citation_count.get(paper_id, 0)) / math.log1p(max_local_citation or 1),
                pagerank.get(paper_id, 0.0) * 100,
                f1 / math.log1p(self._paper_age(paper) + 1),
            ]
        self._predict_ranks(oracle)
        return {"oracle_papers": oracle}
