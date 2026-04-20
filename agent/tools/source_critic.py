from datetime import datetime
from typing import Any

from .sbert_client import SentenceTransformerClient
from .tool_config import ToolConfig
from .utils import cosine_similarity_matrix


class MissingPaperCheck:
    def __init__(self, config: ToolConfig):
        self.eval_date = config.evaluation_date
        self.sbert = SentenceTransformerClient(config.sbert_server_url)

    def _filter_oracles(self, citations: dict[str, Any], oracle_papers: list[dict[str, Any]], topics: list[str]):
        valid_oracles = []
        for paper in oracle_papers:
            publication_date = paper.get("publication_date")
            # if not publication_date: continue
            if (self.eval_date - datetime.strptime(publication_date, "%Y-%m-%d")).days <= 90: continue
            valid_oracles.append(paper)
        print(f"{len(valid_oracles)} missing oracles")
        if not valid_oracles:
            return []
        max_rank = max(paper.get("rank", 0.0) for paper in valid_oracles)
        valid_oracles = [paper for paper in valid_oracles if max_rank - paper.get("rank", 0.0) <= 0.15]
        valid_oracles.sort(key=lambda x: x.get("rank", 0.0), reverse=True)
        print(f"{len(oracle_papers)} missing oracles, max rank {[x['rank'] for x in valid_oracles[:10]]}")
        if not topics or not citations:
            return valid_oracles

        topic_embeddings = self.sbert.embed(topics)
        cited_items = list(citations.items())
        cited_texts = [f"{info.get('title', '')}. {info.get('abstract', '')}".strip() for _, info in cited_items]
        cited_embeddings = self.sbert.embed(cited_texts)
        oracle_texts = [f"{paper.get('title', '')}. {paper.get('abstract', '')}".strip() for paper in valid_oracles]
        oracle_embeddings = self.sbert.embed(oracle_texts)

        filtered = []
        oracle_topic_sim = cosine_similarity_matrix(oracle_embeddings, topic_embeddings)
        oracle_cited_sim = cosine_similarity_matrix(oracle_embeddings, cited_embeddings)
        for idx, paper in enumerate(valid_oracles):
            topic_idx = int(oracle_topic_sim[idx].argmax())
            topic_sim = float(oracle_topic_sim[idx].max())
            cited_idx = int(oracle_cited_sim[idx].argmax())
            cited_sim = float(oracle_cited_sim[idx].max())
            if cited_sim >= 0.9 or topic_sim < 0.55: continue
            key, similar_info = cited_items[cited_idx]
            paper["topic"] = {"name": topics[topic_idx], "topic_similarity": topic_sim}
            paper["most_similar_paper"] = similar_info.get("title", key)
            paper["paper_similarity"] = cited_sim
            filtered.append(paper)
        print(f"{len(filtered)} missing oracles")
        return filtered

    def __call__(self, citations: dict, oracle_data: dict, anchor_papers: dict, topics: list):
        print(f"{len(anchor_papers)} anchor papers")
        cited_id_set = {info.get("metadata", {}).get("id") for info in citations.values() if info.get("metadata")}
        missing_anchors = [paper for paper in anchor_papers.values() if paper.get("id") not in cited_id_set]
        missing_anchor_ids = {paper.get("id") for paper in missing_anchors}
        missing_oracles = [v for v in oracle_data.values() if v['id'] not in cited_id_set.union(missing_anchor_ids)]
        missing_oracles = self._filter_oracles(citations, missing_oracles, topics)
        missing_oracles.sort(key=lambda paper: paper.get("rank", 0.0), reverse=True)
        return {"source_evals": {"missing_oracles": missing_oracles, "missing_anchors": missing_anchors}}
