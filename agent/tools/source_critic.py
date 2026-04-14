from datetime import datetime
from typing import Any

from .sbert_client import SentenceTransformerClient
from .tool_config import ToolConfig
from .utils import cosine_similarity_matrix


def normalize_topic_names(topics):
    normalized = []
    for topic in topics or []:
        if isinstance(topic, dict):
            name = topic.get("topic_name") or topic.get("label") or topic.get("topic")
        else:
            name = topic
        if name:
            normalized.append(str(name))
    return normalized


class MissingPaperCheck:

    def __init__(self, config: ToolConfig):
        self.eval_date = config.evaluation_date  
        self.sbert = SentenceTransformerClient(config.sbert_server_url) 

    def _filter_oracles(self, citations: dict[str, Any], oracle_papers: list[dict[str, Any]], topics: list):
        # 1. 时间过滤：根据ACL投稿规定，90天以内可以不计未引用。
        oracle_papers = [x for x in oracle_papers if x['publication_date'] and (self.eval_date - datetime.strptime(x['publication_date'], "%Y-%m-%d")).days > 90]
        # 2. rank过滤: 只取前10%。
        # oracle_papers.sort(key=lambda x: x['rank'], reverse=True)
        max_rank = max(x['rank'] for x in oracle_papers)
        oracle_papers = [x for x in oracle_papers if max_rank - x['rank'] <= 0.15]
        oracle_papers.sort(key=lambda x: x['rank'], reverse=True)
        # 3. topic一致性检查
        if oracle_papers:
            # 三部分计算相似性：topic、已引用论文、oracle论文。
            topic_embeddings = self.sbert.embed(topics)
            sentences, paper_keys = [], []
            for k, v in citations.items():
                title = v['title']
                abstract = v['abstract'] if v['abstract'] else ""
                sentences.append(f"{title}. {abstract}".strip())
                paper_keys.append(k)
            cited_embeddings = self.sbert.embed(sentences)
            sentences = [f"{v['title']}. {v['abstract'] if v['abstract'] else ''}".strip() for v in oracle_papers]
            oracle_embeddings = self.sbert.embed(sentences)
            # 检查oracle和topic对齐; 与已引用paper的overlap判定
            filtered_oracles = []
            oracle_topic_sim = cosine_similarity_matrix(oracle_embeddings, topic_embeddings)
            oracle_topic_sim_max = oracle_topic_sim.max(axis=1).tolist()
            oracle_topic_sim_argmax = oracle_topic_sim.argmax(axis=1).tolist()
            oracle_cited_sim = cosine_similarity_matrix(oracle_embeddings, cited_embeddings)
            for i, p in enumerate(oracle_papers):
                m = oracle_topic_sim_max[i]
                am = oracle_topic_sim_argmax[i]
                if (cm := oracle_cited_sim[i]).max() >= 0.9:
                    continue
                if m >= 0.55:
                    p['topic'] = {"name": topics[am], "topic_similarity": m}
                    p['most_similar_paper'] = citations[paper_keys[int(cm.argmax())]]['title']
                    p['paper_similarity'] = float(cm.max())
                    filtered_oracles.append(p)
            oracle_papers = filtered_oracles
        return oracle_papers

    def __call__(self, citations: dict, oracle_data: dict, anchor_papers: dict, topics: list):
        # check what are the missing citations.
        oracle_data = oracle_data['oracle_papers']
        cited_id_set = set(x['metadata']['id'] for x in citations)
        # _identify_missing_anchors
        missing_anchors_set = set()
        missing_anchors, missing_oracles = [], []
        for x in anchor_papers.values():
            if x['id'] not in cited_id_set:
                missing_anchors.append(x)
                missing_anchors_set.add(x['id'])
        # filter_oracles
        for x in oracle_data.values():
            if x['id'] not in cited_id_set and x['id'] not in missing_anchors_set:  # add threshold
                missing_oracles.append(x)
        # identify_missing_oracles
        missing_oracles = self._filter_oracles(citations, missing_oracles, topics)
        missing_oracles.sort(key=lambda x: x['rank'], reverse=True)
        return {"source_evals": {
            "missing_oracles": missing_oracles,
            "missing_anchors": missing_anchors,
        }}


class MissingPaperCheck:
    def __init__(self, config: ToolConfig):
        self.eval_date = config.evaluation_date
        self.sbert = SentenceTransformerClient(config.sbert_server_url)

    def _filter_oracles(self, citations: dict[str, Any], oracle_papers: list[dict[str, Any]], topics: list[str]):
        topics = normalize_topic_names(topics)
        valid_oracles = []
        for paper in oracle_papers:
            publication_date = paper.get("publication_date")
            if not publication_date:
                continue
            if (self.eval_date - datetime.strptime(publication_date, "%Y-%m-%d")).days <= 90:
                continue
            valid_oracles.append(paper)
        if not valid_oracles:
            return []
        max_rank = max(paper.get("rank", 0.0) for paper in valid_oracles)
        valid_oracles = [paper for paper in valid_oracles if max_rank - paper.get("rank", 0.0) <= 0.15]
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
            if cited_sim >= 0.9 or topic_sim < 0.55:
                continue
            key, similar_info = cited_items[cited_idx]
            paper["topic"] = {"name": topics[topic_idx], "topic_similarity": topic_sim}
            paper["most_similar_paper"] = similar_info.get("title", key)
            paper["paper_similarity"] = cited_sim
            filtered.append(paper)
        return filtered

    def __call__(self, citations: dict, oracle_data: dict, anchor_papers: dict, topics: list):
        topics = normalize_topic_names(topics)
        oracle_papers = list((oracle_data or {}).get("oracle_papers", {}).values())
        cited_id_set = {info.get("metadata", {}).get("id") for info in citations.values() if info.get("metadata")}
        missing_anchors = [paper for paper in anchor_papers.values() if paper.get("id") not in cited_id_set]
        missing_anchor_ids = {paper.get("id") for paper in missing_anchors}
        missing_oracles = [paper for paper in oracle_papers if paper.get("id") not in cited_id_set and paper.get("id") not in missing_anchor_ids]
        missing_oracles = self._filter_oracles(citations, missing_oracles, topics)
        missing_oracles.sort(key=lambda paper: paper.get("rank", 0.0), reverse=True)
        return {"source_evals": {"missing_oracles": missing_oracles, "missing_anchors": missing_anchors}}
