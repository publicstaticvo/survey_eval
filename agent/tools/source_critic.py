from datetime import datetime
from typing import Any
from sentence_transformers.util import cos_sim

from .sbert_client import SentenceTransformerClient
from .tool_config import ToolConfig


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
            oracle_topic_sim = cos_sim(oracle_embeddings, topic_embeddings)
            oracle_topic_sim_max = oracle_topic_sim.max(dim=1).tolist()
            oracle_topic_sim_argmax = oracle_topic_sim.argmax(dim=1).tolist()
            oracle_cited_sim = cos_sim(oracle_embeddings, cited_embeddings)
            for i, p in enumerate(oracle_papers):
                m = oracle_topic_sim_max[i]
                am = oracle_topic_sim_argmax[i]
                if (cm := oracle_cited_sim[i]).max() >= 0.9:
                    continue
                if m >= 0.55:
                    p['topic'] = {"name": topics[am], "topic_similarity": m}
                    p['most_similar_paper'] = citations[paper_keys[cm.argmax().item()]]['title']
                    p['paper_similarity'] = cm.max().item()
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
