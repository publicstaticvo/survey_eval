import math
import numpy as np
from datetime import datetime

from .sbert_client import SentenceTransformerClient
from .tool_config import ToolConfig


def normalize(a: np.ndarray) -> np.ndarray:
    norm = np.sum(a * a)
    if norm == 0: return a
    return a / np.sqrt(norm)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    b = normalize(b)
    return float(np.sum(a * b))


class MissingPaperCheck:

    def __init__(self, config: ToolConfig):
        self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)
        self.eval_date = config.evaluation_date
        self.topn = config.topn

    def __call__(self, citations: dict, oracle_data: dict, anchor_papers: dict):
        """
        The format of param citations is:
        {
            "metadata": {"id": ., "year": ., "citation_count": .,}
            "title": "title",
            "abstract": "abstract", 
            "full_content": "full_content" or "abstract" or "title",
            "status": 0-3
        }
        status == 0 -> OK
        status == 1 -> fail to download paper
        status == 2 -> fail to download paper and fetch abstract
        status == 3 -> fail to get information of the citation. Please check its existance.
        The key of oracle_data['oracle_papers'] has changed back to work_id.
        """
        # check what are the missing citations.
        oracle_data = oracle_data['oracle_papers']
        cited_id_set = set(x['metadata'] for x in citations)
        missing_anchors_set = set()
        missing_anchors, missing_oracles = [], []
        for x in anchor_papers.values():
            if x['id'] not in cited_id_set:
                missing_anchors.append(x)
                missing_anchors_set.add(x['id'])
        for x in oracle_data.values():
            if x['id'] not in cited_id_set and x['id'] not in missing_anchors_set:  # add threshold
                missing_oracles.append(x)
        missing_oracles.sort(key=lambda x: x['rank'], reverse=True)
        return {"source_evals": {
            "missing_oracles": missing_oracles[:3],
            "missing_anchors": missing_anchors,
        }}
