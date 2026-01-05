from typing import Dict, Any
from sentence_transformers import util

from .sbert_client import SentenceTransformerClient
from .tool_config import ToolConfig


def get_titles_from_sections(content: dict):
    titles = []
    for section in content['sections']:
        titles.append(section['title'])
        titles.extend(get_titles_from_sections(section))
    return titles


class TopicCoverageCritic:

    def __init__(self, config: ToolConfig):
        self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)
        self.weak_threshold = config.topic_weak_sim_threshold
        self.threshold = config.topic_sim_threshold

    async def __call__(self, golden_subtopics, review_paper) -> Dict[str, Dict[str, Any]]:
        # get vectors of subtopics and paper paragraphs
        titles = get_titles_from_sections(review_paper)
        if not titles:
            return {"topic_evals": [
                {"topic": s, "status": "missing", "best_match": None, "similarity": 0.0}
                for s in golden_subtopics
            ]}
        embeddings = self.sentence_transformer.embed(golden_subtopics + titles)
        # calculate similarity
        num_topics = len(golden_subtopics)
        topic_embeddings = embeddings[:num_topics]
        title_embeddings = embeddings[num_topics:]
        cosine_similarity = util.cos_sim(topic_embeddings, title_embeddings)
        # coverage or max
        results = []
        for i, s in enumerate(golden_subtopics):
            max_sim = cosine_similarity[i].max().item()
            argmax_title = cosine_similarity[i].argmax().item()
            if max_sim < self.weak_threshold: status = "missing"
            elif max_sim > self.threshold: status = "covered"
            else: status = "weakly covered"
            results.append({"topic": s, "status": status, "best_match": titles[argmax_title], "similarity": max_sim})
        return {"topic_evals": results}
