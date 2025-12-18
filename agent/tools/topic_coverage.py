from typing import Dict, Any
from sentence_transformers import util

from .sbert_client import SentenceTransformerClient
from .utils import split_content_to_paragraph
from .tool_config import ToolConfig


class TopicCoverageCritic:

    def __init__(self, config: ToolConfig):
        self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)
        self.threshold = config.topic_similarity_threshold

    def __call__(self, golden_subtopics, review_paper) -> Dict[str, Dict[str, Any]]:
        # get vectors of subtopics and paper paragraphs
        paragraphs = split_content_to_paragraph(review_paper)
        embeddings = self.sentence_transformer.embed(golden_subtopics + paragraphs)
        # calculate similarity
        num_topics = len(golden_subtopics)
        topic_embeddings = embeddings[:num_topics]
        paragraph_embeddings = embeddings[num_topics:]
        cosine_similarity = util.cos_sim(topic_embeddings, paragraph_embeddings)
        # coverage or max
        max_similarity = cosine_similarity.max(1).tolist()
        results = {s: ms for s, ms in zip(golden_subtopics, max_similarity)}
        if self.threshold is None:   
            return {"topic_evals": {"similarity_results": results}} 
        missing_topics = [s for s, ms in zip(golden_subtopics, max_similarity) if ms < self.threshold]   
        return {"topic_evals": {"similarity_results": results, "missing_topics": missing_topics}}
