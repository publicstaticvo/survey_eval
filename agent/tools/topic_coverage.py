import json
import numpy as np
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from sentence_transformers import util
from langchain_core.tools import BaseTool

from sbert_client import SentenceTransformerClient
from utils import split_content_to_paragraph


class TopicCriticInput(BaseModel):
    review_text: str = Field(..., description="The full text of the review.")
    oracle_data: Dict[str, Any] = Field(..., description="Output from DynamicOracleGenerator.")


class TopicCoverageCritic(BaseTool):
    name = "topic_coverage_critic"
    description = (
        "Evaluates whether the review covers the essential subtopics identified by the Oracle. "
        "Uses embedding similarity to match review paragraphs to required topics."
    )
    args_schema: type[BaseModel] = TopicCriticInput

    def __init__(
            self, 
            sentence_transformer: SentenceTransformerClient, 
            threshold: float | None = None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.sentence_transformer = sentence_transformer
        self.threshold = threshold

    def _run(self, paper_content: dict[str, str], oracle_data: Dict[str, Any]):
        # get vectors of subtopics and paper paragraphs
        golden_subtopics = oracle_data['subtopics']
        paragraphs = split_content_to_paragraph(paper_content)
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
        return {"similarity_results": results, "missing_topics": missing_topics}
    
    async def _arun(self, paper_content: dict[str, str], dynamic_oracle_data: str) -> str:
        return await self._run(paper_content, dynamic_oracle_data)
