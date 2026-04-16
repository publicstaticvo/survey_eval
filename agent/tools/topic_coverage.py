from typing import Dict, Any, List

from .sbert_client import SentenceTransformerClient
from .golden_topics import TopicRecord
from .tool_config import ToolConfig
from .utils import cosine_similarity_matrix

debug = True


def get_titles_from_sections(content: dict):
    titles = []
    for section in content['sections']:
        titles.append(section['title'])
        titles.extend(get_titles_from_sections(section))
    return titles


def normalize_topic_records(golden_topics: List[TopicRecord]):
    topic_records = []
    for topic in golden_topics or []:
        if isinstance(topic, dict):
            name = topic.get("topic_name") or topic.get("label") or topic.get("topic")
            if name:
                topic_records.append(topic)
        elif topic:
            topic_records.append({"topic_name": str(topic), "source": "unknown"})
    return topic_records


class TopicCoverageCritic:
    STRUCTURE_SECTIONS = [
        "introduction",
        "background",
        "preliminar",
        "overview",
        "conclusion",
        "discussion",
        "future work",
        "related work",
        "reference",
        "acknowledgment",
        "appendix",
        "method",
    ]

    def __init__(self, config: ToolConfig):
        self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)
        self.weak_threshold = config.topic_weak_sim_threshold
        self.threshold = config.topic_sim_threshold

    async def __call__(self, golden_subtopics: List[TopicRecord], review_paper: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        topic_records = normalize_topic_records(golden_subtopics)
        topic_names = [topic["topic_name"] for topic in topic_records]
        titles = get_titles_from_sections(review_paper)
        if not topic_names:
            return {"topic_evals": {"topic_coverage": [], "irrelevant_sections": [], "metadata": {}}}
        if not titles:
            return {
                "topic_evals": {
                    "topic_coverage": [{"topic": topic, "status": "missing", "best_match": None, "similarity": 0.0} for topic in topic_names],
                    "irrelevant_sections": [],
                    "metadata": {},
                }
            }
        embeddings = self.sentence_transformer.embed(topic_names + titles)
        num_topics = len(topic_names)
        topic_embeddings = embeddings[:num_topics]
        title_embeddings = embeddings[num_topics:]
        cosine_similarity = cosine_similarity_matrix(topic_embeddings, title_embeddings)

        topic_results = []
        for idx, topic in enumerate(topic_names):
            max_sim = float(cosine_similarity[idx].max().item())
            argmax_title = int(cosine_similarity[idx].argmax().item())
            if max_sim < self.weak_threshold:
                status = "missing"
            elif max_sim > self.threshold:
                status = "covered"
            else:
                status = "weakly covered"
            record = topic_records[idx]
            topic_results.append(
                {
                    "topic": topic,
                    "status": status,
                    "best_match": titles[argmax_title],
                    "similarity": max_sim,
                    "source": record.get("source", "unknown"),
                }
            )

        top_level_titles = [section.get("title", "") for section in review_paper.get("sections", [])]
        irrelevant_sections = []
        for title in top_level_titles:
            if not title:
                continue
            title_embedding = self.sentence_transformer.embed([title])
            similarities = cosine_similarity_matrix(title_embedding, topic_embeddings)[0]
            max_sim = float(similarities.max().item())
            if max_sim < self.weak_threshold and all(keyword not in title.lower() for keyword in self.STRUCTURE_SECTIONS):
                best_topic = topic_names[int(similarities.argmax().item())]
                irrelevant_sections.append({"title": title, "similarity": max_sim, "best_match_topic": best_topic})

        return {
            "topic_evals": {
                "topic_coverage": topic_results,
                "irrelevant_sections": irrelevant_sections,
                "metadata": {
                    "topic_confidence": "low"
                    if any(topic.get("source") == "oracle-derived" for topic in topic_records) and not any(topic.get("source") == "anchor-derived" for topic in topic_records)
                    else "normal"
                },
            }
        }
