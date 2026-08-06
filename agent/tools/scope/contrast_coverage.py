from __future__ import annotations

from typing import Any

from ..utility.paper_elements import Paper
from .adequacy_common import content_topics, iter_sentences, sentence_entities, top_comments


class ContrastCoverageAdequacy:
    def __init__(self, *_args, top_k: int = 3, **_kwargs):
        self.top_k = top_k

    async def __call__(self, paper: Paper, *_args, **_kwargs) -> dict[str, Any]:
        topics = set(content_topics(paper))
        contrast_sentences = [s for s in iter_sentences(paper) if s.get("label") in {"COMPARISON", "EVALUATION"}]
        covered_topics = set()
        covered_entities = set()
        for sent in contrast_sentences:
            lowered = sent["text"].lower()
            covered_topics.update(topic for topic in topics if topic.lower() in lowered)
            covered_entities.update(sentence_entities(sent))
        topic_coverage = len(covered_topics) / len(topics) if topics else 1.0
        comments = []
        if topics and topic_coverage < 1.0:
            comments.append({
                "issue_type": "contrast_coverage",
                "issue": "Some extracted survey topics do not appear in contrast or evaluation sentences.",
                "missing_topics": sorted(topics - covered_topics)[:10],
            })
        return {
            "comments": top_comments(comments, self.top_k),
            "metrics": {
                "contrast_topic_coverage": topic_coverage,
                "contrast_sentence_count": len(contrast_sentences),
                "contrast_entity_count": len(covered_entities),
            },
        }
