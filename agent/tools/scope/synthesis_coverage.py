from __future__ import annotations

from typing import Any

from ..utility.paper_elements import Paper
from .adequacy_common import content_topics, iter_sentences, sentence_entities, top_comments


class SynthesisCoverageAdequacy:
    def __init__(self, *_args, top_k: int = 3, **_kwargs):
        self.top_k = top_k

    async def __call__(self, paper: Paper, *_args, **_kwargs) -> dict[str, Any]:
        topics = set(content_topics(paper))
        synth = [s for s in iter_sentences(paper) if s.get("label") == "SYNTHESIS"]
        summary = [s for s in iter_sentences(paper) if s.get("label") == "SUMMARY"]
        covered_topics = set()
        covered_entities = set()
        for sent in synth:
            lowered = sent["text"].lower()
            covered_topics.update(topic for topic in topics if topic.lower() in lowered)
            covered_entities.update(sentence_entities(sent))
        topic_coverage = len(covered_topics) / len(topics) if topics else 1.0
        density = len(synth) / (len(synth) + len(summary)) if (len(synth) + len(summary)) else 0.0
        comments = []
        if topics and topic_coverage < 1.0:
            comments.append({
                "issue_type": "synthesis_coverage",
                "issue": "Some extracted survey topics are not represented in synthesis sentences.",
                "missing_topics": sorted(topics - covered_topics)[:10],
            })
        return {
            "comments": top_comments(comments, self.top_k),
            "metrics": {
                "synthesis_topic_coverage": topic_coverage,
                "synthesis_density": density,
                "synthesis_sentence_count": len(synth),
                "summary_sentence_count": len(summary),
                "synthesis_entity_count": len(covered_entities),
            },
        }
