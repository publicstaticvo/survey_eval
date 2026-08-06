from __future__ import annotations

from typing import Any

from ..utility.paper_elements import Paper
from .adequacy_common import content_topics, iter_sentences, top_comments


class GapFutureWorkAdequacy:
    def __init__(self, *_args, top_k: int = 3, **_kwargs):
        self.top_k = top_k

    async def __call__(self, paper: Paper, *_args, **_kwargs) -> dict[str, Any]:
        topics = set(content_topics(paper))
        gap_sentences = [s for s in iter_sentences(paper) if s.get("label") == "GAP"]
        anchored = []
        weak = []
        for sent in gap_sentences:
            text = sent["text"].lower()
            matched = [topic for topic in topics if topic.lower() in text]
            item = {
                "issue_type": "future_work_anchor",
                "future_work_sentence": sent["text"],
                "section": sent["section"],
                "anchor_type": "topic" if matched else "none",
                "anchor_span_in_previous_context": matched[:3],
                "judgment": "anchored" if matched else "unanchored",
            }
            (anchored if matched else weak).append(item)
        denominator = len(gap_sentences)
        coverage = len(anchored) / denominator if denominator else 0.0
        return {
            "comments": top_comments(weak, self.top_k),
            "metrics": {
                "gap_future_work_anchor_coverage": coverage,
                "gap_future_work_sentence_count": denominator,
                "gap_future_work_unanchored_count": len(weak),
            },
        }
