from __future__ import annotations

from typing import Any

from ..utility.paper_elements import Paper
from .adequacy_common import iter_sentences, top_comments


class ScopeMethodologyAdequacy:
    def __init__(self, *_args, top_k: int = 3, **_kwargs):
        self.top_k = top_k

    async def __call__(self, paper: Paper, *_args, **_kwargs) -> dict[str, Any]:
        scope_sentences = [
            sentence
            for sentence in iter_sentences(paper)
            if sentence.get("label") in {"SCOPE", "CONTRIBUTION+SCOPE"}
        ]
        existence = 1.0 if scope_sentences else 0.0
        comments = []
        if not scope_sentences:
            comments.append({
                "issue_type": "scope_methodology_absent",
                "issue": "No explicit operational scope or inclusion/exclusion methodology was detected.",
            })
        return {
            "comments": top_comments(comments, self.top_k),
            "metrics": {"scope_methodology_existence": existence},
        }
