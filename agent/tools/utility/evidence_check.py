import re
from difflib import SequenceMatcher
from .tool_config import ToolConfig


class EvidenceCheck:
    
    def __init__(self, config: ToolConfig):
        self._mean_cov_weight = config.mean_cov_weight
        self._non_compat_punishment = config.non_compat_punishment
        self._confidence_threshold = config.confidence_threshold

    def _tokenize(self, sequence: str) -> list[str]:
        raw = re.split(r"[\s]+", sequence)
        tokens = []
        for t in raw:        
            if t := re.sub(r"^[^\w]+|[^\w]+$", "", t.lower()): tokens.append(t)
        return tokens

    def _align_anchor(self, anchor_tokens: list[str], doc_tokens: list[str]) -> dict[str, float]:
        matcher = SequenceMatcher(None, anchor_tokens, doc_tokens)
        longest_match = matcher.find_longest_match()

        match_len = longest_match.size
        coverage = match_len / len(anchor_tokens)

        return {
            "match_len": match_len,
            "coverage": coverage,
            "doc_start": longest_match.b,
            "doc_end": longest_match.b + match_len
        }

    def _max_gap_between_spans(self, spans: list[tuple[int, int]]):
        if len(spans) <= 1: return 0
        spans.sort(key=lambda x: x[0])
        return max(spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1))

    def _compute_anchor_stats(self, anchors: list[list[str]], doc_tokens: list[str], threshold: float = 0.6) -> tuple[float, float, bool] | None:
        stats = []

        for anchor in anchors:
            res = self._align_anchor(anchor, doc_tokens)
            if res["coverage"] >= threshold:
                stats.append(res)

        if not stats: return  # no valid evidence

        coverage = [x["coverage"] for x in stats]
        mean_coverage = sum(coverage) / len(coverage)
        hit_ratio = len(stats) / len(anchors)

        spans = [(x["doc_start"], x["doc_end"]) for x in stats]
        return mean_coverage, hit_ratio, self._max_gap_between_spans(spans) <= 300

    def _compute_confidence(self, mean_cov: float, hit_ratio: float, compact: bool) -> float:
        base = self._mean_cov_weight * mean_cov + (1 - self._mean_cov_weight) * hit_ratio
        if compact: return base
        return self._non_compat_punishment * base

    def verify(self, quote: list[str], source_doc: str, min_char_len: int = 20) -> tuple[bool, float]:
        quote_tokens = [self._tokenize(q) for q in quote]
        doc_tokens = self._tokenize(source_doc)

        anchors = [x for x in quote_tokens if sum(len(y) for y in x) >= min_char_len]
        if not anchors: return False, 0

        stats = self._compute_anchor_stats(anchors, doc_tokens)
        if stats is None: return False, 0

        mean_cov, hit_ratio, compact = stats
        confidence = self._compute_confidence(mean_cov, hit_ratio, compact)
        return confidence > self._confidence_threshold, confidence