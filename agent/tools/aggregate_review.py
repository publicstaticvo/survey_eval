from __future__ import annotations

from collections import Counter
from typing import Any


class FinalAggregate:
    """Classify module outputs into weaknesses, comments, and intermediate statistics."""

    def _iter_sections(self, paper: dict[str, Any]):
        def walk(section: dict[str, Any]):
            yield section
            for child in section.get("sections", []) or []:
                if isinstance(child, dict):
                    yield from walk(child)

        for section in paper.get("sections", []) or []:
            if isinstance(section, dict):
                yield from walk(section)

    def _iter_sentences(self, paper: dict[str, Any]):
        def walk(node: Any):
            if isinstance(node, dict):
                if "sentences" in node:
                    yield from walk(node.get("sentences", []) or [])
                    return
                for paragraph in node.get("paragraphs", []) or []:
                    yield from walk(paragraph)
                for section in node.get("sections", []) or []:
                    yield from walk(section)
            elif isinstance(node, list):
                for sentence in node:
                    if isinstance(sentence, dict) and sentence.get("environment_type", "text") == "text":
                        yield sentence

        yield from walk(paper)

    def _fact_items(self, fact_checks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        weaknesses, comments = [], []
        for item in fact_checks or []:
            judgment = item.get("judgment")
            target = weaknesses if judgment == "REFUTED" else comments if judgment == "NEUTRAL" else None
            if target is not None:
                target.append({"module": "fact.fact_check", "type": f"fact_{judgment.lower()}", **item})
        return weaknesses, comments

    def _contribution_weaknesses(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        checks = data.get("checks", []) if isinstance(data, dict) else []
        return [
            {"module": "contribution.contribution_consistent", "type": "unmet_contribution", **check}
            for check in checks
            if not check.get("consistent", True)
        ]

    def _internal_weaknesses(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        checks = data.get("checks", []) if isinstance(data, dict) else []
        return [
            {"module": "contribution.internal_consistent", "type": "section_content_inconsistent", **check}
            for check in checks
            if check.get("inconsistent") or not check.get("internal_consistent", True)
        ]

    def _missing_paper_comments(self, source_evals: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        comments, hallucinations = [], []
        grouped_prospective = source_evals.get("uncited_prospective", {}) or {}
        for claim, papers in grouped_prospective.items() if isinstance(grouped_prospective, dict) else []:
            comments.append({
                "module": "scope.missing_papers",
                "type": "uncited_prospective",
                "claim": claim,
                "alternative_papers": papers,
            })
        for item in source_evals.get("missing_papers", []) or []:
            if item.get("reason") == "uncited_prospective":
                continue
            comments.append({"module": "scope.missing_papers", "type": item.get("reason", "missing_paper"), **item})
        for entity in source_evals.get("uncited_entities", []) or []:
            if not entity.get("matched_papers"):
                hallucinations.append({
                    "module": "scope.missing_papers",
                    "type": "entity_fact_hallucination",
                    "entity_name": entity.get("entity", ""),
                })
        return comments, hallucinations

    def _topic_comments(self, topic_evals: dict[str, Any]) -> list[dict[str, Any]]:
        comments = []
        for item in topic_evals.get("missing_functional_types", []) or []:
            comments.append({"module": "scope.topic_coverage", "type": "missing_functional_type", **item})
        for item in topic_evals.get("missing_content_tags", []) or []:
            comments.append({"module": "scope.topic_coverage", "type": "missing_content_tag", **item})
        return comments

    def _statistics(self, paper: dict[str, Any], fact_checks: list[dict[str, Any]]) -> dict[str, Any]:
        sentences = list(self._iter_sentences(paper))
        sections = list(self._iter_sections(paper))
        sentence_labels = Counter(sentence.get("label", "") for sentence in sentences if sentence.get("label"))
        section_types = Counter(section.get("functional_type", "") for section in sections if section.get("functional_type"))
        content_tags = Counter(tag for section in sections for tag in (section.get("content_tags", []) or []) if tag)
        text_unlabeled = sum(1 for sentence in sentences if not sentence.get("label"))
        section_untyped = sum(1 for section in sections if not section.get("functional_type"))
        section_untagged = sum(1 for section in sections if not section.get("content_tags"))
        cited_claims = [item for item in fact_checks or [] if item.get("citation_keys")]
        background_claims = [item for item in cited_claims if item.get("label") == "BACKGROUND"]
        other_claims = [item for item in cited_claims if item.get("label") != "BACKGROUND"]
        return {
            "total_sentences": len(sentences),
            "sentence_label_counts": dict(sentence_labels),
            "text_sentence_missing_label_count": text_unlabeled,
            "total_sections": len(sections),
            "section_functional_type_counts": dict(section_types),
            "section_content_tag_counts": dict(content_tags),
            "section_missing_functional_type_count": section_untyped,
            "section_missing_content_tags_count": section_untagged,
            "total_cited_claims": len(cited_claims),
            "background_cited_claims": len(background_claims),
            "other_cited_claims": len(other_claims),
        }

    def __call__(self, result: dict[str, Any]) -> dict[str, Any]:
        preprocessing = result.get("preprocessing", {}) or {}
        evaluations = result.get("evaluations", {}) or {}
        paper = preprocessing.get("classified_paper", {})
        fact_checks = evaluations.get("fact_checks", []) or []
        source_evals = evaluations.get("source_evals", {}) or {}
        topic_evals = evaluations.get("topic_evals", {}) or {}

        weaknesses = []
        comments = []
        fact_weaknesses, fact_comments = self._fact_items(fact_checks)
        weaknesses.extend(fact_weaknesses)
        comments.extend(fact_comments)
        weaknesses.extend(self._contribution_weaknesses(evaluations.get("contribution_evals", {})))
        weaknesses.extend(self._internal_weaknesses(evaluations.get("internal_evals", {})))
        missing_comments, hallucinations = self._missing_paper_comments(source_evals)
        comments.extend(missing_comments)
        weaknesses.extend(hallucinations)
        comments.extend(self._topic_comments(topic_evals))
        return {
            "weaknesses": weaknesses,
            "comments": comments,
            "statistics": self._statistics(paper, fact_checks),
        }

