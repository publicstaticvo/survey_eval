from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .utility.tool_config import ToolConfig


CATEGORY_NAMES = {
    1: "internal_consistency",
    2: "hallucination",
    3: "taxonomy_structure",
    4: "argument_support",
    5: "scope",
    6: "gap_future_work",
    7: "contrast",
    8: "synthesis",
    9: "contribution_novelty",
    10: "missing_reference",
    11: "missing_topic",
}


class FinalAggregate:
    """Route detector outputs into an official review and plotting-ready summaries."""

    def __init__(self, config: ToolConfig | None = None):
        path = Path(config.adequacy_thresholds_path) if config and config.adequacy_thresholds_path else None
        self.thresholds = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}

    def _metric(self, data: Any, key: str) -> float | None:
        if not isinstance(data, dict):
            return None
        metrics = data.get("metrics", data)
        if isinstance(metrics, dict) and isinstance(metrics.get("adequacy_metrics"), dict):
            value = metrics["adequacy_metrics"].get(key)
            if isinstance(value, (int, float)):
                return float(value)
        value = metrics.get(key) if isinstance(metrics, dict) else None
        return float(value) if isinstance(value, (int, float)) else None

    def _items(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            return [item for item in value.get("comments", []) or [] if isinstance(item, dict)]
        return [item for item in value or [] if isinstance(item, dict)] if isinstance(value, list) else []

    def _finding_key(self, item: dict[str, Any]) -> tuple[Any, ...]:
        paper = item.get("paper", {})
        paper_title = paper.get("title", "") if isinstance(paper, dict) else ""
        labels = tuple(sorted(str(label) for label in item.get("implicated_labels", []) or []))
        return (
            item.get("category"),
            item.get("type") or item.get("problem_type") or item.get("issue_type"),
            item.get("section", ""),
            item.get("issue") or item.get("comment") or item.get("claim_quote", ""),
            paper_title,
            labels,
        )

    def _dedupe(self, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}
        for finding in findings:
            key = self._finding_key(finding)
            if key not in merged:
                merged[key] = finding
                continue
            existing = merged[key]
            paths = set(existing.get("detection_paths", []) or [])
            paths.add(str(existing.get("detection_path", "")))
            paths.add(str(finding.get("detection_path", "")))
            existing["detection_paths"] = sorted(path for path in paths if path)
            if finding.get("external_quote") and not existing.get("external_quote"):
                existing["external_quote"] = finding["external_quote"]
        return list(merged.values())

    def _fact_findings(self, checks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        weaknesses, comments = [], []
        for item in checks or []:
            judgment = str(item.get("judgment", "NEUTRAL")).upper()
            if judgment == "REFUTED":
                weaknesses.append({
                    "category": 2,
                    "module": "fact.fact_check",
                    "type": "factual_or_citation_hallucination",
                    **item,
                })
            elif (
                judgment == "SUPPORTED"
                and item.get("kind") == "uncited_fact_check"
                and item.get("supported_by_external_search")
            ):
                comments.append({
                    "category": 10,
                    "module": "fact.uncited_claims",
                    "type": "externally_supported_uncited_fact",
                    **item,
                })
        return weaknesses, comments

    def _integrity_findings(self, evaluations: dict[str, Any]) -> list[dict[str, Any]]:
        findings = []
        for check in evaluations.get("internal_evals", {}).get("checks", []) or []:
            if check.get("inconsistent") or not check.get("internal_consistent", True):
                findings.append({
                    "category": 1,
                    "module": "contribution.internal_consistent",
                    "type": "internal_inconsistency",
                    **check,
                })
        for check in evaluations.get("contribution_evals", {}).get("checks", []) or []:
            if not check.get("consistent", True):
                findings.append({
                    "category": 1,
                    "module": "contribution.contribution_consistent",
                    "type": "unmet_contribution",
                    **check,
                })
        for item in self._items(evaluations.get("taxonomy_evals", {})):
            if item.get("report_role") == "Weakness":
                findings.append({"category": 3, "module": "cclass.taxonomy_framework_problem", **item})
        for item in self._items(evaluations.get("evidence_support_evals", {})):
            findings.append({"category": 4, "module": "cclass.evidence_support_insufficient", **item})
        return findings

    def _sufficiency_comments(self, evaluations: dict[str, Any]) -> list[dict[str, Any]]:
        comments = []
        source = evaluations.get("source_evals", {}) or {}
        for item in source.get("missing_papers", []) or []:
            comments.append({
                "category": 10,
                "module": "scope.missing_papers",
                "type": item.get("reason", "missing_reference"),
                **item,
            })

        topic = evaluations.get("topic_evals", {}) or {}
        for item in topic.get("comments", []) or []:
            comments.append({
                "category": 11,
                "module": "scope.topic_coverage",
                "type": item.get("type", "missing_topic"),
                **item,
            })

        category_by_module = {
            "scope.scope_methodology": 5,
            "scope.gap_future_work": 6,
            "scope.contrast": 7,
            "scope.synthesis": 8,
            "scope.contribution": 9,
        }
        for item in topic.get("adequacy_comments", []) or []:
            module = item.get("module", "")
            comments.append({
                "category": category_by_module.get(module, 5),
                "type": item.get("issue_type", "sufficiency_signal"),
                **item,
            })
        return comments

    def _category_summaries(
        self,
        weaknesses: list[dict[str, Any]],
        comments: list[dict[str, Any]],
        evaluations: dict[str, Any],
    ) -> dict[str, float]:
        summaries = {
            CATEGORY_NAMES[index]: float(sum(item.get("category") == index for item in weaknesses))
            for index in range(1, 5)
        }
        topic = evaluations.get("topic_evals", {}) or {}
        metrics = topic.get("adequacy_metrics", {}) if isinstance(topic, dict) else {}
        scope_existence = self._metric(metrics, "scope_methodology_existence")
        summaries["scope"] = 1.0 - scope_existence if scope_existence is not None else 0.0
        for index in range(6, 12):
            risks = [
                float(item.get("risk", 1.0))
                for item in comments
                if item.get("category") == index
            ]
            summaries[CATEGORY_NAMES[index]] = sum(sorted(risks, reverse=True)[:3])
        return summaries

    def _integrity_cap(self, weaknesses: list[dict[str, Any]]) -> int:
        severe = sum(
            item.get("category") in {1, 2}
            or item.get("problem_type") in {"OVERLAPPING_CATEGORIES", "ARGUMENT_SUPPORT_FAILURE"}
            for item in weaknesses
        )
        other = len(weaknesses) - severe
        return max(0, 100 - 22 * severe - 8 * other)

    def _sufficiency_cap(self, summaries: dict[str, float]) -> int:
        fitted = self.thresholds.get("categories", {}) if isinstance(self.thresholds, dict) else {}
        deficits = []
        for category in range(5, 12):
            name = CATEGORY_NAMES[category]
            threshold = fitted.get(name, {}).get("threshold")
            if not isinstance(threshold, (int, float)) or threshold <= 0:
                continue
            deficits.append(min(1.0, summaries[name] / float(threshold)))
        return round(100 * (1.0 - max(deficits, default=0.0)))

    def __call__(self, result: dict[str, Any]) -> dict[str, Any]:
        evaluations = result.get("evaluations", {}) or {}
        fact_weaknesses, fact_comments = self._fact_findings(evaluations.get("fact_checks", []) or [])
        weaknesses = [
            {**item, "report_role": "Weakness"}
            for item in self._dedupe(fact_weaknesses + self._integrity_findings(evaluations))
        ]
        comments = [
            {**item, "report_role": "Comment"}
            for item in self._dedupe(fact_comments + self._sufficiency_comments(evaluations))
        ]
        summaries = self._category_summaries(weaknesses, comments, evaluations)

        integrity_cap = self._integrity_cap(weaknesses)
        sufficiency_cap = self._sufficiency_cap(summaries)
        overall_score = min(integrity_cap, sufficiency_cap)
        all_findings = weaknesses + comments
        evidence_attached = sum(
            bool(
                item.get("evidence")
                or item.get("survey_quote")
                or item.get("claim_quote")
                or item.get("external_quote")
                or item.get("verbatim_evidence")
            )
            for item in all_findings
        )
        category_counts = Counter(int(item["category"]) for item in all_findings)
        analysis = {
            "weakness_count_by_category": {
                str(index): sum(item.get("category") == index for item in weaknesses)
                for index in range(1, 12)
            },
            "comment_count_by_category": {
                str(index): sum(item.get("category") == index for item in comments)
                for index in range(1, 12)
            },
            "finding_count_by_category": {
                str(index): category_counts.get(index, 0) for index in range(1, 12)
            },
            "category_summaries": summaries,
            "evidence_attachment_rate": evidence_attached / len(all_findings) if all_findings else 1.0,
            "plot_rows": [
                {
                    "category": index,
                    "name": CATEGORY_NAMES[index],
                    "weaknesses": sum(item.get("category") == index for item in weaknesses),
                    "comments": sum(item.get("category") == index for item in comments),
                    "summary": summaries[CATEGORY_NAMES[index]],
                }
                for index in range(1, 12)
            ],
        }

        return {
            "summary": result.get("summary") or "AuditSurvey official review",
            "strengths": [] if weaknesses else ["No determinate violation was identified."],
            "weaknesses": weaknesses,
            "comments": comments,
            "overall_score": overall_score,
            "score_justification": (
                f"The optional score is capped at {integrity_cap} by determinate findings and "
                f"at {sufficiency_cap} by fitted category summaries."
            ),
            "caps": {
                "integrity_cap": integrity_cap,
                "sufficiency_cap": sufficiency_cap,
            },
            "category_summaries": summaries,
            "analysis": analysis,
        }
